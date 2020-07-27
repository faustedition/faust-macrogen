"""
Functions to build the graphs and perform their analyses.
"""
import json
import logging
import pickle
import re
from collections import defaultdict, Counter
from datetime import date, timedelta
from io import TextIOWrapper
from operator import itemgetter
from pathlib import Path
from typing import List, Any, Dict, Tuple, Union, Sequence, Optional, Set, Iterable
from warnings import warn
from zipfile import ZipFile, ZIP_DEFLATED

import networkx as nx
import pandas as pd
from dataclasses import dataclass

from macrogen.graphutils import is_orphan, find_reachable_by_edge, path2str
from more_itertools import windowed

from .graphutils import mark_edges_to_delete, remove_edges, in_path, first
from .bibliography import BiblSource
from .config import config
from .datings import build_datings_graph, parse_datestr
from .graphutils import simplify_timeline
from .fes import eades, FES_Baharev, V
from .graphutils import expand_edges, collapse_edges_by_source, add_iweight
from .uris import Reference, Inscription, Witness, AmbiguousRef
from .splitgraph import references, SplitReference, Side

logger: logging.Logger = config.getLogger(__name__)

EARLIEST = date(1749, 8, 28)
LATEST = date.today()
DAY = timedelta(days=1)

Node = Union[date, Reference]
MultiEdge = Tuple[Node, Node, int, Dict[str, Any]]


class NotAcyclic(ValueError):
    pass


def check_acyclic(result_graph: nx.DiGraph, message='Graph is not acyclic'):
    """
    Checks whether the given graph is acyclic. Raises a NotAcyclic error if not.

    Args:
        result_graph: the graph to check
        message: message for the exception

    Raises:
        NotAcyclic error with a useful message

    Returns: True
    """
    if not nx.is_directed_acyclic_graph(result_graph):
        cycles = nx.simple_cycles(result_graph)
        counterexample = next(cycles)
        cyc = " → ".join(map(str, counterexample))
        msg = "{!s}\nCounterexample cycle: {!s}".format(message, cyc)
        raise NotAcyclic(msg)
    else:
        return True


_SIGIL_NORM = re.compile('[. ]+')


def _normalize_sigil(sigil: str) -> str:
    return _SIGIL_NORM.sub('', sigil).lower()


def handle_inscriptions(base):
    methods = config.inscriptions
    if not methods:
        return
    if isinstance(methods, str):
        if "," in methods:
            methods = re.split(r',\s*', methods)
        else:
            methods = [methods]

    if 'inline' in methods:
        inscriptions_inline(base)
    if 'copy' in methods:
        datings_from_inscriptions(base)
    if 'copy-orphans' in methods:
        datings_from_inscriptions(base, orphans_only=True)
    if 'orphans' in methods:
        adopt_orphans(base)


def inscriptions_inline(base):
    if any(isinstance(ref, SplitReference) for ref in base.nodes):
        split_refs = (node for node in base.nodes if isinstance(node, SplitReference))
        split_inscrs = [ref for ref in split_refs if isinstance(ref.reference, Inscription)]
        # add potentially missing nodes
        for inscr in split_inscrs:
            base.add_nodes_from(SplitReference.both(inscr.reference.witness).values())

        for split_inscr in split_inscrs:
            wit_half = SplitReference(split_inscr.reference.witness, split_inscr.side)
            if split_inscr.side == Side.START:
                base.add_edge(wit_half, split_inscr, kind='temp-pre',
                              source=BiblSource('faust://model/inscription/inline'))
            else:
                base.add_edge(split_inscr, wit_half, kind='temp-pre',
                              source=BiblSource('faust://model/inscription/inline'))
    else:
        logger.error(
                f"Ignoring inscription method 'inline' for global model '{config.model}', it is only supported for split models")


def yearlabel(max_date_before: date, min_date_after: date) -> str:
    earliest_year = max_date_before and (max_date_before + DAY).year
    latest_year = min_date_after and (min_date_after - DAY).year
    if earliest_year == latest_year:
        return earliest_year and str(earliest_year) or ""
    return " … ".join(str(year) for year in [earliest_year, latest_year] if year is not None)


class MacrogenesisInfo:
    """
    Results of the analysis.
    """

    def __init__(self, load_from: Optional[Path] = None):
        self.base: nx.MultiDiGraph = None
        self.working: nx.MultiDiGraph = None
        self.dag: nx.MultiDiGraph = None
        self.closure: nx.MultiDiGraph = None
        self.conflicts: List[MultiEdge] = []
        self.simple_cycles: Set[Sequence[Tuple[Node, Node]]] = set()
        self.order: List[Reference] = None
        self.index: Dict[Reference, int] = None
        self.details: pd.DataFrame = None

        if load_from:
            self._load_from(load_from)
        else:
            self.run_analysis()

    def feedback_arcs(self, graph: nx.MultiDiGraph, method=None, light_timeline: Optional[bool] = None):
        """
        Calculates the feedback arc set using the given method and returns a
        list of edges in the form (u, v, key, data)

        This is a wrapper that selects and calls the configured method.

        Args:
            light_timeline: different handling for the timeline nodes – they are enforced to be present, but with a light
                edge weight
            graph: NetworkX DiGraph
            method: 'eades', 'baharev', or 'ip'; if None, look at config
        """
        if method is None:
            method = config.fes_method
        if light_timeline is None:
            light_timeline = config.light_timeline
        if isinstance(method, Sequence) and not isinstance(method, str):
            try:
                threshold = config.fes_threshold
            except AttributeError:
                threshold = 64
            method = method[1] if len(graph.edges) > threshold else method[0]

        logger.info('Calculating MFAS for a %d-node graph using %s, may take a while', graph.number_of_nodes(), method)
        if method == 'eades':
            fes = eades(graph, prepare_timeline_for_keeping(graph) if light_timeline else None)
            return list(expand_edges(graph, fes))
        elif method == 'baharev':
            solver = FES_Baharev(graph, prepare_timeline_for_keeping(graph) if light_timeline else None)
            fes = solver.solve()
            self.simple_cycles |= solver.simple_cycles
            return list(expand_edges(graph, fes))
        else:
            if light_timeline:
                logger.warning('Method %s does not support lightweight timeline', method)
            try:
                from .igraph_wrapper import to_igraph, nx_edges
                igraph = to_igraph(graph)
                iedges = igraph.es[igraph.feedback_arc_set(method=method, weights='weight')]
                return list(nx_edges(iedges, keys=True, data=True))
            except ImportError as e:
                logger.critical('The method %s requires python-igraph, but it is not available: %s', method, e, exc_info=True)


    def run_analysis(self):
        """
        Runs the complete analysis by loading the data, building the graph,
        removing conflicting edges and calculating a transitive closure.

        This calls the following steps, in order:
        1. build the raw datings graph from the XML files and auxilliary information
        2. copies dating edges from inscriptions to their base witnesses (*)
        3. adds edge weights, based on bibliographic info etc.
        4. resolve links to ambiguous nodes to their referencees (*)
        5. adopt 'orphan' nodes, i.e. nodes that aren't directly mentioned but only via object model (*)
        6. collapse parallel edges from the same source
        7. add the inverse weight attribute used for some analyses

        now we have built the 'base' graph we'll work on -> attribute `base`

        8. cleanup the base graph by removing everything we don't want to be considered in the MFES
           analysis, and by adding otherwise unmentioned witnesses -> attribute `working`
        9. calculate the minimum feedback edge set for each strongly connected component and
           join those to get the one for the whole graph
        10. create a DAG by removing the MFES from the working graph -> attribute `dag`
        11. double-check the DAG is a DAG (should fail only if the method used is broken) and re-add
            edges that don't make the graph acyclic (can only happen when the method is a heuristic)

        now we have a DAG, add info back to the base graph:

        12. mark all MFES edges as deleted in the base graph
        13. add information links from inscription to witness in the base graph


        """
        base = build_datings_graph()
        if base.number_of_edges() == 0:
            raise ValueError("Loading macrogenesis data resulted in an empty graph. Is there data at {}?".format(
                    config.path.data.absolute()))
        handle_inscriptions(base)
        add_edge_weights(base)
        resolve_ambiguities(base)
        base = collapse_edges_by_source(base)
        if config.temp_syn and config.temp_syn != 'ignore':
            base = add_syn_nodes(base)
        self.base = base
        add_iweight(base)
        working = cleanup_graph(base).copy()
        self.working = working
        self.add_missing_wits(working)
        sccs = scc_subgraphs(working)

        logger.info('Calculating minimum feedback arc set for %d strongly connected components', len(sccs))

        all_feedback_edges = []
        for scc in sccs:
            feedback_edges = self.feedback_arcs(scc)
            mark_edges_to_delete(scc, feedback_edges)
            all_feedback_edges.extend(feedback_edges)

        selfloops = list(nx.selfloop_edges(working, data=True, keys=True))
        if selfloops:
            logger.warning('Found %d self loops, will also remove those. Affected nodes: %s',
                           len(selfloops), ", ".join(str(u) for u, v, k, attr in selfloops))
            all_feedback_edges.extend(selfloops)

        logger.info('Building DAG from remaining data')
        result_graph = working.copy()
        result_graph.remove_edges_from(all_feedback_edges)

        check_acyclic(result_graph, 'After removing %d conflicting edges, the graph is still not a DAG!' %
                      len(all_feedback_edges))

        logger.info('Double-checking %d removed edges ...', len(all_feedback_edges))
        for u, v, k, attr in sorted(all_feedback_edges, key=lambda edge: edge[3].get('weight', 1), reverse=True):
            result_graph.add_edge(u, v, **attr)
            if nx.is_directed_acyclic_graph(result_graph):
                all_feedback_edges.remove((u, v, k, attr))
                logger.debug('Added edge %s -> %s (%d) back without introducing a cycle.', u, v,
                             attr.get('weight', 1))
            else:
                result_graph.remove_edge(u, v)

        self.dag = result_graph

        logger.info('Marking %d conflicting edges for deletion', len(all_feedback_edges))
        mark_edges_to_delete(base, all_feedback_edges)

        logger.info('Removed %d of the original %d edges', len(all_feedback_edges), len(working.edges))
        self.conflicts = all_feedback_edges

        self.closure = nx.transitive_closure_dag(nx.DiGraph(result_graph))
        add_inscription_links(base)
        self._infer_details()

    @staticmethod
    def _secondary_key(node):
        """Returns a tuple suitable as secondary sort key for a topological sort."""
        if isinstance(node, Reference):
            return node.sort_tuple()
        elif isinstance(node, date):
            return node.year, format(node.month, '02d'), node.day, ''
        else:
            return 99999, "zzzzzz", 99999, "zzzzzz"

    def baseline_order(self) -> pd.Series:
        """
        Sorts the references in the base graph using _only_ the immanent sort key,
        ignoring the graph topology.

        Returns:
            a series mapping References to unique integers as rank numbers

        See Also:
            spearman_rank_correlation
        """
        if any(isinstance(node, SplitReference) for node in self.dag.nodes):
            unsorted_refs = [ref.reference for ref in self.dag.nodes
                             if isinstance(ref, SplitReference) and ref.side == Side.END]
        else:
            unsorted_refs = [node for node in self.dag.nodes if isinstance(node, Reference)]
        refs = sorted(unsorted_refs, key=lambda ref: ref.sort_tuple()[1:])
        return pd.Series(index=refs, data=range(1, len(refs) + 1))

    def spearman_rank_correlation(self) -> float:
        """
        Calculates Spearman’s Rank Correlation Coefficient between a baseline ordering (that ignores
        graph topology) and the topological ordering evoked by the model using its current settings.

        Returns:
            a float between -1 and +1. +1 would mean the ordering hasn’t changed at all by
            the model, -1 means a complete permutation.
        """
        model_order: pd.Series = self.details.position
        baseline_order = self.baseline_order()
        n = model_order.index.size
        if n != baseline_order.index.size:
            delta = model_order.index.symmetric_difference(baseline_order.index)
            logger.error("Warning: Baseline order has %d elements while model order has %d: Δ = %s.",
                         baseline_order.size, n, delta)
        Q = ((model_order - baseline_order) ** 2).sum()
        r_s = 1 - (6 * Q) / (n * (n - 1) * (n + 1))
        return r_s

    def add_missing_wits(self, working: nx.MultiDiGraph):
        """
        Add known witnesses that are not in the graph yet.

        The respective witnesses will be single, unconnected nodes. This doesn't help with the graph,
        but it makes these nodes appear in the topological order.
        """
        all_wits = {wit for wit in Witness.database.values() if isinstance(wit, Witness)}
        if any(isinstance(ref, SplitReference) for ref in working.nodes):
            known_wits = {ref for ref in references(working) if isinstance(ref, Witness)}
            mentioned_refs = {ref.reference for ref in self.base.nodes if isinstance(ref, SplitReference)}
            inscription_bases = {inscr.witness for inscr in mentioned_refs if isinstance(inscr, Inscription)}
            missing_wits = (all_wits | mentioned_refs | inscription_bases) - known_wits
            for wit in sorted(missing_wits, key=lambda ref: ref.sort_tuple()):
                working.add_nodes_from(SplitReference.both(wit).values())
        else:
            known_wits = {wit for wit in working.nodes if isinstance(wit, Witness)}
            mentioned_refs = {ref for ref in self.base.nodes if isinstance(ref, Reference)}
            inscription_bases = {inscr.witness for inscr in mentioned_refs if isinstance(inscr, Inscription)}
            missing_wits = (all_wits | mentioned_refs | inscription_bases) - known_wits
            working.add_nodes_from(sorted(missing_wits, key=lambda ref: ref.sort_tuple()))
        logger.info('Adding %d otherwise unmentioned references to the working graph', len(missing_wits))

    def order_refs(self) -> List[Reference]:
        if self.order:
            return self.order

        logger.info('Creating sort order from DAG')
        nodes = list(nx.lexicographical_topological_sort(self.dag, key=self._secondary_key))
        refs = [node for node in nodes if isinstance(node, Reference)]
        self.order = refs
        self.order_all_nodes = nodes
        self._build_index()
        for ref, index in self.index.items():
            if ref in self.base.nodes:
                self.base.nodes[ref]['index'] = index
            ref.index = index
        return refs

    def order_refs_post_model(self) -> List[Reference]:
        refs = self.order_refs()
        if config.model in ['split', 'split-reverse']:
            refs = [ref for ref in refs if ref.side == Side.END]
        return refs

    def _build_index(self):
        refs_diff = len(self.order) - len(set(self.order))
        if refs_diff > 0:
            refs_counts = Counter(self.order)
            msg = ", ".join(f"{ref} ({n})" for ref, n in refs_counts.most_common(refs_diff))
            logger.error('Some refs appear more than once in the order: %s', msg)
        self.index = {ref: i for (i, ref) in enumerate(self.order, start=1)}

    def _infer_details(self):
        """Prepares a table with details on witnesses"""
        logger.info('Preparing details on references')
        ordered_ref_nodes = self.order_refs()
        is_split = any(isinstance(node, SplitReference) for node in ordered_ref_nodes)
        self.is_split = is_split
        if is_split:
            refs_from_graphs = [ref for ref in ordered_ref_nodes if ref.side == Side.END]  # FIXME Configurable?
            refs_from_data = [ref.reference for ref in refs_from_graphs]
            total_positions = {ref: pos for pos, ref in enumerate(ordered_ref_nodes, start=1)}
            start_positions = {ref.reference: pos for ref, pos in total_positions.items() if ref.side == Side.START}
            end_positions = {ref.reference: pos for ref, pos in total_positions.items() if ref.side == Side.END}
        else:
            refs_from_graphs = ordered_ref_nodes
            refs_from_data = ordered_ref_nodes

        table = pd.DataFrame(data=dict(uri=[ref.uri for ref in refs_from_data],
                                       label=[ref.label for ref in refs_from_data],
                                       kind=[type(ref).__name__ for ref in refs_from_data],
                                       inscription_of=[ref.witness if isinstance(ref, Inscription) else None
                                                       for ref in refs_from_data],
                                       position=list(range(1, len(refs_from_data) + 1))),
                             index=refs_from_data)
        if is_split:
            table['start_pos'] = pd.Series(start_positions)
            table['end_pos'] = pd.Series(end_positions)

            def unsplit_rank(node) -> int:
                preds = self.closure.predecessors(node)
                pred_cleaned = {pred.reference if isinstance(pred, SplitReference) else pred
                                for pred in preds}
                if isinstance(node, SplitReference):
                    pred_cleaned -= {node.reference}
                else:
                    pred_cleaned -= {node}
                return len(pred_cleaned)

            table['rank'] = [unsplit_rank(ref) for ref in refs_from_graphs]

        else:
            table['rank'] = [self.closure.in_degree(ref) for ref in refs_from_graphs]
        for ref in refs_from_data:
            max_before = max(
                    (d for d, _ in self.closure.in_edges(SplitReference(ref, Side.START) if is_split else ref)
                     if isinstance(d, date)), default=None)
            max_abs_before = max((d for d, _ in self.dag.in_edges(SplitReference(ref, Side.START) if is_split else ref)
                                  if isinstance(d, date)), default=None)
            min_after = min((d for _, d in self.closure.out_edges(SplitReference(ref, Side.END) if is_split else ref)
                             if isinstance(d, date)), default=None)
            min_abs_after = min((d for _, d in self.dag.out_edges(SplitReference(ref, Side.END) if is_split else ref)
                                 if isinstance(d, date)), default=None)
            avg = max_before + (min_after - max_before) / 2 if max_before and min_after else max_before or min_after
            table.loc[ref, 'max_before_date'] = max_before
            table.loc[ref, 'max_abs_before_date'] = max_abs_before
            table.loc[ref, 'min_after_date'] = min_after
            table.loc[ref, 'min_abs_after_date'] = min_abs_after
            table.loc[ref, 'avg'] = avg
            table.loc[ref, 'avg_year'] = avg and avg.year
            table.loc[ref, 'yearlabel'] = yearlabel(max_before, min_after)
        table['baseline_position'] = self.baseline_order()
        self.details = table

    def find_conflicts(self, from_: Node, to_: Node) -> List[Tuple[Node, Node, int, dict]]:
        """
        Finds conflict edges from a from_ node to a to_ node

        Args:
            from_: source node, never a `SplitReference`
            to_: end node, never a `SplitReference`

        Returns:
            All info on the actual conflicting edges
        """
        if self.is_split:
            return [(u, v, k, attr) for (u, v, k, attr) in self.conflicts
                    if (u == from_ or isinstance(u, SplitReference) and u.reference == from_)
                    and (v == to_ or isinstance(v, SplitReference) and v.reference == to_)]
        else:
            return [(u, v, k, attr) for (u, v, k, attr) in self.conflicts if u == from_ and v == to_]

    def year_stats(self):
        stats: pd.Series = self.details.avg_year.value_counts()
        stats.index = pd.Int64Index(stats.index)
        return Counter(stats.to_dict())

    def conflict_stats(self):
        return [_ConflictInfo(self, edge) for edge in self.conflicts]

    def save(self, outfile: Path):
        with ZipFile(outfile, mode='w', compression=ZIP_DEFLATED) as zip:
            zip.comment = b'Macrogenesis graph dump, see https://github.com/faustedition/faust-macrogen'
            with zip.open('base.gpickle', 'w') as base_entry:
                nx.write_gpickle(self.base, base_entry)
            with zip.open('simple_cycles.pickle', 'w') as sc_entry:
                pickle.dump(self.simple_cycles, sc_entry)
            with zip.open('ref_info.csv', 'w') as detail_entry:
                text = TextIOWrapper(detail_entry, encoding='utf-8')
                ref_info = self.details.set_index('uri')
                ref_info.inscription_of.apply(lambda ref: ref and ref.uri)
                ref_info.to_csv(text, date_format='iso')
            with zip.open('order.json', 'w') as order_entry:
                text = TextIOWrapper(order_entry, encoding='utf-8')
                json.dump([ref.uri for ref in self.order], text, indent=True)
                text.flush()
            with zip.open('config.yaml', 'w') as config_entry:
                config.save_config(config_entry)
            with zip.open('base.yaml', 'w') as base_entry:
                text = TextIOWrapper(base_entry, encoding='utf-8')
                nx.write_yaml(self.base, text)

    def _load_from(self, load_from: Path):
        # Load base graph and simple cycles:
        logger.info('Loading macrogenesis graphs from %s', load_from)
        with ZipFile(load_from, mode='r') as zip:
            with zip.open('base.gpickle', 'r') as base_entry:
                self.base = nx.read_gpickle(base_entry)
            with zip.open('simple_cycles.pickle', 'r') as sc_entry:
                self.simple_cycles = pickle.load(sc_entry)
            # with zip.open('order.json', 'r') as order_entry:
            #     text = TextIOWrapper(order_entry, encoding='utf-8')
            #     uris = json.load(text)
            #     self.order = [Witness.get(uri) for uri in uris]
            #     self._build_index()

        # Now reconstruct the other data:
        self.working: nx.MultiDiGraph = self.base.copy()
        self.working = cleanup_graph(self.working).copy()
        self.add_missing_wits(self.working)
        self.dag = self.working.copy()

        for u, v, k, attr in self.working.edges(keys=True, data=True):
            try:
                if attr.get('delete', False):
                    self.conflicts.append((u, v, k, attr))
                    self.dag.remove_edge(u, v, k)
                if u is v:
                    self.dag.remove_edge(u, v)
                if attr.get('ignore', False):
                    self.dag.remove_edge(u, v, k)
                if attr.get('kind') in {'orphan', 'inscription'}:
                    self.dag.remove_edge(u, v, k)
            except nx.NetworkXError as e:
                logger.info('Could not remove %s→%s (%d): %s', u, v, k, e)
        check_acyclic(self.dag,
                      f'Base graph from {load_from} is not acyclic after removing conflicting and ignored edges.')
        self.order_refs()
        logger.info('Preparing transitive closure')
        self.closure = nx.transitive_closure_dag(self.dag, self.order_all_nodes)
        self._infer_details()
        logger.info('Finished loading model from %s: %d nodes, %d conflicts', load_from, len(self.order), len(self.conflicts))

    def node(self, spec: Union[Reference, date, str], default=KeyError):
        """
        Returns a node from the graph.
        Args:
            spec: A reference to the node. Can be a node or label or uri.

        Returns:
            a single node.

        Raises:
            KeyError if no node can be found
        """

        try:
            if isinstance(spec, Reference) or isinstance(spec, date):
                return first((node for node in self.base.nodes
                             if node == spec or isinstance(node, SplitReference) and node.reference == spec), checked=True)
            try:
                d = parse_datestr(spec)
                return first((node for node in self.base.nodes if node == d), checked=True)
            except ValueError:
                pass
            except KeyError:
                pass
            except StopIteration:
                pass

            if spec.startswith('faust://'):
                ref = Witness.get(spec)
                return first((node for node in self.base.nodes
                             if node == ref or isinstance(node, SplitReference) and node.reference == ref), checked=True)
            else:
                norm_spec = _normalize_sigil(spec)
                return first((ref for ref in self.base.nodes if isinstance(ref, Reference)
                             and (ref.uri == 'faust://document/faustedition/' + spec
                                  or _normalize_sigil(ref.label) == norm_spec)), checked=True)

        except IndexError:
            if default is KeyError:
                raise KeyError("No node matching {!r} in the base graph.".format(spec))
            else:
                return default

    def nodes(self, node_str: str, check: bool = False, report_errors: bool = False) -> Union[
        List[Node], Tuple[List[Node], List[str]]]:
        """
        Find nodes for a comma-separated list of node strings.

        Args:
            node_str:
            check: if true, raise exception when a node has not been found
            report_errors: if True, return an additional list of strings that couldn’t be resolved

        See also:
            MacrogenesisInfo.node

        Returns:
            unless report_errors, a list of Nodes (i.e. each item is either a Reference or a date).
            Otherwise, a pair: first a list of Nodes, then a list of strings which could not be resolved; both possibly empty.

        """
        nodes: List[Node] = []
        errors: List[str] = []
        if node_str:
            for node_spec in node_str.split(','):
                try:
                    stripped = node_spec.strip()
                    nodes.append(self.node(stripped))
                except KeyError:
                    if report_errors:
                        errors.append(node_spec)
                    if check:
                        raise
                    else:
                        logger.warning('Requested node %s not found in the graph', stripped)
        if report_errors:
            return nodes, errors
        else:
            return nodes

    def add_path(self, graph: nx.MultiDiGraph, source: Node, target: Node, weight='iweight', method='dijkstra',
                 must_exist=False, edges_from: Optional[nx.MultiDiGraph] = None):
        """
        Finds the shortest path from source to target in the base graph and adds it to the given graph.

        Args:
            graph: The graph in which to add the path. This is modified, of course.
            source: source node
            target: target node
            weight: Attribute name to be used as weight
            method: see `nx.shortest_path`
            must_exist: if True, raise an exception if there is no path from source to target

        Returns:
            the path as list of nodes, if any.
        """
        try:
            if edges_from is None:
                edges_from = self.dag
            path = nx.shortest_path(edges_from, source, target, weight, method)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('Shortest path from %s to %s: %s', source, target, path2str(path))
            edges = list(expand_edges(edges_from, nx.utils.pairwise(path), filter=True))
            graph.add_edges_from(edges)
            return path
        except nx.NetworkXException as e:
            if must_exist:
                raise e
            else:
                logger.debug('No path from %s to %s', source, target)

    def subgraph(self, *nodes: Node, context: bool = True, path_to: Iterable[Node] = {}, abs_dates: bool = True,
                 path_from: Iterable[Node] = {}, paths: Iterable[Node] = {}, paths_without_timeline: bool = False,
                 paths_between_nodes: Union[bool, str] = 'all', keep_timeline: bool = False, direct_assertions: bool = False,
                 temp_syn_context: bool = False, include_syn_clusters: bool = False, include_inscription_clusters = False) \
            -> nx.MultiDiGraph:
        """
        Extracts a sensible subgraph from the base graph.

        Args:
            *nodes: node or nodes to include  in the graph
            context: If true, we add the context of the given node (or nodes), i.e. the predecessors and successors from
                     the dag
            path_to: Nodes to which the shortest path should be included, if any
            path_from: Node from which the shortest path should be included, if any
            paths: Node(s) from / to which the spp should be included, if any
            paths_without_timeline: Paths should be built without considering the timeline
            paths_between_nodes: If `'all'` or `True` (the default), include the shortest paths between the given nodes
                                on the base graph. If `'dag'`, only allow assertions from the dag.
            keep_timeline: if True, keep the timeline as is, otherwise remove useless date nodes
            direct_assertions: if True, include all edges induced by the given node(s)

        Description:
            This method can be used to extract an 'interesting' subgraph around one or more nodes from the base
            graph. The resulting graph is constructed as follows:

            1. The given nodes in the positional arguments are added to the set of relevant nodes.
            2. If `context` is True, the set of relevant nodes is extended by all direct neighbours in the base graph.
            3. The subgraph induced by the relevant nodes is extracted.
            4. If `abs_dates` is True, for each node in nodes, we look for the closest earlier and later dating node.
               If it is not present, we add the shortest path to it.
            5. We add the shortest paths from all nodes in path_from and pathes to each node in nodes.
            6. We add the shortest paths from each node in nodes to each node in pathes and path_from.

        Returns:
            The constructed subgraph
        """
        if any(isinstance(node, SplitReference) for node in self.base.nodes):
            central_nodes = set()
            for graph_node in self.base:
                if graph_node in nodes or isinstance(graph_node, SplitReference) and graph_node.reference in nodes:
                    central_nodes.add(graph_node)
                    if isinstance(graph_node, SplitReference):
                        central_nodes.add(graph_node.other)
        else:
            central_nodes = set(nodes)
        relevant_nodes = set(central_nodes)
        if context:
            for node in central_nodes:
                relevant_nodes |= set(self.dag.pred[node]).union(self.dag.succ[node])

        if include_syn_clusters:
            for node in central_nodes:
                cluster = find_reachable_by_edge(self.base, node, 'kind', 'temp-syn')
                logger.info("syn-cluster: including %s", cluster)
                relevant_nodes |= cluster

        if temp_syn_context:
            for node in set(relevant_nodes):
                if isinstance(node, SynAnchor):
                    relevant_nodes |= set(self.dag.pred[node]).union(self.dag.succ[node]).union(node.syn_group)

        if include_inscription_clusters:
            for node in set(central_nodes):
                relevant_nodes |= find_reachable_by_edge(self.base, node, 'kind', 'inscription', symmetric=True)

        subgraph: nx.MultiDiGraph = nx.subgraph(self.base, relevant_nodes).copy()
        sources = set(path_from).union(paths)
        targets = set(path_to).union(paths)

        if paths_without_timeline:
            path_base = remove_edges(self.dag, lambda u, v, attr: attr.get('kind') == 'timeline')
        else:
            path_base = self.dag

        for node in central_nodes:
            if abs_dates:
                try:
                    prev = max((d for d in self.closure.pred[node] if isinstance(d, date)), default=None)
                    if prev is not None and prev not in central_nodes:
                        self.add_path(subgraph, prev, node, edges_from=self.dag)
                    next_ = min((d for d in self.closure.succ[node] if isinstance(d, date)), default=None)
                    if next_ is not None and next not in central_nodes:
                        self.add_path(subgraph, node, next_, edges_from=self.dag)
                except KeyError as e:
                    logger.warning('%s is not in closure, so no date justification', node)

            for source in sources:
                self.add_path(subgraph, source, node, edges_from=path_base)
            for target in targets:
                self.add_path(subgraph, node, target, edges_from=path_base)
            if paths_between_nodes in {True, 'dag', 'all'}:
                inbetween_base = self.dag if paths_between_nodes == 'dag' else self.base
                for other in central_nodes:
                    if other != node:
                        self.add_path(subgraph, node, other, edges_from=inbetween_base)
                        self.add_path(subgraph, other, node, edges_from=inbetween_base)

            if direct_assertions:
                subgraph.add_edges_from(self.base.in_edges(node, keys=True, data=True))
                subgraph.add_edges_from(self.base.out_edges(node, keys=True, data=True))

        # make sure the central nodes actually are in the subgraph
        subgraph.add_nodes_from(central_nodes)

        if not keep_timeline:
            subgraph = simplify_timeline(subgraph)

        return subgraph


    def order_graph(self, graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
        """
        This returns a version of the given graph that will be ordered left to right by sort order when layed out
        using dot.
        """
        # noinspection PyTypeChecker  -- graph.copy() will return the correct type
        result: nx.MultiDiGraph = graph.copy()
        ref_order = [ref for ref in self.order_refs() if ref in result.nodes]
        for u, v, k, attr in result.edges(keys=True, data=True):
            attr['weight'] = 0

        if len(ref_order) > 1:
            for u, v in windowed(ref_order, 2):
                if result.has_edge(u, v, 0):
                    result[u][v][0]['weight'] = 100
                    result[u][v][0]['topo'] = True
                else:
                    result.add_edge(u, v, penwidth=0, dir='none', weight=100, topo=True)
        return result






def macrogenesis_graphs() -> MacrogenesisInfo:
    warn("macrogenesis_graphs() is deprecated, instantiate MacrogenesisInfo directly instead", DeprecationWarning)
    return MacrogenesisInfo()


def scc_subgraphs(graph: nx.MultiDiGraph) -> List[nx.MultiDiGraph]:
    """
    Extracts the smallest conflicted subgraphs of the given graph, i.e. the
    non-trivial (more than one node) strongly connected components.

    Args:
        graph: the base graph, or some modified version of it

    Returns:
        List of subgraphs, ordered by number of nodes. Note the subgraphs
        are views on the original graph
    """
    sccs = [scc for scc in nx.strongly_connected_components(graph) if len(scc) > 1]
    by_node_count = sorted(sccs, key=len)
    logger.info('Extracted %d subgraphs with conflicts', len(by_node_count))
    return [nx.subgraph(graph, scc_nodes) for scc_nodes in by_node_count]


# def analyse_conflicts(graph):
#     """
#     Dumps some statistics on the conflicts in the given graph.
#
#     Args:
#         graph:
#
#
#     Todo: is this still up to date?
#     """
#
#     conflicts_file_name = 'conflicts.tsv'
#     with open(conflicts_file_name, "wt") as conflicts_file:
#         writer = csv.writer(conflicts_file, delimiter='\t')
#         writer.writerow(
#                 ['Index', 'Size', 'References', 'Edges', 'Sources', 'Types',
#                  'Nodes'])
#         for index, subgraph in enumerate(scc_subgraphs(graph), start=1):
#             nodes = subgraph.nodes
#             size = subgraph.number_of_nodes()
#             refs = len([node for node in nodes if isinstance(node, Reference)])
#             if size > 1:
#                 logger.debug('  - Subgraph %d, %d refs', index, refs)
#                 edges_to_remove = feedback_arcs(subgraph)
#                 edge_count = len(subgraph.edges)
#                 sources = {str(attr['source'].uri) for u, v, attr in subgraph.edges.data() if 'source' in attr}
#                 node_types = {str(attr['kind']) for u, v, attr in subgraph.edges.data()}
#                 writer.writerow(
#                         [index, size, refs, edge_count, ", ".join(sources), ", ".join(node_types),
#                          " / ".join(map(str, nodes))])
#                 conflicts_file.flush()
#                 mark_edges_to_delete(subgraph, edges_to_remove)
#     return [('List of conflicts', conflicts_file_name)]


def prepare_timeline_for_keeping(graph: nx.MultiDiGraph, weight=0.1) -> List[Tuple[V, V]]:
    result = []
    for u, v, k, attr in graph.edges(keys=True, data=True):
        if attr['kind'] == 'timeline':
            result.append((u, v))
            if weight == 'auto':
                attr['weight'] = (v - u).days / 365.25
            else:
                attr['weight'] = weight
    return result


def add_edge_weights(graph: nx.MultiDiGraph):
    """Adds a 'weight' attribute, coming from the node kind or the bibliography, to the given graph"""
    for u, v, k, data in graph.edges(data=True, keys=True):
        if 'weight' not in data:
            if data['kind'] == 'timeline':
                data['weight'] = 0.00001 if config.light_timeline else 2 ** 31
            if 'source' in data:
                data['weight'] = data['source'].weight


def resolve_ambiguities(graph: nx.MultiDiGraph):
    """
    Replaces ambiguous refs with the referenced nodes, possibly duplicating edges.

    Args:
        graph: The graph in which we work (inplace)
    """
    ambiguities = [node for node in graph.nodes if isinstance(node, AmbiguousRef)]
    for ambiguity in ambiguities:
        for u, _, k, attr in list(graph.in_edges(ambiguity, keys=True, data=True)):
            for witness in ambiguity.witnesses:
                attr['from_ambiguity'] = ambiguity
                graph.add_edge(u, witness, k, **attr)
            graph.remove_edge(u, ambiguity, k)
        for _, v, k, attr in list(graph.out_edges(ambiguity, keys=True, data=True)):
            for witness in ambiguity.witnesses:
                attr['from_ambiguity'] = ambiguity
                graph.add_edge(witness, v, k, **attr)
            graph.remove_edge(ambiguity, v, k)
        graph.remove_node(ambiguity)


def datings_from_inscriptions(base: nx.MultiDiGraph, orphans_only=False):
    """
    Copy datings from inscriptions to witnesses.

    Args:
        base:

    Returns:

    """
    logger.info('Copying datings from inscriptions to witnesses')
    inscriptions_by_wit: Dict[Witness, List[Inscription]] = defaultdict(list)
    for inscription in [node for node in base.nodes if isinstance(node, Inscription)]:
        if inscription.witness in base.nodes:
            inscriptions_by_wit[inscription.witness].append(inscription)
    for witness, inscriptions in inscriptions_by_wit.items():
        iin = [edge for i in inscriptions for edge in base.in_edges(i, data=True, keys=True)]
        before = [edge for edge in iin if isinstance(edge[0], date)]
        iout = [edge for i in inscriptions for edge in base.out_edges(i, data=True, keys=True)]
        after = [edge for edge in iout if isinstance(edge[1], date)]
        if before and not any(isinstance(pred, date) for pred in base.predecessors(witness)) and not (orphans_only and not is_orphan(witness, base)):
            for d, i, k, attr in before:
                base.add_edge(d, witness, copy=(d, i, k), **attr)
        if after and not any(isinstance(succ, date) for succ in base.successors(witness)) and not (orphans_only and not is_orphan(witness, base)):
            for i, d, k, attr in after:
                base.add_edge(witness, d, copy=(d, i, k), **attr)


def adopt_orphans(graph: nx.MultiDiGraph):
    """
    Introduces auxilliary edges to witnesses that are referenced by an inscription or ambiguous ref, but are not
    used otherwise in the graph.
    """
    nodes = set(graph.nodes)
    for node in nodes:
        if isinstance(node, Inscription):
            if node.witness not in nodes and isinstance(node.witness, Witness):
                graph.add_edge(node, node.witness, kind='orphan', source=BiblSource('faust://orphan/adoption'),
                               comments=(), xml='')
                logger.info('Adopted %s from inscription %s', node.witness, node)
        if isinstance(node, AmbiguousRef):
            for witness in node.witnesses:
                if witness not in nodes:
                    graph.add_edge(node, witness, kind='orphan', source=BiblSource('faust://orphan/adoption'),
                                   comments=(), xml='')
                    logger.info('Adopted %s from ambiguous ref %s', witness, node)


def add_inscription_links(base: nx.MultiDiGraph):
    """
    Add an edge from each inscription to its parent witness.
    """
    for node in list(base.nodes):
        if isinstance(node, Inscription):
            base.add_edge(node, node.witness, kind='inscription', source=BiblSource('faust://model/inscription'))


def cleanup_graph(A: nx.MultiDiGraph) -> nx.MultiDiGraph:
    logger.info('Removing edges to ignore')

    def zero_weight(u, v, attr):
        return attr.get('weight', 1) == 0

    def is_syn(u, v, attr):
        return attr['kind'] == 'temp-syn'

    def is_ignored(u, v, attr):
        return attr.get('ignore', False)

    for u, v, k, attr in A.edges(keys=True, data=True):
        if zero_weight(u, v, attr) or is_syn(u, v, attr):
            attr['ignore'] = True

    return remove_edges(A, is_ignored)


@dataclass(frozen=True, order=True)
class SynAnchor:
    syn_group: frozenset
    side: Side

    def __str__(self):
        return f"{self.side.label}({', '.join(ref.label for ref in self.syn_group)})"


def add_syn_nodes(source_graph: nx.MultiDiGraph, mode: Optional[str] = None) -> nx.MultiDiGraph:
    """
    Creates a copy of the graph with appropriate handling of temp-syn edges.

    An edge u –temp-syn→ v models that u and v have been written approximately at the same time. Semantically,
    these are symmetric (or undirected) edges, so all nodes connected via (only) temp-syn edges form a cluster
    (more formally, these are the non-trivial weakly connected components of the subgraph induced by the temp-syn
    edges).

    This function places two artificial nodes of class `SynAnchor` before and after each cluster, connecting the
    SynAnchor nodes to the `Reference`s in the cluster using regular ``temp-pre`` edges, and connects the SynAnchor
    nodes using the given modes:

    - ``ignore``: No SynAnchors, just ignore the nodes completely
    - ``copy``: Copy all in-edges that come from nodes not in the cluster to the anchor before, and all out-edges that
       connect to nodes outside the cluster to the ancher after the cluster.
    - ``closest``: Connect the anchors to the closest date nodes of any of the witnesses in the group
    - ``farthest``: Connect the anchors to the farthest date nodes of any of the witnesses in the group

    Args:
        source_graph: The graph to work on. Is not modified
        mode: ignore, copy, closest or farthest; if None, take the value from the config option ``temp_syn``

    Returns:
        a modified copy of the input graph

    """
    if mode is None:
        mode = config.temp_syn
    if not mode or mode == 'ignore':
        logger.warning('No temp-syn handling (mode==%s)', mode)
        return source_graph

    syn_groups = temp_syn_groups(source_graph)
    logger.info('Adding temp-syn nodes in mode %s for %d clusters', mode, len(syn_groups))
    result = source_graph.copy()
    for component in syn_groups:
        syn_group: Set[Reference] = frozenset(component)
        in_edge_view = source_graph.in_edges(nbunch=syn_group, keys=True, data=True)
        out_edge_view = source_graph.out_edges(nbunch=syn_group, keys=True, data=True)
        in_edges = [(u, v, k, attr) for u, v, k, attr in in_edge_view if u not in syn_group]
        out_edges = [(u, v, k, attr) for u, v, k, attr in out_edge_view if v not in syn_group]
        before = SynAnchor(syn_group, Side.START)
        after = SynAnchor(syn_group, Side.END)
        for ref in syn_group:
            result.add_edge(before, ref, kind='temp-pre', source=BiblSource('faust://temp-syn'))  # TODO add orig source
            result.add_edge(ref, after, kind='temp-pre', source=BiblSource('faust://temp-syn'))

            if mode == 'copy':
                result.add_edges_from([(u, before, k, attr) for u, v, k, attr in in_edges])
                result.add_edges_from([(after, v, k, attr) for u, v, k, attr in out_edges])
            elif mode == 'closest':
                closest_before = max([edge for edge in in_edges if isinstance(edge[0], date)],
                                     key=itemgetter(0), default=None)
                closest_after = min([edge for edge in out_edges if isinstance(edge[0], date)],
                                    key=itemgetter(0), default=None)
                if closest_before:
                    result.add_edge(closest_before[0], before, **closest_before[-1])
                if closest_after:
                    result.add_edge(after, closest_after[1], **closest_after[-1])
            elif mode == 'farthest':
                farthest_before = min([edge for edge in in_edges if isinstance(edge[0], date)],
                                      key=itemgetter(0), default=None)
                farthest_after = max([edge for edge in out_edges if isinstance(edge[0], date)],
                                     key=itemgetter(0), default=None)
                if farthest_before:
                    result.add_edge(farthest_before[0], before, **farthest_before[-1])
                if farthest_after:
                    result.add_edge(after, farthest_after[1], **farthest_after[-1])

    return result


def temp_syn_groups(source_graph: nx.MultiDiGraph):
    syn_only = source_graph.edge_subgraph((u, v, k) for (u, v, k, kind) in source_graph.edges(keys=True, data='kind')
                                          if kind == 'temp-syn')
    syn_groups = [comp for comp in nx.weakly_connected_components(syn_only) if len(comp) > 1]
    return syn_groups


class _ConflictInfo:
    def __init__(self, graphs: MacrogenesisInfo, edge: MultiEdge):
        self.u, self.v, self.k, self.attr = edge
        self.parallel_edges = list(expand_edges(graphs.base, [(self.u, self.v)]))
        self.conflict_weight = self.attr.get('weight', 0)
        self.shortest_path = nx.shortest_path(graphs.base, self.v, self.u, weight='iweight')
        self.sp_weight = sum(attr.get('weight', 0) for u, v, k, attr in expand_edges(graphs.base,
                                                                                     nx.utils.pairwise(
                                                                                             self.shortest_path,
                                                                                             cyclic=True)))
        self.involved_cycles = {cycle for cycle in graphs.simple_cycles if in_path((self.u, self.v), cycle, True)}
        counter_edges = {edge for cycle in self.involved_cycles for edge in nx.utils.pairwise(cycle)} - {
            (self.u, self.v)}
        self.removed_sources = {attr['source'] for u, v, k, attr in self.parallel_edges}

    def stat(self):  # TODO better name
        return dict(
                u=self.u,
                v=self.v,
                edges=len(self.removed_edges),
                conflict_weight=self.conflict_weight,
                sp_weight=self.sp_weight,
                involved_cycles=len(self.involved_cycles),
                removed_source_count=len(self.removed_sources)
        )

