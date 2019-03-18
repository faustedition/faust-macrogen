"""
Functions to build the graphs and perform their analyses.
"""
import pickle
from collections import defaultdict, Counter
from datetime import date, timedelta
from pathlib import Path
from typing import List, Any, Dict, Tuple, Union, Sequence, Optional, Set
from warnings import warn
from zipfile import ZipFile, ZIP_DEFLATED

import networkx as nx

from .graphutils import mark_edges_to_delete, remove_edges, in_path
from .bibliography import BiblSource
from .config import config
from .datings import build_datings_graph
from .fes import eades, FES_Baharev, V
from .graphutils import expand_edges, collapse_edges_by_source, add_iweight
from .igraph_wrapper import to_igraph, nx_edges
from .uris import Reference, Inscription, Witness, AmbiguousRef

logger = config.getLogger(__name__)

EARLIEST = date(1749, 8, 28)
LATEST = date.today()
DAY = timedelta(days=1)

Node = Union[date, Reference]
MultiEdge = Tuple[Node, Node, int, Dict[str, Any]]


class MacrogenesisInfo:
    """
    Results of the analysis.
    """

    def __init__(self):
        self.base: nx.MultiDiGraph = None
        self.working: nx.MultiDiGraph = None
        self.dag: nx.MultiDiGraph = None
        self.closure: nx.MultiDiGraph = None
        self.conflicts: List[MultiEdge] = []
        self.simple_cycles: Set[Sequence[Tuple[Node, Node]]] = set()

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
            method = method[0] if len(graph.edges > threshold) else method[1]

        logger.debug('Calculating MFAS for a %d-node graph using %s, may take a while', graph.number_of_nodes(), method)
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
            igraph = to_igraph(graph)
            iedges = igraph.es[igraph.feedback_arc_set(method=method, weights='weight')]
            return list(nx_edges(iedges, keys=True, data=True))

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
        datings_from_inscriptions(base)
        add_edge_weights(base)
        resolve_ambiguities(base)
        adopt_orphans(base)
        base = collapse_edges_by_source(base)
        add_iweight(base)
        working = cleanup_graph(base).copy()
        add_missing_wits(working)
        sccs = scc_subgraphs(working)

        self.base = base
        self.working = working

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

        if not nx.is_directed_acyclic_graph(result_graph):
            logger.error('After removing %d conflicting edges, the graph is still not a DAG!', len(all_feedback_edges))
            cycles = nx.simple_cycles(result_graph)
            logger.error('Counterexample cycle: %s.', next(cycles))
            # FIXME this should raise an exception
        else:
            logger.info('Double-checking removed edges ...')
            for u, v, k, attr in sorted(all_feedback_edges, key=lambda edge: edge[3].get('weight', 1), reverse=True):
                result_graph.add_edge(u, v, **attr)
                if nx.is_directed_acyclic_graph(result_graph):
                    all_feedback_edges.remove((u, v, k, attr))
                    logger.info('Added edge %s -> %s (%d) back without introducing a cycle.', u, v,
                                attr.get('weight', 1))
                else:
                    result_graph.remove_edge(u, v)

        self.dag = result_graph

        logger.info('Marking %d conflicting edges for deletion', len(all_feedback_edges))
        mark_edges_to_delete(base, all_feedback_edges)

        logger.info('Removed %d of the original %d edges', len(all_feedback_edges), len(working.edges))

        self.closure = nx.transitive_closure(result_graph)
        add_inscription_links(base)
        self._augment_details()

    def order_refs(self):
        if hasattr(self, '_order'):
            return self._order

        logger.info('Creating sort order from DAG')

        def secondary_key(node):
            if isinstance(node, Reference):
                return node.sort_tuple()
            elif isinstance(node, date):
                return node.year, format(node.month, '02d'), node.day, ''
            else:
                return 99999, "zzzzzz", 99999, "zzzzzz"

        nodes = nx.lexicographical_topological_sort(self.dag, key=secondary_key)
        refs = [node for node in nodes if isinstance(node, Reference)]
        self._order = refs
        for index, ref in enumerate(refs):
            if ref in self.base.node:
                self.base.node[ref]['index'] = index
        return refs

    def _augment_details(self):
        logger.info('Augmenting refs with data from graphs')
        for index, ref in enumerate(self.order_refs(), start=1):
            ref.index = index
            ref.rank = self.closure.in_degree(ref)
            max_before_date = max((d for d, _ in self.closure.in_edges(ref) if isinstance(d, date)),
                                  default=EARLIEST - DAY)
            max_abs_before_date = max((d for d, _ in self.dag.in_edges(ref) if isinstance(d, date)),
                                      default=None)
            ref.earliest = max_before_date + DAY
            ref.earliest_abs = max_abs_before_date + DAY if max_abs_before_date is not None else None
            min_after_date = min((d for _, d in self.closure.out_edges(ref) if isinstance(d, date)),
                                 default=LATEST + DAY)
            min_abs_after_date = min((d for _, d in self.dag.out_edges(ref) if isinstance(d, date)),
                                     default=None)
            ref.latest = min_after_date - DAY
            ref.latest_abs = min_abs_after_date - DAY if min_abs_after_date is not None else None

            if ref.earliest != EARLIEST and ref.latest != LATEST:
                avg_date = ref.earliest + (ref.latest - ref.earliest) / 2
                ref.avg_year = avg_date.year
            elif ref.latest != LATEST:
                ref.avg_year = ref.latest.year
            elif ref.earliest != EARLIEST:
                ref.avg_year = ref.earliest.year
            else:
                ref.avg_year = None

    def year_stats(self):
        years = [node.avg_year for node in self.base.nodes if hasattr(node, 'avg_year') and node.avg_year is not None]
        return Counter(years)


    def save(self, outfile: Path):
        with ZipFile(outfile, mode='w', compression=ZIP_DEFLATED) as zip:
            zip.comment = b'Macrogenesis graph dump, see https://github.com/faustedition/faust-macrogen'
            with zip.open('base.gpickle', 'w') as base_entry:
                nx.write_gpickle(self.base, base_entry)
            with zip.open('simple_cycles.pickle', 'w') as sc_entry:
                pickle.dump(self.simple_cycles, sc_entry)
            with zip.open('config.yaml', 'w') as config_entry:
                config.save_config(config_entry)



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
            if weight is 'auto':
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


def datings_from_inscriptions(base: nx.MultiDiGraph):
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
        if before and not any(isinstance(pred, date) for pred in base.predecessors(witness)):
            for d, i, k, attr in before:
                base.add_edge(d, witness, copy=(d, i, k), **attr)
        if after and not any(isinstance(succ, date) for succ in base.successors(witness)):
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


def add_missing_wits(working: nx.MultiDiGraph):
    """
    Add known witnesses that are not in the graph yet.

    The respective witnesses will be single, unconnected nodes. This doesn't help with the graph,
    but it makes these nodes appear in the topological order.
    """
    all_wits = {wit for wit in Witness.database.values() if isinstance(wit, Witness)}
    known_wits = {wit for wit in working.nodes if isinstance(wit, Witness)}
    missing_wits = all_wits - known_wits
    logger.debug('Adding %d otherwise unmentioned witnesses to the working graph', len(missing_wits))
    working.add_nodes_from(sorted(missing_wits, key=Witness.sigil_sort_key))


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

class _ConflictInfo:
    def __init__(self, graphs: MacrogenesisInfo, u: Node, v: Node):
        shortest_path = nx.shortest_path(graphs.base, v, u, weight='iweight')
        involved_cycles = {cycle for cycle in graphs.simple_cycles if in_path((u, v), cycle, True)}





def macrogenesis_graphs() -> MacrogenesisInfo:
    warn("macrogenesis_graphs() is deprecated, instantiate MacrogenesisInfo directly instead", DeprecationWarning)
    return MacrogenesisInfo()
