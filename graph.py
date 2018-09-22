import csv
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import List, Callable, Any, Dict, Tuple, Union

import dateutil
import networkx as nx
from dataclasses import dataclass

from datings import base_graph, BiblSource, parse_datestr
from faust_logging import logging
from igraph_wrapper import to_igraph, nx_edges
from uris import Reference, Inscription, Witness, AmbiguousRef

logger = logging.getLogger(__name__)

EARLIEST = date(1749, 8, 28)
LATEST = date.today()
DAY = timedelta(days=1)


def pathlink(*nodes) -> Path:
    node_names: List[str] = []
    for node in nodes:
        if isinstance(node, str):
            if node.startswith('faust://'):
                node = Witness.get(node)
            else:
                try:
                    node = parse_datestr(node)
                except ValueError:
                    pass

        if isinstance(node, Reference):
            node_names.append(node.filename.stem)
        elif isinstance(node, date):
            node_names.append(node.isoformat())
        else:
            logger.warning('Unknown node type: %s (%s)', type(node), node)
            node_names.append(str(hash(node)))
    return Path("--".join(node_names) + '.php')


def subgraphs_with_conflicts(graph: nx.MultiDiGraph) -> List[nx.MultiDiGraph]:
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


def analyse_conflicts(graph):
    conflicts_file_name = 'conflicts.tsv'
    with open(conflicts_file_name, "wt") as conflicts_file:
        writer = csv.writer(conflicts_file, delimiter='\t')
        writer.writerow(
                ['Index', 'Size', 'References', 'Edges', 'Sources', 'Types',
                 'Nodes'])
        for index, subgraph in enumerate(subgraphs_with_conflicts(graph), start=1):
            nodes = subgraph.nodes
            size = subgraph.number_of_nodes()
            refs = len([node for node in nodes if isinstance(node, Reference)])
            if size > 1:
                logger.debug('  - Subgraph %d, %d refs', index, refs)
                edges_to_remove = feedback_arcs(subgraph)
                edge_count = len(subgraph.edges)
                sources = {str(attr['source'].uri) for u, v, attr in subgraph.edges.data() if 'source' in attr}
                node_types = {str(attr['kind']) for u, v, attr in subgraph.edges.data()}
                writer.writerow(
                        [index, size, refs, edge_count, ", ".join(sources), ", ".join(node_types),
                         " / ".join(map(str, nodes))])
                conflicts_file.flush()
                mark_edges_to_delete(subgraph, edges_to_remove)
    return [('List of conflicts', conflicts_file_name)]


def remove_edges(source: nx.MultiDiGraph, predicate: Callable[[Any, Any, Dict[str, Any]], bool]):
    """
    Returns a subgraph of source that does not contain the edges for which the predicate returns true.
    Args:
        source: source graph

        predicate: a function(u, v, attr) that returns true if the edge from node u to node v with the attributes attr should be removed.

    Returns:
        the subgraph of source induced by the edges that are not selected by the predicate.
        This is a read-only view, you may want to use copy() on  the result.
    """
    to_keep = [(u, v, k) for u, v, k, attr in source.edges(data=True, keys=True)
               if not predicate(u, v, attr)]
    return source.edge_subgraph(to_keep)
    # return nx.restricted_view(source, source.nodes, [(u,v,k) for u,v,k,attr in source.edges if predicate(u,v,attr)])


def feedback_arcs(graph: nx.MultiDiGraph, method='auto', auto_threshold=64):
    """
    Calculates the feedback arc set using the given method and returns a
    list of edges in the form (u, v, key, data)

    Args:
        graph: NetworkX DiGraph
        method: 'eades' (approximation, fast) or 'ip' (exact, exponential), or 'auto'
    """
    if method == 'auto':
        method = 'eades' if len(graph.edges) > auto_threshold else 'ip'
    logger.debug('Calculating MFAS for a %d-node graph using %s, may take a while', graph.number_of_nodes(), method)
    igraph = to_igraph(graph)
    iedges = igraph.es[igraph.feedback_arc_set(method=method, weights='weight')]
    logger.debug('%d edges to remove', len(iedges))
    return list(nx_edges(iedges, keys=True, data=True))


def mark_edges_to_delete(graph: nx.MultiDiGraph, edges: List[Tuple[Any, Any, int, Any]]):
    """Marks edges to delete by setting their 'delete' attribute to True. Modifies the given graph."""
    mark_edges(graph, edges, delete=True)


def mark_edges(graph: nx.MultiDiGraph, edges: List[Tuple[Any, Any, int, Any]], **new_attrs):
    for u, v, k, *_ in edges:
        graph.edges[u, v, k].update(new_attrs)


def add_edge_weights(graph: nx.MultiDiGraph):
    """Adds a 'weight' attribute, coming from the node kind or the bibliography, to the given graph"""
    for u, v, k, data in graph.edges(data=True, keys=True):
        if 'weight' not in data:
            if data['kind'] == 'timeline':
                data['weight'] = 2 ** 31
            if 'source' in data:
                data['weight'] = data['source'].weight


def collapse_edges(graph: nx.MultiDiGraph):
    """
    Returns a new graph with all multi- and conflicting edges collapsed.

    Note:
        This is not able to reduce the number of edges enough to let the
        feedback_arc_set method 'ip' work with the

    """
    result = graph.copy()
    multiedges = defaultdict(list)

    for u, v, k, attr in graph.edges(keys=True, data=True):
        multiedges[tuple(sorted([u, v], key=str))].append((u, v, k, attr))

    for (u, v), edges in multiedges.items():
        if len(edges) > 1:
            total_weight = sum(attr['source'].weight * (1 if (u, v) == (w, r) else -1) for w, r, k, attr in edges)
            result.remove_edges_from([(u, v, k) for u, v, k, data in edges])
            if total_weight < 0:
                u, v = v, u
                total_weight = -total_weight
            result.add_edge(u, v,
                            kind='collapsed',
                            weight=total_weight,
                            sources=tuple(attr['source'] for w, r, k, attr in edges))

    return result


def collapse_edges_by_source(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Returns a new graph with all parallel edges from the same source collapsed.
    """
    result = graph.copy()
    edge_groups = defaultdict(list)
    for u, v, k, attr in result.edges(keys=True, data=True):
        if 'source' in attr:
            edge_groups[(u, v, attr['kind'], attr['source'].uri)].append((u, v, k, attr))

    for (u, v, kind, source_uri), group in edge_groups.items():
        if len(group) > 1:
            logger.debug('Collapsing group %s', group)
            group_attr = dict(
                    weight=sum(attr.get('weight', 1) for u, v, k, attr in group),
                    kind=kind,
                    collapsed=len(group),
                    source=BiblSource(source_uri),
                    sources=[attr['source'] for u, v, k, attr in group],
                    xml=[attr['xml'] for u, v, k, attr in group]
            )
            result.remove_edges_from(group)
            result.add_edge(u, v, **group_attr)
    return result


@dataclass
class MacrogenesisInfo:
    base: nx.MultiDiGraph
    working: nx.MultiDiGraph
    dag: nx.MultiDiGraph
    closure: nx.MultiDiGraph
    conflicts: List[Tuple[Union[date, Reference], Union[date, Reference], int, Dict[str, Any]]]

    def __post_init__(self):
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
                attr['from_ambiguity']=ambiguity
                graph.add_edge(u, witness, k, **attr)
            graph.remove_edge(u, ambiguity, k)
        for _, v, k, attr in list(graph.out_edges(ambiguity, keys=True, data=True)):
            for witness in ambiguity.witnesses:
                attr['from_ambiguity']=ambiguity
                graph.add_edge(witness, v, k, **attr)
            graph.remove_edge(ambiguity, v, k)
        graph.remove_node(ambiguity)


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


def add_missing_wits(working: nx.MultiDiGraph):
    all_wits = {wit for wit in Witness.database.values() if isinstance(wit, Witness)}
    known_wits = {wit for wit in working.nodes if isinstance(wit, Witness)}
    missing_wits = all_wits - known_wits
    logger.debug('Adding %d otherwise unmentioned witnesses to the working graph', len(missing_wits))
    working.add_nodes_from(sorted(missing_wits, key=Witness.sigil_sort_key))


def macrogenesis_graphs() -> MacrogenesisInfo:
    """
    Runs the complete analysis by loading the data, building the graph,
    removing conflicting edges and calculating a transitive closure.

    Returns:

    """
    base = base_graph()
    adopt_orphans(base)
    add_edge_weights(base)
    resolve_ambiguities(base)
    base = collapse_edges_by_source(base)
    working = cleanup_graph(base).copy()
    add_missing_wits(working)
    conflicts = subgraphs_with_conflicts(working)

    logger.info('Calculating minimum feedback arc set for %d subgraphs', len(conflicts))

    all_conflicting_edges = []
    for conflict in conflicts:
        conflicting_edges = feedback_arcs(conflict)
        mark_edges_to_delete(conflict, conflicting_edges)
        all_conflicting_edges.extend(conflicting_edges)

    selfloops = list(nx.selfloop_edges(working, data=True, keys=True))
    if selfloops:
        logger.warning('Found %d self loops, will also remove those. Affected nodes: %s',
                       len(selfloops), ", ".join(str(u) for u, v, k, attr in selfloops))
        all_conflicting_edges.extend(selfloops)

    logger.info('Building DAG from remaining data')
    dag = working.copy()
    dag.remove_edges_from(all_conflicting_edges)

    if not nx.is_directed_acyclic_graph(dag):
        logger.error('After removing %d conflicting edges, the graph is still not a DAG!', len(all_conflicting_edges))
        cycles = list(nx.simple_cycles(dag))
        logger.error('It contains %d simple cycles', len(cycles))
    else:
        logging.info('Double-checking removed edges ...')
        for u, v, k, attr in list(all_conflicting_edges):
            dag.add_edge(u, v, **attr)
            if nx.is_directed_acyclic_graph(dag):
                all_conflicting_edges.remove((u, v, k, attr))
                logging.info('Added edge %s -> %s back without introducing a cycle.', u, v)
            else:
                dag.remove_edge(u, v)

    logger.info('Marking %d conflicting edges for deletion', len(all_conflicting_edges))
    mark_edges_to_delete(base, all_conflicting_edges)

    logger.info('Removed %d of the original %d edges', len(all_conflicting_edges), len(working.edges))

    closure = nx.transitive_closure(dag)

    return MacrogenesisInfo(base, working, dag, closure, conflicts)


def cleanup_graph(A: nx.MultiDiGraph) -> nx.MultiDiGraph:
    logger.info('Removing hertz and temp-syn')

    def is_hertz(u, v, attr):
        return 'source' in attr and 'hertz' in attr['source'].uri

    def is_syn(u, v, attr):
        return attr['kind'] == 'temp-syn'

    def is_ignored(u, v, attr):
        return attr.get('ignore', False)

    for u, v, k, attr in A.edges(keys=True, data=True):
        if is_hertz(u, v, attr) or is_syn(u, v, attr):
            attr['ignore'] = True

    return remove_edges(A, is_ignored)
