from faust_logging import logging
import csv
from collections import defaultdict, namedtuple
from typing import List, Callable, Any, Dict, Tuple

import networkx as nx

from datings import base_graph
from igraph_wrapper import to_igraph, nx_edges
from visualize import simplify_graph, write_dot
from uris import Reference

logger = logging.getLogger()

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
                write_dot(subgraph, f"conflict-{index:02d}.dot")
                nx.write_graphml(simplify_graph(subgraph), f"conflict-{index:02d}.graphml")
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


def feedback_arcs(graph: nx.MultiDiGraph, method='auto'):
    """
    Calculates the feedback arc set using the given method and returns a
    list of edges in the form (u, v, key, data)

    Args:
        graph: NetworkX DiGraph
        method: 'eades' (approximation, fast) or 'ip' (exact, exponential), or 'auto'
    """
    if method == 'auto':
        method = 'eades' if len(graph.edges) > 256 else 'ip'
    logger.debug('Calculating MFAS for a %d-node graph using %s, may take a while', graph.number_of_nodes(), method)
    igraph = to_igraph(graph)
    iedges = igraph.es[igraph.feedback_arc_set(method=method, weights='weight')]
    logger.debug('%d edges to remove', len(iedges))
    return list(nx_edges(iedges, keys=True, data=True))


def mark_edges_to_delete(graph: nx.MultiDiGraph, edges: List[Tuple[Any, Any, int, Any]]):
    """Marks edges to delete by setting their 'delete' attribute to True. Modifies the given graph."""
    for u, v, k, _ in edges:
        graph.edges[u, v, k]['delete'] = True


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


def workflow():
    base = base_graph()
    working = cleanup_graph(base)
    conflicts = subgraphs_with_conflicts(working)

    all_conflicting_edges = []
    for conflict in conflicts:
        conflicting_edges = feedback_arcs(conflict)
        mark_edges_to_delete(conflict, conflicting_edges)
        all_conflicting_edges.append(conflicting_edges)

    mark_edges_to_delete(base, all_conflicting_edges)

    dag = working.copy()
    dag.remove_edges_from(all_conflicting_edges)
    closure = nx.transitive_closure(dag)

    MacrogenesisGraphs = namedtuple('MacrogenesisGraphs', ['base', 'working', 'dag', 'closure', 'conflicts'])
    return MacrogenesisGraphs(base, working, dag, closure, conflicts)


def cleanup_graph(A: nx.MultiDiGraph) -> nx.MultiDiGraph:
    logger.info('Removing hertz and temp-syn')

    def is_hertz(u, v, attr):
        return 'source' in attr and 'hertz' in attr['source'].uri

    def is_syn(u, v, attr):
        return attr['kind'] == 'temp-syn'

    without_hertz = remove_edges(A, is_hertz)
    without_syn = remove_edges(without_hertz, is_syn)
    return without_syn