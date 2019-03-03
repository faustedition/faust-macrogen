import itertools
from typing import Any, Tuple, List, Generator
from .config import config
import networkx as nx

logger = config.getLogger(__name__)

def _exhaust_sinks(g: nx.DiGraph, sink: bool = True):
    """
    Produces all sinks until there are no more.

    Warning: This modifies the graph g
    """
    sink_method = g.out_degree if sink else g.in_degree
    while True:
        sinks = [u for (u, d) in sink_method() if d == 0]
        if sinks:
            yield from sinks
            g.remove_nodes_from(sinks)
        else:
            return


def _exhaust_sources(g: nx.DiGraph):
    """
    Produces all sources until there are no more.

    Warning: This modifies the given graph
    """
    return _exhaust_sinks(g, False)


def eades(graph: nx.DiGraph, double_check=True) -> List[Tuple[Any, Any]]:
    """
    Fast heuristic for the minimum feedback arc set.

    Eades’ heuristic creates an ordering of all nodes of the given graph,
    such that each edge can be classified into *forward* or *backward* edges.
    The heuristic tries to minimize the sum of the weights (`weight` attribute)
    of the backward edges. It always produces an acyclic graph, however it can
    produce more conflicting edges than the minimal solution.

    Args:
        graph: a directed graph, may be a multigraph.
        double_check: check whether we’ve _really_ produced an acyclic graph

    Returns:
        a list of edges, removal of which guarantees a

    References:
        **Eades, P., Lin, X. and Smyth, W. F.** (1993). A fast and effective
        heuristic for the feedback arc set problem. *Information Processing
        Letters*, **47**\ (6): 319–23
        doi:\ `10.1016/0020-0190(93)90079-O. <https://doi.org/10.1016/0020-0190(93)90079-O.>`__
        http://www.sciencedirect.com/science/article/pii/002001909390079O
        (accessed 27 July 2018).
    """
    g = graph.copy()
    logger.info('Internal eades calculation for a graph with %d nodes and %d edges', g.number_of_nodes(), g.number_of_edges())
    g.remove_edges_from(list(g.selfloop_edges()))
    start = []
    end = []
    while g:
        for v in _exhaust_sinks(g):
            end.insert(0, v)
        for v in _exhaust_sources(g):
            start.append(v)
        if g:
            u = max(g.nodes, key=lambda v: g.out_degree(v, weight='weight') - g.in_degree(v, weight='weight'))
            start.append(u)
            g.remove_node(u)
    ordering = start + end
    logger.debug('Internal ordering: %s', ordering)
    pos = dict(zip(ordering, itertools.count()))
    feedback_edges = list(graph.selfloop_edges())
    for u, v in graph.edges():
        if pos[u] > pos[v]:
            feedback_edges.append((u, v))
    logger.info('Found %d feedback edges', len(feedback_edges))

    if double_check:
        check = graph.copy()
        check.remove_edges_from(feedback_edges)
        if not nx.is_directed_acyclic_graph(check):
            logger.error('double-check: graph is not a dag!')
            cycles = nx.simple_cycles()
            counter_example = next(cycles)
            logger.error('Counterexample cycle: %s', counter_example)
        else:
            logger.info('double-check: Graph is acyclic.')

    return feedback_edges
