from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import List, Iterable, Tuple, Any, Generator, Union, TypeVar, Callable, Dict, Sequence

import networkx as nx

from macrogen.datings import add_timeline_edges
from .uris import Witness, Reference
from .bibliography import BiblSource
from .datings import parse_datestr
from .config import config

T = TypeVar('T')
S = TypeVar('S')
logger = config.getLogger(__name__)


def pathlink(*nodes) -> Path:
    """
    Creates a file name for the given path.

    The file name consists of the file names for the given nodes, in order, joined by `--`
    """
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


def expand_edges(graph: nx.MultiDiGraph, edges: Iterable[Tuple[Any, Any]], filter: bool = False) \
        -> Generator[Tuple[Any, Any, int, dict], None, None]:
    """
    Expands a 'simple' edge list (of node pairs) to the corresponding full edge list, including keys and data.
    Args:
        graph: the graph with the edges
        edges: edge list, a list of (u, v) node tuples
        filter: if true, remove missing edges instead of raising an exception

    Returns:
        all edges from the multigraph that are between any node pair from edges as tuple (u, v, key, attrs)

    """
    for u, v in edges:
        try:
            atlas = graph[u][v]
            for key in atlas:
                yield u, v, key, atlas[key]
        except KeyError as e:
            if filter:
                logger.warning('Edge %s→%s from edge list not in graph', u, v)
            else:
                raise e


def collapse_edges(graph: nx.MultiDiGraph):
    """
    Returns a new graph with all multi- and conflicting edges collapsed.

    Note:
        This is not able to reduce the number of edges enough to let the
        feedback_arc_set method 'ip' work with the largest component
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


def first(sequence: Iterable[T], default: S = None) -> Union[T, S]:
    try:
        return next(iter(sequence))
    except StopIteration:
        return default


def collapse_timeline(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Returns a new graph in which unneeded datetime nodes are removed.
    """
    g: nx.MultiDiGraph = graph.copy()
    timeline = sorted(node for node in g.nodes() if isinstance(node, date))
    if not timeline:
        return g  # nothing to do
    for node in timeline[1:]:
        pred = first(g.predecessors(node))
        succ = first(g.successors(node))
        if g.in_degree(node) == 1 and g.out_degree(node) == 1 \
                and isinstance(pred, date) and isinstance(succ, date):
            g.add_edge(pred, succ, **g[pred][node][0])
            g.remove_node(node)
    return g


def add_iweight(graph: nx.MultiDiGraph):
    """
    Adds an 'iweight' attribute with the inverse weight for each edge. timeline edges are trimmed to zero.
    """
    for u, v, k, attr in graph.edges(keys=True, data=True):
        if 'weight' in attr:
            if attr.get('kind', '') == 'timeline':
                attr['iweight'] = 0
            elif attr['weight'] > 0:
                attr['iweight'] = 1 / attr['weight']
            else:
                attr['iweight'] = 2_000_000


def mark_edges_to_delete(graph: nx.MultiDiGraph, edges: List[Tuple[Any, Any, int, Any]]):
    """Marks edges to delete by setting their 'delete' attribute to True. Modifies the given graph."""
    mark_edges(graph, edges, delete=True)


def mark_edges(graph: nx.MultiDiGraph, edges: List[Tuple[Any, Any, int, Any]], **new_attrs):
    """Mark all edges in the given graph by updating their attributes with the keyword arguments. """
    for u, v, k, *_ in edges:
        graph.edges[u, v, k].update(new_attrs)


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


def in_path(edge: Tuple[T, T], path: Sequence[T], cycle=False) -> bool:
    """
    Whether edge is part of the given path.

    Args:
        edge: the edge we search, as a pair of nodes
        path: the path we search in, as a sequence of nodes
        cycle: if True, assume path is a cycle, i.e. there is an edge from path[-1] to path[0]
    """
    try:
        first_index = path.index(edge[0])
        if first_index < len(path) - 1:
            return path[first_index + 1] == edge[1]
        elif cycle:
            return path[0] == edge[1]
        else:
            return False
    except ValueError:
        return False


def simplify_timeline(graph: nx.MultiDiGraph):
    """
    Remove superfluous date nodes (and timeline edges) from the graph.

    When creating subgraphs of the base graph, the subgraph will sometimes contain date nodes that
    are not linked to references remaining in the subgraph. This function will remove those nodes
    and link the remaining date nodes instead. So, it will reduce

                    1798-01-01  ->   1709-01-15   ->   1798-02-01
                       `-------------> H.x ---------------^

    to

                    1798-01-01  ->  1798-02-01
                       `-----> H.x -----^
    """
    graph = remove_edges(graph, lambda u, v, attr: attr.get('kind') == 'timeline').copy()
    add_timeline_edges(graph)
    return graph
    # date_nodes = sorted(node for node in graph.nodes if isinstance(node, date))
    # prev = None
    # for node in date_nodes:
    #     if prev is not None and graph.in_degree(node) == graph.out_degree(node) == 1 and isinstance(
    #             one(graph.successors(node)), date):
    #         graph.remove_node(node)
    #     else:
    #         if prev is not None:
    #             graph.add_edge(prev, node, kind='timeline')
    #         prev = node