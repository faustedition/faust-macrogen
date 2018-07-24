from operator import itemgetter
from typing import List, Dict, Any, Iterable, Union, Tuple

import igraph as ig
import networkx as nx


"""
Convenience functions to work with igraph graphs derived from networkx graphs.

:func:`to_igraph` will convert an NetworkX (Multi-)DiGraph to an igraph directed Graph. Since igraph graphs use
integer nodes, we will keep required data in our attributes.

:func:`nx_edges` converts an igraph EdgeList from a graph created with :func:`to_igraph` such that it can be
incorporated in the original NetworkX graph.
"""


def _convert_attr_list(attr_list: List[Dict[Any, Any]], keep_attrs: Union[None, str, Iterable]) -> Dict[Any, List[Any]]:
    """
    Convert the attribute list from nx to igraph.Graph format.

    Args:
        attr_list: List of attribute dicts, each mapping key to attribute value.
        keep_attrs: Attribute names to keep, or 'all' to keep all attributes

    Returns:
        Dictionary mapping each attribute to a list of values, with one value per item
        in attr_list.
    """
    if keep_attrs is None:
        return None
    elif keep_attrs == 'all':
        attr_list = list(attr_list)
        keep_attrs = {key for item_attr in attr_list for key in item_attr.keys()}

    result = {}
    for attr_name in keep_attrs:
        result[attr_name] = [node_attrs[attr_name] if attr_name in node_attrs else None
                             for node_attrs in attr_list]
    return result


def _get_edge_list(graph: nx.DiGraph):
    """
    Returns a list edges, each in the form (u, v, attr). If graph is a MultiDiGraph, attr will receive an additional
    attribute `_key` with the key of the edge.
    """

    def incorporate_key(u, v, k, attr):
        attr = dict(attr)
        attr['_key'] = k
        return u, v, attr

    if graph.is_multigraph():
        return [incorporate_key(u, v, k, attr) for u, v, k, attr in graph.edges(keys=True, data=True)]
    else:
        return list(graph.edges(data=True))


def to_igraph(nx_graph: nx.MultiDiGraph, keep_node_attrs='all', keep_edge_attrs='all'):
    """
    Converts a networkx (Multi)DiGraph to an igraph graph.

    The node labels are converted to integer labels, while the original node labels are stored in a node attribute called `_node`.
    For multi digraphs, edges receive an additional attribute `_key` that contains the key of the original nx graph.

    Args:
        nx_graph: The networkx DiGraph or MultiDiGraph to convert.
        keep_node_attrs: a list of node attributes to keep, or 'all' to keep all existing attributes.
        keep_edge_attrs: a list of edge attributes to keep, or 'all' to keep all existing attributes.

    Returns:

    """
    int_graph: nx.MultiDiGraph = nx.convert_node_labels_to_integers(nx_graph, label_attribute='_node')
    n = nx_graph.number_of_nodes()

    node_attrs_nx = sorted(int_graph.nodes(data=True), key=itemgetter(0))
    node_attrs = _convert_attr_list(map(itemgetter(1), node_attrs_nx), keep_node_attrs)
    edges_nx = _get_edge_list(int_graph)
    edges = [(u, v) for u, v, attr in edges_nx]
    edge_attrs = _convert_attr_list([attr for u, v, attr in edges_nx], keep_edge_attrs)

    # igraph doesn't use 'real' default arguments
    kwargs = {}
    if node_attrs is not None: kwargs['vertex_attrs'] = node_attrs
    if edge_attrs is not None: kwargs['edge_attrs'] = edge_attrs
    igraph = ig.Graph(n, edges=edges, directed=True, **kwargs)
    return igraph


def nx_edges(edges: ig.EdgeSeq, keys=False, data=False) -> List[Tuple]:
    """
    Converts an igraph edge list to a representation similar to :method:`nx.MultiDiGraph.edges`

    This only works with igraohs converted using `to_igraph`.

    Args:
        edges:
        keys: if True, returns tuples (u, v, key) for DiGraphs
        data: if True, returns (u, v, data) or (u, v, key, data)

    Yields:
        tuples (u, v) or (u, v, k) or (u, v, k, data) with u, v = nodes in the original networkx graph,
        k = key in the original networkx graph, data = edge attributes

    """
    edge: ig.Edge
    for edge in edges:
        item = tuple(v['_node'] for v in edge.graph.vs[edge.source, edge.target])
        if keys:
            item += (edge['_key'],)
        if data:
            item += ({k:v for k,v in edge.attributes().items() if k != '_key' and v is not None},)
        yield item

