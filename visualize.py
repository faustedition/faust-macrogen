from datetime import date

import networkx as nx
import yaml
from networkx import MultiDiGraph
from pygraphviz import AGraph

from datings import BiblSource
from uris import Reference
from faust_logging import logging
logger = logging.getLogger()


def simplify_graph(original_graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Creates a copy of the graph that contains only simple types, so it can be serialized to, e.g., GEXF
    """
    graph = original_graph.copy()

    translation = {}
    for node, attrs in graph.nodes.data():
        if isinstance(node, date):
            attrs['kind'] = 'date'
            translation[node] = node.isoformat()
        elif isinstance(node, Reference):
            attrs['kind'] = node.__class__.__name__
            attrs['label'] = node.label
            translation[node] = node.uri
        _simplify_attrs(attrs)

    nx.relabel_nodes(graph, translation, copy=False)

    for u, v, attrs in graph.edges(data=True):
        if 'source' in attrs and not 'label' in attrs:
            attrs['label'] = str(attrs['source'])
        _simplify_attrs(attrs)

    return graph


def _simplify_attrs(attrs):
    for key, value in list(attrs.items()):
        if isinstance(value, BiblSource):
            attrs[key] = value.uri
            if value.detail is not None:
                attrs[key + '_detail'] = value.detail
        elif value is None:
            del attrs[key]
        elif type(value) not in {str, int, float, bool}:
            attrs[key] = str(value)


def _load_style(filename):
    with open(filename, encoding='utf-8') as f:
        return yaml.load(f)


def write_dot(graph, target='base_graph.dot', style=_load_style('styles.yaml')):
    logger.info('Writing %s ...', target)
    simplified: MultiDiGraph = simplify_graph(graph)
    agraph: AGraph = nx.nx_agraph.to_agraph(simplified)
    agraph.edge_attr['fontsize'] = 8
    agraph.graph_attr['fontname'] = 'Ubuntu'

    # extract the timeline
    timeline = agraph.add_subgraph([node for node in agraph.nodes() if node.attr['kind'] == 'date'],
                                   name='cluster_timeline')
    if 'timeline' in style:
        timeline_style = style['timeline']
        for t in ('graph', 'edge', 'node'):
            if t in timeline_style:
                getattr(timeline, t + '_attr', {}).update(timeline_style[t])
                logger.debug('timeline style: %s = %s', t, getattr(timeline, t + '_attr').items())  ## Doesn’t work

    # now style by kind:
    if 'edge' in style:
        for edge in agraph.edges():
            kind = edge.attr['kind']
            if kind in style['edge']:
                edge.attr.update(style['edge'][kind])
            if 'delete' in edge.attr and edge.attr['delete'] and 'delete' in style['edge']:
                edge.attr.update(style['edge']['delete'])

    if 'node' in style:
        for node in agraph.nodes():
            kind = node.attr['kind']
            if kind in style['node']:
                edge.attr.update(style['node'][kind])

    agraph.write(str(target))