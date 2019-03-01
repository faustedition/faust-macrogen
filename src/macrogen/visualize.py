from collections import Sequence
from datetime import date
from multiprocessing.pool import Pool
from pathlib import Path

import networkx as nx
import yaml
from networkx import MultiDiGraph
from pygraphviz import AGraph
from tqdm import tqdm

from .config import config
from .datings import add_timeline_edges
from macrogen import BiblSource
from .graph import pathlink
from .uris import Reference

logger = config.getLogger(__name__)

_render_queue = []


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
        elif isinstance(value, Sequence) and not isinstance(value, str):
            attrs[key] = " ".join(item.uri if hasattr(item, 'uri') else str(item) for item in value)
        elif type(value) not in {str, int, float, bool}:
            attrs[key] = str(value)


def write_dot(graph: nx.MultiDiGraph, target='base_graph.dot', style=None,
              highlight=None, record='auto', edge_labels=True):
    """
    Writes a properly styled graphviz file for the given graph.

    Args:
        graph: the subgraph to draw
        target: dot file that should be written, may be a Path
        style (dict): rules for styling the graph
        highlight: if a node, highlight that in the graph. If a tuple of nodes, highlight the shortest path(s) from the
                   first to the second node
        record: record in the queue for `render_all`. If ``"auto"`` dependent on graph size
        edge_labels (bool): Should we paint edge labels?

    Returns:
        None.
    """
    if style is None:
        style = config.styles
    logger.info('Writing %s ...', target)
    target_path = Path(target)
    target_path.parent.mkdir(exist_ok=True, parents=True)
    if record == 'auto':
        record = len(graph.edges) < 1000

    vis = graph.copy()
    add_timeline_edges(vis)
    for node in vis:
        if isinstance(node, Reference):
            vis.nodes[node]['URL'] = node.filename.stem
            vis.nodes[node]['target'] = '_top'

    if highlight is not None:
        if isinstance(highlight, tuple) and 'highlight' in style['edge']:
            try:
                vis.edges[highlight].update(style['edge']['highlight'])
                if 'highlight' in style['node']:
                    vis.nodes[highlight[0]].update(style['node']['highlight'])
                    vis.nodes[highlight[1]].update(style['node']['highlight'])
            except KeyError:
                logger.warning('Highlight key %s not found while writing %s', highlight, target)
        elif not isinstance(highlight, tuple) and 'highlight' in style['node']:
            try:
                vis.nodes[highlight].update(style['node']['highlight'])
            except KeyError:
                logger.warning('Highlight key %s not found while writing %s', highlight, target)

    simplified: MultiDiGraph = simplify_graph(vis)

    # now style by kind:
    if 'edge' in style:
        for u, v, k, attr in simplified.edges(data=True, keys=True):
            kind = attr.get('kind', None)
            if attr.get('delete', False):
                attr['URL'] = pathlink(u, v).stem
                attr['target'] = '_top'
            if kind in style['edge']:
                simplified.edges[u, v, k].update(style['edge'][kind])
            for styled_attr in attr.keys() & style['edge']:
                if attr[styled_attr]:
                    simplified.edges[u, v, k].update(style['edge'][styled_attr])

    if 'node' in style:
        for node, attr in simplified.nodes(data=True):
            kind = attr.get('kind', None)
            if kind in style['node']:
                simplified.nodes[node].update(style['node'][kind])
            for styled_attr in attr.keys() & style['node']:
                if attr[styled_attr]:
                    attr.update(style['node'][styled_attr])

    if not edge_labels:
        for u, v, k, attr in simplified.edges(data=True, keys=True):
            if 'label' in attr:
                del attr['label']

    agraph: AGraph = nx.nx_agraph.to_agraph(simplified)
    agraph.edge_attr['fontname'] = 'Ubuntu derivative Faust'
    agraph.edge_attr['fontsize'] = 8
    agraph.node_attr['fontname'] = 'Ubuntu derivative Faust'
    agraph.node_attr['fontsize'] = 12
    agraph.graph_attr['rankdir'] = 'LR'
    agraph.graph_attr['stylesheet'] = '/css/webfonts.css'

    # extract the timeline
    timeline = agraph.add_subgraph([node for node in agraph.nodes() if node.attr['kind'] == 'date'],
                                   name='cluster_timeline')

    if 'timeline' in style:
        timeline_style = style['timeline']
        for t in ('graph', 'edge', 'node'):
            if t in timeline_style:
                getattr(timeline, t + '_attr', {}).update(timeline_style[t])
                logger.debug('timeline style: %s = %s', t, getattr(timeline, t + '_attr').items())  ## Doesn’t work

    dotfilename = str(target)
    agraph.write(dotfilename)
    if record:
        _render_queue.append(dotfilename)
    else:
        logger.warning('%s has not been queued for rendering', dotfilename)


def render_file(filename):
    """
    Renders the given dot file to an svg file using dot.
    """
    graph = AGraph(filename=filename)
    try:
        resultfn = filename[:-3] + 'svg'
        graph.draw(resultfn, format='svg', prog='dot')
        return resultfn
    except:
        logger.exception('Failed to render %s', filename)


def render_all():
    with Pool() as pool:
        global _render_queue
        dots, _render_queue = _render_queue, []
        result = list(tqdm(pool.imap_unordered(render_file, dots), desc='Rendering', total=len(dots), unit=' SVGs'))
        failcount = result.count(None)
        logger.info('Rendered %d SVGs, %d failed', len(result) - failcount, failcount)


