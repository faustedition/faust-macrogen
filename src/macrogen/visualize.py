import asyncio
import shutil
import subprocess
from functools import partial
from os import PathLike
from typing import Optional, Union
from collections.abc import Sequence
from datetime import date, timedelta
from time import perf_counter
from multiprocessing.pool import Pool
from pathlib import Path

import networkx as nx
from networkx import MultiDiGraph
from pygraphviz import AGraph

from .config import config
from .datings import add_timeline_edges
from .bibliography import BiblSource
from .graphutils import pathlink, base_n
from .uris import Reference
from .graph import Node
from .splitgraph import SplitReference
import logging

logger: logging.Logger = config.getLogger(__name__)

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
            attrs['label'] = node.label
            translation[node] = node.uri
            if isinstance(node, SplitReference):
                attrs['kind'] = node.side.value
            else:
                attrs['kind'] = node.__class__.__name__
        else:
            attrs['kind'] = type(node).__name__
            attrs['label'] = str(node)
            translation[node] = base_n(hash(node), 62)  # use a stable, short representation
        _simplify_attrs(attrs)

    nx.relabel_nodes(graph, translation, copy=False)

    for u, v, attrs in graph.edges(data=True):
        if 'source' in attrs and 'label' not in attrs:
            source_ = attrs['source']
            if isinstance(source_, Sequence) and not isinstance(source_, str):
                attrs['label'] = '\n'.join(f"{s.citation}: {s.detail}" if s.detail else s.citation for s in source_)
            else:
                attrs['label'] = str(source_)
        _simplify_attrs(attrs)

    # noinspection PyTypeChecker
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


def remove_nondot_keys(graph: nx.MultiDiGraph, inplace=False) -> nx.MultiDiGraph:
    if not inplace:
        graph = graph.copy()
    allowed = set(config.graphviz_attrs)

    def clean_attr(attrs: dict):
        for key in attrs.keys() - allowed:
            del attrs[key]

    for node in graph:
        clean_attr(graph.nodes[node])
    # noinspection PyArgumentList
    for u, v, attr in graph.edges(keys=False, data=True):
        clean_attr(attr)
    return graph


def write_dot(graph: nx.MultiDiGraph, target: Optional[Union[PathLike, str]] = 'base_graph.dot',
              style: Optional[dict] = None,
              highlight: Optional[Union[Node, Sequence[Node]]] = None,
              highlight_path: Optional[tuple[Node, Node]] = None,
              record: Union[bool, str] = 'auto', edge_labels: bool = True) -> AGraph:
    """
    Writes a properly styled graphviz file for the given graph.

    Args:
        graph: the subgraph to draw
        target: dot file that should be written, may be a Path. If none, nothing is written but the AGraph returns
        style (dict): rules for styling the graph
        highlight: if a node, highlight that in the graph.
        highlight_path: If a tuple of nodes, highlight the shortest path(s) from the
                   first to the second node
        record: record in the queue for `render_all`. If ``"auto"`` dependent on graph size
        edge_labels (bool): Should we paint edge labels?

    Returns:
        the AGraph, can be used to write the thing yourself.
    """
    if style is None:
        style = config.styles
    logger.info('Writing %s ...', target)
    try:
        if record == 'auto' and config.render_node_limit >= 0:
            record = graph.number_of_nodes() < config.render_node_limit
            if not record:
                logger.info('%s is too large to be rendered automatically (%d nodes)', target, graph.number_of_nodes())
    except Exception as e:
        logger.warning('Auto edges limit configuration error: %s', e)

    vis = graph.copy()
    add_timeline_edges(vis)
    for node in vis:
        if isinstance(node, Reference):
            vis.nodes[node]['URL'] = node.filename.stem
            vis.nodes[node]['target'] = '_top'

    # single node highlight
    if highlight is not None and not isinstance(highlight, Sequence):
        highlight = [highlight]

    if highlight_path is not None:
        if highlight is None:
            highlight = list(highlight_path)
        else:
            highlight = list(highlight)
            highlight.extend(highlight_path)
            if 'highlight' in style['edge']:
                try:
                    vis.edges[highlight].update(style['edge']['highlight'])
                except KeyError:
                    logger.warning('Highlight key %s not found while writing %s', highlight, target)

    if highlight is not None:
        if not isinstance(highlight, Sequence):
            highlight = [highlight]

        for node in list(highlight):
            if isinstance(node, SplitReference) and node.other:
                highlight.append(node.other)

        if 'highlight' in style['node']:
            for highlight_node in highlight:
                try:
                    vis.nodes[highlight_node].update(style['node']['highlight'])
                except KeyError:
                    logger.warning('Highlight key %s not found while writing %s', highlight, target)

    # noinspection PyTypeChecker
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
            if 'topo' in attr and 'constraint' in attr:
                del attr['constraint']

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

    if config.clean_gv_files:
        remove_nondot_keys(simplified, inplace=True)
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

    if target is not None:
        target_path = Path(target)
        target_path.parent.mkdir(exist_ok=True, parents=True)
        dotfilename = str(target)
        agraph.write(dotfilename)
        if record:
            _render_queue.append(dotfilename)
        else:
            logger.warning('%s has not been queued for rendering', dotfilename)
    return agraph


def render_file(filename):
    """
    Renders the given dot file to an svg file using dot.
    """
    graph = AGraph(filename=filename)
    starttime = perf_counter()
    try:
        resultfn = filename[:-3] + 'svg'
        graph.draw(resultfn, format='svg', prog='dot')
        return resultfn
    except:
        logger.exception('Failed to render %s', filename)
    finally:
        duration = timedelta(seconds=perf_counter() - starttime)
        if duration > timedelta(seconds=5):
            logger.warning('Rendering %s with %d nodes and %d edges took %s',
                           filename, graph.number_of_nodes(), graph.number_of_edges(), duration)


def render_file_alt(filename: PathLike, timeout: Optional[float] = None) -> \
        Union[Path, tuple[Path, Union[subprocess.CalledProcessError, subprocess.TimeoutExpired]]]:
    """
    Calls GraphViz' dot to render the given file to svg, at least if it does not take more than timeout seconds.

    Args:
        filename: The dot file to render
        timeout: Timeout in seconds, or None if we would like to wait endlessly

    Returns:
        result path if everything is ok.
        Tuple of result path and exception if timeout or process error.
    """
    path = Path(filename)
    dot = shutil.which('dot')
    target = path.with_suffix('.svg')
    args = [dot, '-T', 'svg', '-o', target, path]
    try:
        p = subprocess.run(args, capture_output=True, check=True, encoding='utf-8', timeout=timeout)
        if p.stderr:
            logger.warning('Rendering %s: %s', path, p.stderr)
        return target
    except subprocess.CalledProcessError as e:
        logger.error('Rendering %s failed (%d): %s', path, e.returncode, e.stderr)
        return target, e
    except subprocess.TimeoutExpired as e:
        logger.warning('Rendering %s aborted after %g seconds (%s)', path, timeout, e.stderr)
        return target, e


async def render_file_async(filename: PathLike, semaphore: asyncio.Semaphore, timeout: Optional[float] = None):
    path = Path(filename)
    dot = shutil.which('dot')
    target = path.with_suffix('.svg')
    args = ['-T', 'svg', '-o', target, path]
    async with semaphore:
        logger.debug('Starting process for rendering %s', path)
        process = await asyncio.create_subprocess_exec(dot, *args, stdin=None, stdout=asyncio.subprocess.PIPE,
                                                       stderr=asyncio.subprocess.PIPE, close_fds=True)
        logger.debug('Waiting for rendering of %s', path)
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
            if stdout:
                logger.info('Rendering %s: %s', path, stdout)
            if stderr:
                logger.warning('Rendering %s: %s', path, stderr)
            if process.returncode == 0:
                return target
            else:
                logger.error('Rendering %s failed (%d)', path, process.returncode)
                return target, process.returncode
        except asyncio.TimeoutError as e:
            process.terminate()
            stdout, stderr = await process.communicate()
            logger.warning('Rendering %s timed out after %g seconds | %s | %s', path, timeout, stdout, stderr)
            return target, e


def render_all_pool(timeout=None):
    if timeout is None:
        timeout = config.render_timeout
    if timeout is not None and timeout <= 0:
        timeout = None
    with Pool() as pool:
        global _render_queue
        dots, _render_queue = _render_queue, []
        result = list(config.progress(pool.imap_unordered(partial(render_file_alt, timeout=timeout), dots),
                                      desc='Rendering graphs', total=len(dots), unit=' SVGs'))
        not_rendered = [entry for entry in result if isinstance(entry, tuple)]
        timeout = [path for path, err in not_rendered if isinstance(err, subprocess.TimeoutExpired)]
        failed = [path for path, err in not_rendered if isinstance(err, subprocess.CalledProcessError)]
        _render_queue.append(timeout)
    if failed:
        loglevel = logging.ERROR
    elif timeout:
        loglevel = logging.WARNING
    else:
        loglevel = logging.INFO
    logger.log(loglevel, 'Rendered %d SVGs, %d timed out, %d failed', len(result) - len(timeout) - len(failed),
               len(timeout), len(failed))


def render_all(timeout=None, render_mode=None):
    render_mode = render_mode or config.render_mode
    if render_mode == 'async':
        asyncio.run(render_all_async(timeout))
    else:
        render_all_pool(timeout)


async def render_all_async(timeout=None):
    if timeout is None:
        timeout = config.render_timeout
    if timeout is not None and timeout <= 0:
        timeout = None

    global _render_queue
    dots, _render_queue = _render_queue, []
    sem = asyncio.Semaphore(10)
    results = await asyncio.gather(*[render_file_async(file, sem, timeout) for file in dots])
    not_rendered = [entry for entry in results if isinstance(entry, tuple)]
    timeout = [path for path, err in not_rendered if isinstance(err, asyncio.TimeoutError)]
    failed = [path for path, err in not_rendered if isinstance(err, int)]
    _render_queue.append(timeout)
    if failed:
        loglevel = logging.ERROR
    elif timeout:
        loglevel = logging.WARNING
    else:
        loglevel = logging.INFO
    logger.log(loglevel, 'Rendered %d SVGs, %d timed out, %d failed', len(results) - len(timeout) - len(failed),
               len(timeout), len(failed))
