#!/usr/bin/env python3
import csv
import logging
import sys
from collections import Counter, defaultdict
from datetime import date
from logging.config import dictConfig
from pathlib import Path
from typing import Callable, Any, Dict

import networkx as nx
import yaml
from networkx import MultiDiGraph
from pygraphviz import AGraph

from datings import base_graph, BiblSource
from uris import Reference

logger = logging.getLogger('main')


def setup_logging():
    log_config = Path('logging.yaml')
    if log_config.exists():
        dictConfig(yaml.load(log_config.read_text()))
    else:
        logging.basicConfig(level=logging.WARNING)


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


def analyse_conflicts(graph):
    conflicts_file_name = 'conflicts.tsv'
    with open(conflicts_file_name, "wt") as conflicts_file:
        writer = csv.writer(conflicts_file, delimiter='\t')
        writer.writerow(['Index', 'Size', 'References', 'Edges', 'Sources', 'Types', 'Cycle'])
        for index, nodes in enumerate(sorted(nx.strongly_connected_components(graph), key=len, reverse=True)):
            size = len(nodes)
            refs = len([node for node in nodes if isinstance(node, Reference)])
            if size > 1:
                subgraph = nx.subgraph(graph, nodes)  # type: networkx.DiGraph
                edge_count = len(subgraph.edges)
                sources = {str(attr['source'].uri) for u, v, attr in subgraph.edges.data() if 'source' in attr}
                node_types = {str(attr['kind']) for u, v, attr in subgraph.edges.data()}
                writer.writerow(
                        [index, size, refs, edge_count, ", ".join(sources), ", ".join(node_types),
                         " -> ".join(map(str, nodes))])
                write_dot(subgraph, f"conflict-{index:02d}.dot")
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


def get_bibliography_stats(graph: nx.MultiDiGraph):
    bibls = defaultdict(Counter)
    for u, v, attr in graph.edges(data=True):
        if 'source' in attr:
            bibls[attr['source'].uri][attr['kind']] += 1
    kinds = sorted({str(kind) for bibl in bibls.values() for kind in bibl.keys()})
    totals = Counter({ref: sum(types.values()) for ref, types in bibls.items()})
    with open('sources.tsv', 'wt', encoding='utf-8') as out:
        writer = csv.writer(out, delimiter='\t')
        writer.writerow(['Reference', 'Total'] + kinds)
        for bibl, total in totals.most_common():
            writer.writerow([bibl, total] + [bibls[bibl][kind] for kind in kinds])


def _main(argv=sys.argv):
    setup_logging()

    logger.info('Building base graph ...')
    base = base_graph()
    get_bibliography_stats(base)
    logger.info('Removing hertz')
    # without_hertz = nx.restricted_view(base, base.nodes, [(u, v, k) for u, v, k, attr in base.edges(data=True, keys=True)
    #                                                      if 'source' in attr and 'hertz' in attr['source'].uri])
    without_hertz = remove_edges(base, lambda u, v, attr: 'source' in attr and 'hertz' in attr['source'].uri)
    without_syn = remove_edges(without_hertz, lambda u, v, attr: attr['kind'] == 'temp-syn')
    write_dot(without_syn)
    logger.info('Analyzing conflicts ...')
    analyse_conflicts(without_syn)


def _load_style(filename):
    with open(filename, encoding='utf-8') as f:
        return yaml.load(f)


def write_dot(graph, target='base_graph.dot', style=_load_style('styles.yaml')):
    logger.info('Writing %s ...', target)
    simplified: MultiDiGraph = simplify_graph(graph)
    agraph: AGraph = nx.nx_agraph.to_agraph(simplified)

    # extract the timeline
    timeline = agraph.add_subgraph([node for node in agraph.nodes() if node.attr['kind'] == 'date'], name='cluster_timeline')
    if 'timeline' in style:
        timeline_style = style['timeline']
        for t in ('graph', 'edge', 'node'):
            if t in timeline_style:
                getattr(timeline, t + '_attr', {}).update(timeline_style[t])

    # now style by kind:
    if 'edge' in style:
        for edge in agraph.edges():
            kind = edge.attr['kind']
            if kind in style['edge']:
                edge.attr.update(style['edge'][kind])

    if 'node' in style:
        for node in agraph.nodes():
            kind = node.attr['kind']
            if kind in style['node']:
                edge.attr.update(style['node'][kind])

    agraph.write(target)


if __name__ == '__main__':
    import requests_cache
    requests_cache.install_cache(expire_after=86400)
    _main()
