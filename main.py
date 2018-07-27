#!/usr/bin/env python3
import csv
import logging
import sys
from collections import Counter, defaultdict, namedtuple
from datetime import date
from itertools import islice
from logging.config import dictConfig
from pathlib import Path
from typing import Callable, Any, Dict, List, Tuple

import igraph
import networkx as nx
import yaml
from networkx import MultiDiGraph
from pygraphviz import AGraph
from tqdm import tqdm

from datings import base_graph, BiblSource
from igraph_wrapper import to_igraph, nx_edges
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


def write_bibliography_stats(graph: nx.MultiDiGraph):
    bibls = defaultdict(Counter)
    for u, v, attr in graph.edges(data=True):
        if 'source' in attr:
            bibls[attr['source'].uri][attr['kind']] += 1
    kinds = sorted({str(kind) for bibl in bibls.values() for kind in bibl.keys()})
    totals = Counter({ref: sum(types.values()) for ref, types in bibls.items()})
    with open('sources.tsv', 'wt', encoding='utf-8') as out:
        writer = csv.writer(out, delimiter='\t')
        writer.writerow(['Reference', 'Weight', 'Total'] + kinds)
        for bibl, total in totals.most_common():
            writer.writerow([bibl, BiblSource(bibl).weight, total] + [bibls[bibl][kind] for kind in kinds])


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


def cleanup_graph(A: nx.MultiDiGraph) -> nx.MultiDiGraph:
    logger.info('Removing hertz and temp-syn')

    def is_hertz(u, v, attr):
        return 'source' in attr and 'hertz' in attr['source'].uri

    def is_syn(u, v, attr):
        return attr['kind'] == 'temp-syn'

    without_hertz = remove_edges(A, is_hertz)
    without_syn = remove_edges(without_hertz, is_syn)
    return without_syn


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




def _main(argv=sys.argv):
    setup_logging()

    logger.info('Building base graph ...')
    base = base_graph()
    add_edge_weights(base)
    write_bibliography_stats(base)
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

    agraph.write(target)


if __name__ == '__main__':
    import requests_cache

    requests_cache.install_cache(expire_after=86400)
    _main()
