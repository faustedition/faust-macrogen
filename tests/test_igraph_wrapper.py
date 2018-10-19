from datetime import date

import networkx as nx
import pytest

from macrogen.igraph_wrapper import _convert_attr_list, to_igraph, nx_edges


@pytest.fixture(scope='session')
def base_graph():
    from main import base_graph
    G = base_graph()
    return G


@pytest.fixture
def small_graph():
    G = nx.MultiDiGraph()
    G.add_edge(date(1776, 1, 2), date(1777, 11, 30), kind='timeline')
    G.add_edge(date(1777, 11, 30), "foo", source="Hans")
    return G


def test_igraph_roundtrip(small_graph):
    Gi = to_igraph(small_graph)
    edges = list(nx_edges(Gi.es, True, True))
    assert isinstance(edges[0][0], date)
    assert edges[0][3] == {'kind': 'timeline'}


def test_convert_attr():
    source = [dict(a=23, b=42), dict(a=47, b=11), dict(a=5)]
    target = _convert_attr_list(source, 'all')
    assert target['a'] == [23, 47, 5]
    assert target['b'] == [42, 11, None]
