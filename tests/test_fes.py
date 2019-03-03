import pytest
import networkx as nx

from macrogen.fes import _exhaust_sources, _exhaust_sinks, eades

@pytest.fixture
def graph1():
    """
           ←
    1 → 2 → 3 → 4 → 5

    """
    G = nx.DiGraph()
    G.add_path([1,2,3,4,5])
    G.add_edge(3,2)
    return G


def test_all_sinks(graph1):
    assert list(_exhaust_sinks(graph1)) == [5, 4]


def test_all_sources(graph1):
    assert list(_exhaust_sources(graph1)) == [1]


def test_eades(graph1):
    assert list(eades(graph1)) == [(3,2)]
