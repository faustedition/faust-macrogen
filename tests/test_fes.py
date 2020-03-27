import pytest
import networkx as nx

from macrogen.fes import eades, FES_Baharev, Eades


@pytest.fixture
def graph1():
    """
           ←
    1 → 2 → 3 → 4 → 5

    """
    G = nx.DiGraph()
    nx.add_path(G, [1, 2, 3, 4, 5])
    G.add_edge(3, 2)
    return G


def test_all_sinks(graph1):
    eades = Eades(graph1)
    sinks = []
    for sink in eades._exhaust_sinks():
        eades.graph.remove_node(sink)
        sinks.append(sink)
    assert sinks == [5, 4]


def test_eades(graph1):
    assert list(eades(graph1)) == [(3, 2)]


def test_baharev(graph1):
    solver = FES_Baharev(graph1)
    result = solver.solve()
    assert set(result) == {(3, 2)} or set(result) == {(2, 3)}


def test_eades_ff():
    g = nx.DiGraph()
    nx.add_path(g, [1, 2, 3, 4, 5], weight=1)
    g.add_edge(3, 2, weight=2)

    result = Eades(g).solve()
    assert set(result) == {(2, 3)}

    solver = Eades(g, [(4,5), (2, 3)])
    result = set(solver.solve())
    assert set(result) == {(3, 2)}


def test_baharev_ff():
    g = nx.DiGraph()
    nx.add_path(g, [1, 2, 3, 4, 5], weight=1)
    g.add_edge(3, 2, weight=2)

    # This would normally remove (2,3) since its more lightweight than (3,2):
    result = FES_Baharev(g).solve()
    assert set(result) == {(2, 3)}

    # However, when we forbid this, the next best solution will occur:
    result = FES_Baharev(g, [(2, 3)]).solve()
    assert set(result) == {(3, 2)}
