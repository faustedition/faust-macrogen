import networkx as nx
import pytest

from datings import BiblSource
from main import collapse_edges
from uris import Witness


@pytest.fixture
def conflict_03():
    c41 = Witness.get('faust://document/wa_faust/C_41')
    c4 = Witness.get('faust://document/wa_faust/C_4')
    q = Witness.get('faust://document/wa_faust/Q')

    G = nx.MultiDiGraph()
    G.add_edge(c41, c4, source=BiblSource('faust://bibliography/wa_i_15_2', '1'))
    G.add_edge(c4, c41,source=BiblSource('faust://bibliography/wa_i_15_2', '2'))

    G.add_edge(c41, q, source=BiblSource('faust://bibliography/wa_i_15_2', '3'))
    G.add_edge(c41, q, source=BiblSource('faust://bibliography/wa_i_15_2', '2'))
    G.add_edge(c41, q, source=BiblSource('faust://bibliography/wa_i_15_2', '3'))
    G.add_edge(q, c41, source=BiblSource('faust://bibliography/wa_i_15_2', '4'))

    return G


def test_collapse_edges(conflict_03):
    collapsed = collapse_edges(conflict_03)
    assert 2 == len(collapsed.edges)
