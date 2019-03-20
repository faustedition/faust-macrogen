import macrogen
from macrogen.config import config
import pytest

@pytest.fixture('session')
def eades_graphs():
    config.fes_method = 'eades'
    config.lightweight_timeline = False
    return macrogen.graph.MacrogenesisInfo()

def test_conflict_info(eades_graphs):
    cs = eades_graphs.conflict_stats()
    assert cs