import macrogen
from macrogen.config import config
import pytest


@pytest.fixture('session')
@pytest.mark.slow
def eades_graphs():
    config.fes_method = 'eades'
    config.lightweight_timeline = False
    return macrogen.graph.MacrogenesisInfo()


@pytest.mark.slow
def test_conflict_info(eades_graphs):
    cs = eades_graphs.conflict_stats()
    assert cs
