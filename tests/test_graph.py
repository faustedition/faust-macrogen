from datetime import date

import macrogen
from macrogen.config import config
import pytest
from macrogen.graph import yearlabel


@pytest.fixture('session')
def eades_graphs():
    config.fes_method = 'eades'
    config.lightweight_timeline = False
    return macrogen.graph.MacrogenesisInfo()


def test_conflict_info(eades_graphs):
    cs = eades_graphs.conflict_stats()
    assert cs


@pytest.mark.parametrize(
        "start,end,label", [
            (None, None, ""),
            ("1800-07-01", None, "1800"),
            (None, "1800-07-01", "1800"),
            ("1800-07-01", "1800-09-01", "1800"),
            ("1799-12-31", "1801-01-01", "1800"),
            ("1799-12-01", "1801-01-31", "1799 … 1801")
        ])
def test_yearlabel(start, end, label):
    start_ = start and date.fromisoformat(start)
    end_ = end and date.fromisoformat(end)
    assert yearlabel(start_, end_) == label
