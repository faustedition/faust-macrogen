from datetime import date
from pathlib import Path

import macrogen
from macrogen.config import config
import pytest
from macrogen.graph import yearlabel, MacrogenesisInfo



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


def test_save_invariance(eades_graphs: MacrogenesisInfo, tmp_path: Path):
    info_file = tmp_path / "eades.zip"
    eades_graphs.save(info_file)
    loaded = MacrogenesisInfo(info_file)
    assert set(eades_graphs.base.nodes) == set(loaded.base.nodes)
    difference = eades_graphs.details.index.symmetric_difference(loaded.details.index)
    assert difference.empty, difference
