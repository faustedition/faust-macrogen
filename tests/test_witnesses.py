from pathlib import Path

from macrogen import witnesses, config
from macrogen.witnesses import SceneInfo, WitInscrInfo


def test_document():
    doc = witnesses.Document(Path(config.path.data, 'document/faust/2/gsa_391098.xml'))
    assert doc.sigil == '2 H'
    assert doc.uri == 'faust://document/faustedition/2_H'

def test_all_documents():
    docs = witnesses.all_documents()
    assert docs


def test_scenes():
    si = SceneInfo()
    assert si.toplevel[0].n == '1.0.1'
    assert si.scenes[-1].n == '2.5.5'

def test_witinfo():
    wi = WitInscrInfo.get()
