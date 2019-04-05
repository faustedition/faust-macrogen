from pathlib import Path

from macrogen import witnesses, config


def test_document():
    doc = witnesses.Document(Path(config.path.data, 'document/faust/2/gsa_391098.xml'))
    assert doc.sigil == '2 H'
    assert doc.uri == 'faust://document/faustedition/2_H'

def test_all_documents():
    docs = witnesses.all_documents()
    assert docs