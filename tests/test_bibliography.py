from os import fspath
from pathlib import Path

from lxml import etree
from macrogen import BiblSource, config


def test_bibl_eq():
    source1 = BiblSource("faust://bibliography/bohnenkamp1994", "123")
    source2 = BiblSource("faust://bibliography/bohnenkamp1994", "123")
    assert source1 == source2


def test_bibl_ne_detail():
    source1 = BiblSource("faust://bibliography/bohnenkamp1994", "123")
    source2 = BiblSource("faust://bibliography/bohnenkamp1994", "124")
    assert source1 != source2


def test_bibl_ne_uri():
    source1 = BiblSource("faust://bibliography/bohnenkamp1994", "123")
    source2 = BiblSource("faust://bibliography/bruening_hahn2017", "123")
    assert source1 != source2


def test_bibl_str():
    source = BiblSource("faust://bibliography/bohnenkamp1994", "123")
    assert str(source) == "Bohnenkamp 1994\n123"


def test_long_cit():
    source = BiblSource("faust://bibliography/bohnenkamp1994")
    assert source.long_citation == ('Bohnenkamp, Anne: „… das Hauptgeschäft nicht außer Augen lassend“. Die '
                                    'Paralipomena zu Goethes ‚Faust‘, Frankfurt am Main und Leipzig 1994.')

def test_not_found():
    source = BiblSource('faust://bibliography/notfound')
    assert str(source).strip() == "notfound"


def test_all_bibl_defined():
    macrogen: Path = config.path.data / 'macrogenesis'
    all_uris = set()
    for file in macrogen.rglob('*.xml'):
        xml = etree.parse(fspath(file))
        source_els = xml.xpath('//f:source', namespaces=config.namespaces)
        all_uris.update(el.get('uri', '') for el in source_els)
    sources = {BiblSource(uri) for uri in all_uris}
    undef = {source for source in sources if str(source).startswith('faust://')}
    assert sources
    assert undef == set()
