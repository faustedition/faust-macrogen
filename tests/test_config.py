import pytest

from macrogen.config import CachedFile


@pytest.fixture(scope='session')
def cache_dir(tmp_path_factory):
    return tmp_path_factory.mktemp('cache')


def test_cf_url(cache_dir):
    cf = CachedFile('http://faustedition.net/data/paralipomena.js', cache_dir)
    assert cf.is_url
    assert cf.path.name == 'paralipomena.js'


def test_cf_open(cache_dir):
    cf = CachedFile('http://faustedition.net/data/paralipomena.js', cache_dir)
    with cf.open() as content:
        text = content.read()
        assert text
        assert cf.path.exists()
