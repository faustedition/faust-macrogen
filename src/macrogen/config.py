"""
Configuration etc.


Configuration and data files:

- logging.yaml
- styles.yaml
- reference-normalisation.csv
- bibscores.tsv
- sigils.json
- paralipomena.json
- genetic_bar_graph.json

Additionally:

- macrogenesis data
- output directory

Optional:

- graph file(s) to read from

Additional stuff to configure:

- Render / Render graphs up to ...
- algorithm / threshold
"""
import json
from io import BytesIO, StringIO
from os.path import expanduser, expandvars
from pathlib import Path
from typing import Optional, IO, Callable, Any, Tuple, Mapping
from urllib.parse import urlparse
from .bibliography import _parse_bibliography

import pkg_resources
import requests
from lxml import etree
from ruamel.yaml import YAML

import logging

logger = logging.getLogger(__name__ + '.preliminary')


class CachedFile:
    """Loads data from an URL, optionally caching it."""

    def __init__(self, file_or_url: str, cache_dir: Optional[Path] = None):
        """
        Creates a cacheing file loader.

        Args:
            file_or_url: the url or path to the file to load
            cache_dir: if present, a directory where to cache the file. If absent, don’t cache.
        """
        url = urlparse(file_or_url)

        if url.scheme:
            path = Path(url.path)
            self.url = file_or_url
            self.url_parsed = url
            self.path = None
            self.is_url = True
        else:
            path = Path(file_or_url)
            self.url = None
            self.path = path
            self.is_url = False

        if self.is_url and cache_dir is not None:
            self.path = cache_dir.joinpath(path.name)

    def open(self, offline=False, mode="rt") -> IO:
        """
        Opens the file or URL.

        Args:
            offline: Never access the internet.
            mode: file mode for `open`

        Returns:
            open IO, either to the cached file or to the remotely fetched content
        """
        if self.is_url and not offline:
            # fetch remote to cache
            logger.debug('fetching %s', self.url)
            response = requests.get(self.url)

            if self.path:
                # dump to cache and serve from cache file
                logger.debug('saving as %s', self.path)
                if "b" in mode:
                    with self.path.open("wb") as cache_file:
                        cache_file.write(response.content)
                else:
                    with self.path.open("wt", encoding='utf-8') as cache_file:
                        cache_file.write(response.text)
            else:
                if "b" in mode:
                    return BytesIO(response.content)
                else:
                    return StringIO(response.text)

        return self.path.open(mode=mode, encoding='utf-8-sig')


class LazyConfigLoader:
    """
    Descriptor that lazily loads stuff from configured paths.
    """

    def __init__(self, name: str, parser: Optional[Callable[[IO], Any]] = None,
                 fallback_resource=Optional[Tuple[str, str]]):
        self.name = name
        self.parser = parser
        self.resource = fallback_resource

    def __get__(self, instance, owner):
        if not hasattr(instance, '_data'):
            instance._data = {}
        if self.name not in instance.data:
            self.load_data(instance)
        return instance._data[self.name]

    def load_data(self, instance):
        source = instance.config.get(self.name, None)
        if source:
            logger.info('Loading %s from %s', self.name, source)
            cache = Path(instance.config.get('cache', '.cache'))
            offline = instance.config.get('offline', False)
            cached_file = CachedFile(source, cache)
            with cached_file.open(offline) as file:
                self.parse_data(file, instance)
        elif self.resource:
            logger.debug('Loading %s from internal configuration %s', self.name, self.resource)
            with pkg_resources.resources_stream(*self.resource) as file:
                self.parse_data(file, instance)
        else:
            raise ValueError(
                    f"Cannot access property {self.name}: Neither configured source nor fallback resource available")

    def parse_data(self, file, instance):
        instance.data[self.name] = self.parser(file) if callable(self.parser) else file.read()


_yaml = YAML(typ='rt')
_cfg = 'macrogen'


class _Proxy:

    def __init__(self, constructor, *args, **kwargs):
        self._constructor = constructor
        self._args = args
        self._kwargs = args
        self._target = None

    def _init_proxy(self):
        if self._target is None:
            self._target = self._constructor(*self._args, **self._kwargs)

    def __getattr__(self, item):
        self._init_proxy()
        return getattr(self._target, item)

    def __setattr__(self, key, value):
        self._init_proxy()
        return setattr(self._target, key, value)

    def __delattr__(self, item):
        self._init_proxy()
        return delattr(self._target, item)

class _Accessor:

    def __init__(self, accessor_function: Callable[[Any], Any]):
        self._accessor = accessor_function

    def __getattr__(self, item):
        return self._accessor(item)


class Configuration:
    """
    Ready to use configuration data for the application.

    Data that is coming from files can be loaded lazily.
    """

    logging = LazyConfigLoader('logging', _yaml.load, (_cfg, 'etc/logging.yaml'))
    styles = LazyConfigLoader('styles', _yaml.load, (_cfg, 'etc/styles.yaml'))
    # reference-normalization.csv
    # bibscores.tsv
    sigils = LazyConfigLoader('sigils', json.load)
    paralipomena = LazyConfigLoader('paralipomena', json.load)
    genetic_bar_graph = LazyConfigLoader('bargraph', json.load)
    bibliography = LazyConfigLoader('bibliography', _parse_bibliography)

    def __init__(self, config_override=None):
        if config_override is None:
            config_override = {}
        self.config_override = config_override

        def get_path(key):
            return Path(self.key)

        self.path = _Accessor(get_path)

    @property
    def config_override(self):
        return self._config_override

    @config_override.setter
    def _set_config_override(self, value):
        if hasattr(self, 'config'):
            logger.warning('Configuration has already been loaded. Some override values may not have any effect.')
            self._apply_override(value)
        self._config_override = value

    def _apply_override(self, override=None):
        if override is None:
            override = self.config_override
        for key, value in self.config_override:
            if value is not None:
                self.config[key] = value

    def __getattr__(self, item):
        if item == 'config':
            self._load_config()
            return self.__dict__['config']
        if item in self.config:
            return self.config[item]
        raise AttributeError(f'No configuration item {item}')

    def _load_config(self):
        # First, load the default config
        logger.debug("Loading default configuration")
        with pkg_resources.resource_stream(_cfg, 'etc/default.yaml') as f:
            config: Mapping = _yaml.load(f)
        # now work through all config files configured in the default config
        # if they exist
        if 'config_files' in config:
            for fn in config['config_files']:
                p = Path(expanduser(expandvars(fn)))
                if p.exists():
                    logger.info('Loading configuration file %s', p)
                    with p.open() as f:
                        config.update(_yaml.load(f))
        # now update using command line options etc.
        self.config.update(self._config_override)

        # finally, let’s configure logging
        self._init_logging()

    def _init_logging(self):
        global logger
        from logging.config import dictConfig
        dictConfig(self.logging)
        logger = logging.getLogger(__name__)

    def getLogger(self, name):
        return _Proxy(logging.getLogger, name)

    def relative_path(self, absolute_path):
        return Path(absolute_path).relative_to(self.path.data)


config = Configuration()
