"""
Configuration etc.


Configuration and data files:

- logging.yaml
- styles.yaml
- reference-normalisation.csv
- bibscores.tsv
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
import argparse
import csv
import json
import logging
from logging import Logger
import traceback
from collections import namedtuple, defaultdict
from functools import partial
from io import BytesIO, StringIO, TextIOWrapper
from os.path import expanduser, expandvars
from pathlib import Path
from typing import Optional, IO, Callable, Any, Tuple, Mapping, Union, Dict
from urllib.parse import urlparse

import pkg_resources
import requests
from lxml import etree
from ruamel.yaml import YAML

# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__ + '.preliminary')

BibEntry = namedtuple('BibEntry', ['uri', 'citation', 'reference', 'weight'])


def parse_bibliography(bibxml: Union[str, IO]) -> Dict[str, BibEntry]:
    """Parses the bibliography file at url. Returns a dictionary mapping an URI to a corresponding bibliography entry."""
    db: Dict[str, BibEntry] = {}
    scores = config.bibscores
    et = etree.parse(bibxml)
    for bib in et.xpath('//f:bib', namespaces=config.namespaces):
        uri = bib.get('uri')
        citation = bib.find('f:citation', namespaces=config.namespaces).text
        reference = bib.find('f:reference', namespaces=config.namespaces).text
        db[uri] = BibEntry(uri, citation, reference, scores[uri])
    return db


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
                if not self.path.parent.exists():
                    self.path.parent.mkdir(parents=True, exist_ok=True)
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
                 fallback_resource: Optional[Tuple[str, str]] = None):
        self.name = name
        self.parser = parser
        self.resource = fallback_resource

    def __get__(self, instance, owner):
        if not hasattr(instance, '_data'):
            instance._data = {}
        if self.name not in instance._data:
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
            with pkg_resources.resource_stream(*self.resource) as file:
                self.parse_data(file, instance)
        else:
            raise ValueError(
                    f"Cannot access property {self.name}: Neither configured source nor fallback resource available")

    def parse_data(self, file, instance):
        try:
            instance._data[self.name] = self.parser(file) if callable(self.parser) else file.read()
        except ValueError as e:
            logger.exception('%s parsing %s (for %s)', e, file, self.name)
            instance._data[self.name] = {}


_yaml = YAML(typ='rt')
_config_package = 'macrogen'


class _Proxy:

    def __init__(self, constructor, *args, **kwargs):
        self.__dict__.update(dict(
                _constructor=constructor,
                _args=args,
                _kwargs=kwargs,
                _target=None))

    def _init_proxy(self):
        if self.__dict__['_target'] is None:
            self._target = self._constructor(*self._args, **self._kwargs)

    def __getattr__(self, item):
        self._init_proxy()
        return getattr(self._target, item)

    def __setattr__(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
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


def parse_kvcsv(file: IO, default=None, value_type=None, **kwargs):
    """
    Parses a two-column key-value csv file to a key: value dictionary
    """
    text = TextIOWrapper(file, encoding='utf-8')
    reader = csv.reader(text, **kwargs)
    next(reader)  # skip header
    result = {row[0]: row[1] for row in reader if row[1]}
    if value_type is not None:
        result = {k: value_type(v) for k, v in result.items()}
    if default is not None:
        return defaultdict(lambda: default, result)
    else:
        return result

class Configuration:
    """
    Ready to use configuration data for the application.

    Data that is coming from files can be loaded lazily.
    """

    logging = LazyConfigLoader('logging', _yaml.load, (_config_package, 'etc/logging.yaml'))
    styles = LazyConfigLoader('styles', _yaml.load, (_config_package, 'etc/styles.yaml'))
    # reference-normalization.csv
    # bibscores.tsv
    genetic_bar_graph = LazyConfigLoader('genetic_bar_graph', json.load)
    bibliography = LazyConfigLoader('bibliography', parse_bibliography)
    uri_corrections = LazyConfigLoader('uri_corrections', parse_kvcsv, (_config_package, 'etc/uri-corrections.csv'))
    bibscores = LazyConfigLoader('bibscores', partial(parse_kvcsv, default=1, value_type=int, delimiter='\t'),
                                 (_config_package, 'etc/bibscores.tsv'))
    scenes_xml = LazyConfigLoader('scenes_xml', etree.parse, (_config_package, 'etc/scenes.xml'))
    graphviz_attrs = LazyConfigLoader('graphviz_attrs', _yaml.load, (_config_package, 'etc/graphviz_attrs.yaml'))

    def __init__(self, config_override=None):
        self._config_override = {}
        self.config_loaded = False
        if config_override is None:
            config_override = {}
        self.config_override = config_override

        def get_path(key):
            return Path(getattr(self, key))

        self.path = _Accessor(get_path)

    @property
    def config_override(self):
        return self._config_override

    @config_override.setter
    def config_override(self, value):
        if hasattr(self, 'config'):
            logger.warning('Configuration has already been loaded. Some override values may not have any effect.')
            self._apply_override(value)
        self._config_override = value

    def _apply_override(self, override=None):
        if override is None:
            override = self.config_override
        for key, value in override.items():
            if value is not None:
                if key in self.config:
                    logger.info('Overriding %s=%s with %s', key, self.config[key], value)
                    self.config[key] = value

    def __getattr__(self, item):
        if item == 'config' and not self.config_loaded:
            self._load_config()
            return self.__dict__['config']
        if item in self.config:
            logger.debug('Config %s -> %s', item, self.config[item])
            return self.config[item]
        raise AttributeError(f'No configuration item {item}')

    def _load_config(self):
        self.config_loaded = True
        # First, load the default config
        logger.debug("Loading default configuration.\n%s", "".join(traceback.format_stack()))
        with pkg_resources.resource_stream(_config_package, 'etc/default.yaml') as f:
            config: Mapping = _yaml.load(f)
            self.config = config
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
        logger.debug('Reconfiguring logging')
        dictConfig(self.logging)
        logger.debug('Reconfigured logging')
        logger = logging.getLogger(__name__)

    def getLogger(self, name) -> Union[Logger, _Proxy]:
        return _Proxy(logging.getLogger, name)

    def progress(self, iterable, *args, **kwargs):
        if self.progressbar:
            try:
                from tqdm import tqdm
                from tqdm.contrib.logging import tqdm_logging_redirect
                if 'dynamic_ncols' not in kwargs:
                    kwargs['dynamic_ncols'] = True
                with tqdm_logging_redirect():
                    yield from tqdm(iterable, *args, **kwargs)
            except ImportError:
                pass
        return iterable

    def relative_path(self, absolute_path):
        return Path(absolute_path).relative_to(self.path.data)

    def prepare_options(self, argparser: argparse.ArgumentParser):
        """
        Configures the given argument parser from the current options.

        The method walks through the currently active configuration.
        Each top-level option that has a comment will be considered
        for the option parser, and a commented option will be generated.

        The namespace object the argparser returns can then be used with
        the override options.
        """
        for key in self.config:
            value = self.config[key]
            try:
                comment = self.config.ca.items[key][2].value
                desc = comment.strip('# ')
                option_name='--' + key.replace('_','-')

                if isinstance(value, list):
                    argparser.add_argument(option_name, nargs="*", dest=key,
                                           help=f"{desc} ({' '.join(value)}")
                elif value is not None:
                    argparser.add_argument(option_name, dest=key, action='store', type=_yaml_from_string,
                                           help=f"{desc} ({str(value)})")
                else:
                    argparser.add_argument(option_name, dest=key, action='store', type=_yaml_from_string,
                                           help=desc)
            except KeyError:
                logger.debug('No argument for uncommented config option %s', key)
            except AttributeError:
                logger.debug('Could not extract comment from config option %s', key)



    def save_config(self, output: Optional[Union[Path, str, BytesIO]]):
        """
        Dumps the current configuration.

        Args:
            output: If Path or str, path describing the target file. If it is a stream,
                    we simply write to the stream, closing is at the client's discretion.
                    If None, dump the configuration to the log (at INFO level).
        """
        if output is None:
            target = StringIO()
        elif hasattr(output, 'write'):
            target = output
        else:
            if not isinstance(output, Path):
                output = Path(output)
            target = output.open('wb')

        try:
            with YAML(typ='rt', output=target) as y:
                y.dump(self.config, output)

            if output is None:
                logger.info('Configuration:\n%s', target.getvalue())
            else:
                logger.debug('Saved effective configuration to %s', target)
        finally:
            if target is not output:
                target.close()

def _yaml_from_string(arg):
    yaml = YAML(typ='safe')
    return yaml.load(StringIO(arg))

config = Configuration()
