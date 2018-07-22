"""
This is the result of parsing the respective files.
"""
import logging
from abc import ABCMeta, abstractmethod
from typing import List, Tuple, TypeVar, Optional, Any

import networkx as nx
from lxml import etree
import datetime

from more_itertools import pairwise

import faust
from uris import Witness, Reference

logger = logging.getLogger(__name__)


def _parse_datestr(datestr: str) -> datetime.date:
    if datestr is None:
        return None

    dt = datetime.datetime.strptime(datestr, '%Y-%m-%d')
    if dt is not None:
        return dt.date()
    else:
        return None


class BiblSource:
    """
    A bibliographic source
    """

    def __init__(self, uri, detail):
        self.uri = uri
        self.detail = detail

    def __eq__(self, other):
        if isinstance(other, BiblSource):
            return self.uri == other.uri and self.detail == other.detail
        else:
            return super().__eq__(other)

    def __hash__(self):
        return hash(self.uri) ^ hash(self.detail)

    def __str__(self):
        result = self.uri
        if self.detail is not None:
            result += ' ' + self.detail
        return result


class _AbstractDating(metaclass=ABCMeta):

    def __init__(self, el: etree._Element):
        self.items: List[Reference] = [Witness.get(uri) for uri in el.xpath('f:item/@uri', namespaces=faust.namespaces)]
        self.sources = tuple(BiblSource(source.get('uri'), source.text)
                                   for source in el.xpath('f:source', namespaces=faust.namespaces))
        self.comments = tuple(comment.text for comment in el.xpath('f:comment', namespaces=faust.namespaces))
        self.xmlsource: Tuple[str, int] = (el.getroottree().docinfo.URL, el.sourceline)

    @abstractmethod
    def add_to_graph(self, G: nx.MultiDiGraph):
        ...


def _firstattr(object, *args: Tuple[str]) -> Optional[Tuple[str, Any]]:
    for attribute in args:
        value = getattr(object, attribute, None)
        if value is not None:
            return attribute, value
    return None, None


class AbsoluteDating(_AbstractDating):

    def __init__(self, el: etree._Element):
        super().__init__(el)
        self.from_ = _parse_datestr(el.get('from', None))
        self.to = _parse_datestr(el.get('to', None))
        self.not_before = _parse_datestr(el.get('notBefore', None))
        self.not_after = _parse_datestr(el.get('notAfter', None))
        self.when = _parse_datestr(el.get('when', None))
        self.normalized = el.get('type', '') == 'normalized'

        if self.start is None and self.end is None:
            xml = etree.tostring(el, pretty_print=True, encoding='unicode')
            logger.warning('Absolute dating without a date: %s at %s:%d', xml, *self.xmlsource)

    @property
    def start_attr(self):
        return _firstattr(self, 'from_', 'when', 'not_before')

    @property
    def end_attr(self):
        return _firstattr(self, 'to', 'when', 'not_after')

    @property
    def start(self):
        return self.start_attr[1]

    @property
    def end(self):
        return self.end_attr[1]

    @property
    def date_before(self):
        return self.start - datetime.timedelta(days=1) if self.start is not None else None

    @property
    def date_after(self):
        return self.end + datetime.timedelta(days=1) if self.end is not None else None

    def add_to_graph(self, G: nx.MultiDiGraph):
        for item in self.items:
            for source in self.sources:
                if self.start is not None:
                    G.add_edge(self.date_before, item, kind=self.start_attr[0], source=source, dating=self)
                if self.end is not None:
                    G.add_edge(item, self.date_after, kind=self.end_attr[0], source=source, dating=self)


class RelativeDating(_AbstractDating):

    def __init__(self, el: etree._Element):
        super().__init__(el)
        self.kind = el.get('name')

    def add_to_graph(self, G: nx.MultiDiGraph) -> None:
        """
        Adds nodes for the given relative dating to the graph.

        Args:
            G: the graph to work on
        """
        G.add_nodes_from(self.items)
        for source in self.sources:
            G.add_edges_from(pairwise(self.items),
                             kind=self.kind,
                             source=source,
                             comments=self.comments)


def _parse_file(filename: str):
    tree = etree.parse(filename)
    for element in tree.xpath('//f:relation', namespaces=faust.namespaces):
        yield RelativeDating(element)
    for element in tree.xpath('//f:date', namespaces=faust.namespaces):
        yield AbsoluteDating(element)


def _parse_files():
    for file in faust.macrogenesis_files():
        yield from _parse_file(file)


def _add_timeline_edges(graph):
    date_nodes = sorted(node for node in graph.nodes if isinstance(node, datetime.date))
    for earlier, later in pairwise(date_nodes):
        if earlier != later and (earlier, later) not in graph.edges:
            graph.add_edge(earlier, later, kind='timeline')


def base_graph():
    graph = nx.MultiDiGraph()
    for dating in _parse_files():
        dating.add_to_graph(graph)
    _add_timeline_edges(graph)
    return graph
