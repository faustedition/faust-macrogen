"""
Functions to parse the XML datings and build a graph out of them
"""
from abc import ABCMeta, abstractmethod
from os import fspath
from pathlib import Path
from typing import Optional, Any, Union
from collections.abc import Generator

import networkx as nx
from datetime import date, timedelta, datetime
from lxml import etree
from rich.progress import Progress
from macrogen.splitgraph import start_end_graph
from more_itertools import pairwise

from .bibliography import BiblSource
from .uris import Witness, Reference
from .config import config

logger = config.getLogger(__name__)


def parse_datestr(datestr: str) -> Optional[date]:
    """
    Parses a date str like 1799-01-01 to a date.

    Returns:
        None if the string could not be parsed
    """
    if datestr is None:
        return None

    dt = datetime.strptime(datestr, '%Y-%m-%d')
    if dt is not None:
        return dt.date()
    else:
        return None


class InvalidDatingError(ValueError):
    """The given dating is invalid."""

    def __init__(self, msg, element: Optional[etree._Element] = None):
        if element is not None:
            xml = etree.tostring(element, pretty_print=True, encoding='unicode')
            msg = f"{msg}:\n{xml} at {element.getroottree().docinfo.URL}:{element.sourceline}"
        super().__init__(msg)


class _AbstractDating(metaclass=ABCMeta):
    """
    Abstract base class for assertions on datings.
    """

    def __init__(self, el: etree._Element):
        """
        Initializes the assertion from the given XML element.

        This base class methods mainly initializes the properties items, sources, comments, xmlsource, and details

        Args:
            el: The basic assertion element. This will usually be <relation> or <date>.
        """
        self.items: list[Reference] = [Witness.get(uri) for uri in
                                       el.xpath('f:item/@uri', namespaces=config.namespaces)]
        self.sources = tuple(BiblSource(source.get('uri'), source.text)
                             for source in el.xpath('f:source', namespaces=config.namespaces))
        self.comments = tuple(comment.text for comment in el.xpath('f:comment', namespaces=config.namespaces))
        self.xmlsource: tuple[str, int] = (config.relative_path(el.getroottree().docinfo.URL), el.sourceline)
        self.ignore = el.get('ignore', 'no') == 'yes'

    @abstractmethod
    def add_to_graph(self, G: nx.MultiDiGraph):
        """
        Add the dating to the given graph.

        Implementations should create and add all nodes and edges required to represent this dating.

        Args:
            G: the graph to create.
        """
        ...


def _firstattr(object: Any, *args: str) -> tuple[Optional[str], Optional[Any]]:
    """
    Returns the first of the given attributes together with its value. E.g., when an object o has the attributes bar=1
    and baz=2 and none else and you call ``_firstattr(o, 'foo', 'bar', 'baz')`` it will return ``'bar', 1``

    Args:
        object: The object to query
        *args: The attribute names to query, in order

    Returns:
        key, value if found, `None, None` otherwise

    """
    for attribute in args:
        value = getattr(object, attribute, None)
        if value is not None:
            return attribute, value
    return None, None


class AbsoluteDating(_AbstractDating):
    """
    An absolute dating. Create this from the `<date>` elements
    """

    def __init__(self, el: etree._Element):
        super().__init__(el)
        self.from_ = parse_datestr(el.get('from', None))
        self.to = parse_datestr(el.get('to', None))
        self.not_before = parse_datestr(el.get('notBefore', None))
        self.not_after = parse_datestr(el.get('notAfter', None))
        self.when = parse_datestr(el.get('when', None))
        self.normalized = el.get('type', '') == 'normalized'

        if self.start is None and self.end is None:
            raise InvalidDatingError('Absolute dating without a date', el)
        elif self.date_before is not None and self.date_after is not None and not self.date_before < self.date_after:
            raise InvalidDatingError('Backwards dating (%s), this would have caused a conflict' % self, el)

    @property
    def start_attr(self) -> tuple[str, date]:
        """
        The attribute representing the start of the interval.

        Returns:
            Tuple attribute name, parsed date
        """
        return _firstattr(self, 'from_', 'when', 'not_before')

    @property
    def end_attr(self) -> tuple[str, date]:
        """
        The attribute representing the end of the interval.

        Returns:
            Tuple attribute name, parsed date
        """
        return _firstattr(self, 'to', 'when', 'not_after')

    @property
    def start(self) -> Optional[date]:
        """The start date, regardless of the detailed semantics"""
        return self.start_attr[1]

    @property
    def end(self) -> Optional[date]:
        """The end date, regardless of the detailed semantics"""
        return self.end_attr[1]

    @property
    def date_before(self) -> Optional[date]:
        return self.start - timedelta(days=1) if self.start is not None else None

    @property
    def date_after(self) -> Optional[date]:
        return self.end + timedelta(days=1) if self.end is not None else None

    def __str__(self):
        if self.when is not None:
            result = self.when.isoformat()
        else:
            result = ''
            if self.start is not None: result += self.start.isoformat()
            result += ' .. '
            if self.end is not None: result += self.end.isoformat()
        return result

    def add_to_graph(self, G: nx.MultiDiGraph):
        for item in self.items:
            for source in self.sources:
                if self.start is not None:
                    G.add_edge(self.date_before, item, kind=self.start_attr[0], source=source, dating=self,
                               xml=self.xmlsource, ignore=self.ignore, comments=self.comments)
                    if self.end is None and config.half_interval_mode == 'always' and not self.ignore:
                        G.add_edge(item, min(self.date_before + timedelta(config.half_interval_correction),
                                             date(1832, 3, 23)), kind='not_after',
                                   trigger_node=self.date_before,
                                   source=BiblSource('faust://heuristic', source.citation), xml=self.xmlsource)
                if self.end is not None:
                    G.add_edge(item, self.date_after, kind=self.end_attr[0], source=source, dating=self,
                               xml=self.xmlsource, ignore=self.ignore, comments=self.comments)
                    if self.start is None and config.half_interval_mode == 'always' and not self.ignore:
                        G.add_edge(self.date_after - timedelta(config.half_interval_correction), item,
                                   kind='not_before',
                                   trigger_node=self.date_after,
                                   source=BiblSource('faust://heuristic', source.citation), xml=self.xmlsource)


class RelativeDating(_AbstractDating):
    """
    A relative dating. Represents a sequence of items which are in a kind of relation.
    """

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
                             comments=self.comments,
                             dating=self,
                             xml=self.xmlsource,
                             ignore=self.ignore)


def _parse_file(filename: Union[Path, str]) -> Generator[_AbstractDating, None, None]:
    """
    Parses the given macrogenesis XML file and returns the datings from there.
    Args:
        filename: name or uri to the xml file to parse.

    Returns:

    """
    tree = etree.parse(fspath(filename))
    for element in config.progress(tree.xpath('//f:relation', namespaces=config.namespaces), desc=f"{filename} relations", transient=True):
        yield RelativeDating(element)
    for element in config.progress(tree.xpath('//f:date', namespaces=config.namespaces), desc=f"{filename} datings", transient=True):
        try:
            yield AbsoluteDating(element)
        except InvalidDatingError as e:
            logger.error(str(e))


def _parse_files() -> Generator[_AbstractDating, None, None]:
    """
    Parses the files in the macrogenesis folder and returns the datings from there.
    Returns:

    """

    path = Path(config.data, 'macrogenesis')
    files = list(path.rglob('**/*.xml'))
    logger.info('Looking for macrogenesis files below %s', path.absolute())
    for file in config.progress(files, desc="Parsing macrogenesis files", transient=True):
        yield from _parse_file(file)


def add_timeline_edges(graph):
    """
    Add missing timeline edges to the graph.

    Afterwards, each date node in the graph will have an edge to the next date represented in the graph.
    """
    date_nodes = sorted(node for node in graph.nodes if isinstance(node, date))
    for earlier, later in pairwise(date_nodes):
        if earlier != later and (earlier, later) not in graph.edges:
            graph.add_edge(earlier, later, kind='timeline')


def build_datings_graph() -> nx.MultiDiGraph:
    """
    Builds the raw datings graph by parsing the datings from all macrogenesis files from the default directory,
    adding them to a new graph and adjusting the timeline.
    """
    graph = nx.MultiDiGraph()
    logger.info('Reading data to build base graph ...')
    for dating in _parse_files():
        dating.add_to_graph(graph)

    if config.half_interval_mode == 'light':
        for node in list(graph.nodes):
            if isinstance(node, Reference):
                post = [(v, attr) for (_, v, attr) in graph.out_edges(node, data=True) if isinstance(v, date)]
                pre = [(u, attr) for (u, _, attr) in graph.in_edges(node, data=True) if isinstance(u, date)]
                if pre and not post:
                    other_limit = max(pre, key=lambda item: item[1]['source'].weight)
                    graph.add_edge(node, other_limit[0] + timedelta(config.half_interval_correction),
                                   source=BiblSource('faust://heuristic', other_limit[1]['source'].citation),
                                   kind='not_after', xml=other_limit[1]['xml'])
                elif post and not pre:
                    other_limit = max(post, key=lambda item: item[1]['source'].weight)
                    graph.add_edge(other_limit[0] - timedelta(config.half_interval_correction), node,
                                   source=BiblSource('faust://heuristic', other_limit[1]['source'].citation),
                                   kind='not_before', xml=other_limit[1]['xml'])

    if config.model in ('split', 'split-reverse'):
        graph = start_end_graph(graph, config.model)
    add_timeline_edges(graph)
    return graph


def strongly_connected_subgraphs(graph: nx.Graph):
    """
    Extracts the strongly connected components of the given graph. Those components
    that consist of more than one nodes are then returned, sorted by size (in nodes,
    descending)

    Args:
        graph: a graph

    Returns:
        list of subgraphs
    """
    sccs = [scc for scc in nx.strongly_connected_components(graph) if len(scc) > 1]
    sorted_sccs = sorted(sccs, key=len, reverse=True)
    return [graph.subgraph(scc) for scc in sorted_sccs]


_datings: Optional[list[Union[RelativeDating, AbsoluteDating]]] = None


def get_datings():
    global _datings
    if _datings is None:
        _datings = list(_parse_files())
    return _datings
