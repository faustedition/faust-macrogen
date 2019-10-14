import csv
import json
import os
import urllib
from collections import defaultdict, Counter
from datetime import date, datetime, timedelta
from functools import partial
from html import escape
from io import StringIO
from itertools import chain, groupby, zip_longest
from operator import itemgetter
from pathlib import Path
from typing import Iterable, List, Dict, Mapping, Tuple, Sequence, Union, Generator, Optional, Set
from urllib.parse import urlencode

import networkx as nx
import pandas as pd
import pkg_resources
from lxml.builder import ElementMaker
from lxml.etree import Comment
from more_itertools import pairwise
from pandas import DataFrame

from .witnesses import WitInscrInfo, DocumentCoverage, InscriptionCoverage, SceneInfo
from .bibliography import BiblSource
from .config import config
from .datings import get_datings, AbsoluteDating
from .graph import MacrogenesisInfo, EARLIEST, LATEST, DAY, Node, temp_syn_groups
from .graphutils import pathlink, collapse_timeline, expand_edges, in_path, remove_edges
from .uris import Reference, Witness, Inscription, UnknownRef, AmbiguousRef
from .splitgraph import Side, SplitReference, side_node
from .visualize import write_dot, simplify_graph

logger = config.getLogger(__name__)

RELATION_LABELS = {'not_before': 'nicht vor',
                   'not_after': 'nicht nach',
                   'from_': 'von',
                   'to': 'bis',
                   'when': 'am',
                   'temp-syn': 'ca. gleichzeitig',
                   'temp-pre': 'zeitlich vor',
                   None: '???'
                   }


class HtmlTable:
    """
    Helper class to create a simple HTML table from some kind of data.
    """

    def __init__(self, **table_attrs):
        """
        Creates a new table.

        Args:
            **table_attrs: attributes for the `<table>` elements. For the attribute names, leading and trailing `_` are
             stripped and remaining `_` are transformed to hyphens, so ``HtmlTable(class_='example', data_order='<')``
             will generate ``<table class="example" data-order="&lt;"/>``
        """
        self.titles = []
        self.attrs = []
        self.header_attrs = []
        self.formatters = []
        self.table_attrs = table_attrs
        self.rows = []
        self.row_attrs = []

    def column(self, title='', format_spec=None, attrs={}, **header_attrs):
        """
        Adds a column to this table.

        Args:
            title: The column header as a string (possibly containing an HTML fragment)
            format_spec: How to format each column value. This can be either a format string (see :meth:`str.format`),
                    a format spec (see :func:`format`), a one-argument function or other callable or None (which will
                    result in ``format(value)``)
            **attrs: Attributes to be supplied to each `<td>`

        Returns: self
        """
        self.titles.append(title)

        if format_spec is None:
            formatter = format
        elif callable(format_spec):
            formatter = format_spec
        elif isinstance(format_spec, str) and '{' in format_spec:
            formatter = format_spec.format
        else:
            formatter = lambda data: format(data, format_spec)

        self.formatters.append(formatter)
        self.attrs.append(attrs)
        self.header_attrs.append(header_attrs)
        return self

    def row(self, row, **row_attrs):
        """
        Adds a row to this table's stored data. This data will be used by the zero-argument version of :meth:`format_table`

        The rows are exposed as a list via the property `rows`

        Args:
            row: A tuple (or, in fact, any iterable) of values, prefereably as many as there are rows.
            **row_attrs: HTML Attributes for the row. See the note for `__init__`

        Returns:
            self
        """
        if len(row) > len(self.formatters):
            raise IndexError('{} values in row, but only {} formatters: {}', len(row), len(self.formatters), row)
        self.rows.append(row)
        self.row_attrs.append(row_attrs)
        return self

    def __len__(self):
        return len(self.rows)

    @staticmethod
    def _build_attrs(attrdict: Dict):
        """Converts a dictionary of attributes to a string that may be pasted into an HTML start tag."""
        return ''.join(
                ' {}="{!s}"'.format(attr.strip('_').replace('_', '-'), escape(value)) for attr, value in
                attrdict.items())

    def _format_column(self, index, data) -> str:
        """
        Returns the HTML for the given cell.

        This looks up the configured formatter for the column and calls it to
        rendr the data, and it will also configure the cell attributes.

        Args:
            index: index of the column
            data:  data to display

        Returns:
            String containing HTML `<td>` element
        """
        try:
            attributes = self._build_attrs(self.attrs[index])
            content = self.formatters[index](data)
            return f'<td{attributes}>{content}</td>'
        except Exception as e:
            raise ValueError('Error formatting column %s' % self.titles[index]) from e

    def _format_row(self, row: Iterable, **rowattrs) -> str:
        """
        Returns the HTML for a column with the given row.
        Args:
            row: the row to format
            **rowattrs: attributes for the row

        Returns: a string containing an HTML `<tr>` element

        """
        attributes = self._build_attrs(rowattrs)
        try:
            return f'<tr{attributes}>' + ''.join(
                    self._format_column(index, column) for index, column in enumerate(row)) + '</tr>'
        except:
            row_str = ", ".join(f"{title!s}: {value!r}" for title, value in zip_longest(self.titles, row))
            logger.exception('Error formatting row (%s)', row_str)
            return f'<tr class="pure-alert pure-error"><td colspan="{len(self.titles)}">Error formatting row ({row_str})</td></tr>'

    def _format_rows(self, rows: Iterable[Iterable]) -> Generator[str, None, None]:
        """Formats the given rows."""
        for row in rows:
            yield self._format_row(row)

    def _format_header(self):
        """
        Formats the table header.
        """
        column_headers = ''.join(['<th{1}>{0}</th>'.format(title, self._build_attrs(attrs))
                                  for title, attrs in zip(self.titles, self.header_attrs)])
        return '<table class="pure-table"{1}><thead>{0}</thead><tbody>'.format(column_headers,
                                                                               self._build_attrs(self.table_attrs))

    def _format_footer(self):
        return '</tbody></table>'

    def format_table(self, rows=None, row_attrs=None, sort_key=None):
        """
        Actually formats the table.

        In the zero-argument form, this uses the rows added previously using the `row` method. Otherwise, the given
        data is used.

        Args:
            rows: If given, the rows to format. This is a list of n-tuples, for n columns.
            row_attrs: If given, this should be a list that contains a mapping with the attributes for each row.
            sort_key: If given, a function that returns a sort key for each row

        Returns:
            string containing HTML code for the table
        """
        if rows is None:
            rows = self.rows
            row_attrs = self.row_attrs
        rows = list(rows)
        if row_attrs is None:
            row_attrs = [{} for _ in range(len(rows))]
        if sort_key is None:
            order = range(len(rows))
        else:
            order = [i for (i, _) in sorted(enumerate(rows), key=lambda er: sort_key(er[1]))]

        formatted_rows = [self._format_row(rows[i], **row_attrs[i]) for i in order]

        return self._format_header() + '\n' + '\n'.join(formatted_rows) + '\n' + self._format_footer()


def write_html(filename: Path, content: str, head: str = None, breadcrumbs: List[Dict[str, str]] = [],
               graph_id: str = None,
               graph_options: Dict[str, object] = dict(controlIconsEnabled=True)) -> None:
    """
    Writes a html page.
    Args:
        filename: out file path
        content: formatted contents for the main part
        head: heading, will become the last of the breadcrumbs
        breadcrumbs: list of dictionaries with keys _caption_ and _link_, first (macrogenesis) and last (head) omitted
        graph_id: if present, initialize javascript for graph with the given id
        graph_options: if present, options for the svg viewer js
    """
    if head is not None:
        breadcrumbs = breadcrumbs + [dict(caption=head)]
    breadcrumbs = [dict(caption='Makrogenese-Lab', link='/macrogenesis')] + breadcrumbs
    prefix = f"""<?php include "../includes/header.php"?>
    <!-- Generiert: {datetime.now().isoformat()} -->
     <section>"""
    require = "requirejs(['faust_common', 'svg-pan-zoom'], function(Faust, svgPanZoom)"
    if graph_id is not None:
        init = f"""
        graph = document.getElementById('{graph_id}');
        bbox = graph.getBoundingClientRect();
        if (bbox.height > (window.innerHeight - bbox.top)) {{
            graph.height = window.innerHeight - bbox.top;
            graph.width = '100%';
        }}
        svgPanZoom('#{graph_id}', {json.dumps(graph_options)})
        """
    else:
        init = ''
    suffix = f"""</section>
    <script type="text/javascript">
        requirejs(['faust_common', 'svg-pan-zoom', 'sortable', 'jquery', 'jquery.table'],
        function(Faust, svgPanZoom, Sortable, $, $table) {{
            document.getElementById('breadcrumbs').appendChild(Faust.createBreadcrumbs({json.dumps(breadcrumbs)}));
            {init}
            Sortable.init();
            $("table[data-sortable]").fixedtableheader();
        }});
    </script>
    <?php include "../includes/footer.php"?>"""
    with open(filename, 'wt', encoding='utf-8') as f:
        f.write(prefix)
        f.write(content)
        f.write(suffix)


def report_components(graphs: MacrogenesisInfo):
    target = config.path.report_dir
    logger.info('Writing component overview to %s', target)
    target.mkdir(parents=True, exist_ok=True)
    report = f"""<h3>{len(graphs.conflicts)} stark zusammenhängende Komponenten</h3>
    <p>Stark zusammenhängende Komponenten sind Teilgraphen, in denen jeder Knoten von
    jedem anderen erreichbar ist. Hier ist keine Ordnung möglich, ohne dass Kanten entfernt
    werden.</p>
    """
    sccs = [component for component in nx.strongly_connected_components(graphs.base) if len(component) > 1]
    scc_subgraphs = [nx.subgraph(graphs.base, scc) for scc in sccs]
    scc_table = _report_subgraphs(scc_subgraphs, target, 'scc-{0:02d}')
    report += scc_table.format_table()

    wccs = [nx.subgraph(graphs.working, component) for component in nx.weakly_connected_components(graphs.working)]
    report += f"""<h3>{len(wccs)} schwach zusammenhängende Komponenten</h3>
    <p>Zwischen unterschiedlichen schwach zusammenhängenden Komponenten gibt es keine Verbindungen.</p>"""

    wcc_table = _report_subgraphs(wccs, target, 'wcc-{0:02d}')
    report += wcc_table.format_table()
    write_html(target / 'components.php', report, head="Komponenten")


def _report_subgraphs(removed_edges, out, pattern, breadcrumbs=[], head_pattern='%d'):
    table = HtmlTable().column('Nummer', format_spec='<a href="' + pattern + '">{0}</a>') \
        .column('Dokumente') \
        .column('Relationen') \
        .column('Entfernte Relationen') \
        .column('Quellen', format_spec=lambda s: ", ".join(map(str, s))) \
        .column('Entfernte Quellen', format_spec=lambda s: ", ".join(map(str, s)))
    for index, subgraph in enumerate(removed_edges, start=1):
        refs = len([node for node in subgraph.nodes if isinstance(node, Reference)])
        all_edges = list(subgraph.edges(keys=True, data=True))
        removed_edges = [(u, v, k, attr) for (u, v, k, attr) in all_edges if 'delete' in attr and attr['delete']]
        relations = [(u, v, k, attr) for (u, v, k, attr) in all_edges if 'delete' not in attr or not attr['delete']]
        sources = {attr['source'].citation for u, v, k, attr in relations if 'source' in attr}
        conflict_sources = {attr['source'].citation for u, v, k, attr in removed_edges}
        table.row((index, refs, len(relations), len(removed_edges), sources, conflict_sources))
        write_dot(subgraph, out / (pattern + "-graph.dot").format(index))
        write_html(out / (pattern + '.php').format(index),
                   f"""<object id="refgraph" class="refgraph" type="image/svg+xml"
                          data="{pattern.format(index)}-graph.svg"></object>\n""",
                   graph_id='refgraph', breadcrumbs=breadcrumbs, head=head_pattern.format(index))
    return table


def write_bibliography_stats(graph: nx.MultiDiGraph):
    bibls = defaultdict(Counter)
    for u, v, attr in graph.edges(data=True):
        if 'source' in attr:
            bibls[attr['source'].uri][attr['kind']] += 1
    kinds = sorted({str(kind) for bibl in bibls.values() for kind in bibl.keys()})
    totals = Counter({ref: sum(types.values()) for ref, types in bibls.items()})
    with open('sources.tsv', 'wt', encoding='utf-8') as out:
        writer = csv.writer(out, delimiter='\t')
        writer.writerow(['Reference', 'Weight', 'Total'] + kinds)
        for bibl, total in totals.most_common():
            writer.writerow([bibl, BiblSource(bibl).weight, total] + [bibls[bibl][kind] for kind in kinds])


def _fmt_node(node: Union[Reference, object]):
    """Formats a node by creating a link of possible"""
    if isinstance(node, Reference):
        return f'<a href="{node.filename.stem}">{node}</a>'
    else:
        return format(node)


def _edition_link(ref: Reference):
    """Creates a link or links into the edition for the node."""
    if isinstance(ref, Witness):
        return f'<a href="/document?sigil={ref.sigil_t}">{ref}</a>'
    elif isinstance(ref, Inscription):
        return f'Inskription {ref.inscription} von {_edition_link(ref.witness)}'
    elif isinstance(ref, UnknownRef):
        return ''
    elif isinstance(ref, AmbiguousRef):
        return ", ".join(_edition_link(wit) for wit in ref.witnesses)
    else:
        return format(ref)


def _subgraph_link(*nodes: List[Node], html_content: Optional[str] = None, **options) -> str:
    """
    Creates a link to the dynamic subgraph page for the given nodes.

    Args:
        *nodes: The nodes to visualize.
        html_content: HTML content for the link. If none given, a slider icon is shown.
        **options: Additional parameters for the subgraph generation.

    Returns:
        String containing a HTML link.
    """
    if not config.subgraph_links:
        return ""

    if html_content is None:
        html_content = '<i class="fa fa-sliders"></i>'
    nodestr = ", ".join(str(node) for node in nodes)
    params = dict(nodes=nodestr)
    params.update(options)
    return f'<a href="subgraph?{urllib.parse.urlencode(params)}">{html_content}</a>'


def _fmt_date(d: date, delta: timedelta = timedelta(), na: str = '') -> str:
    """
    Helper to format a given date.

    Args:
        d: The date to format
        delta: If present, a difference to add to d before formatting
        na: When date is a N/A value, return this string instead

    Returns:
        iso-formatted, delta corrected date or the given na string
    """
    if not d or pd.isna(d):
        return na
    else:
        return (d + delta).isoformat()


class RefTable(HtmlTable):
    """
    Builds a table of references.
    """

    def __init__(self, graphs: MacrogenesisInfo, **table_attrs):
        super().__init__(data_sortable="true", **table_attrs)
        (self.column('Nr.', data_sortable_type="numericplus")
         .column('(BL)', data_sortable_type="numericplus", format_spec=lambda f: str(int(f)) if f else '')
         .column('Knoten davor', data_sortable_type="numericplus")
         .column('Objekt', data_sortable_type="sigil", format_spec=_fmt_node)
         .column('Typ / Edition', data_sortable_type="sigil", format_spec=_edition_link)
         .column('nicht vor', data_sortable_type="alpha", format_spec=partial(_fmt_date, delta=DAY))
         .column('nicht nach', data_sortable_type="alpha", format_spec=partial(_fmt_date, delta=-DAY))
         .column('erster Vers', data_sortable_type="numericplus")
         .column('Aussagen', data_sortable_type="numericplus")
         .column('<a href="conflicts">Konflikte</a>', data_sortable_type="numericplus"))
        self.graphs = graphs
        self.base = graphs.base

    def reference(self, ref: Reference, index: Optional[int] = None, write_subpage: bool = False):
        """
        Adds the given reference to the table.

        Args:
            ref: the object for which the row will be created
            index: if present, the index to display for the node.
            write_subpage: if true, create a subpage with all assertions for the reference
        """
        details = self.graphs.details
        if ref in self.base:
            if index is None:
                index = self.graphs.index.get(ref, -1)
            if isinstance(ref, SplitReference):
                wit = ref.reference
                start_node = side_node(self.base, wit, Side.START)
                end_node = side_node(self.base, wit, Side.END)
            else:
                start_node, wit, end_node = ref, ref, ref
            assertions = list(
                    chain(self.base.in_edges(start_node, data=True), self.base.out_edges(end_node, data=True)))
            conflicts = [assertion for assertion in assertions if 'delete' in assertion[2] and assertion[2]['delete']]
            self.row(
                    (f'<a href="refs#idx{index}">{index}</a>', details.baseline_position[wit], details.loc[wit, 'rank'],
                     wit, wit, details.max_before_date[wit], details.min_after_date[wit],
                     getattr(wit, 'min_verse', ''), len(assertions), len(conflicts)),
                    id=f'idx{index}', class_=type(wit).__name__)
            if write_subpage:
                self._last_ref_subpage(ref)
        else:
            wit = ref.reference if isinstance(ref, SplitReference) else ref
            self.row(
                    (f'<a href="refs#idx{index}">{index}</a>', details.baseline_position[wit], details.loc[wit, 'rank'],
                     wit, wit, details.max_before_date[wit], details.min_after_date[wit],
                     getattr(wit, 'min_verse', ''), 0, 0),
                    # (index, 0, format(wit), wit, '', '', getattr(wit, 'min_verse', ''), ''),
                    class_='pure-fade-40', title='Keine Macrogenesedaten', id=f'idx{index}')

    def _last_ref_subpage(self, ref):
        """Writes a subpage for ref, but only if it’s the last witness we just wrote"""
        target = config.path.report_dir
        if isinstance(ref, SplitReference):
            start_node = side_node(self.base, ref, Side.START)
            end_node = side_node(self.base, ref, Side.END)
            wit = ref.reference
        else:
            start_node, end_node, wit = ref, ref, ref
        basename = target / wit.filename

        ref_subgraph = self.graphs.subgraph(ref, context=True, abs_dates=True, direct_assertions=True)
        write_dot(ref_subgraph, basename.with_name(basename.stem + '-graph.dot'), highlight=ref)
        report = f"<!-- {repr(ref)} -->\n"
        report += self.format_table(self.rows[-1:])
        report += f"""<object id="refgraph" class="refgraph" type="image/svg+xml" data="{basename.with_name(
                basename.stem + '-graph.svg').name}"></object>
                {_subgraph_link(wit, abs_dates=True, assertions=True, ignored=True, induced_edges=True)}\n"""
        kinds = {'not_before': 'nicht vor',
                 'not_after': 'nicht nach',
                 'from_': 'von',
                 'to': 'bis',
                 'when': 'am',
                 'temp-syn': 'ca. gleichzeitig',
                 'temp-pre': 'entstanden nach',
                 'orphan': '(Verweis)',
                 'inscription': 'Inskription von',
                 'progress': 'Schreibverlauf',
                 None: '???'
                 }
        assertionTable = (HtmlTable(data_sortable='true')
                          .column('berücksichtigt?', data_sortable_type='alpha')
                          .column('Aussage', data_sortable_type='alpha')
                          .column('Bezug', data_sortable_type='sigil', format_spec=_fmt_node)
                          .column('Quelle', data_sortable_type='bibliography', format_spec=_fmt_source)
                          .column('Kommentare', format_spec=_fmt_comments)
                          .column('XML', format_spec=_fmt_xml))
        for (u, v, attr) in self.base.in_edges(start_node, data=True):
            delete_ = 'delete' in attr and attr['delete']
            assertionTable.row((
                f'<a href="{pathlink(u, v)}">nein</a>' if attr.get('delete', False) else \
                    'ignoriert' if attr.get('ignore', False) else 'ja',
                kinds[attr['kind']],
                u + DAY if isinstance(u, date) else u,
                attr,
                attr.get('comments', []),
                attr.get('xml', [])),
                    class_='delete' if delete_ else str(attr['kind']))
        kinds['temp-pre'] = 'entstanden vor'
        for (u, v, attr) in self.base.out_edges(end_node, data=True):
            delete_ = 'delete' in attr and attr['delete']
            assertionTable.row((f'<a href="{pathlink(u, v).stem}">nein</a>' if delete_ else 'ja',
                                kinds[attr['kind']],
                                v - DAY if isinstance(v, date) else v,
                                attr,
                                attr.get('comments', []),
                                attr.get('xml', [])),
                               class_='delete' if delete_ else str(attr['kind']))
        write_html(basename.with_suffix('.php'), report + assertionTable.format_table(),
                   breadcrumbs=[dict(caption='Referenzen', link='refs')],
                   head=ref.label, graph_id='refgraph')


class AssertionTable(HtmlTable):

    def __init__(self, **table_attrs):
        super().__init__(data_sortable='true', **table_attrs)
        (self.column('berücksichtigt?', data_sortable_type="alpha")
         .column('Subjekt', _fmt_node, data_sortable_type="sigil")
         .column('Relation', RELATION_LABELS.get, data_sortable_type="alpha")
         .column('Objekt', _fmt_node, data_sortable_type="sigil")
         .column('Quelle', _fmt_source, data_sortable_type="bibliography")
         .column('Kommentare', _fmt_comments, data_sortable_type="alpha")
         .column('XML', _fmt_xml, data_sortable_type="alpha"))

    def edge(self, u: Reference, v: Reference, attr: Dict[str, object]):
        classes = [attr['kind']] if 'kind' in attr and attr['kind'] is not None else ['unknown-kind']
        if attr.get('ignore', False): classes.append('ignore')
        if attr.get('delete', False): classes.append('delete')
        self.row((
            f'<a href="{Path(pathlink(u, v)).stem}">nein</a>' if attr.get('delete', False) else \
                'ignoriert' if attr.get('ignore', False) else 'ja',
            u,
            attr['kind'],
            v,
            attr,
            attr.get('comments', []),
            attr.get('xml', '')),
                class_=' '.join(classes))


def _fmt_source(attrs):
    source: BiblSource = attrs['source']
    result = f'<a href="{source.filename}">{source.citation}</a>'
    if 'sources' in attrs:
        result += ' ' + '; '.join(s.detail for s in attrs['sources'])
    elif source.detail:
        result += ' ' + source.detail
    return result


def _source_link(file: Path, line: Optional[Union[int, str]] = None, text: Optional[Union[int, str]] = None) -> str:
    file = file.relative_to('macrogenesis')
    if text is None:
        text = file
        if line is not None:
            text += ': ' + str(line)
    href = os.path.join(config.xmlroot, os.fspath(file))
    if line is not None:
        href += '#L' + str(line)
    return f'<a href="{href}">{text}</a>'


def _fmt_xml(xml: Union[Tuple[Path, int], Sequence[Tuple[str, int]]]):
    if not xml:
        return ""

    if isinstance(xml[0], Path):
        return _source_link(xml[0], xml[1], f"{xml[0].relative_to('macrogenesis')}: {xml[1]}")
    else:
        result = []
        for file, lines in groupby(xml, itemgetter(0)):
            result.append(_source_link(file) + ': ' + ", ".join(_source_link(file, line, line) for (_, line) in lines))
        return "; ".join(result)


def _fmt_comments(comments):
    if not comments:
        return ""
    return " / ".join(str(comment) for comment in comments if comment)


def report_downloads(graphs: MacrogenesisInfo):
    target = config.path.report_dir
    target.mkdir(exist_ok=True, parents=True)

    simplified = simplify_graph(graphs.base)
    nx.write_gexf(simplified, str(target / 'base.gexf'))
    nx.write_edgelist(simplified, str(target / 'base.edges'))
    graphs.save(target / "macrogen-info.zip")

    write_html(target / 'downloads.php', """
    <section>
    <p>Downloadable files for the base graph in various formats:</p>
    <ol>
        <li><a href='base.gexf'>GEXF</a></li>
        <li><a href='base.edges'>Edge List</a></li>
        <li><a href='macrogen-info.zip'>MacrogenesisInfo for the faust-macrogen library</a></li>
    <ol>
    </section>
    <h4>Nodes</h4>
    <p>The <strong>nodes</strong> are either URIs or dates in ISO 8601 format. URIs of the form
       <code>faust://document/<var>scheme</var>/<var>sigil</var></code> denote a witness (document)
       that has the identifier <var>sigil</var> in the respective identifier scheme.
       <code>faust://inscription/<var>scheme</var>/<var>sigil</var>/<var>id</var></code> denote an
       inscription (single “writing event”) on the respective document.</p>
        <p>If some URI has a <var>scheme</var> ≠ <code>faustedition</code>, then it was not possible to map it to
       a document in the edition. You may still try the sigil with the search. Otherwise, the document can be
       displayed at <code>http://faustedition.net/document?sigil=<var>sigil</var></code>.
       </p>
    <p>Dates are always of the form YYYY-MM-DD.</p>

    <h4>Edges</h4>
    <p>The edges have attributes that describe them further:</p>
    <table class="pure-table">
      <tr>
        <th>Attribute</th><th>Value(s)</th><th>Meaning (<var>u</var> → <var>v</var>)</th>
      </tr>
      <tr>
        <td rowspan="3">kind</td>
        <td>
            <code>not_before</code>, <code>not_after</code>, <code>from_</code>, <code>to_</code>, <code>when</code>,
            <code>timeline</code>
        </td>
        <td>These all essentially mean <var>u</var> happened before <var>v</var></td>
      </tr>
      <tr>
        <td><code>timeline</code></td>
        <td><var>v</var> is the next date after date node <var>u</var></td>
      </tr>
      <tr>
        <td><code>temp-syn</code></td>
        <td><var>u</var> and <var>v</var> happened about at the same time
      </tr>
      <tr>
        <td>ignore</td>
        <td>boolean</td>
        <td>if present and true, this edge is to be ignored for philological reasons</td>
      </tr>
      <tr>
         <td>delete</td>
         <td>boolean</td>
         <td>if present and true, this edge has been removed by the minimum feedback edge set heuristics</td>
      </tr>
      <tr>
          <td>weight</td>
          <td>positive integer</td>
          <td>trust in edge, generated mainly from sources</td>
      </tr>
      <tr>
        <td>source</td>
        <td>URI</td>
        <td>URI identifying the source for this assertion</td>
      </tr>
      <tr>
         <td>xml</td>
         <td></td>
         <td>file name and line of the <a href="https://github.com/faustedition/faust-xml/tree/master/xml/macrogenesis">XML file</a>
          with this assertion</td>
      </tr>
    </table>

    <h4>MacrogenesisInfo</h4>
    <p>The <a href='macrogen-info.zip'>macrogen-info.zip</a> file contains the data required to recreate the graph info
    in the <a href='https://github.com/faustedition/faust-macrogen'>faust-macrogen</a> library. To do so, run:</p>
    <pre lang='python'>
    from macrogen import MacrogenesisInfo
    graphs = MacrogenesisInfo('macrogen-info.zip')
    </pre>
    <p>
    """, "Downloads")


def report_refs(graphs: MacrogenesisInfo):
    # Fake dates for when we don’t have any earliest/latest info

    target = config.path.report_dir
    target.mkdir(exist_ok=True, parents=True)

    nx.write_yaml(simplify_graph(graphs.base), str(target / 'base.yaml'))
    nx.write_yaml(simplify_graph(graphs.working), str(target / 'working.yaml'))
    nx.write_yaml(simplify_graph(graphs.dag), str(target / 'dag.yaml'))

    nx.write_gpickle(graphs.dag, str(target / 'dag.gpickle'))
    nx.write_gpickle(graphs.working, str(target / 'working.gpickle'))
    nx.write_gpickle(graphs.base, str(target / 'base.gpickle'))

    refs = graphs.order_refs_post_model()
    overview = RefTable(graphs)

    for index, ref in enumerate(refs, start=1):
        overview.reference(ref, index, write_subpage=True)

    write_html(target / 'refs.php', overview.format_table(), head="Referenzen")

    # write_dot(simplify_graph(graphs.base), str(target / 'base.dot'), record=False)
    # write_dot(simplify_graph(graphs.working), str(target / 'working.dot'), record=False)
    # write_dot(simplify_graph(graphs.dag), str(target / 'dag.dot'), record=False)


def _invert_mapping(mapping: Mapping) -> Dict:
    result = defaultdict(set)
    for key, value in mapping.items():
        result[value].add(key)
    return result


def _flatten(items: List) -> List:
    """
    Flattens and cleans a potentially nested list by removing intermediate list layers and
    contained Nones as well as removing duplicates.

    Examples:
        >>> _flatten([1, 2, [3, [4, None]]])
        [1, 2, 3, 4]
    """
    result = []
    for item in items:
        if isinstance(item, list):
            result.extend(_flatten(item))
        elif item is not None:
            result.append(item)
    return sorted(set(result))


def report_missing(graphs: MacrogenesisInfo):
    target = config.path.report_dir
    refs = {node.reference if isinstance(node, SplitReference) else node
            for node in graphs.base.nodes if isinstance(node, Reference)}
    all_wits = {wit for wit in Witness.database.values() if isinstance(wit, Witness)}
    used_wits = {wit for wit in refs if isinstance(wit, Witness)}
    unknown_refs = {wit for wit in refs if isinstance(wit, UnknownRef)}
    missing_wits = all_wits - used_wits
    inscr = {inscr: inscr.witness for inscr in refs if isinstance(inscr, Inscription)}
    wits_with_inscr = _invert_mapping(inscr)
    report = f"""
    <h2>Fehlende Zeugen</h2>
    <p>Für {len(missing_wits)} von insgesamt {len(used_wits)} Zeugen <a href="#missing">liegen keine Makrogenesedaten
       vor</a>. Bei {len(missing_wits & wits_with_inscr.keys())} davon sind zumindest Informationen über
       Inskriptionen hinterlegt. Umgekehrt gibt es <a href="#unknown">zu {len(unknown_refs)} Referenzen in der
       Makrogenese</a> keine Entsprechung in der Edition.</p>
    <h3 id="missing">Zeugen ohne Makrogenesedaten</h3>
    """
    missing_table = (HtmlTable()
                     .column('Zeuge ohne Daten', format_spec=_edition_link)
                     .column('Inskriptionen', format_spec=lambda refs: ", ".join(map(_fmt_node, refs))))
    report += missing_table.format_table(
            (ref, wits_with_inscr[ref]) for ref in sorted(missing_wits, key=lambda r: r.sort_tuple()))
    report += '\n<h3 id="unknown">Unbekannte Referenzen</h3>'
    unknown_table = (HtmlTable()
                     .column('Referenz', _fmt_node)
                     .column('URI')
                     .column('XML', _fmt_xml, style='width:50%'))
    for ref in sorted(unknown_refs):
        assertions: nx.MultiDiGraph = graphs.subgraph(ref, context=False, abs_dates=False, direct_assertions=True)
        unknown_table.row((ref, ref.uri, _flatten([x for u, v, x in assertions.edges(data='xml')])))
    report += unknown_table.format_table()
    write_html(target / 'missing.php', report, head="Fehlendes")
    for ref in missing_wits:
        missing_path = target / ref.filename.with_suffix('.php')
        logger.info('Writing missing page for %s to %s', ref, missing_path)
        write_html(missing_path,
                   f"""
                   <p class="pure-alert pure-alert-warning"><strong>Für <a href="/document?sigil={ref.filename.stem}">{ref}</a> liegen noch keine Makrogenesedaten vor.</strong>
                   Ggf. fehlt auch nur die Zuordnung zur richtigen Sigle – siehe in der <a href="refs">Liste der Referenzen</a>.</p>
                   """,
                   head=format(ref))


def _format_collapsed_path(path: List[Node]):
    collapsed = []
    for i in range(len(path)):
        if 0 < i < len(path) - 1 and isinstance(path[i], date) and isinstance(path[i - 1], date) and isinstance(
                path[i + 1], date):
            continue
        collapsed.append(path[i])
    return " → ".join(map(_fmt_node, collapsed))


def _report_conflict(graphs: MacrogenesisInfo, u, v):
    target = config.path.report_dir
    reportfile = pathlink(u, v)
    cyclefile = reportfile.with_name(reportfile.stem + '-cycles.php')
    graphfile = reportfile.with_name(reportfile.stem + '-graph.dot')
    cyclegraphfile = reportfile.with_name(graphfile.stem + '-graph.dot')
    relevant_nodes = {u} | set(graphs.base.predecessors(u)) | set(graphs.base.successors(u)) \
                     | {v} | set(graphs.base.predecessors(v)) | set(graphs.base.successors(v))
    counter_path = []
    try:
        counter_path = nx.shortest_path(graphs.dag, v, u, weight='iweight')
        involved_cycles = {cycle for cycle in graphs.simple_cycles if in_path((u, v), cycle, True)}
        relevant_nodes = set(counter_path)

        # counter_desc = " → ".join(map(_fmt_node, counter_path))
        counter_desc = _format_collapsed_path(counter_path)
        counter_html = f'<p><strong><a href="{reportfile.stem}">Pfad in Gegenrichtung:</a></strong> {counter_desc}</p>'
    except nx.NetworkXNoPath:
        counter_html = f'<p>kein Pfad in Gegenrichtung ({_fmt_node(v)} … {_fmt_node(u)}) im Sortiergraphen</p>'
    except nx.NodeNotFound:
        logger.exception('Node not found!? %s or %s', u, v)
        counter_html = ''
    counter_graph: nx.MultiDiGraph = nx.subgraph(graphs.base, relevant_nodes)
    for _, _, trigger_node in counter_graph.edges(data='trigger_node'):
        if trigger_node:
            relevant_nodes.add(trigger_node)
            logger.info('Adding trigger node %s to %s', trigger_node, reportfile.stem)
    counter_graph: nx.MultiDiGraph = nx.subgraph(graphs.base, relevant_nodes).copy()

    if involved_cycles:
        counter_html += f"<p>Teil von mindestens <a href='{cyclefile.stem}'>{len(involved_cycles)} einfachen Zyklen</a></p>"

    # Highlight conflicting edges, counter path and the two nodes of the conflicting edge(s)
    for v1, v2 in [(u, v)] + list(pairwise(nx.shortest_path(counter_graph, v, u, weight='iweight'))):
        for k, attr in counter_graph.get_edge_data(v1, v2).items():
            attr['highlight'] = True
    counter_graph.node[u]['highlight'] = True
    counter_graph.node[v]['highlight'] = True

    counter_graph = collapse_timeline(counter_graph)
    cycle_graph = counter_graph.copy()

    for cycle in involved_cycles:
        cycle_graph.add_edges_from(expand_edges(graphs.base, pairwise(cycle)))

    write_dot(counter_graph, str(target / graphfile))
    write_dot(cycle_graph, str(target / cyclegraphfile))

    table = AssertionTable()
    for k, attr in graphs.base.get_edge_data(u, v).items():
        table.edge(u, v, attr)

    write_html(target / reportfile,
               f"""
               {table.format_table()}
               {counter_html}
               <object id="refgraph" type="image/svg+xml" data="{graphfile.with_suffix('.svg')}"></object>
               {_subgraph_link(u, v)}
               """,
               graph_id='refgraph',
               head=f'Entfernte Kante {u} → {v}', breadcrumbs=[dict(caption="Entfernte Kanten", link='conflicts')])
    write_html(target / cyclefile,
               f"""
               {table.format_table()}
               {counter_html}
               <object id="refgraph" type="image/svg+xml" data="{cyclegraphfile.with_suffix('.svg')}"></object>
               """,
               graph_id='refgraph',
               head=f'Entfernte Kante {u} → {v}', breadcrumbs=[dict(caption="Entfernte Kanten", link='conflicts')])

    return reportfile


def report_conflicts(graphs: MacrogenesisInfo):
    target = config.path.report_dir
    table = AssertionTable()
    removed_edges = [(u, v, k, attr) for (u, v, k, attr) in graphs.base.edges(keys=True, data=True) if
                     'delete' in attr and attr['delete']]
    for index, (u, v, k, attr) in enumerate(sorted(removed_edges, key=lambda t: getattr(t[0], 'index', 0)), start=1):
        reportfile = _report_conflict(graphs, u, v)
        table.edge(u, v, attr)
    write_html(target / 'conflicts.php', table.format_table(), head=f'{len(removed_edges)} entfernte Kanten')


def report_sources(graphs: MacrogenesisInfo):
    target = config.path.report_dir
    by_source = defaultdict(list)
    for u, v, k, attr in graphs.base.edges(keys=True, data=True):
        if 'source' in attr:
            by_source[attr['source'].uri].append((u, v, k, attr))

    def _fmt_source(uri):
        source = BiblSource(uri)
        return f'<a href="{source.filename}">{source}</a>'

    sources_table = (HtmlTable(data_sortable='true')
                     .column('Quelle', format_spec=_fmt_source, data_sortable_type='bibliography')
                     .column('Aussagen', data_sortable_type='numericplus')
                     .column('entfernt', data_sortable_type='numericplus')
                     .column('Nutzungsanteil', format_spec='{:.1f} %', data_sortable_type='numericplus')
                     .column('Gewicht', data_sortable_type='numericplus')
                     .column('Zeugen', data_sortable_type='numericplus'))
    with (target / "sources_data.csv").open('wt', encoding='utf-8') as sources_csv_file:
        csv_writer = csv.writer(sources_csv_file)
        csv_writer.writerow(('Source', 'Label', 'Weight', 'Assertions', 'Conflicts', 'Ignored', 'Usage', 'Witnesses'))
        for uri, edges in sorted(by_source.items()):
            source = BiblSource(uri)
            filename = target / (source.filename + '.php')
            graphfile = filename.with_name(filename.stem + '-graph.dot')
            logger.info('%d assertions from %s', len(edges), source.citation)
            # subgraph = graphs.base.edge_subgraph([(u,v,k) for u,v,k,attr in edges])
            subgraph = graphs.base.subgraph({u for u, v, k, attr in edges} | {v for u, v, k, attr in edges})
            all_edges = {edge[:3] for edge in edges}
            ignored = {edge[:3] for edge in edges if edge[3].get('ignore', False)}
            conflicts = {edge[:3] for edge in edges if edge[3].get('delete', False)} - ignored
            used = all_edges - conflicts - ignored
            ratio = len(used) / len(conflicts | used) if conflicts | used else 0

            write_dot(subgraph, graphfile)
            witness_count = len([node for node in subgraph.nodes if isinstance(node, Reference)])
            sources_table.row((uri,
                               len(edges),
                               len(conflicts),
                               ratio * 100,
                               source.weight,
                               witness_count))
            csv_writer.writerow((uri, source.citation, source.weight, len(edges), len(conflicts), len(ignored), ratio,
                                 witness_count))
            current_table = AssertionTable()
            for u, v, k, attr in edges:
                current_table.edge(u, v, attr)
            write_html(target / (source.filename + '.php'),
                       f"""<p><a href="/bibliography#{source.filename}">{source.citation} in der Bibliographie</a></p>
                       <object id="refgraph" type="image/svg+xml" data="{graphfile.with_suffix('.svg').name}"></object>
                           {current_table.format_table()}""",
                       graph_id='refgraph',
                       breadcrumbs=[dict(caption='Quellen', link='sources')],
                       head=source.citation)

    write_html(target / 'sources.php', sources_table.format_table(), head='Quellen')


def report_index(graphs):
    target = config.path.report_dir

    pages = [('refs', 'Zeugen', 'Alle referenzierten Dokumente in der erschlossenen Reihenfolge'),
             ('scenes', 'nach Szene', 'Die relevanten Zeugen für jede Szene'),
             ('conflicts', 'entfernte Aussagen',
              'Aussagen, die algorithmisch als Konflikt identifiziert und entfernt wurden'),
             ('components', 'Komponenten', 'stark und schwach zusammenhängende Komponenten des Ausgangsgraphen'),
             ('missing', 'Fehlendes',
              'Zeugen, zu denen keine Aussagen zur Makrogenese vorliegen, und unbekannte Zeugen'),
             ('sources', 'Quellen', 'Aussagen nach Quelle aufgeschlüsselt'),
             ('inscriptions', 'Inskriptionen', 'Inskriptionen in Makrogenese bzw. Transkript'),
             ('dag', 'sortierrelevanter Gesamtgraph',
              'Graph aller für die Sortierung berücksichtigter Aussagen (einzoomen!)'),
             ('tred', 'transitive Reduktion',
              '<a href="https://de.wikipedia.org/w/index.php?title=Transitive_Reduktion">Transitive Reduktion</a> des Gesamtgraphen'),
             ('syn', 'Zeitliche Nähe', 'Zeugengruppen, für die Aussagen über ungefähre Gleichzeitigkeit vorliegen'),
             ('timeline', 'Zeitstrahl', 'Zeitstrahl datierter Zeugen'),
             ('help', 'Legende', 'Legende zu den Graphen'),
             ('stats', 'Statistik', 'Der Graph in Zahlen'),
             ('config', 'Konfiguration', 'Einstellungen, mit denen der Graph generiert wurde'),
             ('downloads', 'Downloads', 'Graphen zum Download'),
             ]
    links = "\n".join(
            ('<tr><td><a href="{}" class="pure-button pure-button-tile">{}</td><td>{}</td></tr>'.format(*page) for page
             in pages))
    report = f"""
      <p>
        Dieser Bereich der Edition enthält experimentelle Informationen zur Makrogenese, er wurde zuletzt
        am {datetime.now()} generiert.
      </p>
      <section class="center pure-g-r">

        <article class="pure-u-1">
            <table class="pure-table">
                {links}
            </table>
        </article>

      </section>
    """

    write_html(target / "index.php", report)
    logger.info('Writing DAG ...')
    write_dot(graphs.dag, target / 'dag-graph.dot')
    write_html(target / 'dag.php', '<object type="image/svg+xml" data="dag-graph.svg" id="refgraph"/>',
               graph_id='refgraph', head='Effektiver Gesamtgraph (ohne Konflikte)',
               graph_options=dict(controlIconsEnabled=True, maxZoom=200))

    logger.info('Creating transitive reduction ...')
    tred_base = nx.MultiDiGraph(nx.transitive_reduction(graphs.dag))
    tred = nx.edge_subgraph(graphs.dag, tred_base.edges)
    write_dot(tred, target / 'tred-graph.dot')
    write_html(target / 'tred.php', '<object type="image/svg+xml" data="tred-graph.svg" id="refgraph"/>',
               graph_id='refgraph', head='Transitive Reduktion',
               graph_options=dict(controlIconsEnabled=True, maxZoom=200))


def report_help(info: Optional[MacrogenesisInfo] = None):
    target = config.path.report_dir

    def demo_graph(u, v, extend=None, **edge_attr) -> nx.MultiDiGraph:
        G = nx.MultiDiGraph() if extend is None else extend.copy()
        G.add_edge(u, v, **edge_attr)
        return G

    w1 = Witness({'uri': 'faust://help/wit1', 'sigil': 'Zeuge 1', 'sigil_t': 'Zeuge_1'})
    w2 = Witness({'uri': 'faust://help/wit2', 'sigil': 'Zeuge 2', 'sigil_t': 'Zeuge_2'})
    d1 = date(1799, 1, 17)
    d2 = date(1799, 2, 5)
    d3 = date(1799, 1, 28)

    g1 = demo_graph(w1, w2, kind='temp-pre', label='Quelle 1')
    g1a = g1.copy()
    g1a.add_edge(w2, w1, kind='temp-pre', delete=True, label='Quelle 2')
    g1a.add_edge(w2, w1, kind='temp-pre', ignore=True, label='Quelle 3')
    g2 = demo_graph(w1, w2, kind='temp-syn', label='Quelle 2')
    g3 = demo_graph(d1 - DAY, w1, kind='not_before', source='Quelle 1')
    g3.add_edge(w1, d2 + DAY, kind='not_after', source='Quelle 1')
    g4 = demo_graph(d1 - DAY, w1, kind='from_', source='Quelle 2')
    g4.add_edge(w1, d2 + DAY, kind='to_', source='Quelle 2')
    g5 = demo_graph(d3 - DAY, w2, kind='when', source='Quelle 2')
    g5.add_edge(w2, d3 + DAY, kind='when', source='Quelle 2')

    i1 = Witness.get('faust://inscription/faustedition/2_IV_H.19/i_uebrige')
    i1w = i1.witness
    g_orphan = demo_graph(i1, i1w, kind='orphan', source='Datierungsinhalt für')

    i1 = Inscription(w1, 'i_1')
    g6 = nx.MultiDiGraph()
    g6.add_edge(d1 - DAY, i1, kind='not_before', label='Quelle 1')
    g6.add_edge(i1, w1, kind='inscription', label='Inskription von')
    g6.add_edge(d1 - DAY, w1, copy=True, kind='not_before', label='Quelle 1')

    g7 = nx.MultiDiGraph()
    hp47 = Witness.get('faust://document/faustedition/H_P47')
    g7.add_edge(date(1808, 9, 30), hp47, source='Bohnenkamp 1994')
    g7.add_edge(hp47, date(1809, 3, 31), source=BiblSource('faust://heuristic', 'Bohnenkamp 1994'))

    help_graphs = dict(pre=g1, conflict=g1a, syn=g2, dating=g3, interval=g4, when=g5, orphan=g_orphan, copy=g6,
                       heuristic=g7)
    for name, graph in help_graphs.items():
        write_dot(graph, str(target / f'help-{name}.dot'))

    report = f"""
    <p>Die Graphen repräsentieren Aussagen aus der Literatur zu zeitlichen Verhältnissen von Zeugen.</p>
    <p>Die <strong>Zeugen</strong> sind durch Ovale mit der Sigle oder Inskriptionsbezeichnung repräsentiert.
       Zeugen, die nicht in der Edition sind, steht <code>siglentyp: </code> voran. Die Zeugenbezeichnungen
       sind klickbar und führen zur Makrogeneseseite des entsprechenden Zeugen. Verwenden Sie den Link in der
       Spalte <em>Edition</em> der oberen Tabelle, um zur Darstellung des Zeugen in der Dokumentenansicht zu
       gelangen.</p>
    <p><strong>Pfeile</strong> bedeuten immer <em>zeitlich vor</em>. Im Vergleich zu termini a quo bzw. ad quem
    sind die Datumsangaben deshalb um einen Tag versetzt.</p>
    <p><a href="/intro_macrogenesis">Lesen Sie eine ausführlichere Einführung zum Makrogenese-Lab.</a></p>
    <table class="pure-table">
        <thead><th>Graph</th><th>Bedeutung</th></thead>
        <tbody>
        <tr><td><img src="help-pre.svg" /></td>
            <td>Laut Quelle 1 entstand {w1} vor {w2}</td></tr>
        <tr><td><img src="help-conflict.svg" /></td>
            <td>Laut Quelle 1 entstand {w1} vor {w2},
                laut Quelle 2 und Quelle 3 entstand {w2} vor {w1}.<br/>
                Die Aussage von Quelle 2 wird bei der heuristischen Konfliktbeseitugung aus dem Graphen entfernt.<br/>
                Die Aussage von Quelle 3 bereits vor der heuristischen Konfliktbeseitigung aus dem Graphen entfernt (Quelle allgemein unglaubwürdig oder konkrete Aussage unbegründet).</td></tr>
        <tr><td><img src="help-syn.svg"/></td>
            <td>Laut Quelle 2 entstand {w1} etwa gleichzeitig mit {w2}.</td></tr>
        <tr><td><img src="help-dating.svg"/></td>
            <td>Laut Quelle 1 entstand {w1} nicht vor {d1} und nicht nach {d2}.
                Der gepunktete Pfeil repräsentiert den Zeitstrahl, er führt immer zum nächsten im Graph repräsentierten Datum.
            </td></tr>
        <tr><td><img src="help-interval.svg"/></td>
            <td>Laut Quelle 2 wurde vom {d1} bis zum {d2} an {w1} gearbeitet.</td></tr>
        <tr><td><img src="help-when.svg"/></td>
            <td>Laut Quelle 2 entstand {w2} am {d3}.</td></tr>
        <tr><td><img src="help-orphan.svg"/></td>
            <td>{i1w} wird in den Makrogenesedaten nur indirekt über {i1} referenziert
                und über eine künstliche Kante angebunden.</td></tr>
        <tr><td><img src="help-copy.svg"/></td>
            <td>Die Aussage über {i1} aus den Makrogenesedaten wurde kopiert, sodass sie auch für {w1} gilt.</td></tr>
        <tr><td><img src="help-heuristic.svg"/></td>
            <td>Da zu {hp47} nur ein terminus a quo bekannt ist, wurde ein künstlicher terminus ante quem 6 Monate nach der Datierung durch Bohnenkamp 1994 ergänzt.</td></tr>
        </tbody>
    </table>
    """

    write_html(target / 'help.php', report, 'Legende')


def report_scenes(graphs: MacrogenesisInfo):
    target = config.path.report_dir
    sceneTable = (HtmlTable(data_sortable="true")
                  .column('#')
                  .column('Szene')
                  .column('Verse', format_spec=lambda t: '{} – {}'.format(*t))
                  .column('Dokumente')
                  .column('Inskriptionen')
                  .column('Gesamt'))
    for scene in SceneInfo.get().scenes:
        items = WitInscrInfo.get().by_scene[scene]
        witnessTable = RefTable(graphs)
        scene_docs = [doc for doc in items if isinstance(doc, DocumentCoverage)]
        scene_inscr = [inscr for inscr in items if isinstance(inscr, InscriptionCoverage)]
        scene_refs = scene_docs + scene_inscr
        scene_wits = {graphs.node(doc.uri, default=None) for doc in scene_refs} - {None}
        scene_graph = graphs.subgraph(*scene_wits, context=False, abs_dates=True, paths_between_nodes=False)
        for wit in sorted(scene_wits, key=lambda ref: graphs.index.get(ref, 0)):
            witnessTable.reference(wit)
        basename = 'scene_' + scene.n.replace('.', '-')
        subgraph_page = Path(basename + '-subgraph.php')
        graph_name = Path(basename + '-graph.dot')
        sceneTable.row((scene.n, f'<a href="{basename}">{scene.title}</a>', (scene.first, scene.last),
                        len(scene_docs), len(scene_inscr), len(scene_wits)))
        write_dot(scene_graph, target / graph_name)
        write_html(target / subgraph_page,
                   f"""<object id="refgraph" type="image/svg+xml" data="{graph_name.with_suffix(
                           '.svg')}"></object>""",
                   graph_id='refgraph',
                   head="Szenengraph", breadcrumbs=[dict(caption='nach Szene', link='scenes'),
                                                    dict(caption=scene.title, link=basename)])
        write_html(target / (basename + '.php'),
                   f"""
                       <p><a href="{subgraph_page.stem}">Szenengraph</a> ·
                          {_subgraph_link(*scene_wits)} ·
                       <a href="/genesis_bargraph?rangeStart={scene.first}&amp;rangeEnd={scene.last}">Balkendiagramm</a></p>
                       {witnessTable.format_table()}""",
                   head=scene.title, breadcrumbs=[dict(caption='nach Szene', link='scenes')])

    write_html(target / "scenes.php", sceneTable.format_table(), head='nach Szene')


def report_syngroups(graphs: MacrogenesisInfo):
    clusters = temp_syn_groups(graphs.base)
    target: Path = config.path.report_dir
    table = (HtmlTable(data_sortable="true")
             .column('Referenzen', data_sortable_type='sigil')
             .column('Anzahl', data_sortable_type='numericplus')
             .column('Aussagen', data_sortable_type='numericplus'))
    for cluster in clusters:
        representant = min(cluster, key=lambda item: item.filename)
        file_stem = f"syn-{representant.filename.stem}"
        subgraph = graphs.subgraph(*cluster)
        write_dot(subgraph, target / (file_stem + "-graph.dot"), highlight=list(cluster))
        assertions = AssertionTable()
        for u, v, attr in graphs.base.subgraph(cluster).edges(data=True):
            if attr.get('kind', None) == 'temp-syn':
                assertions.edge(u, v, attr)
        write_html(target / (file_stem + '.php'),
                   f"""<object id="refgraph" type="image/svg+xml" data="{file_stem}-graph.svg"></object>\n""" +
                   _subgraph_link(*cluster) +
                   assertions.format_table(),
                   graph_id='refgraph', head=f'Zeitgleich mit {representant.label}',
                   breadcrumbs=[dict(caption='ca. gleichzeitig', link='syn')])
        table.row((f'<a href="{file_stem}">{", ".join(map(str, cluster))}</a>', len(cluster), len(assertions)))
    write_html(target / 'syn.php', table.format_table(), head='ca. gleichzeitig')


def report_unused(graphs: MacrogenesisInfo):
    unused_nodes = set(node for node in graphs.base.node if isinstance(node, Reference)) - set(graphs.dag.node)
    not_in_dag_table = RefTable(graphs)
    for node in unused_nodes:
        not_in_dag_table.reference(node)

    unindexed = [node for node in graphs.base.node if isinstance(node, Reference) and not node in graphs.index]
    unindexed_table = RefTable(graphs)
    for node in unindexed:
        unindexed_table.reference(node)

    write_html(config.path.report_dir / 'unused.php',
               f"""<p>{len(unused_nodes)} Zeugen existieren im Ausgangsgraphen, aber nicht im DAG:</p>
               {not_in_dag_table.format_table()}
               <p>{len(unindexed)} Knoten haben auf unerklärliche Weise keinen Index:</p>
               {unindexed_table.format_table()}
               """,
               "Nicht eingeordnete Zeugen")


def write_order_xml(graphs: MacrogenesisInfo):
    order_xml: Path = config.path.order or config.path.report_dir / 'order.xml'
    logger.debug('Writing order file to %s', order_xml.absolute())
    ordered_refs = graphs.order_refs_post_model()
    if any(isinstance(ref, SplitReference) for ref in ordered_refs):
        ordered_refs = [sr.reference for sr in ordered_refs]
    ordered_wits = [ref for ref in ordered_refs if isinstance(ref, Witness)]
    F = ElementMaker(namespace='http://www.faustedition.net/ns', nsmap=config.namespaces)
    root = F.order(
            Comment('This file has been generated from the macrogenesis data. Do not edit.'),
            *[F.item(format(row.Index),
                     index=format(row.position),
                     uri=row.uri,
                     sigil_t=row.Index.sigil_t,
                     earliest=(row.max_before_date + DAY).isoformat() if not pd.isna(row.max_before_date) else '',
                     latest=(row.min_after_date - DAY).isoformat() if not pd.isna(row.min_after_date) else '',
                     yearlabel=row.yearlabel)
              for row in graphs.details.query('kind == "Witness"').itertuples(index=True)],
            generated=datetime.now().isoformat())
    order_xml.parent.mkdir(parents=True, exist_ok=True)
    root.getroottree().write(str(order_xml), pretty_print=True)

    stats = graphs.year_stats()
    data = dict(max=max(stats.values()), counts=stats)
    config.path.report_dir.mkdir(exist_ok=True, parents=True)
    with (config.path.report_dir / 'witness-stats.json').open('wt', encoding='utf-8') as out:
        json.dump(data, out)

    graphs.details.to_csv(config.path.report_dir / 'ref_info.csv', date_format='iso')


def report_stats(graphs: MacrogenesisInfo):
    """
    Anzahl Zeugen; Referenzen
    Anzahl absolute, relative Datierungen
    Anzahl datierte Zeugen
    Anzahl automatisch datierte Zeugen
    Auto Datierungsintervallänge

    Args:
        graphs:

    Returns:

    """
    refs = graphs.order_refs()
    extra_refs_count = len(refs) - len(set(refs))
    logger.warning('Extra ref count: %s', extra_refs_count)
    wits = [ref for ref in refs if isinstance(ref, Witness)]
    stat: DataFrame = pd.DataFrame(index=refs, columns=['kind', 'pred', 'pred_dates', 'pre', 'post', 'succ',
                                                        'succ_dates', 'auto_pre', 'auto_post', 'auto_len'])

    # now collect some info per witness:
    for ref in refs:
        try:
            preds = list(graphs.base.pred[ref]) if ref in graphs.base else []
            succs = list(graphs.base.succ[ref]) if ref in graphs.base else []

            pred_dates = [p for p in preds if isinstance(p, date)]
            succ_dates = [s for s in succs if isinstance(s, date)]

            if ref in graphs.closure:
                auto_pre = max((d for d in graphs.closure.pred[ref] if isinstance(d, date)), default=None)
                auto_post = min((d for d in graphs.closure.succ[ref] if isinstance(d, date)), default=None)
            else:
                auto_pre = auto_post = None
            row = dict(
                    kind=ref.__class__.__name__,
                    pred=len(preds),
                    pred_dates=len(pred_dates),
                    pre=max((d for d in pred_dates), default=None),
                    post=min((d for d in succ_dates), default=None),
                    succ=len(succs),
                    succ_dates=len(succ_dates),
                    in_closure=ref in graphs.closure,

                    auto_pre=auto_pre,
                    auto_post=auto_post,
                    auto_len=auto_post - auto_pre if auto_pre and auto_post else None
            )
            stat.loc[ref, :] = row
        except Exception as e:
            logger.error("Could not record %s in stats", ref, exc_info=True)

    def _dating_table():
        for dating in get_datings():
            if isinstance(dating, AbsoluteDating):
                for item in dating.items:
                    for source in dating.sources:
                        yield item, item.__class__.__name__, dating.start, dating.end, source

    dating_stat = pd.DataFrame(list(_dating_table()), columns='item kind start end source'.split())

    edge_df = pd.DataFrame([dict(start=u, end=v, key=k, **attr)
                            for u, v, k, attr in graphs.base.edges(keys=True, data=True)])
    edge_df['delete'] = edge_df['delete'].fillna(False) if 'delete' in edge_df.columns else False
    edge_df.ignore = edge_df.ignore.fillna(False)

    html = f"""
    <h2>Kanten (Aussagen)</h2>
    <p>{len(edge_df)} Kanten, {(edge_df.kind != 'timeline').sum()} Datierungsaussagen:</p>
    <pre>{edge_df.kind.value_counts()}</pre>
    <p>{edge_df.ignore.sum()} Aussagen (manuell) ignoriert,
    {edge_df.delete.sum()} widersprüchliche Aussagen (automatisch) ausgeschlossen
    ({len(edge_df[edge_df.delete].groupby(['start', 'end']))} ohne Parallelaussagen)
    </p>
    
    <h2>Sortierung</h2>
    <p>Korrelation um <strong><var>ρ</var> = {graphs.spearman_rank_correlation():+.5f}</strong> ∈ [-1, +1] 
       mit einer Sortierung ohne Makrogeneseinformationen.</p>

    <h2>Absolute Datierungen</h2>
    <table class="pure-table">

        <thead><tr><td/><th>direkt</th><th>erschlossen</th><th>angepasst</th><th>fehlend</th></tr></thead>
        <tbody>
        <tr><th>Datumsuntergrenze</th>
            <td>{(stat.pred_dates > 0).sum()}</td>
            <td>{(~stat.auto_pre.isna() & ~(stat.pred_dates > 0)).sum()}</td>
            <td>{(~stat.auto_pre.isna() & ~stat.pre.isna() & (stat.auto_pre != stat.pre)).sum()}</td>
            <td>{(stat.auto_pre.isna()).sum()}</td>
        </tr>
        <tr><th>Datumsobergrenze</th>
            <td>{(stat.succ_dates > 0).sum()}</td>
            <td>{(~stat.auto_post.isna() & ~(stat.succ_dates > 0)).sum()}</td>
            <td>{(~stat.auto_post.isna() & ~stat.post.isna() & (stat.auto_post != stat.post)).sum()}</td>
            <td>{(stat.auto_post.isna()).sum()}</td>
        </tr>
        </tbody>
    </table>
    """
    write_html(config.path.report_dir / "stats.php", html, "Statistik")

    return stat, dating_stat, edge_df


def report_timeline(graphs: MacrogenesisInfo):
    witinfo = WitInscrInfo.get()

    def ref_info(row) -> dict:
        ref = row.Index
        result = dict(start=row.max_before_date.isoformat(),
                      end=row.min_after_date.isoformat(),
                      content=_fmt_node(ref),
                      id=ref.filename.stem,
                      index=row.position)
        info = witinfo.get().by_uri.get(ref.uri, None)
        if info:
            result['scenes'] = sorted([scene.n for scene in info.max_scenes])
            result['groups'] = sorted(info.groups)
            result['group'] = info.group
        else:
            result['scenes'] = []
            result['groups'] = []
            result['group'] = None
        return result

    rows = graphs.details.itertuples(index=True)
    data = list()  # FIXME list comprehension after #25 is resolved
    known = set()
    for row in rows:
        if not (pd.isna(row.max_before_date) or pd.isna(row.min_after_date)):
            info = ref_info(row)
            if info['id'] not in known:
                data.append(info)
                known.add(info['id'])
    with (config.path.report_dir / 'timeline.json').open("wt") as data_out:
        json.dump(data, data_out)
    (config.path.report_dir / 'timeline.html').write_bytes(pkg_resources.resource_string('macrogen', 'timeline.html'))


def report_inscriptions(info: MacrogenesisInfo):
    # all documents that have inscriptions in their textual transcript
    from .witnesses import all_documents
    docs = all_documents()
    witinfo = WitInscrInfo.get()
    docs_by_uri = {doc.uri: doc for doc in docs}
    tt_inscriptions = {doc.uri: doc.inscriptions for doc in docs if doc.inscriptions}

    # all documents that have inscriptions in the graph
    inscriptions_from_graph = [ref for ref in info.base if isinstance(ref, Inscription)]
    graph_inscriptions: Dict[str, Set[str]] = defaultdict(set)
    for inscr in inscriptions_from_graph:
        graph_inscriptions[inscr.witness.uri].add(inscr)

    relevant_uris = {uri for uri in set(tt_inscriptions.keys()).union(graph_inscriptions.keys())}
    stripped = remove_edges(info.base,
                            lambda _, __, attr: attr.get('copy') or attr.get('kind') in ['inscription', 'orphan'])

    table = (HtmlTable(data_sortable="true")
             .column('Dokument', lambda uri: _fmt_node(Witness.get(uri)), data_sortable_type='sigil')
             .column('Inskriptionen Makrogenese')
             .column('Inskriptionen Transkript')
             .column('Dok.-Aussagen', data_sortable_type='numericplus')
             .column('Graph'))

    def uri_idx(uri):
        wit = info.node(uri, None)
        return getattr(wit, 'index', 9999)

    def ghlink(path: Path):
        ghroot = config.xmlroot[:config.xmlroot.rindex('/') + 1]  # strip /macrogenesis
        relative = str(path.relative_to(config.path.data))
        return ghroot + relative

    for doc_uri in sorted(relevant_uris, key=uri_idx):
        relevance = 1
        wit = info.node(doc_uri, None)
        if doc_uri in docs_by_uri:
            document = docs_by_uri[doc_uri]
            doc_tt_inscriptions = document.inscriptions
            graph_inscr_uris = {inscr.uri for inscr in graph_inscriptions[doc_uri]}
            tt_inscr_uris = set(document.inscription_uris) if doc_tt_inscriptions else set()
            if graph_inscr_uris == tt_inscr_uris:
                relevance -= 1
            elif graph_inscr_uris - tt_inscr_uris:
                relevance += 1
            if document.text_transcript:
                ttlink = ghlink(document.text_transcript)
                if doc_tt_inscriptions:
                    transcript_links = '<br/>'.join(
                            f"""<a href="{ttlink}#L{i.getparent().sourceline}">{i}</a> <small class="pure-fade">({witinfo.resolve(doc_uri, inscription=i).covered_lines()} V.)</small>"""
                            for i in sorted(doc_tt_inscriptions))
                else:
                    transcript_links = f'<a href="{ttlink}">(keine)</a>'
            else:
                transcript_links = f'<a href="{ghlink(document.source)}">(kein Transkript)</a>'
        else:
            transcript_links = '(unbekanntes Dokument)'

        doc_assertions = (stripped.in_degree(wit) + stripped.out_degree(wit)) if wit and wit in stripped else 0
        if doc_assertions:
            relevance += 1

        if not isinstance(Witness.get(doc_uri), Witness):
            relevance += 1

        relevance_class = ['ignore', '', 'warning', 'error'][max(0, min(relevance, 3))]

        table.row((doc_uri,
                   '<br/>'.join(f'<a href="{wit.filename.stem}">{wit.inscription}</a>'
                                for wit in sorted(graph_inscriptions[doc_uri])),
                   transcript_links,
                   str(doc_assertions) if doc_assertions else '',
                   _subgraph_link(wit, *sorted(graph_inscriptions.get(doc_uri, [])), assertions=True, ignored=True)),
                  class_=relevance_class)
    write_html(config.path.report_dir / 'inscriptions.php',
               """<style>
               .warning td { background-color: rgba(220,160,0,0.2); }
               .error td { background-color: rgba(190,0,0,0.1); }
               .ignore td * { color: lightgray; }
               </style>""" +
               table.format_table() + """
               <div class="pure-g-r">
               <section class="pure-u-1">
               <h2>Erläuterungen</h2>
               <p>
               Im Idealfall stehen in den Makrogenese- und Transkriptspalten jedes Dokuments genau dieselben Inskriptionen und 
               in der Spalte <em>Dok.-Aussagen</em> eine 0. Abweichungen können wie folgt erklärt werden:</p>
               <ul>
               <li>Inskriptionen, die <strong>nur in der Makrogenesespalte</strong> auftauchen, sind entweder falsch
                   benannt oder nicht im Textuellen Transkript verzeichnet. Über den Link zur Inskription in der 
                   Makrogenesespalte kommt man zu allen Aussagen über die Inskription, mit Links in die XML-Quellen.</li>
               <li>Für Inskriptionen, die <strong>nur in der Transkriptspalte</strong> auftauchen, gibt es keine
                   Aussagen in den Makrogenesedaten, oder nur mit falschen Referenzen</li>
               <li><strong>(eingeklammertes)</strong> in der Transkriptspalte deutet auf Fehler in den Makrogenese-Verweisen hin</li> 
               <li>Die Spalte <strong>Dok.-Aussagen</strong> enthält die Anzahl der Makrogenese-Aussagen über das 
                   <em>Dokument</em> direkt statt über die Inskriptionen. Hier müsste das Verhältnis zwischen
                   Dokument und Inskriptionen geklärt werden. (In den Graphen sind von den Inskriptionen
                   kopierte Aussagen mit einer leeren Pfeilspitze dargestellt, direkte Aussagen mit einer schwarzen)
               </li>
               </ul>
               </section></div>
               """,
               'Inskriptionen')


def report_config(info: MacrogenesisInfo):
    models = {
        'single': 'Datenmodell, bei dem jeder Zeuge und jede Inskription durch je einen Knoten repräsentiert wird.',
        'split': 'Datenmodell, bei dem jeder Zeuge und jede Inskription durch je einen Knoten für den Beginn und einen '
                 'Knoten für das Ende des Schreibprozesses repräsentiert wird. '
                 'Eine Kante stellt sicher, dass der Beginn vor dem Ende liegt. '
                 'Eine Aussage wie <em>A vor B</em> wird interpretiert als '
                 '<em>A wurde beendet, bevor mit B begonnen wurde</em>.',
        'split-reverse': 'Datenmodell, bei dem jeder Zeuge und jede Inskription durch je einen Knoten für den Beginn und einen '
                         'Knoten für das Ende des Schreibprozesses repräsentiert wird. '
                         'Eine Kante stellt sicher, dass der Beginn vor dem Ende liegt. '
                         'Eine Aussage wie <em>A vor B</em> wird interpretiert als '
                         '<em>A wurde begonnen, bevor B beendet wurde</em>.',
    }

    inscriptions = {
        'orphan': 'Falls für einen Zeugen <var>w</var> keine Makrogeneseaussagen vorliegen, aber für mindestens eine '
                  'Inskription <var>w<sub>i</sub></var> von <var>w</var>, so wird für jede Inskription von <var>w</var>'
                  'eine Kante <var>w<sub>i</sub> → w</var> eingezogen.',
        'copy': 'Alle Aussagen über Inskriptionen werden auf die zugehörigen Zeugen kopiert.',
        'inline': 'Eine Inskription <var>w<sub>i</sub></var> eines Zeugen <var>w</var> wird so eingebunden, dass '
                  'sie nach dem Beginn von <var>w</var> beginnt und vor dem Ende von <var>w</var> endet.',
    }
    half_interval_mode = {
        'off': 'Abgeleitete Intervallgrenzen wurden niemals hinzugefügt.',
        'light': 'Nur wenn für einen Zeugen keine einzige belegte Angabe der einen Intervallgrenze einer absoluten '
                 'Datierung, aber mindestens '
                 'eine Angabe der anderen Intervallgrenze existiert, wird eine künstliche Intervallgrenze eingezogen,'
                 f'und zwar um <strong>{config.half_interval_correction}</strong> (<code>half_interval_correction</code>) '
                 'Tage von der dem Zeugen nächsten Intervallgrenze entfernt.',
        'always': 'Gibt eine Quelle für einen Zeugen nur eine Intervallgrenze zur absoluten Datierung an, so wird '
                  f'automatisch eine um ± <strong>{config.half_interval_correction}</strong> '
                  '(<code>half_interval_correction</code> Tage entfernte andere Intervallgrenze ergänzt.)'
    }
    content = f"""
    <p>Durch die Konfiguration kann die makrogenetische Analyse beeinflusst werden. Diese Seite zeigt und erläutert 
    wichtige Einstellungen, wie sie bei der Generierung dieser Daten aktuell waren.</p>
    <h2 id="model">Graphmodell (<code>model</code>)</h2>
    <dl>
    <dt><code>{config.model}</code></dt>
    <dd>{models[config.model]}</dd>
    </dl>
    <h3 id="inscriptions">Inskriptionsbehandlung</h3>
    <p>Die folgenden Regeln wurden angewandt, um Inskriptionen und die zugehörigen Zeugen zu verbinden:</p>
    <ul>
    """
    for option in config.inscriptions or []:
        content += f'<li>(<code>{option}</code>) {inscriptions.get(option, "…")}</li>\n'
    content += f"""
    </ul>
    <h3 id="heuristics">Ergänzende Heuristiken</h3>
    <p>Für manche Knoten existiert nur eine »halbe« absolute Datierung, d. h. nur die Angabe entweder eines frühesten
       Beginns oder eines spätesten Endes. Eine Heuristik kann ggf. die andere Seite ergänzen 
       (<code>half_interval_mode</code>):</p>
    <dl><dt><code>{config.half_interval_mode}</code></dt><dd>{half_interval_mode[config.half_interval_mode]}</dd></dl>
    """

    content += """<h2 id="bibscores">Gewichte für bibliographische Quellen</h2>
    <table class="pure-table" data-sortable="true">
    <thead><tr><th data-sortable-type="alpha">Quelle</th><th data-sortable-type="numeric">Gewicht</th></tr></thead>
    <tbody>
    """
    for uri, value in sorted(config.bibscores.items(), key=itemgetter(1)):
        source = BiblSource(uri)
        content += f"<tr><td><a href='{source.filename}' title='{source.long_citation}'>{source}</td><td class='pure-right'>{value}</td></tr>"
    content += """</tbody></table>"""

    config_io = StringIO()
    config.save_config(config_io)
    content += f"""
    <h2 id="config.yaml">Konfigurationsdatei</h2>
    <pre class="lang-yaml">
    {config_io.getvalue()}
    </pre>
    """

    write_html(config.path.report_dir / "config.php", content, head="Konfiguration")


def generate_reports(info: MacrogenesisInfo):
    report_functions = [fn for name, fn in globals().items() if name.startswith('report_')]
    for report in report_functions:
        logger.info('Running %s', report.__name__)
        report(info)
