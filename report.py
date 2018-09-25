import json
from datetime import date, datetime
from itertools import chain, repeat, groupby
from operator import itemgetter

import requests
from lxml import etree
from lxml.builder import ElementMaker
from lxml.etree import Comment
from more_itertools import pairwise

from faust_logging import logging

import csv
from collections.__init__ import defaultdict, Counter
from html import escape
from pathlib import Path
from typing import Iterable, List, Dict, Mapping, Tuple, Sequence, Union

import networkx as nx

import faust
from datings import BiblSource
from graph import MacrogenesisInfo, pathlink, EARLIEST, LATEST, DAY
from uris import Reference, Witness, Inscription, UnknownRef, AmbiguousRef
from visualize import write_dot, simplify_graph

logger = logging.getLogger(__name__)
target = Path(faust.config.get('macrogenesis', 'output-dir'))

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
        self.formatters = []
        self.table_attrs = table_attrs
        self.rows = []
        self.row_attrs = []

    def column(self, title='', format_spec=None, **attrs):
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

    @staticmethod
    def _build_attrs(attrdict: Dict):
        return ''.join(
                ' {}="{}"'.format(attr.strip('_').replace('-', '_'), escape(value)) for attr, value in attrdict.items())

    def _format_column(self, index, data):
        attributes = self._build_attrs(self.attrs[index])
        content = self.formatters[index](data)
        return f'<td{attributes}>{content}</td>'

    def _format_row(self, row: Iterable, **rowattrs) -> str:
        attributes = self._build_attrs(rowattrs)
        return f'<tr{attributes}>' + ''.join(
                self._format_column(index, column) for index, column in enumerate(row)) + '</tr>'

    def _format_rows(self, rows: Iterable[Iterable]):
        for row in rows:
            yield self._format_row(row)

    def _format_header(self):
        column_headers = ''.join('<th>{}</th>'.format(title) for title in self.titles)
        return '<table class="pure-table"><thead>{}</thead><tbody>'.format(column_headers)

    def _format_footer(self):
        return '</tbody></table>'

    def format_table(self, rows=None, row_attrs=None):
        """
        Actually formats the table.

        In the zero-argument form, this uses the rows added previously using the `row` method. Otherwise, the given
        data is used.

        Args:
            rows: If given, the rows to format. This is a list of n-tuples, for n columns.
            row_attrs: If given, this should be a list that contains a mapping with the attributes for each row.

        Returns:
            string containing HTML code for the table
        """
        if rows is None:
            rows = self.rows
            row_attrs = self.row_attrs
        if row_attrs is None:
            row_attrs = repeat({})
        return self._format_header() + ''.join(
                (self._format_row(row, **attrs) for row, attrs in zip(rows, row_attrs))) + self._format_footer()


def write_html(filename: Path, content: str, head: str = None, breadcrumbs: List[Dict[str,str]] = [],
               graph_id: str = None,
               graph_options: Dict[str,object] = dict(controlIconsEnabled=True)) -> None:
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
    prefix = """<?php include "../includes/header.php"?>
     <section>"""
    if graph_id is not None:
        require = "requirejs(['faust_common', 'svg-pan-zoom'], function(Faust, svgPanZoom)"
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
        require = "requirejs(['faust_common'], function(Faust)"
        init = ''
    suffix = f"""</section>
    <script type="text/javascript">
        {require} {{
            document.getElementById('breadcrumbs').appendChild(Faust.createBreadcrumbs({json.dumps(breadcrumbs)}));
            {init}
        }});
    </script>
    <?php include "../includes/footer.php"?>"""
    with open(filename, 'wt', encoding='utf-8') as f:
        f.write(prefix)
        f.write(content)
        f.write(suffix)


def report_components(graphs: MacrogenesisInfo):
    logger.info('Writing component overview to %s', target)
    target.mkdir(parents=True, exist_ok=True)
    report = f"""<h3>{len(graphs.conflicts)} stark zusammenhängende Komponenten</h3>
    <p>Stark zusammenhängende Komponenten sind Teilgraphen, in denen jeder Knoten von
    jedem anderen erreichbar ist. Hier ist keine Ordnung möglich, ohne dass Kanten entfernt
    werden.</p>
    """
    scc_table = _report_subgraphs(graphs.conflicts, target, 'scc-{0:02d}')
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


def _fmt_node(node):
    if isinstance(node, Reference):
        return f'<a href="{node.filename.stem}">{node}</a>'
    else:
        return format(node)


def _edition_link(ref: Reference):
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


class RefTable(HtmlTable):

    def __init__(self, base: nx.MultiDiGraph, **table_attrs):
        super().__init__(**table_attrs)
        (self.column('Nr.')
             .column('Knoten davor')
             .column('Objekt', format_spec=_fmt_node)
             .column('Typ / Edition', format_spec=_edition_link)
             .column('nicht vor', format_spec=lambda d: format(d) if d != EARLIEST else "")
             .column('nicht nach', format_spec=lambda d: format(d) if d != LATEST else "")
             .column('erster Vers')
             .column('Aussagen')
             .column('<a href="conflicts">Konflikte</a>'))
        self.base = base


    def reference(self, ref, index=None, write_subpage=False):
        if ref in self.base:
            if index is None:
                index = self.base.node[ref]['index']
            assertions = list(chain(self.base.in_edges(ref, data=True), self.base.out_edges(ref, data=True)))
            conflicts = [assertion for assertion in assertions if 'delete' in assertion[2] and assertion[2]['delete']]
            self.row((f'<a href="refs#idx{index}">{index}</a>', ref.rank, ref, ref, ref.earliest, ref.latest,
                          getattr(ref, 'min_verse', ''), len(assertions), len(conflicts)),
                         id=f'idx{index}', class_=type(ref).__name__)
            if write_subpage:
                self._last_ref_subpage(DAY, ref)
        else:
            self.row((index, 0, format(ref), ref, '', '', getattr(ref, 'min_verse', ''), ''),
                         class_='pure-fade-40', title='Keine Macrogenesedaten', id=f'idx{index}')

    def _last_ref_subpage(self, DAY, ref):
        """Writes a subpage for ref, but only if it’s the last witness we just wrote"""
        basename = target / ref.filename
        relevant_nodes = {ref} | set(self.base.predecessors(ref)) | set(self.base.successors(ref))
        if ref.earliest != EARLIEST:
            relevant_nodes |= set(nx.shortest_path(self.base, ref.earliest - DAY, ref))
        if ref.latest != LATEST:
            relevant_nodes |= set(nx.shortest_path(self.base, ref, ref.latest + DAY))
        ref_subgraph = self.base.subgraph(relevant_nodes)
        write_dot(ref_subgraph, basename.with_name(basename.stem + '-graph.dot'), highlight=ref)
        report = f"<!-- {repr(ref)} -->\n"
        report += self.format_table(self.rows[-1:])
        report += f"""<object id="refgraph" class="refgraph" type="image/svg+xml" data="{basename.with_name(basename.stem+'-graph.svg').name}"></object>\n"""
        kinds = {'not_before': 'nicht vor',
                 'not_after': 'nicht nach',
                 'from_': 'von',
                 'to': 'bis',
                 'when': 'am',
                 'temp-syn': 'ca. gleichzeitig',
                 'temp-pre': 'entstanden nach',
                 'orphan': '(Verweis)',
                 'inscription': 'Inskription von',
                 None: '???'
                 }
        assertionTable = (HtmlTable()
                          .column('berücksichtigt?')
                          .column('Aussage')
                          .column('Bezug', format_spec=_fmt_node)
                          .column('Quelle', format_spec=_fmt_source)
                          .column('Kommentare', format_spec="/".join)
                          .column('XML', format_spec=_fmt_xml))
        for (u, v, attr) in self.base.in_edges(ref, data=True):
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
        for (u, v, attr) in self.base.out_edges(ref, data=True):
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
                   head=str(ref), graph_id='refgraph')


class AssertionTable(HtmlTable):

    def __init__(self, **table_attrs):
        super().__init__(**table_attrs)
        (self.column('berücksichtigt?')
             .column('Subjekt', _fmt_node)
             .column('Relation', RELATION_LABELS.get)
             .column('Objekt', _fmt_node)
             .column('Quelle', _fmt_source)
             .column('Kommentare', ' / '.join)
             .column('XML', _fmt_xml))

    def edge(self, u: Reference, v: Reference, attr: Dict[str,object]):
        classes = [attr['kind']] if 'kind' in attr and attr['kind'] is not None else ['unknown-kind']
        if attr.get('ignore', False): classes.append('ignore')
        if attr.get('delete', False): classes.append('delete')
        self.row((
            f'<a href="{pathlink(u, v)}">nein</a>' if attr.get('delete', False) else \
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


def _fmt_xml(xml: Union[Tuple[str, int], Sequence[Tuple[str, int]]]):
    if not xml:
        return ""
    if isinstance(xml[0], str):
        return f"{xml[0].replace('macrogenesis/', '')}: {xml[1]}"
    else:
        result = []
        for file, lines in groupby(xml, itemgetter(0)):
            result.append(file.replace('macrogenesis/', '') + ': ' + ", ".join(map(str, map(itemgetter(1), lines))))
        return "; ".join(result)



def report_refs(graphs: MacrogenesisInfo):
    # Fake dates for when we don’t have any earliest/latest info

    target.mkdir(exist_ok=True, parents=True)

    nx.write_yaml(simplify_graph(graphs.base), str(target / 'base.yaml'))
    nx.write_yaml(simplify_graph(graphs.working), str(target / 'working.yaml'))
    nx.write_yaml(simplify_graph(graphs.dag), str(target / 'dag.yaml'))

    nx.write_gpickle(graphs.dag, str(target / 'dag.gpickle'))
    nx.write_gpickle(graphs.working, str(target / 'working.gpickle'))
    nx.write_gpickle(graphs.base, str(target / 'base.gpickle'))

    refs = graphs.order_refs()
    overview = RefTable(graphs.base)

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


def report_missing(graphs: MacrogenesisInfo):
    refs = {node for node in graphs.base.nodes if isinstance(node, Reference)}
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
                     .column('Referenz')
                     .column('URI'))
    report += unknown_table.format_table((ref, ref.uri) for ref in sorted(unknown_refs))
    write_html(target / 'missing.php', report, head="Fehlendes")
    for ref in missing_wits:
        missing_path = target / ref.filename.with_suffix('.php')
        logger.info('Writing missing page for %s to %s', ref, missing_path)
        write_html(missing_path,
                   f"""
                   <p class="pure-alert pure-alert-warning"><strong>Für {ref} liegen noch keine Makrogenesedaten vor.</strong>
                   Ggf. fehlt auch nur die Zuordnung zur richtigen Sigle – siehe in der <a href="refs">Liste der Referenzen</a>.</p>
                   """,
                   head=format(ref))


def _report_conflict(graphs: MacrogenesisInfo, u, v):
    reportfile = pathlink(u, v)
    graphfile = reportfile.with_name(reportfile.stem + '-graph.dot')
    relevant_nodes = {u} | set(graphs.base.predecessors(u)) | set(graphs.base.successors(u)) \
                     | {v} | set(graphs.base.predecessors(v)) | set(graphs.base.successors(v))
    counter_path = []
    try:
        counter_path = nx.shortest_path(graphs.dag, v, u)
        relevant_nodes = set(counter_path)
        counter_desc = " → ".join(map(_fmt_node, counter_path))
        counter_html = f'<p><strong>Pfad in Gegenrichtung:</strong> {counter_desc}</p>'
    except nx.NetworkXNoPath:
        counter_html = f'<p>kein Pfad in Gegenrichtung ({_fmt_node(v)} … {_fmt_node(u)}) im Sortiergraphen</p>'
    except nx.NodeNotFound:
        logger.exception('Node not found!? %s or %s', u, v)
        counter_html = ''
    subgraph: nx.MultiDiGraph = nx.subgraph(graphs.base, relevant_nodes).copy()

    # Highlight conflicting edges, counter path and the two nodes of the conflicting edge(s)
    for v1, v2 in [(u, v)] + list(pairwise(counter_path)):
        for k, attr in subgraph.get_edge_data(v1, v2).items():
            attr['highlight'] = True
    subgraph.node[u]['highlight'] = True
    subgraph.node[v]['highlight'] = True

    write_dot(subgraph, str(target / graphfile))

    table = AssertionTable()
    for k, attr in graphs.base.get_edge_data(u, v).items():
        table.edge(u, v, attr)

    write_html(target / reportfile,
               f"""
               {table.format_table()}
               {counter_html}
               <object id="refgraph" type="image/svg+xml" data="{graphfile.with_suffix('.svg')}"></object>
               """,
               graph_id='refgraph',
               head=f'Entfernte Kante {u} → {v}', breadcrumbs=[dict(caption="Entfernte Kanten", link='conflicts')])

    return reportfile


def report_conflicts(graphs: MacrogenesisInfo):
    table = AssertionTable()
    removed_edges = [(u, v, k, attr) for (u, v, k, attr) in graphs.base.edges(keys=True, data=True) if
                     'delete' in attr and attr['delete']]
    for index, (u, v, k, attr) in enumerate(sorted(removed_edges, key=lambda t: getattr(t[0], 'index', 0)), start=1):
        reportfile = _report_conflict(graphs, u, v)
        table.edge(u, v, attr)
    write_html(target / 'conflicts.php', table.format_table(), head='entfernte Kanten')


def report_sources(graphs: MacrogenesisInfo):
    by_source = defaultdict(list)
    for u, v, k, attr in graphs.base.edges(keys=True, data=True):
        if 'source' in attr:
            by_source[attr['source'].uri].append((u, v, k, attr))

    def _fmt_source(uri):
        source = BiblSource(uri)
        return f'<a href="{source.filename}">{source}</a>'

    sources_table = (HtmlTable()
                     .column('Quelle', format_spec=_fmt_source)
                     .column('Aussagen')
                     .column('Zeugen'))
    for uri, edges in sorted(by_source.items()):
        source = BiblSource(uri)
        filename = target / (source.filename + '.php')
        graphfile = filename.with_name(filename.stem + '-graph.dot')
        logger.info('%d assertions from %s', len(edges), source.citation)
        # subgraph = graphs.base.edge_subgraph([(u,v,k) for u,v,k,attr in edges])
        subgraph = graphs.base.subgraph({u for u, v, k, attr in edges} | {v for u, v, k, attr in edges})
        write_dot(subgraph, graphfile)
        sources_table.row((uri, len(edges), len([node for node in subgraph.nodes if isinstance(node, Reference)])))
        current_table = AssertionTable()
        for u, v, k, attr in edges:
            current_table.edge(u, v, attr)
        write_html(target / (source.filename + '.php'),
                   f"""<object id="refgraph" type="image/svg+xml" data="{graphfile.with_suffix('.svg').name}"></object>
                       {current_table.format_table()}""",
                   graph_id='refgraph',
                   breadcrumbs=[dict(caption='Quellen', link='sources')],
                   head=source.citation)

    write_html(target / 'sources.php', sources_table.format_table(), head='Quellen')


def report_index(graphs):

    pages = [('refs', 'Zeugen', 'Alle referenzierten Dokumente in der erschlossenen Reihenfolge'),
             ('scenes', 'nach Szene', 'Die relevanten Zeugen für jede Szene'),
             ('conflicts', 'entfernte Aussagen', 'Aussagen, die algorithmisch als Konflikt identifiziert und entfernt wurden'),
             ('components', 'Komponenten', 'stark und schwach zusammenhängende Komponenten des Ausgangsgraphen'),
             ('missing', 'Fehlendes', 'Zeugen, zu denen keine Aussagen zur Makrogenese vorliegen, und unbekannte Zeugen'),
             ('sources', 'Quellen', 'Aussagen nach Quelle aufgeschlüsselt'),
             ('dag', 'sortierrelevanter Gesamtgraph', 'Graph aller für die Sortierung berücksichtigter Aussagen (einzoomen!)'),
             ('tred', 'transitive Reduktion', '<a href="https://de.wikipedia.org/w/index.php?title=Transitive_Reduktion">Transitive Reduktion</a> des Gesamtgraphen'),
             ('help', 'Legende', 'Legende zu den Graphen')]
    links = "\n".join(('<tr><td><a href="{}" class="pure-button pure-button-tile">{}</td><td>{}</td></tr>'.format(*page) for page in pages))
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
    write_dot(graphs.dag, target / 'dag-graph.dot', record=True)
    write_html(target / 'dag.php', '<object type="image/svg+xml" data="dag-graph.svg" id="refgraph"/>',
               graph_id='refgraph', head='Effektiver Gesamtgraph (ohne Konflikte)',
               graph_options=dict(controlIconsEnabled=True, maxZoom=200))

    logger.info('Creating transitive reduction ...')
    tred_base = nx.MultiDiGraph(nx.transitive_reduction(graphs.dag))
    tred = nx.edge_subgraph(graphs.dag, tred_base.edges)
    write_dot(tred, target / 'tred-graph.dot', record=True)
    write_html(target / 'tred.php', '<object type="image/svg+xml" data="tred-graph.svg" id="refgraph"/>',
               graph_id='refgraph', head='Transitive Reduktion',
               graph_options=dict(controlIconsEnabled=True, maxZoom=200))

def report_help():
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
    g_orphan = demo_graph(i1, i1w, kind='orphan', source='faust://orphan/adoption')

    help_graphs = dict(pre=g1, conflict=g1a, syn=g2, dating=g3, interval=g4, when=g5, orphan=g_orphan)
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
    <table class="pure-table">
        <thead><th>Graph</th><th>Bedeutung</th></thead>
        <tbody>
        <tr><td><img src="help-pre.svg" /></td>
            <td>Laut Quelle 1 entstand {w1} vor {w2}</td></tr>
        <tr><td><img src="help-conflict.svg" /></td>
            <td>Laut Quelle 1 entstand {w1} vor {w2},
                laut Quelle 2 entstand {w2} vor {w1}.
                Diese Aussage von Quelle 2 wird nicht berücksichtigt.
                Die Aussage von Quelle Quelle 3 wird von vornherein ignoriert.</td></tr>
        <tr><td><img src="help-syn.svg"/></td>
            <td>Laut Quelle 2 entstand {w1} etwa zeitgleich zu {w2}.</td></tr>
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
        </tbody>
    </table>
    """

    write_html(target / 'help.php', report, 'Legende')

def _yearlabel(ref: Reference):
    earliest_year = ref.earliest.year
    latest_year = ref.latest.year
    if earliest_year == latest_year:
        return str(earliest_year)
    else:
        sep = " … "
        result = ""
        if ref.earliest != EARLIEST:
            result += str(earliest_year)
        result += sep
        if ref.latest != LATEST:
            result += str(latest_year)
        return result if result != sep else ""


class ByScene:
    def __init__(self, graphs: MacrogenesisInfo):
        scene_xml = etree.parse('scenes.xml')
        self.scenes = scene_xml.xpath('//f:scene[@first-verse]', namespaces=faust.namespaces)
        bargraph_info = requests.get('http://dev.digital-humanities.de/ci/job/faust-gen-fast/lastSuccessfulBuild/artifact/target/www/data/genetic_bar_graph.json').json()
        self.intervals = {Witness.get('faust://document/faustedition/' + doc['sigil_t']): doc['intervals'] for doc in bargraph_info}
        self.ordering = list(enumerate(graphs.order_refs()))
        self.graphs = graphs

    def report(self):
        sceneTable = (HtmlTable()
                      .column('#')
                      .column('Szene')
                      .column('Verse', format_spec=lambda t: '{} – {}'.format(*t))
                      .column('Zeugen'))
        for scene in self.scenes:
            witnessTable = RefTable(self.graphs.base)
            title = scene.xpath('f:title/text()', namespaces=faust.namespaces)[0]
            start, end = int(scene.get('first-verse')), int(scene.get('last-verse'))
            scene_wits = [(index, wit) for index, wit in self.ordering if self.relevant(wit, start, end)]
            for index, witness in scene_wits:
                witnessTable.reference(witness, index)
            scene_wits = {wit for _, wit in scene_wits}
            scene_nodes = scene_wits | {node   for wit in scene_wits if wit in self.graphs.base
                                        for node in chain(self.graphs.base.predecessors(wit),
                                                          self.graphs.base.successors(wit))
                                        if isinstance(node, date)}
            scene_subgraph = self.graphs.base.subgraph(scene_nodes)
            basename = 'scene_' + scene.get('n').replace('.', '-')
            subgraph_page = Path(basename + '-subgraph.php')
            graph_name = Path(basename + '-graph.dot')
            sceneTable.row((scene.get('n'), f'<a href="{basename}">{title}</a>', (start, end), len(scene_wits)))
            write_dot(scene_subgraph, target / graph_name)
            write_html(target / subgraph_page,
                       f"""<object id="refgraph" type="image/svg+xml" data="{graph_name.with_suffix('.svg')}"></object>""",
                       graph_id='refgraph',
                       head="Szenengraph", breadcrumbs=[dict(caption='nach Szene', link='scenes'),
                                                        dict(caption=title, link=basename)])
            write_html(target / (basename + '.php'),
                       f"""
                       <p><a href="{subgraph_page.stem}">Szenengraph</a> ·
                       <a href="/genesis_bargraph?rangeStart={start}&amp;rangeEnd={end}">Balkendiagramm</a></p>
                       {witnessTable.format_table()}""",
                       head=title, breadcrumbs=[dict(caption='nach Szene', link='scenes')])
        write_html(target / "scenes.php", sceneTable.format_table(), head='nach Szene')

    def relevant(self, witness: Reference, first_verse: int, last_verse: int) -> bool:
        try:
            for interval in self.intervals[witness]:
                if first_verse <= interval['start'] <= last_verse or \
                        first_verse <= interval['end'] <= last_verse or \
                        interval['start'] <= first_verse <= interval['end']  or \
                        interval['start'] <= last_verse <= interval['end']:
                    return True
        except KeyError:
            return False
        return False

def report_scenes(graphs):
    ByScene(graphs).report()

def write_order_xml(graphs):
    F = ElementMaker(namespace='http://www.faustedition.net/ns', nsmap=faust.namespaces)
    root = F.order(
            Comment('This file has been generated from the macrogenesis data. Do not edit.'),
            *[F.item(format(witness),
                     index=format(index),
                     uri=witness.uri,
                     sigil_t=witness.sigil_t,
                     earliest=witness.earliest.isoformat() if witness.earliest != EARLIEST else '',
                     latest=witness.latest.isoformat() if witness.latest != LATEST else '',
                     yearlabel=_yearlabel(witness))
              for index, witness in enumerate(graphs.order_refs(), start=1)
              if isinstance(witness, Witness)],
            generated=datetime.now().isoformat())
    target.mkdir(parents=True, exist_ok=True)
    root.getroottree().write(str(target / 'order.xml'), pretty_print=True)
