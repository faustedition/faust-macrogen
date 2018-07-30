from datetime import timedelta, date
from itertools import chain

from faust_logging import logging
from graph import order_refs, MacrogenesisInfo

import csv
from collections.__init__ import defaultdict, Counter
from html import escape
from pathlib import Path
from typing import Iterable, List

import networkx as nx

import faust
from datings import BiblSource
from uris import Reference
from visualize import write_dot, simplify_graph

logger = logging.getLogger()
target = Path(faust.config.get('macrogenesis', 'output-dir'))


class HtmlTable:

    def __init__(self, **table_attrs):
        self.titles = []
        self.attrs = []
        self.formatters = []
        self.table_attrs = table_attrs
        self.rows = []

    def column(self, title='', format_spec=None, **attrs):
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

    def row(self, row):
        self.rows.append(row)
        return self

    def _format_column(self, index, data):
        attributes = ''.join(' {}="{}"'.format(attr, escape(value)) for attr, value in self.attrs[index].items())
        content = self.formatters[index](data)
        return f'<td{attributes}>{content}</td>'

    def _format_row(self, row: Iterable) -> str:
        return '<tr>' + ''.join(self._format_column(index, column) for index, column in enumerate(row)) + '</tr>'

    def _format_rows(self, rows: Iterable[Iterable]):
        for row in rows:
            yield self._format_row(row)

    def _format_header(self):
        column_headers = ''.join('<th>{}</th>'.format(title) for title in self.titles)
        return '<table class="pure-table"><thead>{}</thead><tbody>'.format(column_headers)

    def _format_footer(self):
        return '</tbody></table>'

    def format_table(self, rows=None):
        if rows is None:
            rows = self.rows
        return self._format_header() + ''.join(self._format_row(row) for row in rows) + self._format_footer()


def write_html(filename, content, head=None):
    title = head if head is not None else "Faustedition"
    prefix = """<?php include "../includes/header.php"?>
     <section>""".format(escape(title))
    suffix = """</section>
    <?php include "../includes/footer.php"?>"""
    with open(filename, 'wt', encoding='utf-8') as f:
        f.write(prefix)
        if head is not None:
            f.write('<h1>{}</h1>'.format(escape(head)))
        f.write(content)
        f.write(suffix)


def report_conflicts(conflicts: List[nx.MultiDiGraph]):
    out = target / 'conflicts'
    logger.info('Writing conflict overview to %s', out)
    out.joinpath('conflicts').mkdir(parents=True, exist_ok=True)
    table = HtmlTable().column('Nummer', format_spec='<a href="conflict-{0:02d}.svg">{0}</a>') \
        .column('Dokumente') \
        .column('Relationen') \
        .column('Entfernte Relationen') \
        .column('Quellen', format_spec=lambda s: ", ".join(map(str, s))) \
        .column('Entfernte Quellen', format_spec=lambda s: ", ".join(map(str, s)))

    for index, subgraph in enumerate(conflicts, start=1):
        refs = len([node for node in subgraph.nodes if isinstance(node, Reference)])
        all_edges = list(subgraph.edges(keys=True, data=True))
        conflicts = [(u, v, k, attr) for (u, v, k, attr) in all_edges if 'delete' in attr and attr['delete']]
        relations = [(u, v, k, attr) for (u, v, k, attr) in all_edges if 'delete' not in attr or not attr['delete']]
        sources = {attr['source'].citation for u, v, k, attr in relations if 'source' in attr}
        conflict_sources = {attr['source'].citation for u, v, k, attr in conflicts}
        table.row((index, refs, len(relations), len(conflicts), sources, conflict_sources))
        write_dot(subgraph, out / "conflict-{:02d}.dot".format(index))

    write_html(out / 'index.html', table.format_table(), 'Konfliktgruppen')


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


def report_refs(graphs: MacrogenesisInfo):
    # Fake dates for when we don’t have any earliest/latest info
    EARLIEST = date(1749, 8, 27)
    LATEST = date.today()

    target.mkdir(exist_ok=True, parents=True)

    nx.write_yaml(simplify_graph(graphs.base), str(target / 'base.yaml'))
    nx.write_yaml(simplify_graph(graphs.working), str(target / 'working.yaml'))
    nx.write_yaml(simplify_graph(graphs.dag), str(target / 'dag.yaml'))

    refs = order_refs(graphs.dag)
    overview = (HtmlTable()
                .column('Nr.')
                .column('Rang')
                .column('Sigle', format_spec=_fmt_node)
                .column('nicht vor')
                .column('nicht nach')
                .column('Aussagen')
                .column('<a href="conflicts">Konflikte</a>'))

    for index, ref in enumerate(refs, start=1):
        rank = graphs.closure.in_degree(ref)
        max_before_date = max((d for d, _ in graphs.closure.in_edges(ref) if isinstance(d, date)), default=EARLIEST)
        earliest = max_before_date + timedelta(days=1)
        min_after_date = min((d for _, d in graphs.closure.out_edges(ref) if isinstance(d, date)), default=LATEST)
        latest = min_after_date - timedelta(days=1)
        assertions = list(chain(graphs.base.in_edges(ref, data=True), graphs.base.out_edges(ref, data=True)))
        conflicts = [assertion for assertion in assertions if 'delete' in assertion[2] and assertion[2]['delete']]
        overview.row((index, rank, ref, earliest, latest, len(assertions), len(conflicts)))

        basename = target / ref.filename
        relevant_nodes = {ref} | set(graphs.base.predecessors(ref)) | set(graphs.base.successors(ref))
        if max_before_date != EARLIEST:
            relevant_nodes |= set(nx.shortest_path(graphs.base, max_before_date, ref))
        if min_after_date != LATEST:
            relevant_nodes |= set(nx.shortest_path(graphs.base, ref, min_after_date))
        ref_subgraph = graphs.base.subgraph(relevant_nodes)
        write_dot(ref_subgraph, basename.with_suffix('.dot'), highlight=ref)
        report = f"""<!-- {repr(ref)} -->
        <h1>{ref}</h1>
        <object class="refgraph" type="image/svg+xml" data="{basename.with_suffix('.svg').name}"></object>
        <dl>
            <dt>Nr.</dt><dd>{index}</dd>
            <dt>Rang</dt><dd>{rank}</dd>
            <dt>nicht vor</dt><dd>{earliest}</dd>
            <dt>nicht nach</dt><dd>{latest}</dd>
        </dl>
        """
        kinds = {'not_before': 'nicht vor',
                 'not_after': 'nicht nach',
                 'from_': 'von',
                 'to': 'bis',
                 'when': 'am',
                 'temp-syn': 'ca. gleichzeitig',
                 'temp-pre': 'früherer Zeuge:',
                 None: '?'
                 }
        assertionTable = (HtmlTable()
                          .column('berücksichtigt?')
                          .column('Relation')
                          .column('als …', format_spec=_fmt_node)
                          .column('Quelle')
                          .column('Kommentare'))
        for (u, v, attr) in graphs.base.in_edges(ref, data=True):
            assertionTable.row(('nein' if 'delete' in attr and attr['delete'] else 'ja',
                                kinds[attr['kind']],
                                u,
                                attr['source'],
                                '<br/>'.join(attr.get('comments', []))))
        kinds['temp-pre'] = 'späterer Zeuge:'
        for (u, v, attr) in graphs.base.out_edges(ref, data=True):
            assertionTable.row(('nein' if 'delete' in attr and attr['delete'] else 'ja',
                                kinds[attr['kind']],
                                v,
                                attr['source'],
                                '<br/>'.join(attr.get('comments', []))))
        write_html(basename.with_suffix('.php'), report + assertionTable.format_table())

    write_html(target / 'index.php', overview.format_table(), head="Referenzen")

    write_dot(simplify_graph(graphs.base), str(target / 'base.dot'))
    write_dot(simplify_graph(graphs.working), str(target / 'working.dot'))
    write_dot(simplify_graph(graphs.dag), str(target / 'dag.dot'))
