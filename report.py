from faust_logging import logging
logger = logging.getLogger()

import csv
from collections.__init__ import defaultdict, Counter
from datetime import date
from html import escape
from pathlib import Path
from typing import Iterable, List, Optional

import networkx as nx

import faust
from datings import BiblSource
from uris import Reference
from visualize import write_dot

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
        return '<table><thead>{}</thead><tbody>'.format(column_headers)

    def _format_footer(self):
        return '</tbody></table>'

    def format_table(self, rows=None):
        if rows is None:
            rows = self.rows
        return self._format_header() + ''.join(self._format_row(row) for row in rows) + self._format_footer()


def write_html(filename, content, head=None):
    title = head if head is not None else "Faustedition"
    prefix = """<html xmlns="http://www.w3.org/1999/html>
    <head>    
        <meta charset="utf-8" />
        <title>{0}</title>
    </head>
    <body>""".format(escape(title))
    suffix = """</body></head>"""
    with open(filename, 'wt', encoding='utf-8') as f:
        f.write(prefix)
        if head is not None:
            f.write('<h1>{}</h1>'.format(escape(head)))
        f.write(content)
        f.write(suffix)


def report_conflicts(conflicts: List[nx.MultiDiGraph]):
    out = target / 'conflicts'
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
        conflicts = [(u, v, k, attr) for (u, v, k, attr) in all_edges if attr['delete']]
        relations = [(u, v, k, attr) for (u, v, k, attr) in all_edges if not attr['delete']]
        sources = {attr['source'] for u, v, k, attr in relations}
        conflict_sources = {attr['source'] for u, v, k, attr in conflicts}
        table.row(index, refs, len(relations), len(conflicts), sources, conflict_sources)
        write_dot(subgraph, out / "conflict-{:02d}.dot".format(index))

    write_html(out / 'index.html', table.format_table(), 'Konfliktgruppen')


def order_refs(dag: nx.MultiDiGraph):
    def secondary_key(node):
        if isinstance(node, Reference):
            return node.sort_tuple()
        elif isinstance(node, date):
            return date.year, format(date.month, '02d'), date.day, ''
        else:
            return 99999, "zzzzzz", 99999, "zzzzzz"

    nodes = nx.lexicographical_topological_sort(dag, key=secondary_key)
    refs = [node for node in nodes if isinstance(node, Reference)]
    return refs


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