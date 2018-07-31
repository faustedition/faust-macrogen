import json
from datetime import timedelta, date
from itertools import chain

from faust_logging import logging
from graph import MacrogenesisInfo, EARLIEST, LATEST, DAY

import csv
from collections.__init__ import defaultdict, Counter
from html import escape
from pathlib import Path
from typing import Iterable, List, Dict, Mapping

import networkx as nx

import faust
from datings import BiblSource
from uris import Reference, Witness, Inscription, UnknownRef
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


def write_html(filename, content, head=None, breadcrumbs=[]):
    if head is not None:
        breadcrumbs = breadcrumbs + [dict(caption=head)]
    breadcrumbs = [dict(caption='Makrogenese', link='/macrogenesis')] + breadcrumbs
    prefix = """<?php include "../includes/header.php"?>
     <section>"""
    suffix = """</section>
    <script type="text/javascript">
        requirejs(['faust_common'], function(Faust) {{
            document.getElementById('breadcrumbs').appendChild(Faust.createBreadcrumbs({}));
        }});
    </script>
    <?php include "../includes/footer.php"?>""".format(json.dumps(breadcrumbs))
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


def _report_subgraphs(removed_edges, out, pattern):
    table = HtmlTable().column('Nummer', format_spec='<a href="'+pattern+'.svg">{0}</a>') \
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
        write_dot(subgraph, out / (pattern+".dot").format(index))
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
    else:
        return format(ref)



def report_refs(graphs: MacrogenesisInfo):
    # Fake dates for when we don’t have any earliest/latest info

    target.mkdir(exist_ok=True, parents=True)

    nx.write_yaml(simplify_graph(graphs.base), str(target / 'base.yaml'))
    nx.write_yaml(simplify_graph(graphs.working), str(target / 'working.yaml'))
    nx.write_yaml(simplify_graph(graphs.dag), str(target / 'dag.yaml'))

    refs = graphs.order_refs()
    overview = (HtmlTable()
                .column('Nr.')
                .column('Knoten davor')
                .column('Objekt', format_spec=_fmt_node)
                .column('Typ / Edition', format_spec=_edition_link)
                .column('nicht vor', format_spec=lambda d: format(d) if d != EARLIEST else "")
                .column('nicht nach', format_spec=lambda d: format(d) if d != LATEST else "")
                .column('erster Vers')
                .column('Aussagen')
                .column('<a href="conflicts">Konflikte</a>'))

    for index, ref in enumerate(refs, start=1):
        assertions = list(chain(graphs.base.in_edges(ref, data=True), graphs.base.out_edges(ref, data=True)))
        conflicts = [assertion for assertion in assertions if 'delete' in assertion[2] and assertion[2]['delete']]
        overview.row((index, ref.rank, ref, ref, ref.earliest, ref.latest, getattr(ref, 'min_verse', ''), len(assertions), len(conflicts)))

        DAY = timedelta(days=1)
        basename = target / ref.filename
        relevant_nodes = {ref} | set(graphs.base.predecessors(ref)) | set(graphs.base.successors(ref))
        if ref.earliest != EARLIEST:
            relevant_nodes |= set(nx.shortest_path(graphs.base, ref.earliest-DAY, ref))
        if ref.latest != LATEST:
            relevant_nodes |= set(nx.shortest_path(graphs.base, ref, ref.latest+DAY))
        ref_subgraph = graphs.base.subgraph(relevant_nodes)
        write_dot(ref_subgraph, basename.with_name(basename.stem+'-graph.dot'), highlight=ref)
        report =  f"<!-- {repr(ref)} -->\n"
        report += overview.format_table(overview.rows[-1:])
        report += f"""<object class="refgraph" type="image/svg+xml" data="{basename.with_name(basename.stem+'-graph.svg').name}"></object>\n"""

        kinds = {'not_before': 'nicht vor',
                 'not_after': 'nicht nach',
                 'from_': 'von',
                 'to': 'bis',
                 'when': 'am',
                 'temp-syn': 'ca. gleichzeitig',
                 'temp-pre': 'entstanden nach',
                 None: '???'
                 }
        assertionTable = (HtmlTable()
                          .column('berücksichtigt?')
                          .column('Aussage')
                          .column('Bezug', format_spec=_fmt_node)
                          .column('Quelle')
                          .column('Kommentare', format_spec="/".join)
                          .column('XML', format_spec=lambda xml: ":".join(map(str, xml))))
        for (u, v, attr) in graphs.base.in_edges(ref, data=True):
            assertionTable.row(('nein' if 'delete' in attr and attr['delete'] else 'ja',
                                kinds[attr['kind']],
                                u+DAY if isinstance(u, date) else u,
                                attr['source'],
                                attr.get('comments', []),
                                attr['xml']))
        kinds['temp-pre'] = 'entstanden vor'
        for (u, v, attr) in graphs.base.out_edges(ref, data=True):
            assertionTable.row(('nein' if 'delete' in attr and attr['delete'] else 'ja',
                                kinds[attr['kind']],
                                v-DAY if isinstance(v, date) else v,
                                attr['source'],
                                attr.get('comments', []),
                                attr['xml']))
        write_html(basename.with_suffix('.php'), report + assertionTable.format_table(),
                   breadcrumbs=[dict(caption='Referenzen', link='refs')],
                   head=str(ref))

    write_html(target / 'refs.php', overview.format_table(), head="Referenzen")

    write_dot(simplify_graph(graphs.base), str(target / 'base.dot'), record=False)
    write_dot(simplify_graph(graphs.working), str(target / 'working.dot'), record=False)
    write_dot(simplify_graph(graphs.dag), str(target / 'dag.dot'), record=False)

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
    report += missing_table.format_table((ref, wits_with_inscr[ref]) for ref in sorted(missing_wits, key=lambda r: r.sort_tuple()))
    report += '\n<h3 id="unknown">Unbekannte Referenzen</h3>'
    unknown_table = (HtmlTable()
                     .column('Referenz')
                     .column('URI'))
    report += unknown_table.format_table((ref, ref.uri) for ref in sorted(unknown_refs))
    write_html(target / 'missing.php', report, head="Fehlendes")

def report_conflicts(graphs: MacrogenesisInfo):

    kinds = {'not_before': 'nicht vor',
             'not_after': 'nicht nach',
             'from_': 'von',
             'to': 'bis',
             'when': 'am',
             'temp-syn': 'ca. gleichzeitig',
             'temp-pre': 'zeitlich vor',
             None: '???'
             }
    table = (HtmlTable()
             .column('#', format_spec=_fmt_node)
             .column('u', format_spec=_fmt_node)
             .column('Relation', format_spec=kinds.get)
             .column('v', format_spec=_fmt_node)
             .column('Quelle')
             .column('Kommentare', format_spec="/".join)
             .column('XML', format_spec=lambda xml: ":".join(map(str, xml))))
    removed_edges = [(u, v, k, attr) for (u, v, k, attr) in graphs.base.edges(keys=True, data=True) if 'delete' in attr and attr['delete']]
    for index, (u, v, k, attr) in enumerate(sorted(removed_edges, key=lambda t: getattr(t[0], 'index', 0)), start=1):
        graphfile = Path(f"conflict-{index:03d}.dot")
        table.row((f"""<a href="{graphfile.with_suffix('.svg')}">{index}</a>""",
                   u+DAY if isinstance(u, date) else u,
                   attr['kind'],
                   v-DAY if isinstance(v, date) else v,
                   attr['source'],
                   attr.get('comments', []),
                   attr['xml']))
        relevant_nodes =   {u} | set(graphs.base.predecessors(u)) | set(graphs.base.successors(u)) \
                         | {v} | set(graphs.base.predecessors(v)) | set(graphs.base.successors(v))
        subgraph = nx.subgraph(graphs.base, relevant_nodes)
        write_dot(subgraph, str(target / graphfile), highlight=(u,v,k))
    write_html(target / 'conflicts.php', table.format_table(), head='entfernte Kanten')

def report_index():
    report = f"""
      <p>
        Dieser Bereich der Edition enthält experimentelle Informationen zur Makrogenese, er wurde zuletzt
        am {date.today()} generiert.
      </p>
      <section class="center pure-g-r">
        
        <div class="pure-u-1-5"></div>
        
        <article class="pure-u-3-5 pure-center">
            <p>
             <a href="refs" class="pure-button pure-button-tile">Zeugen</a>
             <a href="conflicts" class="pure-button pure-button-tile">entfernte Relationen</a>
             <a href="components" class="pure-button pure-button-tile">Komponenten</a>
             <a href="missing" class="pure-button pure-button-tile">Fehlendes</a>
            </p>
        
        </article>
        
        <div class="pure-u-1-5"></div>

      </section>
    """
    write_html(target / "index.php", report)
