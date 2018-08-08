import json
from datetime import timedelta, date, datetime
from itertools import chain, repeat

from lxml.builder import ElementMaker
from lxml.etree import Comment

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
from uris import Reference, Witness, Inscription, UnknownRef, AmbiguousRef
from visualize import write_dot, simplify_graph

logger = logging.getLogger(__name__)
target = Path(faust.config.get('macrogenesis', 'output-dir'))


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
        self.rows.append(row)
        self.row_attrs.append(row_attrs)
        return self

    @staticmethod
    def _build_attrs(attrdict: Dict):
        return ''.join(' {}="{}"'.format(attr.strip('_').replace('-', '_'), escape(value)) for attr, value in attrdict.items())

    def _format_column(self, index, data):
        attributes = self._build_attrs(self.attrs[index])
        content = self.formatters[index](data)
        return f'<td{attributes}>{content}</td>'

    def _format_row(self, row: Iterable, **rowattrs) -> str:
        attributes = self._build_attrs(rowattrs)
        return f'<tr{attributes}>' + ''.join(self._format_column(index, column) for index, column in enumerate(row)) + '</tr>'

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
            row_attrs=repeat({})
        return self._format_header() + ''.join((self._format_row(row, **attrs) for row, attrs in zip(rows, row_attrs))) + self._format_footer()


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
    elif isinstance(ref, UnknownRef):
        return ''
    elif isinstance(ref, AmbiguousRef):
        return ", ".join(_edition_link(wit) for wit in ref.witnesses)
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
        if ref in graphs.base:
            _report_single_ref(index, ref, graphs, overview)
        else:
            overview.row((index, 0, format(ref), ref, '', '', getattr(ref, 'min_verse', ''), ''),
                         class_='pure-fade-40', title='Keine Macrogenesedaten', id=f'idx{index}')

    write_html(target / 'refs.php', overview.format_table(), head="Referenzen")

    write_dot(simplify_graph(graphs.base), str(target / 'base.dot'), record=False)
    write_dot(simplify_graph(graphs.working), str(target / 'working.dot'), record=False)
    write_dot(simplify_graph(graphs.dag), str(target / 'dag.dot'), record=False)


def _report_single_ref(index, ref, graphs, overview):
    assertions = list(chain(graphs.base.in_edges(ref, data=True), graphs.base.out_edges(ref, data=True)))
    conflicts = [assertion for assertion in assertions if 'delete' in assertion[2] and assertion[2]['delete']]
    overview.row((f'<a href="refs#idx{index}">{index}</a>', ref.rank, ref, ref, ref.earliest, ref.latest,
                  getattr(ref, 'min_verse', ''), len(assertions), len(conflicts)),
                 id=f'idx{index}', class_=type(ref).__name__)
    DAY = timedelta(days=1)
    basename = target / ref.filename
    relevant_nodes = {ref} | set(graphs.base.predecessors(ref)) | set(graphs.base.successors(ref))
    if ref.earliest != EARLIEST:
        relevant_nodes |= set(nx.shortest_path(graphs.base, ref.earliest - DAY, ref))
    if ref.latest != LATEST:
        relevant_nodes |= set(nx.shortest_path(graphs.base, ref, ref.latest + DAY))
    ref_subgraph = graphs.base.subgraph(relevant_nodes)
    write_dot(ref_subgraph, basename.with_name(basename.stem + '-graph.dot'), highlight=ref)
    report = f"<!-- {repr(ref)} -->\n"
    report += overview.format_table(overview.rows[-1:])
    report += f"""<object class="refgraph" type="image/svg+xml" data="{basename.with_name(basename.stem+'-graph.svg').name}"></object>\n"""
    kinds = {'not_before': 'nicht vor',
             'not_after': 'nicht nach',
             'from_': 'von',
             'to': 'bis',
             'when': 'am',
             'temp-syn': 'ca. gleichzeitig',
             'temp-pre': 'entstanden nach',
             'orphan': '(Verweis)',
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
        delete_ = 'delete' in attr and attr['delete']
        assertionTable.row(('nein' if delete_ else 'ja',
                            kinds[attr['kind']],
                            u + DAY if isinstance(u, date) else u,
                            attr['source'],
                            attr.get('comments', []),
                            attr['xml']),
                           class_='delete' if delete_ else str(attr['kind']))
    kinds['temp-pre'] = 'entstanden vor'
    for (u, v, attr) in graphs.base.out_edges(ref, data=True):
        delete_ = 'delete' in attr and attr['delete']
        assertionTable.row(('nein' if delete_ else 'ja',
                            kinds[attr['kind']],
                            v - DAY if isinstance(v, date) else v,
                            attr['source'],
                            attr.get('comments', []),
                            attr['xml']),
                           class_='delete' if delete_ else str(attr['kind']))
    write_html(basename.with_suffix('.php'), report + assertionTable.format_table(),
               breadcrumbs=[dict(caption='Referenzen', link='refs')],
               head=str(ref))


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
                   attr['xml']),
                  class_=attr['kind'])
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
             <a href="help" class="pure-button pure-button-tile">Legende</a>
            </p>
        
        </article>
        
        <div class="pure-u-1-5"></div>

      </section>
    """
    write_html(target / "index.php", report)



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
    g2 = demo_graph(w1, w2, kind='temp-syn', label='Quelle 2')
    g3 = demo_graph(d1 - DAY, w1, kind='not_before', source='Quelle 1')
    g3.add_edge(w1, d2+DAY, kind='not_after', source='Quelle 1')
    g4 = demo_graph(d1 - DAY, w1, kind='from_', source='Quelle 2')
    g4.add_edge(w1, d2+DAY, kind='to_', source='Quelle 2')
    g5 = demo_graph(d3 - DAY, w2, kind='when', source='Quelle 2')
    g5.add_edge(w2, d3+DAY, kind='when', source='Quelle 2')

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
                Diese Aussage von Quelle 2 wird nicht berücksichtigt.</td></tr>
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


def write_order_xml(graphs):
    F = ElementMaker(namespace='http://www.faustedition.net/ns', nsmap=faust.namespaces)
    root = F.order(
            Comment('This file has been generated from the macrogenesis data. Do not edit.'),
            *[F.item(format(witness), index=format(index), uri=witness.uri, sigil_t=witness.sigil_t)
                    for index, witness in enumerate(graphs.order_refs(), start=1)
                    if isinstance(witness, Witness)],
                   generated=datetime.now().isoformat())
    target.mkdir(parents=True, exist_ok=True)
    root.getroottree().write(str(target / 'order.xml'), pretty_print=True)