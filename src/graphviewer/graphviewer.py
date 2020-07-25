import codecs
import subprocess
from pathlib import Path

import networkx as nx
from flask import Flask, render_template, request, Response, flash, jsonify
from markupsafe import Markup
from networkx import DiGraph

from macrogen import MacrogenesisInfo, write_dot
from macrogen.graphutils import remove_edges, simplify_timeline, expand_edges, collapse_edges, collapse_parallel_edges

app = Flask(__name__)

try:
    from gv_config import SECRET_KEY
    app.secret_key = SECRET_KEY
except ImportError:
    app.logger.error('No secret key config -- using unsafe default')
    app.secret_key = b'not-really-a-secret-key'


default_model = MacrogenesisInfo('target/macrogenesis/macrogen-info.zip')  # FIXME evil

modelFiles = Path('target/macrogenesis').glob('*-*/macrogen-info.zip')
models = {p.parent.stem: MacrogenesisInfo(p) for p in modelFiles}
models['default'] = default_model



class NoNodes(ValueError):
    pass

def prepare_agraph():
    node_str = request.values.get('nodes')
    model = request.values.get('model', 'default')
    info = models.get(model, default_model)
    nodes, errors = info.nodes(node_str, report_errors=True)
    if errors:
        flash('Die folgenden zentralen Knoten wurden nicht gefunden: ' + ', '.join(errors), 'warning')
    context = request.values.get('context', False)
    abs_dates = request.values.get('abs_dates', False)
    extra, errors = info.nodes(request.values.get('extra', ''), report_errors=True)
    if errors:
        flash('Die folgenden Pfadziele wurden nicht gefunden: ' + ', '.join(errors), 'warning')
    induced_edges = request.values.get('induced_edges', False)
    ignored_edges = request.values.get('ignored_edges', False)
    direct_assertions = request.values.get('assertions', False)
    paths_wo_timeline = request.values.get('paths_wo_timeline', False)
    no_edge_labels = request.values.get('no_edge_labels', False)
    tred = request.values.get('tred', False)
    nohl = request.values.get('nohl', False)
    syn = request.values.get('syn', False)
    inscriptions = request.values.get('inscriptions', False)
    order = request.values.get('order', False)
    collapse = request.values.get('collapse', False)
    direction = request.values.get('dir', 'LR').upper()
    central_paths = request.values.get('central_paths', 'all').lower()
    if direction not in {'LR', 'RL', 'TB', 'BT'}:
        direction = 'LR'
    if nodes:
        g = info.subgraph(*nodes, context=context, abs_dates=abs_dates, paths=extra, keep_timeline=True,
                          paths_between_nodes=central_paths,
                          paths_without_timeline=paths_wo_timeline,
                          direct_assertions=direct_assertions, include_syn_clusters=syn,
                          include_inscription_clusters=inscriptions)
        if induced_edges:
            g = info.base.subgraph(g.nodes).copy()
        if not ignored_edges or tred:
            g = remove_edges(g, lambda u, v, attr: attr.get('ignore', False) and not attr.get('kind', '') == 'temp-syn')
        if not syn:
            g = remove_edges(g, lambda u, v, attr: attr.get('kind', None) == "temp-syn")
        if tred:
            g = remove_edges(g, lambda u, v, attr: attr.get('delete', False))
        if tred:
            if nx.is_directed_acyclic_graph(g):
                reduction = nx.transitive_reduction(g)
                g = g.edge_subgraph([(u, v, k) for u, v, k, _ in expand_edges(g, reduction.edges)])
            else:
                flash('Cannot produce DAG – subgraph is not acyclic!?', 'error')
        g = simplify_timeline(g)
        if collapse:
            g = collapse_parallel_edges(g)
        g.add_nodes_from(nodes)
        if order:
            g = info.order_graph(g)
        agraph = write_dot(g, target=None, highlight=None if nohl else nodes, edge_labels=not no_edge_labels)
        agraph.graph_attr['basename'] = ",".join([str(node.filename.stem if hasattr(node, 'filename') else node) for node in nodes])
        agraph.graph_attr['bgcolor'] = 'transparent'
        agraph.graph_attr['rankdir'] = direction
        if order:
            agraph.graph_attr['ranksep'] = '0.2'
        return agraph
    else:
        raise NoNodes('No nodes in graph')

def _normalize_args(args):
    result = dict(args)
    for field in ['nodes', 'extra']:
        if field in result:
            result[field] = ", ".join(str(node) for node in default_model.nodes(result[field]))
    return result

@app.route('/macrogenesis/subgraph')
def render_form():
    # try:
    #     agraph = prepare_agraph()
    #     output = subprocess.check_output(['dot', '-T', 'svg'], input=codecs.encode(agraph.to_string()), timeout=30)
    #     svg = Markup(codecs.decode(output))
    # except NoNodes:
    #     flash(Markup('<strong>Keine Knoten im Graphen.</strong> Bitte mindestens einen Knoten im Feld <em>Zentrale Knoten</em> eingeben.'), 'danger')
    #     svg = ''
    return render_template('form.html', query=codecs.decode(request.query_string), **_normalize_args(request.args), models=models)


@app.route('/macrogenesis/subgraph/pdf')
def render_pdf():
    agraph = prepare_agraph()
    output = subprocess.check_output(['dot', '-T', 'pdf'], input=codecs.encode(agraph.to_string()), timeout=30)
    # p = subprocess.run(['dot', '-T', 'pdf'], input=codecs.encode(agraph.to_string()), capture_output=True, timeout=30)
    response = Response(output, mimetype='application/pdf')
    response.headers['Content-Disposition'] = f'attachment; filename="{agraph.graph_attr["basename"]}.pdf"'
    return response


@app.route('/macrogenesis/subgraph/dot', methods=['GET', 'POST'])
def render_dot():
    try:
        agraph = prepare_agraph()
        response = Response(agraph.to_string(), mimetype='text/vnd.graphviz')
        response.headers['Content-Disposition'] = f'attachment; filename="{agraph.graph_attr["basename"]}.dot"'
        return response
    except NoNodes as e:
        return Response(Markup('<strong>Keine Knoten im Graphen.</strong> Bitte mindestens einen Knoten im Feld <em>Zentrale Knoten</em> eingeben.'), status=404)

@app.route('/macrogenesis/subgraph/svg')
def render_svg():
    agraph = prepare_agraph()
    output = subprocess.check_output(['dot', '-T', 'svg'], input=codecs.encode(agraph.to_string()), timeout=30)
    response = Response(output, mimetype='image/svg+xml')
    response.headers['Content-Disposition'] = f'attachment; filename="{agraph.graph_attr["basename"]}.svg"'
    return response

@app.route('/macrogenesis/subgraph/help')
def render_help():
    return render_template('help.html')


@app.route('/macrogenesis/subgraph/check-nodes')
def check_nodes():
    node_str = request.values.get('nodes')
    model = request.values.get('model', 'default')
    info = models.get(model, default_model)
    nodes, errors = info.nodes(node_str, report_errors=True)
    return jsonify(dict(
            normalized=', '.join(map(str, nodes)),
            not_found=errors,
            error_msg="Unbekannte Knoten ignoriert: " + ', '.join(errors) if errors else ''
    ))
