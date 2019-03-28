import codecs
import subprocess

from flask import Flask, render_template, request, Response
from markupsafe import Markup

from macrogen import MacrogenesisInfo, write_dot
from macrogen.graphutils import remove_edges, simplify_timeline

app = Flask(__name__)

info = MacrogenesisInfo('target/macrogenesis/macrogen-info.zip')  # FIXME evil


class NoNodes(ValueError):
    pass


def prepare_agraph():
    node_str = request.args.get('nodes')
    nodes = info.nodes(node_str)
    context = request.args.get('context', False)
    abs_dates = request.args.get('abs_dates', False)
    extra = info.nodes(request.args.get('extra', ''))
    induced_edges = request.args.get('induced_edges', False)
    ignored_edges = request.args.get('ignored_edges', False)
    direct_assertions = request.args.get('assertions', False)
    tred = request.args.get('tred', False)
    if nodes:
        g = info.subgraph(*nodes, context=context, abs_dates=abs_dates, pathes=extra, keep_timeline=True,
                          direct_assertions=direct_assertions)
        if induced_edges:
            g = info.base.subgraph(g.nodes).copy()
        if not ignored_edges or tred:
            g = remove_edges(g, lambda u, v, attr: attr.get('ignore', False))
        if tred:
            g = remove_edges(g, lambda u, v, attr: attr.get('delete', False))
        g = simplify_timeline(g)
        agraph = write_dot(g, target=None, highlight=nodes[0])
        return agraph
    else:
        raise NoNodes('No nodes in graph')

def _normalize_args(args):
    result = dict(args)
    for field in ['nodes', 'extra']:
        if field in result:
            result[field] = ", ".join(str(node) for node in info.nodes(result[field]))
    return result

@app.route('/macrogenesis/subgraph')
def render_form():
    try:
        agraph = prepare_agraph()
        output = subprocess.check_output(['dot', '-T', 'svg'], input=codecs.encode(agraph.to_string()), timeout=30)
        svg = Markup(codecs.decode(output))
    except NoNodes:
        svg = 'Bitte Knoten und Optionen im Formular angeben.'
    return render_template('form.html', svg=svg, query=codecs.decode(request.query_string), **_normalize_args(request.args))


@app.route('/macrogenesis/subgraph/pdf')
def render_pdf():
    agraph = prepare_agraph()
    output = subprocess.check_output(['dot', '-T', 'pdf'], input=codecs.encode(agraph.to_string()), timeout=30)
    # p = subprocess.run(['dot', '-T', 'pdf'], input=codecs.encode(agraph.to_string()), capture_output=True, timeout=30)
    return Response(output, mimetype='application/pdf')


@app.route('/macrogenesis/subgraph/dot')
def render_dot():
    agraph = prepare_agraph()
    return Response(agraph.to_string(), mimetype='text/vnd.graphviz')

@app.route('/macrogenesis/subgraph/svg')
def render_svg():
    agraph = prepare_agraph()
    output = subprocess.check_output(['dot', '-T', 'svg'], input=codecs.encode(agraph.to_string()), timeout=30)
    return Response(output, mimetype='image/svg+xml')