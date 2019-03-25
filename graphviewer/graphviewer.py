import subprocess

import networkx as nx
from flask import Flask, render_template, request, flash
from markupsafe import Markup

from macrogen import MacrogenesisInfo, write_dot
from macrogen.graphutils import remove_edges, simplify_timeline

app = Flask(__name__)
from pathlib import Path

info = MacrogenesisInfo('target/macrogenesis/macrogen-info.zip')

@app.route('/')
def hello_world():
    node_str = request.args.get('nodes')
    nodes = parse_nodes(node_str)
    context = request.args.get('context', False)
    abs_dates = request.args.get('abs_dates', False)
    extra = parse_nodes(request.args.get('extra', ''))
    induced_edges = request.args.get('induced_edges', False)
    ignored_edges = request.args.get('ignored_edges', False)
    tred = request.args.get('tred', False)
    if nodes:
        g = info.subgraph(*nodes, context=context, abs_dates=abs_dates, pathes=extra, keep_timeline=True)
        if induced_edges:
            g = info.base.subgraph(g.nodes).copy()
        if not ignored_edges or tred:
            g = remove_edges(g, lambda u,v,attr: attr.get('ignore', False))
        if tred:
            g = remove_edges(g, lambda u, v, attr: attr.get('delete', False))
        g = simplify_timeline(g)
        agraph = write_dot(g, target=None, highlight=nodes[0])
        p = subprocess.run(['dot', '-T', 'svg'], input=agraph.to_string(), capture_output=True, timeout=30, encoding='UTF-8')
        svg = Markup(p.stdout)
    else:
        svg = ':( Keine Knoten'



    return render_template('form.html', svg=svg, **request.args)


def parse_nodes(node_str):
    nodes = []
    if node_str:
        for node_spec in node_str.split(','):
            try:
                nodes.append(info.node(node_spec.strip()))
            except KeyError:
                ...  # flash("Knoten »%s« nicht gefunden", node_spec.strip())
    return nodes