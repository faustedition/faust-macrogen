import codecs
import subprocess
from collections import Mapping

import pydantic
from dataclasses import dataclass, Field
from enum import Enum
from typing import TypeVar, Hashable, Callable, List, Dict, Optional, Iterable

import networkx as nx
from macrogen import MacrogenesisInfo, write_dot

import uvicorn
from fastapi import FastAPI, Depends, Request, Response
from fastapi.templating import Jinja2Templates
from macrogen.graph import Node
from macrogen.graphutils import remove_edges, expand_edges, simplify_timeline, collapse_parallel_edges
from pydantic import BaseModel
from pygraphviz import AGraph

app = FastAPI()

S, T = TypeVar('S'), TypeVar('T')


class LazyLoader(Mapping):
    _items: Dict[S, T]
    _loaded: Dict[S, bool]

    def __init__(self, loader: Callable[[S], T], keys: Optional[Iterable[S]] = None):
        self._load = loader
        self._loaded = {}
        self._items = {}
        for key in keys or []:
            self._load[key] = False

    def __getitem__(self, item: S) -> T:
        if not self._loaded.get(item):
            self._items[item] = self._load(item)
            self._load[item] = True
        return self._items[item]

    def __iter__(self):
        return iter(self._loaded.keys())

    def __len__(self):
        return len(self._loaded)


models: Dict[str, MacrogenesisInfo] = {}


class NodeInput(BaseModel):
    nodes: str
    model: str = 'default'


class NodeReport(BaseModel):
    nodes: List[str]
    normalized: str
    not_found: Optional[List[str]]


class Direction(str, Enum):
    """Major direction for the graph (rankdir)"""
    LR = "LR"
    TB = "TB"
    BT = "BT"
    RL = "RL"


class CentralPaths(str, Enum):
    NONE = "no"
    ALL = "all"
    DAG = "dag"


class ExportFormat(str, Enum):
    SVG = 'svg'
    PDF = 'pdf'
    PNG = 'png'
    JPEG = 'jpg'


MIME_TYPES: Dict[ExportFormat, str] = {
    ExportFormat.SVG: 'image/svg+xml',
    ExportFormat.PDF: 'application/pdf',
    ExportFormat.PNG: 'image/png',
    ExportFormat.JPEG: 'image/jpeg'
}


@app.on_event('startup')
def load_models():
    # TODO load other models if available
    # TODO configurability
    models['default'] = MacrogenesisInfo('target/macrogenesis/macrogen-info.zip')


templates = Jinja2Templates(directory='src/graphviewer/templates')


@app.get('/macrogenesis/subgraph')
async def render_form(request: Request):
    return templates.TemplateResponse('form.html', {'request': request})


@app.get('/macrogenesis/subgraph/help')
async def render_form(request: Request):
    return templates.TemplateResponse('help.html', {'request': request})


@app.get('/macrogenesis/subgraph/check-nodes', response_model=NodeReport)
def check_nodes(nodeinfo: NodeInput = Depends()) -> NodeReport:
    """
    Normalizes the given node string.
    """
    model = models[nodeinfo.model]
    nodes, errors = model.nodes(nodeinfo.nodes, report_errors=True)
    return NodeReport(nodes=[str(node) for node in nodes],
                      normalized=', '.join(map(str, nodes)),
                      not_found=errors)

@dataclass
class _AGraphInfo:
    graph: AGraph
    nodes: List[Node]
    extra_nodes: List[Node]
    unknown_nodes: List[str]
    basename: str

@pydantic.dataclasses.dataclass
class AGraphInfo:
    dot: str
    nodes: List[str]
    extra_nodes: List[str]
    unknown_nodes: List[str]


def agraph(nodeinfo: NodeInput = Depends(),
           context: bool = False,
           abs_dates: bool = False,
           induced_edges: bool = False,
           ignored_edges: bool = False,
           assertions: bool = False,
           extra: str = '',
           paths_wo_timeline: bool = False,
           tred: bool = False,
           nohl: bool = False,
           syn: bool = False,
           inscriptions: bool = False,
           order: bool = False,
           collapse: bool = False,
           dir: Direction = Direction.LR,
           central_paths: CentralPaths = CentralPaths.ALL,
           no_edge_labels: bool = False) -> _AGraphInfo:
    """
    Creates the actual graph.
    """

    # retrieve the nodes by string
    model = models[nodeinfo.model]
    nodes, unknown_nodes = model.nodes(nodeinfo.nodes, report_errors=True)
    extra_nodes, unknown_extra_nodes = model.nodes(extra, report_errors=True)

    # extract the basic subgraph
    g = model.subgraph(*nodes, context=context, abs_dates=abs_dates, paths=extra_nodes,
                       paths_without_timeline=paths_wo_timeline,
                       paths_between_nodes=central_paths.value,
                       direct_assertions=assertions, include_syn_clusters=syn,
                       include_inscription_clusters=inscriptions)

    if induced_edges:  # TODO refactor into model.subgraph?
        g = model.base.subgraph(g.nodes).copy()

    # now remove ignored or conflicting edges, depending on the options
    if not ignored_edges or tred:
        g = remove_edges(g, lambda u, v, attr: attr.get('ignore', False) and not attr.get('kind', '') == 'temp-syn')
    if not syn:
        g = remove_edges(g, lambda u, v, attr: attr.get('kind', None) == 'temp-syn')
    if tred:
        g = remove_edges(g, lambda u, v, attr: attr.get('delete', False))

        # now actual tred
        if nx.is_directed_acyclic_graph(g):
            reduction = nx.transitive_reduction(g)
            g = g.edge_subgraph([(u, v, k) for u, v, k, _ in expand_edges(g, reduction.edges)])
        else:
            raise ValueError(
                    'Cannot produce transitive reduction – the subgraph is not acyclic after removing conflict edges!')

    g = simplify_timeline(g)

    if collapse:
        g = collapse_parallel_edges(g)

    # make sure the central nodes are actually in the subgraph. They might theoretically have fallen out by
    # one of the reduction operations before, e.g., the edge subgraph required for tred
    g.add_nodes_from(nodes)

    # now we have our subgraph. All following operations are visualisation focused
    if order:
        g = model.order_graph(g)  # adjusts weights & invisible edges to make graphviz layout the nodes in
        # a straight line according to the order of the model
    agraph = write_dot(g, target=None, highlight=None if nohl else nodes, edge_labels=not no_edge_labels)
    basename = ",".join(
            [str(node.filename.stem if hasattr(node, 'filename') else node) for node in nodes])
    agraph.graph_attr['bgcolor'] = 'transparent'
    agraph.graph_attr['rankdir'] = dir.value
    if order:
        agraph.graph_attr['ranksep'] = '0.2'
    return _AGraphInfo(agraph, nodes, extra_nodes, unknown_nodes + unknown_extra_nodes, basename)


@app.get('/macrogenesis/subgraph/extract')
def extract_dot(info: _AGraphInfo = Depends(agraph)) -> AGraphInfo:
    return AGraphInfo(dot=info.graph.to_string(),
                      nodes=[str(node) for node in info.nodes],
                      extra_nodes=[str(node) for node in info.extra_nodes],
                      unknown_nodes=info.unknown_nodes)


@app.get('/macrogenesis/subgraph/dot')
def render_dot(info: _AGraphInfo = Depends(agraph)):
    return Response(content=info.graph.to_string(),
                    media_type='text/vnd.graphviz',
                    headers={'Content-Disposition': f'attachment; filename="{info.basename}.dot"'})


@app.get('/macrogenesis/subgraph/{format}')
def render_image(format: ExportFormat, agraph_info: _AGraphInfo = Depends(agraph)):
    # TODO convert to asyncio.subprocess
    output = subprocess.check_output(['dot', '-T', format.value], input=codecs.encode(agraph_info.graph.to_string()), timeout=30)
    return Response(output,
                    media_type=MIME_TYPES[format],
                    headers={
                        'Content-Disposition': f'attachment; filename="{agraph_info.basename}.{format.value}"'})


if __name__ == '__main__':
    uvicorn.run('gvfa:app', host="0.0.0.0", port=5000, reload=True, reload_dirs=["src/graphviewer"])
