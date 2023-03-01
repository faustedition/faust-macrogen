import asyncio
import codecs
import logging
from asyncio import create_subprocess_exec, wait_for
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TypeVar, Callable, List, Dict, Optional, Iterable

import networkx as nx
import pydantic
import uvicorn
from fastapi import FastAPI, Depends, Request, Response, HTTPException
from fastapi.exception_handlers import http_exception_handler
from fastapi.templating import Jinja2Templates
from fastapi.exceptions import HTTPException
from macrogen import MacrogenesisInfo, write_dot
from macrogen.graph import Node
from macrogen.graphutils import remove_edges, expand_edges, simplify_timeline, collapse_parallel_edges
from macrogen.config import config
from starlette.responses import HTMLResponse

logger = logging.getLogger(__name__)

from pydantic import BaseModel, BaseSettings
from pygraphviz import AGraph


class Settings(BaseSettings):
    default_model: Path = Path('target/macrogenesis/macrogen-info.zip')
    extra_models: dict[str, Path] | list[Path] | str = '*-*/macrogen-info.zip'
    class Config:
        env_file = '.env'

settings = Settings()
app = FastAPI(debug=True)

S = TypeVar('S')
T = TypeVar('T')


@app.exception_handler(HTTPException)
async def logging_exception_handler(request, exc):
    logger.exception(exc)
    return http_exception_handler(request, exc)


class LazyLoader(Mapping):
    _items: Dict[S, T]
    _loaded: Dict[S, bool]

    def __init__(self, loader: Callable[[S], T], keys: Optional[Iterable[S]] = None):
        self._load = loader
        self._loaded = {}
        self._items = {}
        for key in keys or []:
            self._loaded[key] = False

    def __getitem__(self, item: S) -> T:
        if not self._loaded.get(item):
            self._items[item] = self._load(item)
            self._loaded[item] = True
        return self._items[item]

    def __iter__(self):
        return iter(self._loaded.keys())

    def __len__(self):
        return len(self._loaded)


models: Mapping[str, MacrogenesisInfo] = {}


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
    global models

    by_key = {'default': settings.default_model}
    if isinstance(settings.extra_models, dict):
        by_key.update({k: Path(v) for (k, v) in settings.extra_models.items()})
    else:
        if isinstance(settings.extra_models, list):
            model_files = [Path(entry) for entry in settings.extra_models]
        else:
            model_files = [entry for entry in Path().glob(settings.extra_models)]
        by_key.update({p.parent.stem: p for p in model_files if p not in by_key.values()})

    def load_model(key):
        logger.info('Loading model %s ...', key)
        return MacrogenesisInfo(by_key[key])

    models = LazyLoader(load_model, by_key.keys())
    logger.info('Found %d models: %s', len(by_key), by_key.keys())


templates = Jinja2Templates(directory='src/graphviewer/templates')


@app.get('/macrogenesis/subgraph', response_class=HTMLResponse)
async def render_form(request: Request) -> templates.TemplateResponse:
    """
    Returns a HTML page with the subgraph viewer’s frontend.
    """
    return templates.TemplateResponse('form.html', {'request': request, 'models': models})


@app.get('/macrogenesis/subgraph/help', response_class=HTMLResponse)
async def render_form(request: Request) -> templates.TemplateResponse:
    """
    Returns a HTML page with help on the options of the subgraph viewer.
    """
    return templates.TemplateResponse('help.html', {'request': request, 'models': models})


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
            raise HTTPException(500, dict(
                    error="not-acyclic",
                    msg='Cannot produce transitive reduction – the subgraph is not acyclic after removing conflict edges!',
                    dot=write_dot(g, target=None, highlight=nodes).to_string()))

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


@app.get('/macrogenesis/subgraph/dot', responses={200: {'content': {'text/vnd.graphviz': {}}}})
def render_dot(info: _AGraphInfo = Depends(agraph)) -> Response:
    """
    Returns a dot file with the (unlayouted) graph.
    """
    return Response(content=info.graph.to_string(),
                    media_type='text/vnd.graphviz',
                    headers={'Content-Disposition': f'attachment; filename="{info.basename}.dot"'})


@app.get('/macrogenesis/subgraph/{format}', responses={
    404: {'description': 'None of the given nodes have been found in the graph.'},
    513: {'description': 'The rendering process failed due to a GraphViz error.'},
    504: {'description': 'The rendering process timed out, probably due to a too complex graph. '
                         'You should get a dot file using the /dot endpoint and run GraphViz locally.'}
})
async def render_image(format: ExportFormat, agraph_info: _AGraphInfo = Depends(agraph)):
    """
    Layouts and renders a graph.
    
    When successful, this will return an image in the given format. Note that this shouldn’t be used
    for too complex cases since it has quite a low timeout. 
    """
    if not agraph_info.nodes:
        raise HTTPException(404, dict(error="empty", msg="No nodes in graph", unknown_nodes=agraph_info.unknown_nodes))
    proc = await create_subprocess_exec('dot', '-T', format.value, stdin=asyncio.subprocess.PIPE,
                                        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        output, errors = await wait_for(proc.communicate(codecs.encode(agraph_info.graph.to_string())), timeout=10)
        if proc.returncode != 0:
            raise HTTPException(513, f'Rendering {format} failed:\n\n{codecs.decode(errors)}')
        return Response(output,
                        media_type=MIME_TYPES[format],
                        headers={
                            'Content-Disposition': f'attachment; filename="{agraph_info.basename}.{format.value}"'})
    except asyncio.TimeoutError:
        proc.terminate()
        raise HTTPException(504, dict(error="timeout",
                                      msg="This layout is to complex to be rendered within the server's limits.\n"
                                          "Download a .dot file and run graphviz on your local machine to get a rendering."))

def run_dev_server():
    print(logger.name)
    uvicorn.run('gvfa:app', host="0.0.0.0", port=5000, reload=True, reload_dirs=["src/graphviewer"])


if __name__ == '__main__':
    run_dev_server()
