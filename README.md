Macrogenetic Analysis for the Faustedition
==========================================

These scripts analyze the macrogenetic data from the data/xml/macrogenesis folder. They create the macrogenesis lab area of the edition, i.e. everything below macrogenesis/, and an order of witnesses used in the bargraph and in the variant apparatus.

We only support running this on Linux.

### What’s in the box

- the macrogenesis python package contains all code to work with the data. Its main nentry point is the Macrogenesis class, which can either build an analysis structore from the XML data or load a graph structure from a previous run.
- the `macrogen` command line script is the main script to run the analysis or the reporting or both.
- the interactive subgraph viewer (graphviewer) is a FastAPI based service that can be used to interactively display parts of the graph.

### Installation and Usage – Basic Steps

Python ≥ 3.9 and GraphViz ≤ 2.38 or ≥ 2.41 need to be installed separately.

```bash
git submodules update --init --remote
pip install -e .
macrogen
```

will produce the output.

### Installation – Details

The main supported way of installation is to clone the repository and then run `pip install .` to install the package, potentially into a virtual environment. For development, install Poetry and run `poetry install .`. 

There is one supported optional features (or 'extra'):

* __fastapi__: run `pip install .[fastapi]` to get FastAPI, uvicorn, gunicorn and everything else needed to run the interactive graphviewer

There are two additional historical extras, which may still work but are no longer supported:

* `graphviewer` for the old, Flask-based interactive graphviewer
* `igraph` for the old, igraph-based solver that can be optionally configured.

#### The bootstrapped environment of the Gradle task

It is possible to __bootstrap__ a Python environment with everything required to run macrogen. This is implemented in the installMacrogen gradle task of the build.gradle script. This is used by the global Gradle task in `faust-gen`, and it can be triggered by running `./gradlew installMacrogen`. 

The bootstrapping process will download micromamba and then use that to build a python environment in `build/envs/macrogen/` using conda packages. This is controlled using the [environment.yml])(environment.yml) YAML file. Afterwards, `pip` is used to install the local package into that environment (to make sure we have everything from pyproject.toml). 


### Additional Configuration

Macrogenesis data structure is documented elsewhere (TODO Link).

Use --help to see a list of options.

* `src/macrogen/etc/default.yaml` is the main configuration file that can be copied and edited. It links to various extra files:
* `logging.yaml` contains the logging configuration for the main script. It’s a YAML file containing the data in the [dictConfig](https://docs.python.org/3/library/logging.config.html#logging.config.dictConfig) format of Python’s logging system.
* `styles.yaml` contains styling information for the graphs.

   It is a YAML file with a top-level mapping with two entries: node and edge, for the node styles and edge styles, respectively. Each of these mappings contain a 2nd level mapping where the keys identify which nodes/edges to style and the values are 3rd level mappings that are directly translated into [GraphViz attributes](https://www.graphviz.org/doc/info/attrs.html#d:stylesheet).

   The keys can be:

   - values of the nodes’ or edges’ `kind` attribute – this corresponds to the class name in nodes and to the relation name in edges.
   - names of additional attributes – in this case the styles are applied when the node/edge has an attribute of that name with a truthy value. 
   
   Examples for the latter case can be
   
   * `highlight` for stuff that is to be highlighted specific graphs
   * `deleted` for conflicting edges
   * `ignored` for ignored edges

* `bibscores.tsv` assigns a default score to each bibliographic source
* `uri-corrections.csv` contains corrections for URIs from the macrogenetic data that could not be identified
