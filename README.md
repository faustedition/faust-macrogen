Macrogenetic Analysis for the Faustedition
==========================================

These scripts analyze the macrogenetic data from the data/xml/macrogenesis folder. They create the macrogenesis lab area of the edition, i.e. everything below macrogenesis/, and an order of witnesses used in the bargraph and in the variant apparatus.

### Installation and Usage

Python ≥ 3.6 and GraphViz ≤ 2.38 or ≥ 2.41 need to be installed separately.

```bash
git submodules update --init --remote
pip install pipenv
pipenv install
pipenv run ./main.py
```

will produce the output.

### Additional Configuration

Macrogenesis data structure is documented elsewhere (TODO Link).

* `faust.ini` is an ini-style file that contains some configuration options (paths to the data etc.)
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
