## Interactive Subgraph Viewer

Some graphs are too complex, some are too simple: So let’s build an interactive tool that renders extracts from the base graph that can be customized.

Basically, we take an already rendered complete graph and generate subgraphs from it. Subgraph generation can be customized, and rendering a new graph will be quite fast – so there is no adjustment of the FES possible. 

## Options to specify

All steps are applied from top to bottom to create the graph actually plotted.

### Choose core node(s)

One or more nodes around which the graph should be drawn. This can be sigils or dates.

### Add relevant context nodes

* [ ] __Context__: the neighbours of each core node
* [ ] __justify absolute datings__: For each core node, a path to the closest inferred absolute datings
* __extra paths__: list of nodes – if any of the nodes is reachable from one of the context node(s), include the shortest path to that.

### Which edges to include

* [ ] __all induced edges__: include all edges between nodes in the graph
* [ ] __remove ignored__: remove ignored (gray) edges
* [ ] __transitive reduction__: remove structurally redundant paths [^1]

[^1]: since the transitive reduction can only be calculated for DAGs, this either generally does not work with cyclic graphs or we need to remove the conflict edges first, then calculate the TRED and finally maybe add the conflict edges back again.

### Styling

* [ ] __collapse parallel edges__
* [ ] __show edge labels__

## Implementation Notes

* core nodes, context, absolute datings and extra path are implemented in MacrogenesisInfo.subgraph()
* all induced edges: `G = info.base.subgraph(G.nodes)`
* remove ignored: `G = macrogen.graphutils.remove_edges(G, lambda u, v, attr: attr.get('ignore'))`
* TRED: basically `G.edge_subgraph(macrogen.graphutils.expand_edges(G, nx.transitive_reduction(G).edges))` plus workaround for `delete` and `ignore` edges
* collapse: We’ve a function for that
* edge labels: `write_dot` has an option for that
