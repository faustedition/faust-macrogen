from enum import Enum
from pathlib import Path
from typing import Set, Optional

import networkx as nx
from macrogen.uris import Reference
from macrogen.bibliography import BiblSource
from more_itertools import first


class Side(Enum):
    START = 'start'
    END = 'end'

    @property
    def label(self):
        return "Anfang" if self == Side.START else "Ende"


class SplitReference(Reference):
    """
    Represents either the start or the end of the process of working on a reference.
    """

    def __init__(self, reference: Reference, side: Side, other: Optional['SplitReference'] = None):
        if isinstance(reference, SplitReference):
            raise TypeError(f'Cannot create a SplitReference from SplitReference {reference}')
        super().__init__(reference.uri + '#' + side.value)
        self.reference = reference
        self.side = side
        self.other = other

    def __getattr__(self, item):
        if hasattr(self.reference, item):
            return getattr(self.reference, item)
        else:
            raise AttributeError(f"Neither the split reference nor the base object has attribute {item}")

    @property
    def filename(self) -> Path:
        return self.reference.filename

    @property
    def label(self) -> str:
        return self.reference.label  # XXX start/end?

    def __str__(self):
        return f"{self.reference} ({self.side.label})"

    @classmethod
    def both(cls, reference):
        start = cls(reference, Side.START)
        end = cls(reference, Side.END, other=start)
        start.other = end
        return {Side.START: start, Side.END: end}


def start_end_graph(orig: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Converts a graph containing References into a graph that represents start and end of the writing process for
    each reference by separate nodes.

    Args:
        orig: A graph that does not contain `SplitReference`s.

    Returns:
        A new graph with all edges u,v of the original graph, where u is replaced by SplitReference(u, Sides.END)
        if it is a `Reference`, and v replaced by SplitReference(v, Sides.START) if it is a reference. Additionally,
        edges ref.start → ref.end with kind='progress' are added for all references.
    """
    refs = {node: SplitReference.both(node) for node in orig.nodes if isinstance(node, Reference)}
    assert not any(isinstance(node, SplitReference) for node in refs)
    result = nx.MultiDiGraph()
    for u, v, k, attr in orig.edges(keys=True, data=True):
        if u in refs: u = refs[u][Side.END]
        if v in refs: v = refs[v][Side.START]
        result.add_edge(u, v, k, **attr)

    for ref, sides in refs.items():
        result.add_edge(sides[Side.START], sides[Side.END], kind="progress", source=BiblSource('faust://progress'))

    return result


def references(graph: nx.MultiDiGraph) -> Set[Reference]:
    return {node.reference if isinstance(node, SplitReference) else node
            for node in graph.nodes if isinstance(node, Reference)}


def side_node(graph: nx.MultiDiGraph, ref: Reference, side: Side) -> SplitReference:
    if isinstance(ref, SplitReference):
        ref = ref.reference
    match = SplitReference(ref, side)
    return first(node for node in graph.nodes if node == match)


def side_nodes(graph: nx.MultiDiGraph, side: Side) -> Set[SplitReference]:
    return {node for node in graph.nodes if isinstance(node, SplitReference) and node.side == side}
