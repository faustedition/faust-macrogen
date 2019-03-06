# -*- coding: utf-8 -*-
import itertools
from collections import defaultdict
from typing import Tuple, List, Generator, TypeVar, Iterable, Sequence, Optional
from .config import config
import networkx as nx
import numpy as np
import cvxpy as cp

logger = config.getLogger(__name__)

V = TypeVar('V')


class Eades:

    def to_start(self, node):
        """
        Removes the node from the graph and appends it to the start sequence.
        """
        self.start.append(node)
        self.graph.remove_node(node)

    def to_end(self, node):
        self.end.insert(0, node)
        self.graph.remove_node(node)

    def _exhaust_sinks(self, sink: bool = True):
        """
        Produces all sinks until there are no more.

        Warning: This modifies the graph g
        """
        sink_method = self.graph.out_degree if sink else self.graph.in_degree
        while True:
            sinks = [u for (u, d) in sink_method() if d == 0]
            if sinks:
                yield from sinks
            else:
                return

    def _exhaust_sources(self):
        """
        Produces all sources until there are no more.

        Warning: This modifies the given graph
        """
        return self._exhaust_sinks(False)

    def __init__(self, graph: nx.DiGraph, double_check=True):
        """
        Fast heuristic for the minimum feedback arc set.

        Eades’ heuristic creates an ordering of all nodes of the given graph,
        such that each edge can be classified into *forward* or *backward* edges.
        The heuristic tries to minimize the sum of the weights (`weight` attribute)
        of the backward edges. It always produces an acyclic graph, however it can
        produce more conflicting edges than the minimal solution.

        Args:
            graph: a directed graph, may be a multigraph.
            double_check: check whether we’ve _really_ produced an acyclic graph

        Returns:
            a list of edges, removal of which guarantees a

        References:
            **Eades, P., Lin, X. and Smyth, W. F.** (1993). A fast and effective
            heuristic for the feedback arc set problem. *Information Processing
            Letters*, **47**\ (6): 319–23
            doi:\ `10.1016/0020-0190(93)90079-O. <https://doi.org/10.1016/0020-0190(93)90079-O.>`__
            http://www.sciencedirect.com/science/article/pii/002001909390079O
            (accessed 27 July 2018).
        """
        self.original_graph = graph
        g = graph.copy()
        self.graph = g
        self.logger = config.getLogger(__name__ + '.' + self.__class__.__name__)
        self.logger.debug('Internal eades calculation for a graph with %d nodes and %d edges', g.number_of_nodes(),
                          g.number_of_edges())
        self.start = self.end = None
        self.feedback_edges = None

    def solve(self) -> List[Tuple[V, V]]:
        self.start = []
        self.end = []
        self.graph.remove_edges_from(list(self.graph.selfloop_edges()))
        while self.graph:
            for v in self._exhaust_sinks():
                self.to_end(v)
            for v in self._exhaust_sources():
                self.to_start(v)
            if self.graph:
                u = max(self.graph.nodes, key=lambda v: self.graph.out_degree(v, weight='weight')
                                                        - self.graph.in_degree(v, weight='weight'))
                self.to_start(u)
        ordering = self.start + self.end
        pos = dict(zip(ordering, itertools.count()))
        feedback_edges = list(self.original_graph.selfloop_edges())
        for u, v in self.original_graph.edges():
            if pos[u] > pos[v]:
                feedback_edges.append((u, v))
        logger.debug('Found %d feedback edges', len(feedback_edges))
        self.feedback_edges = feedback_edges
        return feedback_edges

    def double_check(self):
        check = self.graph.copy()
        check.remove_edges_from(self.feedback_edges)
        if not nx.is_directed_acyclic_graph(check):
            logger.error('double-check: graph is not a dag!')
            cycles = nx.simple_cycles()
            counter_example = next(cycles)
            logger.error('Counterexample cycle: %s', counter_example)


def induced_cycles(graph: nx.DiGraph, fes: Iterable[Tuple[V, V]]) -> Generator[Iterable[V], None, None]:
    """
    Produce a simple cycle from the graph for every edge in the given feedback edge set (if it is an actual feedback
    edge). For edge `u` → `v`, the cycle is formed by the shorted path from `v` to `u` plus the edge `u`→`v` to
    finish the cycle. If no such path can be found in the given graph, the edge is silently skipped.

    Args:
        graph: the directed graph we work on
        fes: a set of feedback edges.

    Yields:
        paths (sequences of nodes) that form simple cycles

    See Also:
        fes_evans

    """
    for u, v in fes:
        try:
            path = nx.shortest_path(graph, v, u)
            path.append(v)  # complete simple cycle
            yield tuple(path)
        except nx.NetworkXNoPath:
            logger.debug('no feedback edge from %s to %s', u, v)


def eades(graph: nx.DiGraph, double_check: bool = False):
    solver = Eades(graph)
    result = solver.solve()
    if double_check:
        solver.double_check()
    return result


class FES_Baharev:
    """
    Calculates the minimum feedback edge set for a given graph using the
    algorithm presented by Baharev et al. (2015).

    Creating an instance of this class will initialize the data model by building a
    reduced version of the given graph (no multi-edges, no attributes except for weight
    and index) and defining an arbitrary ordering of the edges used for the vector
    representations. Calling the `solve` method will perform the actual optimization
    and return the set of edges, which is also available via attributes.

    Attributes:
        original_graph: The original graph we are called on
        graph: Simplified version: No parallel edges, weight and index attributes only
        edges: list of (u, v) tuples of edges in graph such that edges[i] has index i
        weights: weight vector for edges
        m: number of edges in graph
        solver_args: configured arguments to the solver
        solution: List of edges in feedback edge set
        solution_vector: binary vector of edges
        objective: sum of weights in feedback edge set
        iterations: number of baharev iterations used to calculate the edge set

    Notes:

        The behaviour can be configured using the macrogen configuration mechanism:
        - solvers contains a list of solvers to try
        - solver_options contains a dictionary with options for every (configurable) solver,
          `all` is used for default options


    Description:
        Baharev's algorithm is based on a formulation of the minimum feedback edge set problem as a mixed integer
        program: The objective to be minimized is the sum of the weights of the edges in the feedback set,
        and the constraint is that at least one edge of every simple cycle of the graph must be in the feedback set.
        Since the number of simple cycles may be exponential, the set of simple cycles is built in an iterative
        fashion.

    References:
        **Baharev, A., Schichl, H. and Neumaier, A.** (2015). An exact method
        for the minimum feedback arc set problem. : 34
        http://www.mat.univie.ac.at/~neum/ms/minimum_feedback_arc_set.pdf.
    """

    def __init__(self, graph: nx.DiGraph, force_forward_edges: Optional[List[Tuple[V, V]]] = None):
        self.original_graph = graph
        self.logger = config.getLogger(__name__ + '.' + self.__class__.__name__)

        # We need to address the edges by index, and we're only
        # interested in their weights. We collapse multi-edges,
        # and self loops are feedback edges anyway, so ignore them.
        edge_weights = defaultdict(int)
        for u, v, weight in graph.edges(data='weight', default=1):
            if u != v:
                edge_weights[u, v] += weight
        G = nx.DiGraph()
        weights = []
        edges = []
        for i, ((u, v), w) in enumerate(edge_weights.items()):
            if w < 0:
                raise ValueError(f"Edge {u} -> {v} has negative weights sum {w}, this is not supported.")
            G.add_edge(u, v, weight=w, index=i)
            weights.append(w)
            edges.append((u, v))

        if force_forward_edges is None:
            force_forward_edges = []

        self.graph = G
        self.weights = np.array(weights)
        self.edges = edges
        self.m = len(self.edges)
        self.force_forward_edges = force_forward_edges
        self.force_forward_vec = self.edge_vector(force_forward_edges)
        self.solver_args = {}
        self.solution_vector = None
        self.solution = None
        self.objective = None
        self.iterations = None
        self._load_solver_args()

    def _load_solver_args(self):
        """
        Loads the configuration
        """
        solvers: List[str] = config.solvers
        installed = cp.installed_solvers()
        index = 0
        while index < len(solvers):
            if solvers[index] not in installed:
                del solvers[index]
            else:
                index += 1
        if solvers and solvers[0]:
            solver = solvers[0]
        else:
            solver = None
        options = {}
        if 'all' in config.solver_options:
            options.update(config.solver_options['all'])
        if solver and solver in config.solver_options:
            options.update(config.solver_options[solver])
        if solver:
            options['solver'] = solver
        self.solver_args = options
        self.logger.info('configured solver: %s, options: %s (installed solvers: %s)',
                         solver, options, ', '.join(installed))

    def edge_vector(self, edges: Iterable[Tuple[V, V]]) -> np.ndarray:
        """
        Converts a list of edges to a boolean vector which is True if the corresponding edge is contained in the edge list.
        """
        edge_set = frozenset(edges)
        return np.array([(edge in edge_set) for edge in self.edges])

    def edges_for_vector(self, edge_vector: Sequence[bool]) -> List[Tuple[V, V]]:
        """
        Translates an edge vector back to a list of edges.

        Args:
            edge_vector: sequence of m booleans indicating whether edge m is present or not

        Returns:
            list of edges which are true in edge vector as (source, target) node pairs

        See Also:
            edge_vector
        """
        S = [self.edges[i] for i in range(self.m) if edge_vector[i]]  # TODO optimize
        return S

    def solve(self):
        """
        Runs the actual optimization. This may take a while ...

        Returns:
            the edge set as list of (u,v) tuples
        """
        initial_fes = eades(self.graph)
        initial_fes_vec = self.edge_vector(initial_fes)

        # bounds for the objective
        lower_bound = 0
        upper_bound = np.sum(initial_fes_vec * self.weights)

        self.logger.info('Calculating FES for graph with %d edges, max %d feedback edges', self.m, len(initial_fes))

        simple_cycles = set(induced_cycles(self.graph, initial_fes))

        for iteration in itertools.count(1):
            self.logger.info('Baharev iteration %d, %g <= objective <= %g, %d simple cycles', iteration, lower_bound,
                             upper_bound, len(simple_cycles))

            # Formulate and solve the problem for this iteration:
            y = cp.Variable(self.m, boolean=True, name="y")
            objective = cp.Minimize(cp.sum(y * self.weights))

            cycle_vectors = [self.edge_vector(nx.utils.pairwise(cycle)) for cycle in simple_cycles]
            constraints = [cp.sum(a * y) >= 1 for a in cycle_vectors]
            constraints.append(cp.sum(y * self.force_forward_vec) == 0)  # no force forward vec may be in the result set
            problem = cp.Problem(objective, constraints)
            resolution = problem.solve(**self.solver_args)
            if problem.status != 'optimal':
                self.logger.warning('Optimization solution is %s. Try solver != %s?', problem.status,
                                    problem.solver_stats.solver_name)
            self.logger.debug(
                "Solved optimization problem with %d constraints: %s -> %s (%g + %g seconds, %d iterations, solver %s)",
                len(constraints), resolution, problem.solution.status,
                problem.solver_stats.solve_time or 0, problem.solver_stats.setup_time or 0,
                problem.solver_stats.num_iters or 0, problem.solver_stats.solver_name)
            current_solution = np.abs(y.value) >= 0.5  # y.value = vector of floats each ≈ 0 or 1
            current_fes = self.edges_for_vector(current_solution)
            self.logger.debug('Iteration %d, resolution: %s, %d feedback edges', iteration, resolution,
                              len(current_fes))
            # S, the feedback edge set calculated using the constraint subset, can be an incomplete solution
            # (i.e. cycles remain after removing S from the graph). So lets compare this with the upper bound
            # from the heuristic
            lower_bound = max(lower_bound, objective.value)
            if lower_bound == upper_bound:
                self.logger.info('upper == lower bound == %g, optimal solution found', lower_bound)
                break  # y.value is the optimal solution

            if resolution > upper_bound:
                self.logger.error('Solution %g > upper bound %g!', resolution, upper_bound)
                break

            Gi = self.graph.copy()
            Gi.remove_edges_from(current_fes)
            if nx.is_directed_acyclic_graph(Gi):
                self.logger.info('Graph is acyclic, optimal solution found')
                break  # y.value is the optimal solution

            # The solution is not yet ideal. So we take G^(i), the graph still containing some feedback edges,
            # calculate a heuristic on it and use the heuristic (= over-estimation) to adjust upper bound and
            # determine additional simple cycles (= constraints)
            Fi = eades(Gi)
            yi = self.edge_vector(Fi) | current_solution
            zi = np.sum(yi * self.weights)
            if zi < upper_bound:
                upper_bound = zi
                current_solution = yi
            simple_cycles |= set(induced_cycles(Gi, Fi))

        self.solution_vector = current_solution
        self.solution = self.edges_for_vector(current_solution)
        self.objective = objective.value
        self.iterations = iteration
        return self.solution
