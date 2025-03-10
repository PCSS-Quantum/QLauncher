from typing import Literal

import numpy as np
import networkx as nx

from quantum_launcher.problems.problem_initialization import TSP

from quantum_launcher.hampy import Equation
import quantum_launcher.hampy as hampy
from quantum_launcher.hampy.utils import shift_affected_qubits


def make_non_collision_hamiltonian(node_count: int, quadratic=False):
    """
    Creates a Hamiltonian representing constraints for the TSP problem. (Each node visited only once, one node per timestep)
    Qubit mapping: [step1:[node1, node2,... noden]],[step2],...[stepn]

    Args:
        node_count: Number of nodes in the TSP problem
        quadratic: Whether to encode as a QUBO problem

    Returns:
        np.ndarray: Hamiltonian representing the constraints
    """

    n = node_count**2
    eq = Equation(n)

    # hampy.one_in_n() takes a long time to execute with large qubit counts.
    # We can exploit the nature of tsp constraints and create the operator for the first timestep and then shift it for the rest of the timesteps
    # Same with nodes below.
    # I'm pretty sure this works as intended...

    # Ensure that at each timestep only one node is visited
    t0_op = hampy.one_in_n(
        [i for i in range(node_count)],
        eq.size,
        quadratic=quadratic
    )

    for timestep in range(node_count):
        shift = shift_affected_qubits(t0_op, timestep * node_count)
        eq += shift

    # Ensure that each node is visited only once
    n0_op = hampy.one_in_n(
        [timestep * node_count for timestep in range(node_count)],
        eq.size,
        quadratic=quadratic
    )

    for node in range(node_count):
        shift = shift_affected_qubits(n0_op, node)
        eq += shift

    return -1 * eq.hamiltonian


def make_connection_hamiltonian(edge_costs: np.ndarray, return_to_start: bool = True) -> np.ndarray:
    """
    Creates a Hamiltonian that represents the costs of picking each path.

    Args:
        tsp_matrix: Edge cost matrix of the TSP problem

    Returns:
        np.ndarray: Optimal chain of nodes to visit
    """
    node_count = edge_costs.shape[0]
    if len(edge_costs.shape) != 2 or edge_costs.shape[1] != node_count:
        raise ValueError("edge_costs must be a square matrix")

    n = node_count**2
    eq = Equation(n)

    for timestep in range(node_count - 1):
        for node in range(node_count):
            for next_node in range(node_count):
                if node == next_node:
                    continue
                and_hamiltonian = (
                    eq[node + timestep * node_count]
                    & eq[next_node + (timestep + 1) * node_count]
                )
                eq += edge_costs[node, next_node] * and_hamiltonian

    if not return_to_start:
        return eq.hamiltonian

    # Add cost of returning to the first node
    for node in range(node_count):
        for node2 in range(node_count):
            and_hamiltonian = (
                eq[node + (node_count - 1) * node_count] & eq[node2]
            )
            eq += edge_costs[node, node2] * and_hamiltonian

    return eq.hamiltonian


def problem_to_hamiltonian(problem: TSP, constraints_weight: int = 5, costs_weight: int = 1, return_to_start: bool = True, onehot: Literal['exact', 'quadratic'] = 'exact') -> np.ndarray:
    """
    Creates a Hamiltonian for the TSP problem.

    Args:
        problem: TSP problem instance
        quadratic: Whether to encode as a quadratic Hamiltonian
        constraints_weight: Weight of the constraints in the Hamiltonian
        costs_weight: Weight of the costs in the Hamiltonian

    Returns:
        np.ndarray: Hamiltonian representing the TSP problem
    """
    instance_graph = problem.instance

    edge_costs = nx.to_numpy_array(instance_graph)
    # discourage breaking the constraints
    edge_costs += np.eye(len(edge_costs)) * np.max(edge_costs)
    scaled_edge_costs = edge_costs.astype(np.float32) / np.max(edge_costs)

    node_count = len(instance_graph.nodes)

    constraints = make_non_collision_hamiltonian(node_count, quadratic=(onehot == 'quadratic'))
    costs = make_connection_hamiltonian(scaled_edge_costs, return_to_start=return_to_start)

    return constraints * constraints_weight + costs * costs_weight
