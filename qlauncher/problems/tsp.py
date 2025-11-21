from typing import Literal

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import SamplingMinimumEigensolverResult

import qlauncher.hampy as hampy
from qlauncher.base import Problem
from qlauncher.hampy import Equation
from qlauncher.hampy.utils import shift_affected_qubits


class TSP(Problem):
	"""
	Traveling Salesman Problem (TSP) definition.
	"""

	def __init__(self, instance: nx.Graph, instance_name: str = 'unnamed'):
		"""
		Args:
			instance (nx.Graph): Graph representing the TSP instance.
			instance_name (str): Name of the instance.
			quadratic (bool): Whether to use quadratic constraints
		"""
		super().__init__(instance=instance, instance_name=instance_name)

	@property
	def setup(self) -> dict:
		return {'instance_name': self.instance_name}

	def _get_path(self) -> str:
		return f'{self.name}@{self.instance_name}'

	def _solution_to_node_chain(self, solution: SamplingMinimumEigensolverResult | str) -> np.ndarray:
		"""
		Converts the solution of the TSP problem to a chain of nodes (order of visiting)

		Args:
			solution: Solution of the TSP problem. If type is str, qiskit-style ordering is assumed (MSB first)

		Returns:
			np.ndarray: Solution chain of nodes to visit
		"""
		bitstring = solution[::-1] if isinstance(solution, str) else solution.best_measurement['bitstring'][::-1]  # type: ignore

		node_count = int(len(bitstring) ** 0.5)
		chain = []

		for i in range(0, len(bitstring), node_count):
			step_string = bitstring[i : i + node_count]
			chosen_node = np.argmax([int(x) for x in step_string])
			chain.append(chosen_node)

		return np.array(chain)

	def _calculate_solution_cost(self, solution: SamplingMinimumEigensolverResult | str | list[int]) -> float:
		cost = 0
		chain = solution if isinstance(solution, list) else self._solution_to_node_chain(solution)

		for i in range(len(chain) - 1):
			cost += self.instance[chain[i]][chain[i + 1]]['weight']
		cost += self.instance[chain[-1]][chain[0]]['weight']
		return cost

	def _visualize_solution(self, solution: SamplingMinimumEigensolverResult | str | list[int]) -> None:
		chain = solution if isinstance(solution, list) else self._solution_to_node_chain(solution)

		import matplotlib.pyplot as plt

		pos = nx.spring_layout(self.instance, weight=None, seed=42)  # set seed for same node graphs in plt
		problem_edges = list(self.instance.edges)

		marked_edges = []
		for i in range(len(chain) - 1):
			marked_edges.append({chain[i], chain[i + 1]})
		marked_edges.append({chain[-1], chain[0]})

		draw_colors = []  # Limegreen for marked edges, gray for unmarked
		draw_widths = []  # Draw marked edges thicker

		path_cost = 0
		for edge in problem_edges:
			draw_colors.append('limegreen' if set(edge) in marked_edges else 'gray')
			draw_widths.append(2 if set(edge) in marked_edges else 1)

			path_cost += self.instance[edge[0]][edge[1]]['weight'] if set(edge) in marked_edges else 0

		plt.figure(figsize=(8, 6))
		nx.draw(
			self.instance,
			pos,
			edge_color=draw_colors,
			width=draw_widths,
			with_labels=True,
			node_color='skyblue',
			node_size=500,
			font_size=10,
			font_weight='bold',
		)

		labels = nx.get_edge_attributes(self.instance, 'weight')
		nx.draw_networkx_edge_labels(
			self.instance,
			pos,
			edge_labels=labels,
			rotate=False,
			font_weight='bold',
			label_pos=0.45,
		)

		plt.title('TSP Solution Visualization')
		plt.suptitle(f'Path cost:{path_cost}')
		plt.show()

	def _visualize_problem(self) -> None:
		"""
		Show plot of the TSP instance.
		"""

		pos = nx.spring_layout(self.instance, weight=None, seed=42)
		plt.figure(figsize=(8, 6))

		nx.draw(
			self.instance,
			pos,
			with_labels=True,
			node_color='skyblue',
			node_size=500,
			edge_color='gray',
			font_size=10,
			font_weight='bold',
		)

		labels = nx.get_edge_attributes(self.instance, 'weight')
		nx.draw_networkx_edge_labels(
			self.instance,
			pos,
			edge_labels=labels,
			rotate=False,
			font_weight='bold',
			node_size=500,
			label_pos=0.45,
		)

		plt.title('TSP Instance Visualization')
		plt.show()

	def visualize(self, solution: SamplingMinimumEigensolverResult | str | list[int] | None = None) -> None:
		if solution is None:
			self._visualize_problem()
		else:
			self._visualize_solution(solution)

	@staticmethod
	def from_preset(instance_name: str = 'default', **kwargs) -> 'TSP':
		"""
		Generate TSP instance from a preset name.

		Args:
			instance_name (str): Name of the preset instance
			quadratic (bool, optional): Whether to use quadratic constraints. Defaults to False

		Returns:
			TSP: TSP instance
		"""
		match instance_name:
			case 'default':
				edge_costs = np.array([[0, 1, 2, 3], [1, 0, 4, 5], [2, 4, 0, 6], [3, 5, 6, 0]])
			case _:
				raise ValueError('Unknown instance name')
		G = nx.Graph()
		n = edge_costs.shape[0]
		for i in range(n):
			for j in range(i + 1, n):  # No connections to self
				G.add_edge(i, j, weight=edge_costs[i, j])

		return TSP(instance=G, instance_name=instance_name)

	@staticmethod
	def generate_tsp_instance(num_vertices: int, min_distance: float = 1.0, max_distance: float = 10.0, **kwargs) -> 'TSP':
		"""
		Generate a random TSP instance.

		Args:
			num_vertices (int): Number of vertices in the graph
			min_distance (float, optional): Minimum distance between vertices. Defaults to 1.0
			max_distance (float, optional): Maximum distance between vertices. Defaults to 10.0
			quadratic (bool, optional): Whether to use quadratic constraints. Defaults to False

		Returns:
			TSP: TSP instance
		"""
		if num_vertices < 2:
			raise ValueError('num_vertices must be at least 2')

		if min_distance <= 0:
			raise ValueError('min_distance must be greater than 0')

		g = nx.Graph()
		rng = np.random.default_rng()
		for i in range(num_vertices):
			for j in range(i + 1, num_vertices):
				g.add_edge(i, j, weight=int(rng.uniform(low=min_distance, high=max_distance)))

		return TSP(instance=g, instance_name='generated')

	def _make_non_collision_hamiltonian(self, node_count: int, quadratic: bool = False) -> SparsePauliOp:
		"""
		Creates a Hamiltonian representing constraints for the TSP problem. (Each node visited only once, one node per timestep)
		Qubit mapping: [step1:[node_1, node_2,... node_n]],[step_2],...[step_n]

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
		t0_op = hampy.one_in_n(list(range(node_count)), eq.size, quadratic=quadratic)

		for timestep in range(node_count):
			shift = shift_affected_qubits(t0_op, timestep * node_count)
			eq += shift

		# Ensure that each node is visited only once
		n0_op = hampy.one_in_n([timestep * node_count for timestep in range(node_count)], eq.size, quadratic=quadratic)

		for node in range(node_count):
			shift = shift_affected_qubits(n0_op, node)
			eq += shift

		return -1 * eq.hamiltonian

	def _make_connection_hamiltonian(self, edge_costs: np.ndarray, return_to_start: bool = True) -> SparsePauliOp:
		"""
		Creates a Hamiltonian that represents the costs of picking each path.

		Args:
			tsp_matrix: Edge cost matrix of the TSP problem

		Returns:
			np.ndarray: Optimal chain of nodes to visit
		"""
		node_count = edge_costs.shape[0]
		if len(edge_costs.shape) != 2 or edge_costs.shape[1] != node_count:
			raise ValueError('edge_costs must be a square matrix')

		n = node_count**2
		eq = Equation(n)

		for timestep in range(node_count - 1):
			for node in range(node_count):
				for next_node in range(node_count):
					if node == next_node:
						continue
					and_hamiltonian = eq[node + timestep * node_count] & eq[next_node + (timestep + 1) * node_count]
					eq += edge_costs[node, next_node] * and_hamiltonian

		if not return_to_start:
			return eq.hamiltonian

		# Add cost of returning to the first node
		for node in range(node_count):
			for node2 in range(node_count):
				and_hamiltonian = eq[node + (node_count - 1) * node_count] & eq[node2]
				eq += edge_costs[node, node2] * and_hamiltonian

		return eq.hamiltonian

	def to_hamiltonian(
		self,
		constraints_weight: int = 5,
		costs_weight: int = 1,
		return_to_start: bool = True,
		onehot: Literal['exact', 'quadratic'] = 'exact',
	) -> SparsePauliOp:
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
		instance_graph = self.instance

		edge_costs = nx.to_numpy_array(instance_graph)
		# discourage breaking the constraints
		edge_costs += np.eye(len(edge_costs)) * np.max(edge_costs)
		scaled_edge_costs = edge_costs.astype(np.float32) / np.max(edge_costs)

		node_count = len(instance_graph.nodes)

		constraints = self._make_non_collision_hamiltonian(node_count, quadratic=(onehot == 'quadratic'))
		costs = self._make_connection_hamiltonian(scaled_edge_costs, return_to_start=return_to_start)

		return constraints * constraints_weight + costs * costs_weight
