"""This module contains the EC class."""

import ast
from collections import defaultdict
from typing import Literal

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from qiskit.quantum_info import SparsePauliOp

from qlauncher import hampy
from qlauncher.base import Problem
from qlauncher.base.problem_like import QUBO, Hamiltonian


class EC(Problem):
	"""
	Class for exact cover problem.

	The exact cover problem is a combinatorial optimization problem that involves finding a subset of a given set
	of elements such that the subset covers all elements and the number of elements in the subset is minimized.
	The class contains an instance of the problem, so it can be passed into QLauncher.

	Attributes:

	onehot (str): The one-hot encoding used for the problem.
	instance (any): The instance of the problem.
	instance_name (str | None): The name of the instance.
	instance_path (str): The path to the instance file.

	"""

	def __init__(self, instance: list[set[int]], instance_name: str = 'unnamed') -> None:
		super().__init__(instance=instance, instance_name=instance_name)

	@staticmethod
	def from_preset(instance_name: Literal['micro', 'default'], **kwargs) -> 'EC':
		match instance_name:
			case 'micro':
				instance = [{1, 2}, {1}]
			case 'default':
				instance = [{1, 4, 7}, {1, 4}, {4, 5, 7}, {3, 5, 6}, {2, 3, 6, 7}, {2, 7}]
			case _:
				raise TypeError
		return EC(instance=instance, instance_name=instance_name, **kwargs)

	@classmethod
	def from_file(cls, path: str, **kwargs) -> 'EC':
		with open(path, encoding='utf-8') as file:
			read_file = file.read()
		instance = ast.literal_eval(read_file)
		return EC(instance, **kwargs)

	def visualize(self, marked: str | None = None) -> None:
		G = nx.Graph()
		size = len(self.instance)
		ec = [set(map(str, x)) for x in self.instance]
		names = set.union(*ec)
		for i in range(len(ec)):
			G.add_node(i)
		for i in sorted(names):
			G.add_node(i)
		covered = defaultdict(int)
		for node, edges in enumerate(ec):
			for goal_node in edges:
				G.add_edge(node, goal_node)

		edge_colors = []
		for goal_node in G.edges():
			node, str_node = goal_node
			if marked is None:
				edge_colors.append('black')
				continue
			if marked[node] == '1':
				edge_colors.append('red')
				covered[str_node] += 1
			else:
				edge_colors.append('gray')
		color_map = []
		for node in G:
			if isinstance(node, int):
				color_map.append('lightblue')
			elif covered[node] == 0:
				color_map.append('yellow')
			elif covered[node] == 1:
				color_map.append('lightgreen')
			else:
				color_map.append('red')
		pos = nx.bipartite_layout(G, nodes=range(size))
		nx.draw(G, pos, node_color=color_map, with_labels=True, edge_color=edge_colors)
		plt.title('Exact Cover Problem Visualization')
		plt.show()

	@staticmethod
	def generate_ec_instance(n: int, m: int, p: float = 0.5, **kwargs) -> 'EC':
		graph = nx.bipartite.random_graph(n, m, p)
		right_nodes = [n for n, d in graph.nodes(data=True) if d['bipartite'] == 0]
		instance = [set() for _ in right_nodes]
		for left, right in graph.edges():
			instance[left].add(right)
		return EC(instance=instance, **kwargs)

	def to_hamiltonian(self, onehot: Literal['exact', 'quadratic'] = 'exact') -> Hamiltonian:
		onehots = []
		for ele in set().union(*self.instance):
			ohs = set()
			for i, subset in enumerate(self.instance):
				if ele in subset:
					ohs.add(i)
			onehots.append(ohs)

		equation = hampy.Equation(len(self.instance))

		for ohs in onehots:
			if onehot == 'exact':
				part = ~hampy.one_in_n(list(ohs), len(self.instance))
			elif onehot == 'quadratic':
				part = hampy.one_in_n(list(ohs), len(self.instance), quadratic=True)
			equation += part

		return Hamiltonian(
			equation,
			mixer_hamiltonian=self.get_mixer_hamiltonian(),
		)

	def get_mixer_hamiltonian(self, amount_of_rings: int | None = None) -> hampy.Equation:
		"""generates mixer hamiltonian"""

		# looking for all rings in a data and creating a list with them
		ring, x_gate = [], []
		main_set = []
		for element_set in self.instance:
			for elem in element_set:
				if elem not in main_set:
					main_set.append(elem)
		constraints = []
		for element in main_set:
			element_set = set()
			for index, _ in enumerate(self.instance):
				if element in self.instance[index]:
					element_set.add(index)
			if len(element_set) > 0 and element_set not in constraints:
				constraints.append(element_set)

		ring.append(max(constraints, key=len))

		ring_qubits = set.union(*ring)

		for set_ in constraints:
			if len(set_.intersection(ring_qubits)) == 0:
				ring.append(set_)
				ring_qubits.update(set_)

		if amount_of_rings is not None:
			max_amount_of_rings, user_rings = len(ring), []
			if amount_of_rings > max_amount_of_rings:
				raise ValueError(f'Too many rings. Maximum amount is {max_amount_of_rings}')
			if amount_of_rings == 0:
				ring_qubits = []
			else:
				current_qubits = ring[0]
				for index in range(amount_of_rings):
					user_rings.append(ring[index])
					current_qubits = current_qubits.union(ring[index])
				ring_qubits = current_qubits
		x_gate.extend(id for id, _ in enumerate(self.instance) if id not in ring_qubits)

		# connecting all parts of mixer hamiltonian together
		mix_ham = hampy.Equation(len(self.instance))
		for set_ in ring:
			mix_ham += ring_ham(set_, len(self.instance))

		x_gate_ham = hampy.Equation(len(self.instance))
		# creating mixer hamiltonians for all qubits that aren't in rings (in other words applying X gate to them)
		for elem in x_gate:
			x_gate_ham += SparsePauliOp.from_sparse_list([('X', [elem], 1)], len(self.instance))

		return mix_ham + x_gate_ham

	def to_qubo(self) -> QUBO:
		n = len(self.instance)
		# Calculate length tabs
		len_routes = []
		for route in self.instance:
			len_routes.append(len(route))

		qubo = np.zeros((n, n))

		Jrr_dict = {}
		indices = np.triu_indices(n, 1)
		for i1, i2 in zip(indices[0], indices[1], strict=False):
			Jrr_dict[(i1, i2)] = len(set(self.instance[i1]).intersection(set(self.instance[i2]))) / 2

		hr_dict = {}
		for i in range(n):
			i_sum = sum(len(set(r).intersection(set(self.instance[i]))) for r in self.instance)

			hr_dict[i] = (i_sum - len(self.instance[i]) * 2) / 2

		# Space
		for key, value in Jrr_dict.items():
			qubo[key[0]][key[1]] = value
			qubo[key[1]][key[0]] = qubo[key[0]][key[1]]

		for i in hr_dict:
			qubo[i][i] = -hr_dict[i]

		return QUBO(qubo, 0)


def ring_ham(set_ring: set[int], n: int) -> hampy.Equation:
	total = hampy.Equation(n)
	ring = list(set_ring)
	for index in range(len(ring) - 1):
		total += SparsePauliOp.from_sparse_list(
			[
				('XX', [ring[index], ring[index + 1]], 1),
				('YY', [ring[index], ring[index + 1]], 1),
			],
			n,
		)
	total += SparsePauliOp.from_sparse_list(
		[
			('XX', [ring[-1], ring[0]], 1),
			('YY', [ring[-1], ring[0]], 1),
		],
		n,
	)
	return total
