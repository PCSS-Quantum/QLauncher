"""This module contains the MaxCut class."""

from typing import Literal

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from qlauncher import hampy
from qlauncher.base import Problem
from qlauncher.base.problem_like import QUBO, Hamiltonian


class MaxCut(Problem):
	"""
	Class for MaxCut Problem.

	This class represents MaxCut Problem which is a combinatorial optimization problem that involves partitioning the
	vertices of a graph into two sets such that the number of edges between the two sets is maximized. The class contains
	an instance of the problem, so it can be passed into QLauncher.

	Args:
		instance (nx.Graph): The graph instance representing the problem.

	"""

	def __init__(self, instance: nx.Graph, instance_name: str = 'unnamed'):
		self.instance: nx.Graph
		super().__init__(instance, instance_name)

	@property
	def setup(self) -> dict:
		return {'instance_name': self.instance_name}

	@staticmethod
	def from_preset(instance_name: Literal['default'], **kwargs) -> 'MaxCut':
		match instance_name:
			case 'default':
				node_list = list(range(6))
				edge_list = [(0, 1), (0, 2), (0, 5), (1, 3), (1, 4), (2, 4), (2, 5), (3, 4), (3, 5)]
		graph = nx.Graph()
		graph.add_nodes_from(node_list)
		graph.add_edges_from(edge_list)
		return MaxCut(graph, instance_name=instance_name)

	def visualize(self, bitstring: str | None = None) -> None:
		pos = nx.spring_layout(self.instance, seed=42)
		plt.figure(figsize=(8, 6))
		cmap = 'skyblue' if bitstring is None else ['crimson' if bit == '1' else 'skyblue' for bit in bitstring]
		nx.draw(self.instance, pos, with_labels=True, node_color=cmap, node_size=500, edge_color='gray', font_size=10, font_weight='bold')
		plt.title('Max-Cut Problem Instance Visualization')
		plt.show()

	@staticmethod
	def generate_maxcut_instance(num_vertices: int, edge_probability: float) -> 'MaxCut':
		graph = nx.gnp_random_graph(num_vertices, edge_probability)
		return MaxCut(graph, instance_name='Generated')

	def to_hamiltonian(self) -> Hamiltonian:
		size = self.instance.number_of_nodes()
		hamiltonian = hampy.Equation(size)
		for edge in self.instance.edges():
			hamiltonian += ~hampy.one_in_n(list(edge), size)
		if hamiltonian is None:
			raise TypeError
		return Hamiltonian(hamiltonian)

	def to_qubo(self) -> QUBO:
		n = len(self.instance)
		Q = np.zeros((n, n))
		for i, j in self.instance.edges:
			Q[i, i] += -1
			Q[j, j] += -1
			Q[i, j] += 1
			Q[j, i] += 1

		return QUBO(Q, 0)
