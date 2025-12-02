"""This module contains the Graph Coloring class."""

import pickle
from itertools import product
from random import randint
from typing import Literal

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pyqubo import Array, Binary

from qlauncher.base import Problem
from qlauncher.base.problem_like import QUBO, Hamiltonian
from qlauncher.hampy import Equation


class GraphColoring(Problem):
	"""
	Class for Graph Coloring Problem which is a combinatorial problem involving assigning labels to vertices of the graph such that no two adjacent vertices share the same label.

	Attributes:
		instance (nx.Graph): The graph for which the coloring problem is to be solved.


	"""

	def __init__(
		self,
		instance: nx.Graph,
		num_colors: int,
		instance_name: str = 'unnamed',
	) -> None:
		super().__init__(instance=instance, instance_name=instance_name)
		self.pos = None
		self.num_colors = num_colors

	@property
	def setup(self) -> dict:
		return {'instance_name': self.instance_name}

	@staticmethod
	def from_preset(instance_name: Literal['default', 'small'], **kwargs) -> 'GraphColoring':
		match instance_name:
			case 'default':
				graph = nx.petersen_graph()
				gc = GraphColoring(graph, instance_name=instance_name, num_colors=3)
				gc.pos = nx.shell_layout(gc.instance, nlist=[list(range(5, 10)), list(range(5))])
				return gc
			case 'small':
				graph = nx.Graph()
				graph.add_edges_from([(0, 1), (1, 2), (0, 2), (3, 2)])
				return GraphColoring(graph, instance_name=instance_name, num_colors=3)
			case _:
				raise ValueError(f'Preset f{instance_name} not defined')

	@classmethod
	def from_file(cls: type['GraphColoring'], path: str) -> 'Problem':
		with open(path, 'rb') as f:
			graph, num_colors = pickle.load(f)
		return cls(graph, instance_name=path, num_colors=num_colors)

	# def to_file(self, path: str) -> None:
	# 	with open(path, 'wb') as f:
	# 		pickle.dump((self.instance, self.num_colors), f, pickle.HIGHEST_PROTOCOL)

	def _get_path(self) -> str:
		return f'{self.name}@{self.instance_name}'

	def visualize(self, solution: list[int] | None = None) -> None:
		if self.pos is None:
			self.pos = nx.spring_layout(self.instance, seed=42)  # set seed for same node graphs in plt
		plt.figure(figsize=(8, 6))
		if solution is not None:
			nx.draw_networkx_nodes(self.instance, self.pos, node_size=500, node_color=solution, cmap='Accent')
			colors = []
			for n1, n2 in self.instance.edges:
				if solution[n1] == solution[n2]:
					colors.append('r')
				else:
					colors.append('gray')
			nx.draw_networkx_edges(self.instance, self.pos, edge_color=colors, width=4)
		else:
			nx.draw_networkx_nodes(self.instance, self.pos, node_size=500, node_color='skyblue')
		nx.draw_networkx_edges(self.instance, self.pos, edge_color='gray')
		nx.draw_networkx_labels(self.instance, self.pos, font_size=10, font_weight='bold')
		plt.title(f'Graph {self.num_colors}-Coloring Problem Instance Visualization')
		plt.show()

	@staticmethod
	def generate_graph_coloring_instance(num_vertices: int, edge_probability: int, num_colors: int) -> 'GraphColoring':
		graph = nx.gnp_random_graph(num_vertices, edge_probability)
		return GraphColoring(graph, num_colors=num_colors)

	@staticmethod
	def randomly_choose_a_graph(num_colors: int) -> 'GraphColoring':
		graphs = nx.graph_atlas_g()
		graph = graphs[randint(0, len(graphs) - 1)]
		return GraphColoring(graph, num_colors=num_colors)

	# Penalty for assigning the same colors to neighboring vertices
	def _color_duplication_hamiltonian(self, num_qubits: int, color_bit_length: int) -> Equation:
		eq = Equation(num_qubits)
		for node1, node2 in self.instance.edges:
			for ind, comb in enumerate(product(range(2), repeat=color_bit_length)):
				if ind >= self.num_colors:
					break
				eq_inner = None
				for i in range(color_bit_length):
					qubit1 = eq[node1 * color_bit_length + i]
					qubit2 = eq[node2 * color_bit_length + i]
					exp = qubit1 & qubit2 if comb[i] else ~qubit1 & ~qubit2
					if eq_inner is None:
						eq_inner = exp
					else:
						eq_inner &= exp
				if eq_inner is not None:
					eq += eq_inner
		return eq

	# Penalty for using excessive colors
	def _excessive_colors_use_hamiltonian(self, num_qubits: int, color_bit_length: int) -> Equation:
		eq = Equation(num_qubits)
		for node in self.instance.nodes:
			for ind, comb in enumerate(product(range(2), repeat=color_bit_length)):
				if ind < self.num_colors:
					continue
				eq_inner = None
				for i in range(color_bit_length):
					qubit = eq[node * color_bit_length + i]
					exp = qubit if comb[i] else ~qubit
					if eq_inner is None:
						eq_inner = exp
					else:
						eq_inner &= exp
				if eq_inner is not None:
					eq += eq_inner
		return eq

	def to_hamiltonian(self, constraints_weight: float = 1, costs_weight: float = 1) -> Hamiltonian:
		color_bit_length = int(np.ceil(np.log2(self.num_colors)))
		num_qubits = self.instance.number_of_nodes() * color_bit_length

		eq = self._color_duplication_hamiltonian(num_qubits, color_bit_length)
		eq2 = self._excessive_colors_use_hamiltonian(num_qubits, color_bit_length)

		return Hamiltonian((eq * costs_weight + eq2 * constraints_weight).hamiltonian.simplify())

	def to_qubo(self) -> QUBO:
		"""Returns Qubo function"""
		num_qubits = self.instance.number_of_nodes() * self.num_colors
		x = Array.create('x', shape=(self.instance.number_of_nodes(), self.num_colors), vartype='BINARY')
		qubo: Binary = 0
		for node in self.instance.nodes:
			expression: Binary = 1 - sum(x[node, i] for i in range(self.num_colors))
			qubo += expression * expression
		for n1, n2 in self.instance.edges:
			for c in range(self.num_colors):
				qubo += x[n1, c] * x[n2, c]
		model = qubo.compile()
		qubo_dict, offset = model.to_qubo()
		Q_matrix = np.zeros((num_qubits, num_qubits))
		for i in range(num_qubits):
			for j in range(num_qubits):
				n1, c1 = i // self.num_colors, i % self.num_colors
				n2, c2 = j // self.num_colors, j % self.num_colors
				key = ('x[' + str(n1) + '][' + str(c1) + ']', 'x[' + str(n2) + '][' + str(c2) + ']')
				if key in qubo_dict:
					Q_matrix[i, j] = qubo_dict[key]
		return QUBO(Q_matrix, offset)
