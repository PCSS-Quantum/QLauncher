from typing import Literal

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pyqubo import Array, Binary

from qlauncher.base import Problem
from qlauncher.base.problem_like import QUBO


class VertexCover(Problem):
	"""
	Class for the Vertex Cover problem which is a combinatorial problem involving choosing a subset of graph vertices such that each edge of the graph has at least one vertex in the chosen subset.

	Attributes:
		instance (nx.Graph): The graph for which the coloring problem is to be solved.


	"""

	def __init__(self, instance: nx.Graph, instance_name: str = 'unnamed') -> None:
		"""
		Args:
			instance (nx.Graph): Graph representing the TSP instance.
			instance_name (str): Name of the instance.
		"""
		super().__init__(instance=instance, instance_name=instance_name)

	# method to visualize a problem and optionally a solution to it
	def visualize(self, solution: list[int] | None = None) -> None:
		pos = nx.spring_layout(self.instance)
		plt.figure(figsize=(8, 6))
		if solution is not None:
			solution_colors = ['red' if x else 'skyblue' for x in solution]
			nx.draw_networkx_nodes(self.instance, pos, node_size=500, node_color=solution_colors)
		else:
			nx.draw_networkx_nodes(self.instance, pos, node_size=500, node_color='skyblue')
		nx.draw_networkx_edges(self.instance, pos, edge_color='gray')
		nx.draw_networkx_labels(self.instance, pos, font_size=10, font_weight='bold')
		plt.title('Vertex Cover Problem Instance Visualization')
		plt.show()

	# method to load a predefined toy example
	@staticmethod
	def from_preset(instance_name: Literal['default'], **kwargs) -> 'VertexCover':
		match instance_name:
			case 'default':
				node_list = list(range(5))
				edge_list = [(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (3, 4)]
		graph = nx.Graph()
		graph.add_nodes_from(node_list)
		graph.add_edges_from(edge_list)
		return VertexCover(graph, instance_name=instance_name)

	# method which generates ransom problem instance
	@staticmethod
	def generate_vertex_cover_instance(num_vertices: int, edge_probability: int) -> 'VertexCover':
		graph = nx.gnp_random_graph(num_vertices, edge_probability)
		return VertexCover(graph)

	def to_qubo(self, constraint_weight: int = 5, cost_weight: int = 1) -> QUBO:
		vertices = self.instance.nodes()
		edges = self.instance.edges()
		x = Array.create('x', shape=(len(vertices),), vartype='BINARY')

		# penalty for number of vertices used
		qubo: Binary = sum(cost_weight * x[v] for v in vertices)

		# penalty for violating constraint, not covering all edges
		for e in edges:
			qubo += constraint_weight * (1 - x[e[0]] - x[e[1]] + x[e[0]] * x[e[1]])

		qubo_dict, offset = qubo.compile().to_qubo()

		# turn qubo dict into qubo matrix
		Q_matrix = np.zeros((len(vertices), len(vertices)))
		var_labels = [f'x[{k}]' for k in range(len(vertices))]
		for i, vi in enumerate(var_labels):
			for j, vj in enumerate(var_labels):
				key = (vi, vj)
				if key in qubo_dict:
					Q_matrix[i, j] = qubo_dict[key]
		return QUBO(Q_matrix, offset)
