"""  This module contains the MaxCut class."""
from typing import Optional
import networkx as nx
import matplotlib.pyplot as plt

from quantum_launcher.base import Problem


class MaxCut(Problem):
    """
    Class for MaxCut Problem.

    This class represents MaxCut Problem which is a combinatorial optimization problem that involves partitioning the
    vertices of a graph into two sets such that the number of edges between the two sets is maximized. The class contains
    an instance of the problem, so it can be passed into Quantum Launcher.

    Attributes:
        instance (nx.Graph): The graph instance representing the problem.

    """

    def __init__(self, instance: nx.Graph, instance_name='unnamed'):
        super().__init__(instance, instance_name)

    @property
    def setup(self) -> dict:
        return {
            'instance_name': self.instance_name
        }

    @staticmethod
    def from_preset(instance_name: str) -> "MaxCut":
        match instance_name:
            case 'default':
                node_list = list(range(6))
                edge_list = [(0, 1), (0, 2), (0, 5), (1, 3), (1, 4),
                             (2, 4), (2, 5), (3, 4), (3, 5)]
        graph = nx.Graph()
        graph.add_nodes_from(node_list)
        graph.add_edges_from(edge_list)
        return MaxCut(graph, instance_name=instance_name)

    def _get_path(self) -> str:
        return f'{self.name}@{self.instance_name}'

    def visualize(self, bitstring: Optional[str] = None):
        pos = nx.spring_layout(self.instance)
        plt.figure(figsize=(8, 6))
        if bitstring is None:
            cmap = 'skyblue'
        else:
            cmap = ['crimson' if bit == '1' else 'skyblue' for bit in bitstring]
        nx.draw(self.instance, pos, with_labels=True, node_color=cmap,
                node_size=500, edge_color='gray', font_size=10, font_weight='bold')
        plt.title("Max-Cut Problem Instance Visualization")
        plt.show()

    @staticmethod
    def generate_maxcut_instance(num_vertices: int, edge_probability: float) -> "MaxCut":
        graph = nx.gnp_random_graph(num_vertices, edge_probability)
        return MaxCut(graph, instance_name='Generated')
