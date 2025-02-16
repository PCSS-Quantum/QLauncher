"""  This module contains the MaxCut class."""
from typing import Literal, Optional, overload
import networkx as nx

from quantum_launcher.base import Problem


class MaxCut(Problem):
    """ 
    Class for MaxCut Problem.

    This class represents MaxCut Problem which is a combinatorial optimization problem that involves partitioning the
    vertices of a graph into two sets such that the number of edges between the two sets is maximized. The class contains
    an instance of the problem, so it can be passed into Quantum Launcher.

    Attributes:
        instance (nx.Graph): The graph instance representing the problem.

    Methods:
        visualize()
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
                edge_list = [(0, 1), (0, 2), (0, 5), (1, 3), (1, 4),
                             (2, 4), (2, 5), (3, 4), (3, 5)]
        return MaxCut(nx.Graph(edge_list), instance_name=instance_name)

    def _get_path(self) -> str:
        return f'{self.name}@{self.instance_name}'

    def visualize(self, bitstring: Optional[str] = None):
        import matplotlib.pyplot as plt
        pos = nx.spring_layout(self.instance)
        plt.figure(figsize=(8, 6))

        nx.draw(self.instance, pos, with_labels=True, node_color='skyblue',
                node_size=500, edge_color='gray', font_size=10, font_weight='bold')
        plt.title("Max-Cut Problem Instance Visualization")
        plt.show()

    @staticmethod
    def generate_maxcut_instance(num_vertices, edge_probability):
        G = nx.gnp_random_graph(num_vertices, edge_probability)
        return G
