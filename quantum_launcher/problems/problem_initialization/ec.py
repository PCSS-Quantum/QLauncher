"""  This module contains the EC class."""
import ast
from collections import defaultdict
from typing import List, Optional, Set
import networkx as nx
import matplotlib.pyplot as plt

from quantum_launcher.base import Problem


class EC(Problem):
    """
    Class for exact cover problem.

    The exact cover problem is a combinatorial optimization problem that involves finding a subset of a given set
    of elements such that the subset covers all elements and the number of elements in the subset is minimized.
    The class contains an instance of the problem, so it can be passed into Quantum Launcher.

    Attributes:
        onehot (str): The one-hot encoding used for the problem.
        instance (any): The instance of the problem.
        instance_name (str | None): The name of the instance.
        instance_path (str): The path to the instance file.

    """

    def __init__(self, instance: List[Set[int]] = None, instance_name: str = 'unnamed') -> None:
        super().__init__(instance=instance, instance_name=instance_name)

    @property
    def setup(self) -> dict:
        return {
            'instance_name': self.instance_name
        }

    def _get_path(self) -> str:
        return f'{self.name}@{self.instance_name}'

    @staticmethod
    def from_preset(instance_name: str, **kwargs) -> "EC":
        match instance_name:
            case 'micro':
                instance = [{1, 2},
                            {1}]
            case 'toy':
                instance = [{1, 4, 7},
                            {1, 4},
                            {4, 5, 7},
                            {3, 5, 6},
                            {2, 3, 6, 7},
                            {2, 7}]
        return EC(instance=instance, instance_name=instance_name, **kwargs)

    @classmethod
    def from_file(cls, path: str, **kwargs) -> "EC":
        with open(path, 'r', encoding='utf-8') as file:
            read_file = file.read()
        instance = ast.literal_eval(read_file)
        return EC(instance, **kwargs)

    def read_instance(self, instance_path: str):
        with open(instance_path, 'r', encoding='utf-8') as file:
            read_file = file.read()
        self.instance = ast.literal_eval(read_file)

    def visualize(self, marked: Optional[str] = None):
        G = nx.Graph()
        size = len(self.instance)
        ec = list(map(lambda x: set(map(str, x)), self.instance))
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
        nx.draw(G, pos, node_color=color_map, with_labels=True,
                edge_color=edge_colors)
        plt.title('Exact Cover Problem Visualization')
        plt.show()

    @staticmethod
    def generate_ec_instance(n: int, m: int, p: float = 0.5, **kwargs) -> "EC":
        graph = nx.bipartite.random_graph(n, m, p)
        right_nodes = [n for n, d in graph.nodes(data=True) if d["bipartite"] == 0]
        instance = [set() for _ in right_nodes]
        for left, right in graph.edges():
            instance[left].add(right)
        return EC(instance=instance, **kwargs)
