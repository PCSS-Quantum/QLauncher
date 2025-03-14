"""This module contains the Graph Coloring class."""

import pickle
import networkx as nx
from random import randint
import matplotlib.pyplot as plt
from quantum_launcher.base import Problem


class GraphColoring(Problem):
    """
    Class for Graph Coloring Problem which is a combinatorial problem involving assigning labels to vertices of the graph such that no two adjacent
    vertices share the same label.

    Attributes:
        instance (nx.Graph): The graph for which the coloring problem is to be solved.


    """

    def __init__(
        self,
        instance: nx.Graph,
        num_colors: int,
        instance_name: str = "unnamed",
    ) -> None:
        super().__init__(instance=instance, instance_name=instance_name)
        self.pos = None
        self.num_colors = num_colors

    @property
    def setup(self) -> dict:
        return {"instance_name": self.instance_name}

    @staticmethod
    def from_preset(instance_name: str) -> "GraphColoring":
        match instance_name:
            case "default":
                graph = nx.petersen_graph()
                gc = GraphColoring(graph, instance_name=instance_name, num_colors=3)
                gc.pos = nx.shell_layout(gc.instance, nlist=[list(range(5, 10)), list(range(5))])
                return gc
            case "small":
                graph = nx.Graph()
                graph.add_edges_from([(0, 1), (1, 2), (0, 2), (3, 2)])
                return GraphColoring(graph, instance_name=instance_name, num_colors=3)

    @staticmethod
    def from_file(path: str) -> "GraphColoring":
        with open(path, 'rb') as f:
            graph, num_colors = pickle.load(f)
        return GraphColoring(graph, instance_name=path, num_colors=num_colors)

    def to_file(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump((self.instance, self.num_colors), f, pickle.HIGHEST_PROTOCOL)

    def _get_path(self) -> str:
        return f"{self.name}@{self.instance_name}"

    def visualize(self, solution: list[int] | None = None):
        if self.pos is None:
            self.pos = nx.spring_layout(self.instance)
        plt.figure(figsize=(8, 6))
        if solution is not None:
            nx.draw_networkx_nodes(self.instance, self.pos, node_size=500, node_color=solution, cmap="Accent")
        else:
            nx.draw_networkx_nodes(self.instance, self.pos, node_size=500, node_color="skyblue")
        nx.draw_networkx_edges(self.instance, self.pos, edge_color="gray")
        nx.draw_networkx_labels(self.instance, self.pos, font_size=10, font_weight="bold")
        plt.title("Graph Coloring Problem Instance Visualization")
        plt.show()

    @staticmethod
    def generate_graph_coloring_instance(num_vertices: int, edge_probability: int, num_colors: int) -> "GraphColoring":
        graph = nx.gnp_random_graph(num_vertices, edge_probability)
        return GraphColoring(graph, num_colors=num_colors)

    @staticmethod
    def randomly_choose_a_graph(num_colors: int) -> "GraphColoring":
        graphs = nx.graph_atlas_g()
        graph = graphs[randint(0, len(graphs) - 1)]
        return GraphColoring(graph, num_colors=num_colors)
