import networkx as nx

import numpy as np

from quantum_launcher.base import Problem


class TSP(Problem):
    def __init__(self, instance: nx.Graph, instance_name: str = "unnamed", quadratic: bool = False):
        super().__init__(instance=instance, instance_name=instance_name)
        self.quadratic = quadratic

    @property
    def setup(self) -> dict:
        return {"instance_name": self.instance_name}

    def _get_path(self) -> str:
        return f"{self.name}@{self.instance_name}"

    def visualize(self):
        import matplotlib.pyplot as plt

        pos = nx.spring_layout(self.instance, weight=None, seed=42)
        plt.figure(figsize=(8, 6))

        nx.draw(
            self.instance,
            pos,
            with_labels=True,
            node_color="skyblue",
            node_size=500,
            edge_color="gray",
            font_size=10,
            font_weight="bold",
        )

        labels = nx.get_edge_attributes(self.instance, "weight")
        nx.draw_networkx_edge_labels(
            self.instance,
            pos,
            edge_labels=labels,
            rotate=False,
            font_weight="bold",
            node_size=500,
            label_pos=0.45,
        )

        plt.title("TSP Instance Visualization")
        plt.show()

    @staticmethod
    def from_preset(instance_name: str = 'default', **kwargs) -> "TSP":
        match instance_name:
            case 'default':
                edge_costs = np.array(
                    [
                        [0, 1, 2, 3],
                        [1, 0, 4, 5],
                        [2, 4, 0, 6],
                        [3, 5, 6, 0]
                    ]
                )
            case _:
                raise ValueError("Unknown instance name")
        G = nx.Graph()
        n = edge_costs.shape[0]
        for i in range(n):
            for j in range(i + 1, n):  # No connections to self
                G.add_edge(i, j, weight=edge_costs[i, j])

        quadratic = kwargs.get("quadratic", False)

        return TSP(instance=G, instance_name=instance_name, quadratic=quadratic)

    @staticmethod
    def generate_tsp_instance(num_vertices: int, min_distance: float = 1.0, max_distance: float = 10.0, **kwargs) -> "TSP":
        if num_vertices < 2:
            raise ValueError("num_vertices must be at least 2")

        if min_distance <= 0:
            raise ValueError("min_distance must be greater than 0")

        g = nx.Graph()
        for i in range(num_vertices):
            for j in range(i + 1, num_vertices):
                g.add_edge(i, j, weight=int(
                    np.random.uniform(min_distance, max_distance)))

        quadratic = kwargs.get("quadratic", False)

        return TSP(instance=g, instance_name="generated", quadratic=quadratic)
