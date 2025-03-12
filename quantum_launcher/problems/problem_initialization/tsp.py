import networkx as nx

import numpy as np
import matplotlib.pyplot as plt

from qiskit_algorithms import SamplingMinimumEigensolverResult

from quantum_launcher.base import Problem


class TSP(Problem):
    """
    Traveling Salesman Problem (TSP) definition.
    """

    def __init__(self, instance: nx.Graph, instance_name: str = "unnamed"):
        """
        Args:
            instance (nx.Graph): Graph representing the TSP instance.
            instance_name (str): Name of the instance.
            quadratic (bool): Whether to use quadratic constraints
        """
        super().__init__(instance=instance, instance_name=instance_name)

    @property
    def setup(self) -> dict:
        return {"instance_name": self.instance_name}

    def _get_path(self) -> str:
        return f"{self.name}@{self.instance_name}"

    def _solution_to_node_chain(self, solution: SamplingMinimumEigensolverResult | str) -> np.ndarray:
        """
        Converts the solution of the TSP problem to a chain of nodes (order of visiting)

        Args:
            solution: Solution of the TSP problem. If type is str, qiskit-style ordering is assumed (MSB first)

        Returns:
            np.ndarray: Solution chain of nodes to visit
        """
        if isinstance(solution, str):
            bitstring = solution[::-1]
        else:
            bitstring = solution.best_measurement["bitstring"][::-1]

        node_count = int(len(bitstring)**0.5)
        chain = []

        for i in range(0, len(bitstring), node_count):
            step_string = bitstring[i: i + node_count]
            chosen_node = np.argmax([int(x) for x in step_string])
            chain.append(chosen_node)

        return np.array(chain)

    def _calculate_solution_cost(self, solution: SamplingMinimumEigensolverResult | str | list[int]) -> float:
        cost = 0
        if isinstance(solution, list):
            chain = solution
        else:
            chain = self._solution_to_node_chain(solution)

        for i in range(len(chain) - 1):
            cost += self.instance[chain[i]][chain[i + 1]]["weight"]
        cost += self.instance[chain[-1]][chain[0]]["weight"]
        return cost

    def _visualize_solution(self, solution: SamplingMinimumEigensolverResult | str | list[int]):
        if isinstance(solution, list):
            chain = solution
        else:
            chain = self._solution_to_node_chain(solution)

        import matplotlib.pyplot as plt

        pos = nx.spring_layout(self.instance, weight=None, seed=42)
        problem_edges = list(self.instance.edges)

        marked_edges = []
        for i in range(len(chain) - 1):
            marked_edges.append(set((chain[i], chain[i + 1])))
        marked_edges.append(set((chain[-1], chain[0])))

        draw_colors = []  # Limegreen for marked edges, gray for unmarked
        draw_widths = []  # Draw marked edges thicker

        path_cost = 0
        for edge in problem_edges:
            draw_colors.append("limegreen" if set(edge) in marked_edges else "gray")
            draw_widths.append(2 if set(edge) in marked_edges else 1)

            path_cost += self.instance[edge[0]][edge[1]]["weight"] if set(edge) in marked_edges else 0

        plt.figure(figsize=(8, 6))
        nx.draw(
            self.instance,
            pos,
            edge_color=draw_colors,
            width=draw_widths,
            with_labels=True,
            node_color="skyblue",
            node_size=500,
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
            label_pos=0.45,
        )

        plt.title("TSP Solution Visualization")
        plt.suptitle(f"Path cost:{path_cost}")
        plt.show()

    def _visualize_problem(self):
        """
        Show plot of the TSP instance.
        """

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

    def visualize(self, solution=None):
        if solution is None:
            self._visualize_problem()
        else:
            self._visualize_solution(solution)

    @staticmethod
    def from_preset(instance_name: str = 'default', **kwargs) -> "TSP":
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

        return TSP(instance=G, instance_name=instance_name)

    @staticmethod
    def generate_tsp_instance(num_vertices: int, min_distance: float = 1.0, max_distance: float = 10.0, **kwargs) -> "TSP":
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
            raise ValueError("num_vertices must be at least 2")

        if min_distance <= 0:
            raise ValueError("min_distance must be greater than 0")

        g = nx.Graph()
        for i in range(num_vertices):
            for j in range(i + 1, num_vertices):
                g.add_edge(i, j, weight=int(
                    np.random.uniform(min_distance, max_distance)))

        return TSP(instance=g, instance_name="generated")
