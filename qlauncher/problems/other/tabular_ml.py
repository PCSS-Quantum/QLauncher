import numpy as np

from qlauncher.base import ProblemLike


class TabularML(ProblemLike):
	def __init__(self, X: np.ndarray, y: np.ndarray | None = None, instance_name: str = 'unnamed') -> None:
		self.X = X
		self.y = y
		self.name = instance_name

	@staticmethod
	def from_preset(instance_name: str, **kwargs) -> 'TabularML':
		match instance_name:
			case 'default':
				return TabularML(
					np.array(
						[
							[5.2, 3.1],
							[11.3, 2.2],
							[9.8, 10.5],
							[2.1, 1.9],
							[12.0, 15.2],
							[6.4, 9.9],
							[0.0, 11.1],
							[10.1, 10.2],
							[8.8, 7.7],
							[13.3, 0.4],
						]
					),
					np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 1]),
					instance_name=instance_name,
				)
		raise ValueError()

	def visualize(self) -> None:
		print(f"Tabular ML problem '{self.name}'")
		print(self.X[:10])
		print(self.y[:10] if self.y is not None else 'No target variable')
