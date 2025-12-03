import ast
from collections import defaultdict

import networkx as nx
import matplotlib.pyplot as plt

from qlauncher.base import ProblemLike

class DTQW_1D(ProblemLike):

    def __init__(self, instance: list[set[int]] = None, instance_name: str = 'unnamed') -> None:
        super().__init__(instance=instance, instance_name=instance_name)

    @property
    def setup(self) -> dict:
        return {
            'instance_name': self.instance_name
        }

    def _get_path(self) -> str:
        return f'{self.name}@{self.instance_name}'