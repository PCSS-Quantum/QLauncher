""" This module contains raw problem classes"""
from quantum_launcher.base import Problem


class Raw(Problem):
    """
    Class for solving problem implemented in raw mathematical form.

    "Raw mathematical form" means that the problem is defined in format 
    that can be directly read by the quantum algorithm, such as Qubo, Hamiltonian, etc.

    The object contains an instance of the problem written in mentioned raw mathematical form,
    can be passed into Quantum Launcher.

    Attributes:
        instance (any): Formulated problem instance.
    """

    def __init__(self, instance: any = None, instance_name: str | None = None) -> None:
        super().__init__(instance=instance, instance_name=instance_name)

    def _get_path(self) -> str:
        return f'{self.name}/{self.instance_name}'


class Hamiltonian(Raw, Problem):
    """Raw problem in hamiltonian form"""

    def __init__(self, hamiltonian, instance_name: str | None = None) -> None:
        super().__init__(hamiltonian, instance_name)


class Qubo(Raw, Problem):
    """Raw problem in qubo form"""

    def __init__(self, instance, instance_name: str | None = None) -> None:
        super().__init__(instance, instance_name)


class BQM(Raw, Problem):
    """Raw problem in bqm form"""

    def __init__(self, instance, instance_name: str | None = None) -> None:
        super().__init__(instance, instance_name)
