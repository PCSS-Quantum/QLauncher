from qiskit.quantum_info import SparsePauliOp

from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper, QubitMapper
from qiskit_nature.second_q.problems import ElectronicStructureProblem

from qlauncher.base import Problem


class Molecule(Problem):
    def __init__(self, instance: MoleculeInfo, mapper: QubitMapper | None = None, basis_set: str = 'STO-6G', instance_name: str = 'unnamed') -> None:
        self.basis_set = basis_set
        self.mapper = ParityMapper() if mapper is None else mapper
        self.problem: ElectronicStructureProblem = self._get_problem(instance)
        self.operator: SparsePauliOp = self._get_operator(self.problem)
        super().__init__(instance, instance_name)

    @staticmethod
    def from_preset(instance_name: str) -> "Molecule":
        match instance_name:
            case 'H2':
                instance = MoleculeInfo(["H", "H"], [(0.0, 0.0, 0.0), (0.0, 0.0, 0.74)])
            case _:
                raise ValueError(f"Molecule {instance_name} not supported, currently you can use: 'H2'")
        return Molecule(instance, instance_name=instance_name)

    def _get_problem(self, molecule) -> ElectronicStructureProblem:
        driver = PySCFDriver.from_molecule(molecule, basis=self.basis_set)
        problem = driver.run()
        return problem

    def _get_operator(self, problem) -> SparsePauliOp:
        self.mapper.num_particles = problem.num_particles
        hamiltonian = self.mapper.map(problem.hamiltonian.second_q_op())
        if not isinstance(hamiltonian, SparsePauliOp):
            raise ValueError
        return hamiltonian
