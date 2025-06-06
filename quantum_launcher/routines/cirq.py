"""
Routine file for Cirq library
"""
from typing import Literal
from collections.abc import Sequence

import qiskit
from qiskit.primitives import Sampler
from qiskit.primitives.base.sampler_result import SamplerResult
from qiskit.result import QuasiDistribution
from qiskit_algorithms.optimizers import COBYLA

from quantum_launcher.base import Backend
from quantum_launcher.base.translator import Translation
from quantum_launcher.exceptions import DependencyError
try:
    import cirq
    from cirq.sim.sparse_simulator import Simulator
except ImportError as e:
    raise DependencyError(e, install_hint='cirq') from e


class _CirqRunner:
    simulator = Simulator()
    repetitions = 1024

    @classmethod
    def calculate_circuit(cls, circuit: qiskit.QuantumCircuit) -> dict:
        circuit = circuit.measure_all(inplace=False)
        cirq_circ = Translation.get_translation(circuit, 'cirq')
        result = cls.simulator.run(cirq_circ, repetitions=cls.repetitions)
        return cls._result_to_dist(result)

    @classmethod
    def _result_to_dist(cls, result) -> dict:
        return cirq_result_to_probabilities(result)


class CirqSampler(Sampler):
    """ Sampler adapter for Cirq """

    def _call(self, circuits: Sequence[int], parameter_values: Sequence[Sequence[float]], **run_options) -> SamplerResult:
        bound_circuits = []
        for i, value in zip(circuits, parameter_values):
            bound_circuits.append(
                self._circuits[i]
                if len(value) == 0
                else self._circuits[i].assign_parameters(dict(zip(self._parameters[i], value)))
            )
        distributions = [_CirqRunner.calculate_circuit(circuit) for circuit in bound_circuits]
        quasi_dists = list(map(QuasiDistribution, distributions))
        return SamplerResult(quasi_dists, [{} for _ in range(len(parameter_values))])


class CirqBackend(Backend):
    """

    Args:
        Backend (_type_): _description_
    """

    def __init__(self, name: Literal['local_simulator'] = 'local_simulator'):
        self.sampler = self.samplerV1 = CirqSampler()
        self.optimizer = COBYLA()
        super().__init__(name)


def cirq_result_to_probabilities(
    result: cirq.Result,
    integer_keys: bool = False
) -> dict:
    measurements = result.measurements

    sorted_keys = list(measurements.keys())

    bitstrings = []
    num_shots = len(measurements[sorted_keys[0]])

    for shot_index in range(num_shots):
        bits = []
        for key in sorted_keys:
            bits.extend(str(b) for b in measurements[key][shot_index])

        bitstring = "".join(bits)
        bitstrings.append(bitstring)

    counts = {}
    for bs in bitstrings:
        counts[bs] = counts.get(bs, 0) + 1

    total_shots = sum(counts.values())
    if integer_keys:
        prob_dict = {int(k, 2): v / total_shots for k, v in counts.items()}
    else:
        prob_dict = {k: v / total_shots for k, v in counts.items()}

    return prob_dict
