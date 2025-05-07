from qiskit_aer.primitives import Sampler, SamplerV2
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeAlmadenV2
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.primitives import SamplerResult

from quantum_launcher.routines.qiskit_routines.v2_wrapper import SamplerV2Adapter
from quantum_launcher.routines.qiskit_routines import QiskitBackend


def test_v2_sampler_adapter():
    circ = QuantumCircuit(4)
    circ.h(list(range(4)))
    circ.measure_all()

    backend = AerSimulator.from_backend(FakeAlmadenV2())

    sampler_v1 = Sampler()
    sampler_v2_adapted = SamplerV2Adapter(SamplerV2(), backend)

    v1_result = sampler_v1.run(circ).result()
    adapted_result = sampler_v2_adapted.run(circ).result()

    assert len(v1_result.quasi_dists) == len(adapted_result.quasi_dists)

    for d1, d2 in zip(v1_result.quasi_dists, adapted_result.quasi_dists):
        assert len(d1) == len(d2)


def test_v2_sampler_adapter_unnamed_measurements():
    """
    Test whether SamplerV2Adapter correctly outputs results for circuits with unnamed cl_registers.
    """

    backend = QiskitBackend('local_simulator')
    circ = QuantumCircuit(2, 1)

    circ.h(0)
    circ.cx(0, 1)
    circ.measure(1, 0)

    res = backend.samplerV1.run(circ).result()

    assert isinstance(res, SamplerResult)


def test_v2_sampler_adapter_multiname():
    circ = QuantumCircuit(4)
    circ.add_register(ClassicalRegister(2, 'name1'))
    circ.add_register(ClassicalRegister(2, 'name2'))
    circ.h(list(range(4)))
    circ.measure([0, 2], [0, 2])
    circ.measure([1, 3], [1, 3])

    backend = AerSimulator.from_backend(FakeAlmadenV2())

    sampler_v1 = Sampler()
    sampler_v2_adapted = SamplerV2Adapter(SamplerV2(), backend)

    v1_result = sampler_v1.run(circ).result()
    adapted_result = sampler_v2_adapted.run(circ).result()

    assert len(v1_result.quasi_dists) == len(adapted_result.quasi_dists)

    for d1, d2 in zip(v1_result.quasi_dists, adapted_result.quasi_dists):
        assert len(d1) == len(d2)
