from qiskit_aer.primitives import Sampler, SamplerV2
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeAlmadenV2
from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit.primitives import SamplerResult

from qlauncher.routines.qiskit.adapters import SamplerV2ToSamplerV1Adapter, SamplerV1ToSamplerV2Adapter
from qlauncher.routines.qiskit import QiskitBackend


def test_v2_sampler_adapter():
    circ = QuantumCircuit(4)
    circ.h(list(range(4)))
    circ.measure_all()

    backend = AerSimulator.from_backend(FakeAlmadenV2())

    sampler_v1 = Sampler()
    sampler_v2_adapted = SamplerV2ToSamplerV1Adapter(SamplerV2(), backend)

    v1_result = sampler_v1.run(circ).result()
    adapted_result = sampler_v2_adapted.run(circ).result()

    assert len(v1_result.quasi_dists) == len(adapted_result.quasi_dists)

    for d1, d2 in zip(v1_result.quasi_dists, adapted_result.quasi_dists):
        assert len(d1) == len(d2)


def test_v1_sampler_adapter():
    circ = QuantumCircuit(4)
    circ.h(list(range(4)))
    circ.measure_all()

    backend = AerSimulator.from_backend(FakeAlmadenV2())
    circ = transpile(circ, backend, optimization_level=3)

    sampler_v2 = SamplerV2()
    sampler_v1_adapted = SamplerV1ToSamplerV2Adapter(Sampler())

    v2_result = sampler_v2.run([circ]).result()
    adapted_result = sampler_v1_adapted.run([circ]).result()

    assert len(v2_result) == len(adapted_result)

    for d1, d2 in zip(v2_result, adapted_result):
        assert len(d1.data.meas.get_int_counts()) == len(d2.data.meas.get_int_counts())


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
    sampler_v2_adapted = SamplerV2ToSamplerV1Adapter(SamplerV2(), backend)

    v1_result = sampler_v1.run(circ).result()
    adapted_result = sampler_v2_adapted.run(circ).result()

    assert len(v1_result.quasi_dists) == len(adapted_result.quasi_dists)

    for d1, d2 in zip(v1_result.quasi_dists, adapted_result.quasi_dists):
        assert len(d1) == len(d2)
