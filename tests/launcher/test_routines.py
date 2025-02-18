from quantum_launcher.routines.qiskit_routines.v2_wrapper import SamplerV2Adapter
from qiskit_aer.primitives import Sampler, SamplerV2, Estimator, EstimatorV2
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeAlmadenV2
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


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
