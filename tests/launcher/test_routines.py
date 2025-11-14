import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, transpile
from qiskit.circuit import ClassicalRegister, ParameterVector
from qiskit.primitives import SamplerResult
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Estimator, EstimatorV2, Sampler, SamplerV2
from qiskit_ibm_runtime.fake_provider import FakeAlmadenV2

from qlauncher.hampy import Equation
from qlauncher.routines.qiskit import QiskitBackend
from qlauncher.routines.qiskit.adapters import EstimatorV1ToEstimatorV2Adapter, SamplerV1ToSamplerV2Adapter, SamplerV2ToSamplerV1Adapter


def test_v2_estimator_adapter():
	circ = QuantumCircuit(2)
	params = ParameterVector('pv', 2)

	circ.rx(params[0], 0)
	circ.rx(params[1], 1)
	circ.cx(0, 1)

	obs = Equation(2)
	obs = obs[0] & obs[1]

	obs2 = Equation(2)
	obs2 = ~obs2[0] & obs2[1]

	estimator_v1 = Estimator()
	estimator_v1_adapted = EstimatorV1ToEstimatorV2Adapter(estimator_v1)
	estimator_v2 = EstimatorV2()

	for inpt in [
		[(circ, [obs.hamiltonian, obs2.hamiltonian], [1.0, 2.0])],
		[(circ, obs2.hamiltonian, [1.0, 2.0])],
		[(circ, obs.hamiltonian, [1.0, 2.0]), (circ, obs2.hamiltonian, [1.0, 2.0])],
		[(circ, obs.hamiltonian, [[1.0, 2.0], [2.0, 3.0]]), (circ, obs2.hamiltonian, [[1.0, 2.0], [2.0, 3.0]])],
		[(circ, [obs.hamiltonian, obs2.hamiltonian], [[1.0, 2.0], [2.0, 3.0]])],
	]:
		v2_result = estimator_v2.run(inpt).result()
		v1_adapted_result = estimator_v1_adapted.run(inpt).result()

		assert len(v2_result) == len(v1_adapted_result)

		for r1, r2 in zip(v1_adapted_result, v2_result):
			assert np.allclose(r1.data.evs, r2.data.evs, atol=0.05)
			assert np.allclose(r1.data.stds, r2.data.stds, atol=0.05)


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
