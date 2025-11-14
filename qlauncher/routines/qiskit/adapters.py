from collections.abc import Sequence
from typing import Any, Iterable

import math

import numpy as np

from qiskit import transpile
from qiskit.primitives.containers import PubResult
from qiskit.primitives.containers.primitive_result import PrimitiveResult
from qiskit.primitives.containers.sampler_pub_result import SamplerPubResult
from qiskit.result import QuasiDistribution
from qiskit.primitives import SamplerResult, BasePrimitiveJob, BitArray, DataBin
from qiskit.primitives.base import BaseSamplerV1, BaseSamplerV2, BaseEstimatorV1, BaseEstimatorV2, EstimatorResult
from qiskit.primitives.containers.sampler_pub import SamplerPubLike, SamplerPub
from qiskit.primitives.containers.estimator_pub import EstimatorPubLike, EstimatorPub
from qiskit.primitives.primitive_job import PrimitiveJob
from qiskit.quantum_info import SparsePauliOp


def _transpile_circuits(circuits, backend):
	# Transpile qaoa circuit to backend instruction set, if backend is provided
	# ? I pass a backend into SamplerV2 as *mode* but here sampler_v2.mode returns None, why?
	if not backend is None:
		if isinstance(circuits, Sequence):
			circuits = [transpile(circuit) for circuit in circuits]
		else:
			circuits = transpile(circuits)

	return circuits


class SamplerV2ToSamplerV1Adapter(BaseSamplerV1):
	"""
	Adapts a v2 sampler to a v1 interface.
	"""

	def __init__(self, sampler_v2: BaseSamplerV2, backend=None):
		"""
		Args:
		    sampler_v2 (BaseSamplerV2): V2 sampler to be adapted.
		    backend (Backend | None): Backend to transpile circuits to.
		"""
		self.sampler_v2 = sampler_v2
		self.backend = backend
		super().__init__()

	def _get_quasi_meta(self, res):
		data = BitArray.concatenate_bits(list(res.data.values()))
		counts = data.get_int_counts()
		probs = {measurement: counts / data.num_shots for measurement, counts in counts.items()}
		quasi_dists = QuasiDistribution(probs, shots=data.num_shots)

		metadata = res.metadata
		metadata['sampler_version'] = 2  # might be useful for debugging

		return quasi_dists, metadata

	def _run_v2(self, pubs, **run_options):
		job = self.sampler_v2.run(pubs=pubs, **run_options)
		result = job.result()
		quasi_dists, metas = [], []
		for result_single in result:
			quasi_dist, metadata = self._get_quasi_meta(result_single)
			quasi_dists.append(quasi_dist)
			metas.append(metadata)

		return SamplerResult(quasi_dists=quasi_dists, metadata=metas)

	def _run(self, circuits, parameter_values=None, **run_options) -> PrimitiveJob:
		circuits = _transpile_circuits(circuits, self.backend)
		v2_list = list(zip(circuits, parameter_values))

		job = PrimitiveJob(self._run_v2, v2_list, **run_options)
		job._submit()
		return job


class SamplerV1ToSamplerV2Adapter(BaseSamplerV2):
	"""
	Adapts a v1 sampler to a v2 interface.

	Args:
	    BaseSamplerV2 (_type_): _description_
	"""

	def __init__(self, sampler_v1: BaseSamplerV1) -> None:
		super().__init__()
		self.samplerv1 = sampler_v1

	def _run(self, pubs: Iterable[SamplerPubLike], shots: int = 1024):
		circuits, params = [], []
		for pub in pubs:
			coerced = SamplerPub.coerce(pub)
			circuits.append(coerced.parameter_values.bind_all(coerced.circuit).item())
			params.append([])

		out = self.samplerv1.run(circuits, params, shots=shots).result()

		results = []
		for circuit, dist in zip(circuits, out.quasi_dists):
			values: list[int] = []
			for value, relative_frequency in dist.items():
				values += [value] * int(round(relative_frequency * shots, 0))

			required_bits = circuit.num_qubits
			required_bytes = math.ceil(required_bits / 8)
			byte_array = np.array([np.frombuffer(value.to_bytes(required_bytes), dtype=np.uint8) for value in values])

			bit_array = BitArray(byte_array, num_bits=required_bits)

			results.append(SamplerPubResult(data=DataBin(meas=bit_array), metadata={'shots': shots}))

		return PrimitiveResult(results, metadata={'version': 2})

	def run(self, pubs: Iterable[SamplerPubLike], *, shots: int | None = None) -> BasePrimitiveJob[PrimitiveResult[SamplerPubResult], Any]:
		job = PrimitiveJob(self._run, pubs, shots if shots is not None else 1024)
		job._submit()
		return job


class EstimatorV1ToEstimatorV2Adapter(BaseEstimatorV2):
	def __init__(self, estimator: BaseEstimatorV1) -> None:
		super().__init__()
		self.estimator = estimator

	def _construct_v2_result(self, estimator_result: EstimatorResult) -> PubResult:
		var = np.array([meta.get('variance', 0) for meta in estimator_result.metadata])
		shots = np.array([meta.get('shots', 1) for meta in estimator_result.metadata])

		values = estimator_result.values
		if len(values) == 1:
			values = values.squeeze()
			var = var.squeeze()
			shots = shots.squeeze()
		data_bin = DataBin(evs=values, stds=var / np.sqrt(shots), shape=values.shape if isinstance(values, np.ndarray) else tuple())
		return PubResult(
			data_bin,
			metadata={
				'shots': shots,
			},
		)

	def _run(self, pubs: Iterable[EstimatorPub]) -> PrimitiveResult[PubResult]:
		results = []
		for pub in pubs:
			observables = pub.observables
			parameter_values = pub.parameter_values

			param_shape = parameter_values.shape
			param_indices = np.fromiter(np.ndindex(param_shape), dtype=object).reshape(param_shape)
			broadcast_param_indices, broadcast_observables = np.broadcast_arrays(param_indices, observables)

			params_final, final_observables = [], []
			for index in np.ndindex(*broadcast_param_indices.shape):
				param_index = broadcast_param_indices[index]
				params_final.append(parameter_values[param_index].as_array())
				final_observables.append(broadcast_observables[index])

			res = self.estimator.run(
				[pub.circuit] * len(params_final),
				[SparsePauliOp.from_list(observable.items()) for observable in final_observables],
				params_final,
			).result()
			results.append(res)

		return PrimitiveResult([self._construct_v2_result(result) for result in results], metadata={'version': 2})

	def run(self, pubs: Iterable[EstimatorPubLike], *, precision: float | None = None) -> BasePrimitiveJob[PrimitiveResult[PubResult], Any]:
		coerced_pubs = [EstimatorPub.coerce(pub, precision) for pub in pubs]
		job = PrimitiveJob(self._run, coerced_pubs)
		job._submit()
		return job
