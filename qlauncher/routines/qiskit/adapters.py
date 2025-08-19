from collections.abc import Sequence
from typing import Any, Iterable

import math

import numpy as np

from qiskit import transpile, QuantumCircuit
from qiskit.primitives.containers.primitive_result import PrimitiveResult
from qiskit.primitives.containers.sampler_pub_result import SamplerPubResult
from qiskit.result import QuasiDistribution
from qiskit.primitives import SamplerResult, BasePrimitiveJob, BitArray, DataBin
from qiskit.primitives.base import BaseSamplerV1, BaseSamplerV2
from qiskit.primitives.containers.sampler_pub import SamplerPubLike
from qiskit.primitives.primitive_job import PrimitiveJob


class RuntimeJobV2Adapter(BasePrimitiveJob):
    def __init__(self, job, **kwargs):
        super().__init__(job.job_id(), **kwargs)
        self.job = job

    def result(self):
        raise NotImplementedError()

    def cancel(self):
        return self.job.cancel()

    def status(self):
        return self.job.status()

    def done(self):
        return self.job.done()

    def cancelled(self):
        return self.job.cancelled()

    def running(self):
        return self.job.running()

    def in_final_state(self):
        return self.job.in_final_state()


class SamplerV2JobAdapter(RuntimeJobV2Adapter):
    """
    Dummy data holder, returns a v1 SamplerResult from v2 sampler job.
    """

    def __init__(self, job, **kwargs):
        super().__init__(job, **kwargs)

    def _get_quasi_meta(self, res):
        data = BitArray.concatenate_bits(list(res.data.values()))
        counts = data.get_int_counts()
        probs = {k: v/data.num_shots for k, v in counts.items()}
        quasi_dists = QuasiDistribution(probs, shots=data.num_shots)

        metadata = res.metadata
        metadata["sampler_version"] = 2  # might be useful for debugging

        return quasi_dists, metadata

    def result(self):
        res = self.job.result()
        qd, metas = [], []
        for r in res:
            quasi_dist, metadata = self._get_quasi_meta(r)
            qd.append(quasi_dist)
            metas.append(metadata)

        return SamplerResult(quasi_dists=qd, metadata=metas)


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

    def _run(self, circuits, parameter_values=None, **run_options) -> SamplerV2JobAdapter:
        circuits = _transpile_circuits(circuits, self.backend)
        v2_list = list(zip(circuits, parameter_values))
        job = self.sampler_v2.run(pubs=v2_list, **run_options)

        return SamplerV2JobAdapter(job)


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
            if isinstance(pub, QuantumCircuit):
                circuits.append(pub)
                params.append([])
            elif isinstance(pub, tuple):
                circuits.append(pub[0])
                params.append(pub[1] if len(pub) == 2 else [])

        out = self.samplerv1.run(circuits, params, shots=shots).result()
        results = []
        for dist in out.quasi_dists:
            vals: list[int] = []
            max_val = 0
            for k, v in dist.items():
                vals += [k] * int(round(v*shots, 0))
                max_val = max(max_val, k)

            required_bits = math.ceil(math.log2(max_val))
            required_bytes = math.ceil(required_bits/8)

            arr = np.array(
                [
                    np.frombuffer(v.to_bytes(required_bytes), dtype=np.uint8)
                    for v in vals
                ])
            bit_array = BitArray(
                arr,
                num_bits=required_bits
            )
            results.append(SamplerPubResult(data=DataBin(meas=bit_array), metadata={'shots': shots}))

        return PrimitiveResult(results, metadata={'version': 2})

    def run(
        self,
        pubs: Iterable[SamplerPubLike],
        *,
        shots: int | None = None
    ) -> BasePrimitiveJob[PrimitiveResult[SamplerPubResult], Any]:
        job = PrimitiveJob(self._run, pubs, shots if shots is not None else 1024)
        job._submit()
        return job
