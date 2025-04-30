from typing import Sequence

from qiskit import transpile
from qiskit.primitives.base import BaseSamplerV1, BaseSamplerV2
from qiskit.primitives import SamplerResult, BasePrimitiveJob
from qiskit.result import QuasiDistribution


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
        data = next(iter(res.data.values()))
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


class SamplerV2Adapter(BaseSamplerV1):
    """
    V1 adapter for V2 samplers.
    """

    def __init__(self, sampler_v2: BaseSamplerV2, backend=None):
        """
        Args:
            sampler_v2 (BaseSamplerV2): V2 sampler to be adapted.
            backend (Optional[Backend]): Backend to transpile circuits to.
        """
        self.sampler_v2 = sampler_v2
        self.backend = backend
        super().__init__()

    def _run(self, circuits, parameter_values=None, **run_options) -> SamplerV2JobAdapter:
        circuits = _transpile_circuits(circuits, self.backend)
        v2_list = list(zip(circuits, parameter_values))
        job = self.sampler_v2.run(pubs=v2_list, **run_options)

        return SamplerV2JobAdapter(job)
