from typing import get_args

import numpy as np
from qiskit.primitives.containers.sampler_pub import SamplerPub, SamplerPubLike

from qlauncher.routines.circuits import CIRCUIT_FORMATS


def coerce_to_circuit_list(pub: SamplerPubLike | CIRCUIT_FORMATS, shots: int | None = None) -> list[CIRCUIT_FORMATS]:
    if not isinstance(pub, get_args(CIRCUIT_FORMATS)):
        coerced = SamplerPub.coerce(pub, shots)
        bound = coerced.parameter_values.bind_all(coerced.circuit)
        return np.ravel(bound).tolist()
    return [pub]
