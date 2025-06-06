from typing import TypeVar
from qiskit import QuantumCircuit, transpile
from qiskit.providers.backend import BackendV1, BackendV2
from qiskit.primitives import BackendSamplerV2, BackendEstimatorV2
from qiskit_ibm_runtime import EstimatorV2, SamplerV2


AUTO_TRANSPILE_SAMPLER_TYPE = TypeVar('AUTO_TRANSPILE_SAMPLER_TYPE', BackendSamplerV2, SamplerV2)
AUTO_TRANSPILE_ESTIMATOR_TYPE = TypeVar('AUTO_TRANSPILE_ESTIMATOR_TYPE', BackendEstimatorV2, EstimatorV2)


def _get_transpiled_pubs(pubs: list[tuple | QuantumCircuit], backend: BackendV1 | BackendV2, optimization_level: int = 2) -> list[tuple | QuantumCircuit]:
    new_pubs = []
    for pub in pubs:
        if isinstance(pub, QuantumCircuit):
            pub = transpile(pub, backend, optimization_level=optimization_level)
        elif isinstance(pub, tuple):
            pub = (transpile(pub[0], backend, optimization_level=optimization_level), *pub[1:])
        new_pubs.append(pub)
    return new_pubs


def _assign_sampler_pubs(pubs: list[tuple | QuantumCircuit]) -> list[QuantumCircuit]:
    new_pubs = []
    for pub in pubs:
        if isinstance(pub, tuple):
            qc, *params = pub
            pub = qc.assign_parameters(*params)
        new_pubs.append(pub)
    return new_pubs


def _assign_estimator_pubs(pubs: list[tuple]) -> list[tuple]:
    new_pubs = []
    for pub in pubs:
        if len(pub) > 2:
            qc, observable, *params = pub
            pub = (qc.assign_parameters(*params), observable)
        new_pubs.append(pub)
    return new_pubs


def set_sampler_auto_run_behavior(sampler: AUTO_TRANSPILE_SAMPLER_TYPE, auto_transpile: bool = False, auto_transpile_level: int = 2, auto_assign: bool = False) -> AUTO_TRANSPILE_SAMPLER_TYPE:
    """
    Set chosen automatic behavior on a sampler instance.

    Args:
        sampler (AUTO_TRANSPILE_SAMPLER_TYPE): Compatible sampler instance
        auto_transpile (bool, optional): Whether to automatically transpile to the samplers backend. Defaults to False.
        auto_transpile_level (int, optional): What level of optimization to set. Defaults to 2.
        auto_assign (bool, optional): Whether to automatically assign parameters in case of parameterized circuits. Defaults to False.

    Returns:
        AUTO_TRANSPILE_SAMPLER_TYPE: Same instance with modified run() method.
    """
    func = sampler.run

    def run_wrapper(pubs, *args, shots: int | None = None):
        if auto_transpile:
            pubs = _get_transpiled_pubs(pubs, sampler._backend, optimization_level=auto_transpile_level)
        if auto_assign:
            pubs = _assign_sampler_pubs(pubs)

        return func(pubs, *args, shots=shots)

    sampler.run = run_wrapper
    return sampler


def set_estimator_auto_run_behavior(estimator: AUTO_TRANSPILE_ESTIMATOR_TYPE, auto_transpile: bool = False, auto_transpile_level: int = 2, auto_assign: bool = False) -> AUTO_TRANSPILE_ESTIMATOR_TYPE:
    """
    Set chosen automatic behavior on a estimator instance.

    Args:
        estimator (AUTO_TRANSPILE_ESTIMATOR_TYPE): Compatible estimator instance
        auto_transpile (bool, optional): Whether to automatically transpile to the estimators backend. Defaults to False.
        auto_transpile_level (int, optional): What level of optimization to set. Defaults to 2.
        auto_assign (bool, optional): Whether to automatically assign parameters in case of parameterized circuits. Defaults to False.

    Returns:
        AUTO_TRANSPILE_ESTIMATOR_TYPE: Same instance with modified run() method.
    """
    func = estimator.run

    def run_wrapper(pubs, *args, precision: float | None = None):
        if auto_transpile:
            pubs = _get_transpiled_pubs(pubs, estimator._backend, optimization_level=auto_transpile_level)
        if auto_assign:
            pubs = _assign_estimator_pubs(pubs)
        return func(pubs, *args, precision=precision)
    estimator.run = run_wrapper
    return estimator
