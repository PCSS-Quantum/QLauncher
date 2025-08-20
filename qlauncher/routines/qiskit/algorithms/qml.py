from collections.abc import Callable, Sequence
from typing import Any

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.primitives import BaseSampler, BaseSamplerV1, BaseSamplerV2
from qiskit.transpiler.passmanager import PassManager
from qiskit.providers import Options
import numpy as np

from qiskit_machine_learning.kernels.algorithms import QuantumKernelTrainer
from qiskit_machine_learning.kernels import TrainableFidelityQuantumKernel, FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute, BaseStateFidelity

from qlauncher.base.base import Backend, Problem, Algorithm, Result
from qlauncher.routines.qiskit.backends.qiskit_backend import QiskitBackend
from qlauncher.routines.cirq import CirqBackend


class ComputeUncomputeCustom(ComputeUncompute):
    """
        This is just :class:`qiskit_machine_learning.state_fidelities.ComputeUncompute` that checks 
        if a sampler is an instance of BaseSamplerV1 instead of BaseSampler.
        The reason was that classes basing BaseSampler were getting isinstance(cls(),BaseSampler) == False
        probably because of some qiskit shenanigans.
        """

    def __init__(
        self,
        sampler: BaseSampler | BaseSamplerV2,
        *,
        options: Options | None = None,
        local: bool = False,
        pass_manager: PassManager | None = None,
    ) -> None:

        if (not isinstance(sampler, BaseSamplerV1)) and (not isinstance(sampler, BaseSamplerV2)):
            raise ValueError(
                f"The sampler should be an instance of BaseSampler or BaseSamplerV2, "
                f"but got {type(sampler)}"
            )
        self._sampler: BaseSamplerV1 | BaseSamplerV2 = sampler
        self._pass_manager = pass_manager
        self._local = local
        self._default_options = Options()
        if options is not None:
            self._default_options.update_options(**options)
        BaseStateFidelity.__init__(self)  # pylint: disable=non-parent-init-called


class TrainQSVCKernel(Algorithm):
    """
    Train a quantum kernel with additional parameters to be optimized.
    The kernel will be optimized to provide maximum accuracy with a support vector classifier on the provided dataset.
    If no trainable parameters are provided, the algorithm will return
        a :class:`qiskit_machine_learning.kernels.FidelityQuantumKernel` kernel with a sampler assigned to the provided backend.
        Otherwise an instance of :class:`qiskit_machine_learning.kernels.TrainableFidelityQuantumKernel` with optimal 
        parameters and a sampler assigned to the provided backend will be returned.

    Args:
        kernel_circuit(QuantumCircuit): A parametrizable quantum circuit. The measurements will be used to produce kernel output.
        trainable_params(Sequence[Parameter] | None, optional): 
            The parameters to be optimized during training. If None no optimization will be done. Defaults to None.
    """

    _algorithm_format = 'tabular_ml'

    def __init__(
        self,
        kernel_circuit: QuantumCircuit,
        trainable_params: Sequence[Parameter] | None = None,
        **alg_kwargs
    ) -> None:
        super().__init__(**alg_kwargs)
        self.kernel = kernel_circuit
        self.trainable = trainable_params if trainable_params is not None else []

    def run(self, problem: Problem, backend: Backend, formatter: Callable[..., Any]) -> Result:
        X, y = formatter(problem)

        if not isinstance(X, np.ndarray):
            raise ValueError(f"X is not of type np.ndarray: received {type(X)}")

        if not isinstance(y, np.ndarray):
            raise ValueError(f"y is not of type np.ndarray: received {type(y)}")

        if isinstance(backend, (QiskitBackend, CirqBackend)):
            sampler = backend.samplerV1
        else:
            raise ValueError(f"The accepted backends are QiskitBackend and CirqBackend, got {type(backend)}")

        if len(self.trainable) == 0:
            return Result(
                best_bitstring='',
                best_energy=1,
                most_common_bitstring='',
                most_common_bitstring_energy=0,
                distribution={},
                energies={},
                num_of_samples=0,
                average_energy=0,
                energy_std=0,
                result=FidelityQuantumKernel(feature_map=self.kernel, fidelity=ComputeUncomputeCustom(sampler=sampler))
            )
        trainable_kernel = TrainableFidelityQuantumKernel(
            feature_map=self.kernel,
            fidelity=ComputeUncomputeCustom(sampler=sampler),
            training_parameters=self.trainable
        )
        kernel_trainer = QuantumKernelTrainer(trainable_kernel)
        kernel_trainer.fit(X, y)

        return Result(
            best_bitstring='',
            best_energy=1,
            most_common_bitstring='',
            most_common_bitstring_energy=0,
            distribution={},
            energies={},
            num_of_samples=0,
            average_energy=0,
            energy_std=0,
            result=kernel_trainer.quantum_kernel
        )
