"""Linear-Ramp QAOA as a QLauncher Algorithm: classical schedule optimisation + one submission."""

import time
from typing import Any

import numpy as np

try:
    from qlauncher.base import Algorithm, Result
    from qlauncher.base.models import Hamiltonian

    _BASE = Algorithm
except Exception:  # pragma: no cover
    _BASE = object
    Result = Any  # type: ignore
    Hamiltonian = Any  # type: ignore


def ising_diagonal(hamiltonian: Any) -> np.ndarray:
    """Diagonal of an Ising (Z/ZZ) cost Hamiltonian over all 2**n basis states."""
    n = hamiltonian.num_qubits
    dim = 1 << n
    idx = np.arange(dim)
    zval = np.stack([1 - 2 * ((idx >> b) & 1) for b in range(n)])
    diag = np.zeros(dim)
    for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
        if pauli.x.any():
            continue
        contrib = np.ones(dim)
        for b in range(n):
            if pauli.z[b]:
                contrib = contrib * zval[b]
        diag += float(np.real(coeff)) * contrib
    return diag


def lr_qaoa_state(diag: np.ndarray, n: int, p: int, gamma_scale: float, beta_scale: float) -> np.ndarray:
    """Linear-ramp QAOA statevector for a diagonal cost (gamma ramps up, beta ramps down)."""
    dim = 1 << n
    psi = np.ones(dim, dtype=complex) / np.sqrt(dim)
    for k in range(1, p + 1):
        frac = (k - 0.5) / p
        gamma, beta = gamma_scale * frac, beta_scale * (1.0 - frac)
        psi = psi * np.exp(-1j * gamma * diag)
        a = psi.reshape([2] * n)
        c, s = np.cos(beta), -1j * np.sin(beta)
        for q in range(n):
            a = np.moveaxis(a, q, 0)
            sh = a.shape
            a = a.reshape(2, -1)
            a = np.stack([c * a[0] + s * a[1], s * a[0] + c * a[1]]).reshape(sh)
            a = np.moveaxis(a, 0, q)
        psi = a.reshape(dim)
    return psi


def expected_energy(psi: np.ndarray, diag: np.ndarray) -> float:
    return float(np.sum((np.abs(psi) ** 2) * diag))


def normalise(diag: np.ndarray) -> float:
    s = float(np.std(diag))
    return s if s > 1e-12 else 1.0


def optimize_schedule(diag_norm: np.ndarray, n: int, p_grid=(1, 2, 3),
                      starts=((1.0, 1.0), (2.0, 0.5), (2.0, -0.5), (0.5, 1.0))) -> dict:
    """Optimise the two ramp scalars on the exact statevector (multistart Nelder-Mead), 0 QPU calls."""
    from scipy.optimize import minimize

    best: dict | None = None
    evals = 0

    def energy(x: np.ndarray, p: int) -> float:
        nonlocal evals
        evals += 1
        return expected_energy(lr_qaoa_state(diag_norm, n, p, float(x[0]), float(x[1])), diag_norm)

    for p in p_grid:
        for g0, b0 in starts:
            res = minimize(energy, np.array([g0, b0]), args=(p,), method='Nelder-Mead',
                           options={'xatol': 1e-3, 'fatol': 1e-4, 'maxiter': 300})
            if best is None or res.fun < best['energy']:
                best = {'p': p, 'gamma_scale': float(res.x[0]), 'beta_scale': float(res.x[1]), 'energy': float(res.fun)}
    assert best is not None
    best['n_statevector_evals'] = evals
    return best


def build_cost_circuit(hamiltonian: Any, p: int, gamma_scale: float, beta_scale: float):
    """QAOA circuit exp(-i*gamma*H) (RZ/RZZ) + RX mixer, matching lr_qaoa_state."""
    from qiskit import QuantumCircuit

    n = hamiltonian.num_qubits
    terms: list[tuple[list[int], float]] = []
    for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
        if pauli.x.any():
            continue
        qs = [i for i in range(n) if pauli.z[i]]
        if qs:
            terms.append((qs, float(np.real(coeff))))

    qc = QuantumCircuit(n)
    qc.h(range(n))
    for k in range(1, p + 1):
        frac = (k - 0.5) / p
        gamma, beta = gamma_scale * frac, beta_scale * (1.0 - frac)
        for qs, c in terms:
            if len(qs) == 1:
                qc.rz(2 * gamma * c, qs[0])
            elif len(qs) == 2:
                qc.rzz(2 * gamma * c, qs[0], qs[1])
            else:
                for t in range(len(qs) - 1):
                    qc.cx(qs[t], qs[t + 1])
                qc.rz(2 * gamma * c, qs[-1])
                for t in reversed(range(len(qs) - 1)):
                    qc.cx(qs[t], qs[t + 1])
        for q in range(n):
            qc.rx(2 * beta, q)
    qc.measure_all()
    return qc


class LinearRampQAOA(_BASE):
    """Queue-minimal linear-ramp QAOA (QLauncher Algorithm)."""

    def __init__(self, p_grid=(1, 2, 3), shots: int = 1024,
                 starts=((1.0, 1.0), (2.0, 0.5), (2.0, -0.5), (0.5, 1.0)), **alg_kwargs) -> None:
        super().__init__(**alg_kwargs)
        self.name = 'linear_ramp_qaoa'
        self.p_grid = tuple(p_grid)
        self.shots = shots
        self.starts = starts
        self.parameters = ['p_grid', 'shots']

    def run(self, problem: Hamiltonian, backend: Any) -> Result:  # noqa: ANN401
        hamiltonian = problem.hamiltonian
        n = hamiltonian.num_qubits

        t0 = time.perf_counter()
        diag = ising_diagonal(hamiltonian)
        scale = normalise(diag)
        best = optimize_schedule(diag / scale, n, self.p_grid, self.starts)
        t_classical = time.perf_counter() - t0

        circuit = build_cost_circuit(hamiltonian * (1.0 / scale), best['p'], best['gamma_scale'], best['beta_scale'])
        t1 = time.perf_counter()
        counts = backend.sample_circuit(circuit, shots=self.shots)
        t_quantum = time.perf_counter() - t1

        energies = {bs: float(diag[int(bs.replace(' ', ''), 2)]) for bs in counts}
        meta = {
            'schedule': {'p': best['p'], 'gamma_scale': best['gamma_scale'], 'beta_scale': best['beta_scale']},
            'classical_statevector_evals': best['n_statevector_evals'],
            'qpu_submissions': 1,
            'exact_optimum_energy': float(diag.min()),
            't_classical_s': t_classical,
            't_quantum_s': t_quantum,
        }
        return Result.from_counts_energies(counts, energies, result=meta)
