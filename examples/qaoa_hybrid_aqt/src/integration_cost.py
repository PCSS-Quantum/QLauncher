"""Integration-cost accounting for a LinearRampQAOA run."""

from typing import Any


def integration_cost(result: Any, naive_iters: int = 20) -> dict:
    """Classical/quantum time split, quantum duty, and submissions vs a naive trained QAOA."""
    meta = getattr(result, 'result', None) or (result if isinstance(result, dict) else {})
    t_c = float(meta.get('t_classical_s', 0.0))
    t_q = float(meta.get('t_quantum_s', 0.0))
    p = int(meta.get('schedule', {}).get('p', 1))
    submissions = int(meta.get('qpu_submissions', 1))
    total = t_c + t_q
    naive = (2 * p + 1) * naive_iters
    return {
        'qpu_submissions': submissions,
        'naive_submissions': naive,
        't_classical_s': round(t_c, 4),
        't_quantum_s': round(t_q, 4),
        't_total_s': round(total, 4),
        'quantum_duty': round(t_q / total, 4) if total else 0.0,
        'est_wall_saved_vs_naive_s': round((naive - submissions) * t_q, 3) if t_q else None,
    }
