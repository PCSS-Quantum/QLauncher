"""Scoring, baselines, bootstrap CIs, connectivity comparison and figures for a QLauncher Result."""

import itertools
from typing import Any

import networkx as nx
import numpy as np

GW_RATIO = 0.878  # Goemans-Williamson MaxCut approximation guarantee


def cut_value(graph: nx.Graph, bitstring: str) -> float:
    """Cut weight of ``bitstring`` (qubit i == graph node i)."""
    x = int(bitstring.replace(' ', ''), 2)
    return sum(float(a.get('weight', 1.0)) for i, j, a in graph.edges(data=True) if ((x >> i) & 1) != ((x >> j) & 1))


def exact_max_cut(graph: nx.Graph) -> tuple[float, str]:
    """Brute-force optimum (small graphs only, n <= ~20)."""
    n = graph.number_of_nodes()
    best_val, best_bs = -np.inf, '0' * n
    for bits in itertools.product('01', repeat=n):
        bs = ''.join(bits)
        v = cut_value(graph, bs)
        if v > best_val:
            best_val, best_bs = v, bs
    return best_val, best_bs


def random_cut_ratio(graph: nx.Graph) -> float:
    total_w = sum(float(a.get('weight', 1.0)) for *_, a in graph.edges(data=True))
    opt, _ = exact_max_cut(graph)
    return (0.5 * total_w) / opt if opt else 0.0


def _distribution(result: Any) -> tuple[dict[str, float], int]:
    if hasattr(result, 'distribution'):
        return dict(result.distribution), int(getattr(result, 'num_of_samples', 0)) or 1024
    if isinstance(result, dict):
        vals = np.array(list(result.values()), dtype=float)
        if vals.sum() > 1.5:
            n = int(vals.sum())
            return {k: v / n for k, v in result.items()}, n
        return dict(result), 1024
    raise TypeError('result must be a qlauncher Result or a dict')


def score_result(result: Any, graph: nx.Graph, good_frac: float = 0.9, optimum: float | None = None,
                 n_boot: int = 300, seed: int = 11) -> dict:
    """Approximation-ratio metrics with 95% bootstrap CIs and baselines."""
    dist, n_samples = _distribution(result)
    if optimum is None:
        optimum, _ = exact_max_cut(graph)
    bitstrings = list(dist)
    probs = np.array([dist[b] for b in bitstrings]); probs /= probs.sum()
    ratios = np.array([cut_value(graph, b) / optimum for b in bitstrings])

    mean_ratio = float((probs * ratios).sum())
    rand = random_cut_ratio(graph)
    denom = (1.0 - rand) if 1.0 > rand else 1.0

    rng = np.random.default_rng(seed)
    boot = rng.multinomial(n_samples, probs, size=n_boot).astype(float) / n_samples
    mr_b, pg_b = boot @ ratios, boot @ (ratios >= good_frac).astype(float)
    pct = lambda a: [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]

    return {
        'mean_ratio': mean_ratio,
        'mean_ratio_normalised': float((mean_ratio - rand) / denom),
        'mean_ratio_ci95': pct(mr_b),
        'p_good': float(probs[ratios >= good_frac].sum()),
        'p_good_ci95': pct(pg_b),
        'p_opt': float(probs[ratios >= 1.0 - 1e-9].sum()),
        'best_ratio': float(ratios.max()),
        'random_cut_ratio': rand,
        'gw_ratio': GW_RATIO,
        'optimum': float(optimum),
        'num_of_samples': n_samples,
    }


def _qaoa_circuit(graph: nx.Graph, p: int, gammas, betas):
    from qiskit import QuantumCircuit

    n = graph.number_of_nodes()
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for layer in range(p):
        for i, j, a in graph.edges(data=True):
            qc.rzz(2 * gammas[layer] * float(a.get('weight', 1.0)), i, j)
        for q in range(n):
            qc.rx(2 * betas[layer], q)
    qc.measure_all()
    return qc


def connectivity_comparison(backend: Any, graph: nx.Graph, p: int = 3, shots: int = 1200,
                            gammas=None, betas=None, max_ops: int = 1900) -> dict:
    """Native (all-to-all) vs SWAP-routed-to-a-line, on the same backend. Depth auto-fits the
    device's ~2000-ops limit."""
    from qiskit import transpile
    from qiskit.transpiler import CouplingMap

    n = graph.number_of_nodes()
    line = CouplingMap.from_line(n)
    fixed = gammas is not None and betas is not None

    def build(depth):
        gs = gammas if fixed else [0.8 * (k + 0.5) / depth for k in range(depth)]
        bs = betas if fixed else [0.8 * (1.0 - (k + 0.5) / depth) for k in range(depth)]
        nat = _qaoa_circuit(graph, depth, gs, bs)
        rou = transpile(nat, coupling_map=line, basis_gates=['rz', 'rx', 'ry', 'h', 'cx'],
                        optimization_level=3, seed_transpiler=1)
        return nat, rou

    native, routed = build(p)
    while not fixed and p > 1 and len(routed.data) > max_ops:
        p -= 1
        native, routed = build(p)
    while True:
        try:
            counts_native = backend.sample_circuit(native, shots=shots)
            counts_routed = backend.sample_circuit(routed, shots=shots)
            break
        except Exception as exc:  # noqa: BLE001
            if fixed or p <= 1 or not ('at most' in str(exc) or 'too_long' in str(exc) or '2000' in str(exc)):
                raise
            p -= 1
            native, routed = build(p)

    opt, _ = exact_max_cut(graph)
    return {
        'p_used': p,
        'native_all_to_all': score_result(counts_native, graph, optimum=opt),
        'forced_onto_grid': score_result(counts_routed, graph, optimum=opt),
        'native_2q_gates': graph.number_of_edges() * p,
        'routed_2q_gates': sum(1 for inst in routed.data if inst.operation.num_qubits == 2),
        'routed_total_ops': len(routed.data),
    }


def plot_graph_solution(graph: nx.Graph, bitstring: str, path: str, title: str = '') -> str:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    x = int(bitstring.replace(' ', ''), 2)
    side = [(x >> i) & 1 for i in range(graph.number_of_nodes())]
    pos = nx.spring_layout(graph, seed=1)
    cut = [(i, j) for i, j in graph.edges() if side[i] != side[j]]
    kept = [(i, j) for i, j in graph.edges() if side[i] == side[j]]
    fig, ax = plt.subplots(figsize=(6, 5))
    nx.draw_networkx_edges(graph, pos, edgelist=kept, edge_color='#8A8F98', alpha=0.35, ax=ax)
    nx.draw_networkx_edges(graph, pos, edgelist=cut, edge_color='#E8833A', width=2, ax=ax)
    nx.draw_networkx_nodes(graph, pos, node_color=['#0B5FA5' if s == 0 else '#2E9E6B' for s in side], ax=ax)
    nx.draw_networkx_labels(graph, pos, font_color='white', font_size=9, ax=ax)
    ax.set_title(title or 'MaxCut partition'); ax.axis('off')
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


def plot_modes(scores_by_mode: dict[str, dict], optimum: float, path: str) -> str:
    """Grouped bars across execution modes: P(>=0.9) and mean ratio (native runs)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    labels = list(scores_by_mode)
    x = np.arange(len(labels))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.4))

    def err(key):
        return np.array([[abs(scores_by_mode[m][key] - scores_by_mode[m][key + '_ci95'][0]),
                          abs(scores_by_mode[m][key + '_ci95'][1] - scores_by_mode[m][key])] for m in labels]).T

    pg = [scores_by_mode[m]['p_good'] for m in labels]
    a1.bar(x, pg, 0.55, color='#0B5FA5', yerr=err('p_good'), capsize=4)
    for xi, v in zip(x, pg):
        a1.text(xi, v + 0.012, f'{v:.2f}', ha='center', fontsize=9)
    a1.set_xticks(x); a1.set_xticklabels(labels); a1.set_ylabel('P(approx ratio >= 0.9)')
    a1.set_title('Concentration on good solutions', fontsize=10.5); a1.grid(alpha=0.25)

    mr = [scores_by_mode[m]['mean_ratio'] for m in labels]
    a2.bar(x, mr, 0.55, color='#E8833A', yerr=err('mean_ratio'), capsize=4)
    a2.axhline(GW_RATIO, color='#2E9E6B', ls=(0, (3, 1, 1, 1)), label='Goemans-Williamson 0.878')
    rnd = next(iter(scores_by_mode.values()))['random_cut_ratio']
    a2.axhline(rnd, color='#B03A2E', ls='-.', label=f'random cut ({rnd:.2f})')
    a2.set_xticks(x); a2.set_xticklabels(labels); a2.set_ylim(0.5, 1.0)
    a2.set_ylabel('mean approximation ratio'); a2.set_title('Mean solution quality', fontsize=10.5)
    a2.legend(fontsize=8.5, loc='lower left'); a2.grid(alpha=0.25)

    fig.suptitle('QLauncher MaxCut + QAOA on AQT (PIAST-Q): execution modes', y=1.0, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(path, dpi=130); plt.close(fig)
    return path
