"""One configuration: MaxCut + LinearRampQAOA on an AQT backend -> results/<tag>.json.

    python run.py --backend {aqt_ideal|aqt_noise|hardware} --n 10 --seed 7
"""

import argparse
import json
import os
import sys

import networkx as nx

# this file lives in examples/piastq_hpc/ ; the src package is one level up (examples/src)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_backend(kind: str):
    from qlauncher.routines.qiskit import AQTBackend

    # auto_transpile_level=0: AQT-native gate translation without optimisation
    if kind == 'aqt_ideal':
        return AQTBackend('local_simulator', auto_transpile_level=0)
    if kind == 'aqt_noise':
        from qiskit_aqt_provider import AQTProvider
        from qiskit_aqt_provider.aqt_resource import OfflineSimulatorResource

        prov = AQTProvider('OFFLINE')
        rid = prov.get_backend('offline_simulator_no_noise').resource_id
        noisy = OfflineSimulatorResource(prov, rid, with_noise_model=True)
        return AQTBackend('backendv1v2', backendv1v2=noisy, auto_transpile_level=0)
    if kind == 'hardware':
        return AQTBackend('device', dotenv_path=os.environ.get('AQT_DOTENV', '../.env'), auto_transpile_level=0)
    raise ValueError(kind)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', default='aqt_ideal', choices=['aqt_ideal', 'aqt_noise', 'hardware'])
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--density', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--p', type=int, nargs='+', default=[1, 2, 3])
    ap.add_argument('--shots', type=int, default=2000)
    ap.add_argument('--outdir', default='results')
    args = ap.parse_args()

    from qlauncher import QLauncher
    from qlauncher.problems import MaxCut

    from src.hybrid_analysis import exact_max_cut, score_result
    from src.integration_cost import integration_cost
    from src.lr_qaoa import LinearRampQAOA

    graph = nx.gnp_random_graph(args.n, args.density, seed=args.seed)
    problem = MaxCut(graph, instance_name=f'n{args.n}_s{args.seed}')
    algorithm = LinearRampQAOA(p_grid=tuple(args.p), shots=args.shots)
    backend = build_backend(args.backend)

    result = QLauncher(problem, algorithm, backend).run()

    optimum, _ = exact_max_cut(graph)
    record = {
        'config': vars(args),
        'exact_optimum': optimum,
        'schedule': result.result.get('schedule'),
        'score': score_result(result, graph, optimum=optimum),
        'integration_cost': integration_cost(result),
    }
    os.makedirs(args.outdir, exist_ok=True)
    tag = f'{args.backend}_n{args.n}_s{args.seed}'
    with open(os.path.join(args.outdir, f'{tag}.json'), 'w') as fh:
        json.dump(record, fh, indent=2)

    s, c = record['score'], record['integration_cost']
    print(f"{tag}: mean={s['mean_ratio']:.3f}  P(>=0.9)={s['p_good']:.2f}  "
          f"submissions={c['qpu_submissions']} (naive {c['naive_submissions']})  "
          f"t_classical={c['t_classical_s']}s t_quantum={c['t_quantum_s']}s")


if __name__ == '__main__':
    main()
