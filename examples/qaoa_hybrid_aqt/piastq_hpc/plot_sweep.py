"""Figure from results/*.json: AQT ideal vs noise (and PIAST-Q) across seeds -> sweep_comparison.png."""

import argparse
import glob
import json
import os

import numpy as np

BLUE, ORANGE, GREEN, RED, GREY = '#0B5FA5', '#E8833A', '#2E9E6B', '#B03A2E', '#8A8F98'
ORDER = ['aqt_ideal', 'aqt_noise', 'hardware']
LABELS = {'aqt_ideal': 'AQT ideal', 'aqt_noise': 'AQT noise', 'hardware': 'PIAST-Q'}


def load(results_dir: str) -> dict:
    data: dict[str, list] = {}
    for path in glob.glob(os.path.join(results_dir, '**', '*.json'), recursive=True):
        with open(path) as fh:
            d = json.load(fh)
        data.setdefault(d['config']['backend'], []).append(d)
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results')
    ap.add_argument('--out', default='sweep_comparison.png')
    args = ap.parse_args()

    data = load(args.results)
    backs = [b for b in ORDER if b in data] + [b for b in data if b not in ORDER]
    if not backs:
        print('no results in', args.results); return

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.25, 'figure.dpi': 130})
    x = np.arange(len(backs))
    rng = np.random.default_rng(0)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.6))

    def panel(ax, key, color, title, ylabel):
        means, stds, allpts = [], [], []
        for b in backs:
            v = np.array([d['score'][key] for d in data[b]], dtype=float)
            means.append(v.mean()); stds.append(v.std()); allpts.append(v)
        ax.bar(x, means, 0.55, yerr=stds, color=color, alpha=0.85, capsize=4,
               error_kw=dict(ecolor=GREY, lw=1.3))
        for xi, v in zip(x, allpts):                 # individual seeds
            ax.scatter(np.full_like(v, xi) + rng.uniform(-0.12, 0.12, len(v)), v,
                       color='black', s=16, zorder=3, alpha=0.7)
        for xi, m in zip(x, means):
            ax.text(xi, m + (max(stds) + 0.02), f'{m:.2f}', ha='center', fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels([LABELS.get(b, b) for b in backs])
        ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10.5)

    panel(a1, 'p_good', BLUE, 'Concentration on good solutions', 'P(approx ratio >= 0.9)')
    panel(a2, 'mean_ratio', ORANGE, 'Mean solution quality', 'mean approximation ratio')
    rand = np.mean([d['score']['random_cut_ratio'] for b in backs for d in data[b]])
    a2.axhline(0.878, color=GREEN, ls=(0, (3, 1, 1, 1)), lw=1.3, label='Goemans-Williamson 0.878')
    a2.axhline(rand, color=RED, ls='-.', lw=1.2, label=f'random cut ({rand:.2f})')
    a2.set_ylim(0.5, 1.0); a2.legend(fontsize=8.5, loc='lower left')

    n_seeds = len(next(iter(data.values())))
    fig.suptitle(f'QLauncher MaxCut + LinearRampQAOA on AQT (PIAST-Q) — sweep over {n_seeds} seeds '
                 '(1 QPU submission each)', y=1.0, fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.out); plt.close(fig)
    print('wrote', args.out, '| backends:', ', '.join(backs))


if __name__ == '__main__':
    main()
