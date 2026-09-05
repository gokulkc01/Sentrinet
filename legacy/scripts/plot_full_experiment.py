import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_csv(path):
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            # convert numeric fields
            for k in list(r.keys()):
                v = r[k]
                if v is None or v == '':
                    continue
                try:
                    if '.' in v:
                        r[k] = float(v)
                    else:
                        r[k] = int(v)
                except Exception:
                    pass
            rows.append(r)
    return rows


def group_stats(rows, key_system='system', key_x='drop_rate'):
    # structure: stats[system][x] = list of rows
    stats = {}
    for r in rows:
        sys = r.get(key_system)
        x = r.get(key_x)
        if sys is None or x is None:
            continue
        stats.setdefault(sys, {})
        stats[sys].setdefault(x, []).append(r)
    # compute aggregates
    aggs = {}
    for sys, xs in stats.items():
        aggs[sys] = {}
        for x, rs in sorted(xs.items()):
            n = len(rs)
            def mean_field(field):
                vals = [float(r[field]) for r in rs if r.get(field) is not None]
                if not vals:
                    return None, None
                m = sum(vals) / len(vals)
                var = sum((vv - m) ** 2 for vv in vals) / len(vals)
                return m, math.sqrt(var)

            cap_mean, cap_std = mean_field('capture_rate')
            steps_mean, steps_std = mean_field('mean_steps')
            rew_mean, rew_std = mean_field('mean_reward')
            trust_mean, trust_std = mean_field('mean_trust')

            aggs[sys][x] = {
                'capture_mean': cap_mean, 'capture_std': cap_std,
                'steps_mean': steps_mean, 'steps_std': steps_std,
                'reward_mean': rew_mean, 'reward_std': rew_std,
                'trust_mean': trust_mean, 'trust_std': trust_std,
            }
    return aggs


def plot_lines(aggs, field_mean, field_std, ylabel, title, outname):
    outdir = Path('results/plots')
    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7,4))
    for sys, xs in sorted(aggs.items()):
        xs_sorted = sorted(xs.items())
        xs_vals = [x for x,_ in xs_sorted]
        means = [v[field_mean] if v[field_mean] is not None else float('nan') for _,v in xs_sorted]
        stds = [v[field_std] if v[field_std] is not None else 0.0 for _,v in xs_sorted]
        ax.plot(xs_vals, means, marker='o', label=sys)
        ax.fill_between(xs_vals, [m - s for m,s in zip(means,stds)], [m + s for m,s in zip(means,stds)], alpha=0.2)
    ax.set_xlabel('Drop rate')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    p = outdir/outname
    fig.savefig(p)
    print('Wrote', p)


def main():
    path = Path('results/full_experiment.csv')
    if not path.exists():
        print('Missing', path)
        return
    rows = read_csv(path)
    aggs = group_stats(rows)
    plot_lines(aggs, 'capture_mean', 'capture_std', 'Capture rate', 'Capture rate vs drop rate', 'capture_vs_drop.png')
    plot_lines(aggs, 'steps_mean', 'steps_std', 'Mean steps', 'Mean steps vs drop rate', 'steps_vs_drop.png')
    plot_lines(aggs, 'reward_mean', 'reward_std', 'Mean reward', 'Mean reward vs drop rate', 'reward_vs_drop.png')
    plot_lines(aggs, 'trust_mean', 'trust_std', 'Mean trust', 'Mean trust vs drop rate', 'trust_vs_drop.png')


if __name__ == '__main__':
    main()
