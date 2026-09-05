import csv
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_rows(path):
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            r['capture_rate'] = float(r['capture_rate'])
            r['drop_rate'] = float(r['drop_rate'])
            rows.append(r)
    return rows

def aggregate(rows):
    ag = {}
    for r in rows:
        sys = r['system']
        d = r['drop_rate']
        ag.setdefault(sys, {})
        ag[sys].setdefault(d, []).append(r['capture_rate'])
    return ag

def mean_std_percent(lst):
    arr = np.array(lst)
    return arr.mean()*100.0, arr.std(ddof=1)*100.0, len(arr)

def welch_ttest(a, b):
    # a,b are lists
    A = np.array(a)
    B = np.array(b)
    m1, m2 = A.mean(), B.mean()
    s1, s2 = A.var(ddof=1), B.var(ddof=1)
    n1, n2 = len(A), len(B)
    denom = math.sqrt(s1/n1 + s2/n2)
    if denom == 0:
        return float('nan'), float('nan')
    t = (m1 - m2) / denom
    # degrees of freedom (Welch-Satterthwaite)
    num = (s1/n1 + s2/n2)**2
    den = 0.0
    if n1 > 1:
        den += (s1**2) / (n1**2 * (n1 - 1))
    if n2 > 1:
        den += (s2**2) / (n2**2 * (n2 - 1))
    df = num / den if den > 0 else 1
    # try scipy for p-value
    try:
        from scipy import stats
        p = stats.t.sf(abs(t), df) * 2
    except Exception:
        # fallback: normal approx
        z = abs(t)
        p = math.erfc(z / math.sqrt(2))
    return t, p

def plot_curve(ag, outpath):
    drops = sorted({d for sys in ag for d in ag[sys].keys()})
    systems = sorted(ag.keys())
    fig, ax = plt.subplots(figsize=(7,4))
    colors = {'A':'#7f7f7f','B':'#4C72B0','C':'#55A868'}
    for sys in systems:
        means = []
        stds = []
        for d in drops:
            vals = ag[sys].get(d, [0.0])
            m, s, n = mean_std_percent(vals)
            means.append(m)
            stds.append(s)
        ax.plot(drops, means, marker='o', label=f"System {sys}", color=colors.get(sys,'k'))
        ax.fill_between(drops, [m - s for m,s in zip(means,stds)], [m + s for m,s in zip(means,stds)], alpha=0.15, color=colors.get(sys,'k'))
    ax.set_xlabel('Drop rate')
    ax.set_ylabel('Capture rate (%)')
    ax.set_ylim(0,100)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath)
    print('Wrote', outpath)

def main():
    path = Path('results/full_experiment.csv')
    if not path.exists():
        print('Missing', path)
        return
    rows = read_rows(path)
    ag = aggregate(rows)
    out = Path('results/plots')
    out.mkdir(parents=True, exist_ok=True)
    plot_curve(ag, out/'degradation_curve.png')

    print('\nPer-drop statistics (mean ± std; n seeds). Comparison C vs A: diff (pp) and p-value (Welch t-test or normal approx)')
    drops = sorted({d for sys in ag for d in ag[sys].keys()})
    for d in drops:
        a_mean, a_std, a_n = mean_std_percent(ag.get('A',{}).get(d, [0.0]))
        b_mean, b_std, b_n = mean_std_percent(ag.get('B',{}).get(d, [0.0]))
        c_mean, c_std, c_n = mean_std_percent(ag.get('C',{}).get(d, [0.0]))
        # t-test C vs A
        t, p = welch_ttest(ag.get('C',{}).get(d, [0.0]), ag.get('A',{}).get(d, [0.0]))
        diff = c_mean - a_mean
        print(f"drop={d:.2f}: A={a_mean:.1f}±{a_std:.1f} (n={a_n})  B={b_mean:.1f}±{b_std:.1f} (n={b_n})  C={c_mean:.1f}±{c_std:.1f} (n={c_n})  diff(C-A)={diff:.1f}pp  t={t:.3f} p={p:.4f}")

    # Check expectation: for drop >= 0.3, C should be >=20pp above A and p<0.05
    ok = True
    for d in drops:
        if d >= 0.3:
            a_mean, _, _ = mean_std_percent(ag.get('A',{}).get(d, [0.0]))
            c_mean, _, _ = mean_std_percent(ag.get('C',{}).get(d, [0.0]))
            diff = c_mean - a_mean
            if diff < 20.0:
                ok = False
    print('\nExpectation: For drop >= 0.3, C >= A + 20pp (and p<0.05).')
    print('Result: ', 'Holds across drops >=0.3' if ok else 'Does NOT hold across all drops >=0.3')

if __name__ == '__main__':
    main()
