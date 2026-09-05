import pandas as pd
from pathlib import Path
import numpy as np

fn = Path('results/trust_grid.csv')
if not fn.exists():
    print('No file', fn)
    raise SystemExit(1)

df = pd.read_csv(fn)
summary = []
for (alpha, mm), g in df.groupby(['alpha','maxerr_mult']):
    capture_mean = g['capture_rate'].mean()
    capture_std = g['capture_rate'].std()
    trust_mean = g['mean_trust'].mean()
    trust_std = g['mean_trust'].std()
    rew_mean = g['mean_reward'].mean()
    rew_std = g['mean_reward'].std()
    steps_mean = g['mean_steps'].mean()
    steps_std = g['mean_steps'].std()
    summary.append((alpha, mm, capture_mean, capture_std, trust_mean, trust_std, rew_mean, rew_std, steps_mean, steps_std))

print('alpha,maxerr_mult,capture_mean,capture_std,trust_mean,trust_std,mean_reward,rew_std,mean_steps,steps_std')
for row in summary:
    print(','.join(map(str,row)))

# Save aggregated CSV
out = Path('results/trust_grid_summary.csv')
agg = pd.DataFrame(summary, columns=['alpha','maxerr_mult','capture_mean','capture_std','trust_mean','trust_std','mean_reward','rew_std','mean_steps','steps_std'])
agg.to_csv(out,index=False)
print('Saved', out)
