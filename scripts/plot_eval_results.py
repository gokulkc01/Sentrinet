import re
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

res_dir = Path('results')
files = {
    'system_A_gru_seed1': res_dir/'eval_system_A_gru_seed1.txt',
    'system_B_gru_seed1': res_dir/'eval_system_B_gru_seed1.txt',
    'system_C_gru_seed1': res_dir/'eval_system_C_gru_seed1.txt',
}

data = {}
for name, p in files.items():
    if not p.exists():
        print('Missing', p)
        continue
    # Read as bytes and try UTF-8, then UTF-16 (LE/BE) if null bytes present
    raw = p.read_bytes()
    txt = None
    try:
        txt = raw.decode('utf-8')
    except Exception:
        try:
            txt = raw.decode('utf-16')
        except Exception:
            try:
                txt = raw.decode('utf-16-le')
            except Exception:
                txt = raw.decode('latin-1', errors='ignore')
    # Strip null characters that sometimes appear in redirected output
    txt = txt.replace('\x00', '')
    m = re.search(r'Capture rate\s*:\s*([0-9]+\.?[0-9]*)%\s*\((\d+)/(\d+)\)', txt)
    if m:
        rate = float(m.group(1))
        num = int(m.group(2))
        den = int(m.group(3))
        data[name] = (rate, num, den)
    else:
        # try alternate summary line
        m2 = re.search(r'Capture rate\s*:\s*([0-9]+\.?[0-9]*)%', txt)
        if m2:
            rate = float(m2.group(1))
            data[name] = (rate, None, None)
        else:
            data[name] = (0.0, None, None)

# Plot
names = list(data.keys())
rates = [data[n][0] for n in names]

fig, ax = plt.subplots(figsize=(6,4))
ax.bar(names, rates, color=['#4C72B0','#55A868','#C44E52'])
ax.set_ylim(0,100)
ax.set_ylabel('Capture rate (%)')
ax.set_title('Deterministic eval capture rates (300 eps, no attack)')
for i, v in enumerate(rates):
    ax.text(i, v+1, f"{v:.1f}%", ha='center')

out = Path('results/plots')
out.mkdir(parents=True, exist_ok=True)
plot_path = out/'capture_rates_systems.png'
fig.tight_layout()
fig.savefig(plot_path)
print('Wrote', plot_path)
