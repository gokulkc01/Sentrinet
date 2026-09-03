import torch
import csv
from pathlib import Path
from collections import defaultdict

def scan_checkpoints():
    """Scan all checkpoint directories and extract metadata."""
    ckpt_dir = Path('checkpoints')
    results = defaultdict(list)

    for folder in sorted(ckpt_dir.iterdir()):
        if not folder.is_dir():
            continue
        
        # Find latest checkpoint
        pt_files = list(folder.glob('*.pt'))
        if not pt_files:
            continue
        
        # Sort by step number if present in filename
        import re
        def extract_step(path):
            m = re.search(r'(\d+)', path.stem)
            return int(m.group(1)) if m else 0
        
        latest_pt = max(pt_files, key=extract_step)
        
        try:
            ckpt = torch.load(latest_pt, map_location='cpu')
            config = ckpt.get('config', {})
            step = ckpt.get('step', extract_step(latest_pt))
            policy_state = ckpt.get('policy_state_dict', {})
            
            # Extract key config fields
            system = config.get('run_name', folder.name).split('_')[0].upper()
            seed = config.get('seed', '?')
            policy_type = config.get('policy_type', 'unknown')
            obs_dim = config.get('obs_dim', '?')
            
            results[system].append({
                'folder': folder.name,
                'path': str(latest_pt),
                'step': step,
                'seed': seed,
                'policy_type': policy_type,
                'obs_dim': obs_dim,
                'run_name': config.get('run_name', '?'),
            })
        except Exception as e:
            print(f"  Warning: could not load {latest_pt}: {e}")
    
    return results

def find_best_from_csv():
    """Parse results/full_experiment.csv to find best performance per system."""
    csv_path = Path('results/full_experiment.csv')
    if not csv_path.exists():
        return {}
    
    best = defaultdict(lambda: {'capture_rate': 0, 'mean_reward': float('-inf'), 'count': 0})
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sys = row['system']
            seed = row['seed']
            drop_rate = float(row.get('drop_rate', 0))
            cap_rate = float(row.get('capture_rate', 0))
            mean_rew = float(row.get('mean_reward', 0))
            
            # aggregate: at drop_rate=0 (clean), which system/seed performs best?
            if drop_rate == 0.0:
                key = f"{sys}_seed{seed}"
                avg_perf = cap_rate + (mean_rew / 100.0)  # simple aggregate
                
                if avg_perf > best[key]['capture_rate']:
                    best[key]['capture_rate'] = cap_rate
                    best[key]['mean_reward'] = mean_rew
                    best[key]['count'] += 1
    
    return best

def main():
    print("=" * 80)
    print("SCANNING CHECKPOINTS")
    print("=" * 80)
    
    results = scan_checkpoints()
    
    print("\nBEST CHECKPOINTS PER SYSTEM:\n")
    for system in sorted(results.keys()):
        print(f"\n{system}:")
        print("-" * 70)
        for ckpt in sorted(results[system], key=lambda x: x['step'], reverse=True):
            print(f"  Path: {ckpt['path']}")
            print(f"  Folder: {ckpt['folder']}")
            print(f"  Step: {ckpt['step']:>10d}  |  Seed: {ckpt['seed']}  |  Policy: {ckpt['policy_type']}  |  obs_dim: {ckpt['obs_dim']}")
            print()
    
    # Try to add performance info from CSV
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY (from results/full_experiment.csv)")
    print("=" * 80)
    
    csv_path = Path('results/full_experiment.csv')
    if csv_path.exists():
        perf = defaultdict(lambda: {'capture_mean': 0, 'capture_std': 0, 'reward_mean': 0, 'reward_std': 0, 'count': 0})
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sys = row['system']
                if float(row.get('drop_rate', 0)) == 0.0:  # clean conditions
                    cap = float(row['capture_rate'])
                    rew = float(row['mean_reward'])
                    perf[sys]['capture_mean'] += cap
                    perf[sys]['reward_mean'] += rew
                    perf[sys]['count'] += 1
        
        print("\nAt p_drop=0 (clean conditions):\n")
        for sys in sorted(perf.keys()):
            if perf[sys]['count'] > 0:
                avg_cap = perf[sys]['capture_mean'] / perf[sys]['count']
                avg_rew = perf[sys]['reward_mean'] / perf[sys]['count']
                print(f"  {sys}: avg capture={avg_cap:.1f}%  |  avg reward={avg_rew:.2f}  |  (n={perf[sys]['count']} rows)")
    else:
        print("\n  results/full_experiment.csv not found (run evaluations first)")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATION:")
    print("=" * 80)
    print("""
To view the best checkpoint for each system interactively:

  # System A
  python dashboard.py --checkpoint checkpoints\\system_A_gru_seed1\\step_1001472.pt --pybullet

  # System B
  python dashboard.py --checkpoint checkpoints\\system_B_gru_seed1\\step_1001472.pt --pybullet

  # System C
  python dashboard.py --checkpoint checkpoints\\system_C_gru_seed1\\step_1001472.pt --pybullet

To get detailed stats on specific checkpoints:

  python scripts\\inspect_ckpt.py <checkpoint_path>
""")
    print("=" * 80)

if __name__ == '__main__':
    main()
