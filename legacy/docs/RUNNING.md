# Running SentryNet — Quickstart & Demo Commands

This file collects the exact commands to set up the environment, run training/evaluation, and launch the interactive demo (dashboard + optional PyBullet). Commands are shown for Windows PowerShell (primary) and Unix shells where appropriate.

---

## 1) Prepare Python environment (Windows PowerShell)

If the repository already contains `sentrinet_env/` you can activate it directly. Otherwise create and install a virtualenv:

```powershell
# create venv (only if you don't already have sentrinet_env)
python -m venv sentrinet_env

# activate the venv
.\sentrinet_env\Scripts\Activate.ps1

# allow the script to run if your policy blocks it (one-time per shell)
# (run as Administrator if needed)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# upgrade pip and install deps
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Unix / macOS:

```bash
python -m venv sentrinet_env
source sentrinet_env/bin/activate
pip install -r requirements.txt
```

Note: `requirements.txt` contains the primary runtime packages used by the project. If you need GPU-enabled `torch`, install the appropriate `torch` wheel for your CUDA version instead of the generic `torch` in `requirements.txt`.

---

## 2) Inspect checkpoints

```powershell
# prints config and tensor shapes for a checkpoint
python scripts\inspect_ckpt.py checkpoints\system_C_gru_seed1\step_1001472.pt
```

---

## 3) Run a quick deterministic evaluation (headless)

```powershell
# run 200 evaluation episodes (headless) with no domain randomization
python run_trained.py --checkpoint checkpoints\system_C_gru_seed1 --episodes 200 --p_drop 0 --p_spoof 0 --no-domain-rand
```

Replace `--checkpoint` with the path to your `.pt` file or a checkpoint directory.

---

## 4) Run a robustness sweep (example: System A across several drop rates)

PowerShell loop (example):

```powershell
foreach ($drop in @(0.0,0.2,0.4,0.6,0.8)) {
  python run_trained.py `
    --checkpoint checkpoints\system_A_gru_seed1 `
    --capture-mode sustained --sustained-steps 3 `
    --p_drop $drop --p_spoof 0.1 `
    --compromised-drone 1 --no-domain-rand --episodes 100
}
```

This produces per-run text/stat outputs in `results/` (see `scripts/plot_eval_results.py`).

---

## 5) Launch the interactive dashboard (visual demo)

Pygame dashboard with optional PyBullet overlay:

```powershell
# with PyBullet 3D overlay
python dashboard.py --checkpoint checkpoints\system_C_gru_seed1\step_1001472.pt --pybullet

# without PyBullet (pygame UI only)
python dashboard.py --checkpoint checkpoints\system_C_gru_seed1\step_1001472.pt
```

If PyBullet windows fail (physics disconnects), close the PyBullet window and re-run. See Troubleshooting below for automation tips.

---

## 6) Run the validation harness (multi-seed short training / evaluation)

```powershell
# quick validation (multiple configs / seeds; produces CSV/JSON in results/validation)
python validation_harness.py
```

---

## 7) Generate plots and analysis

```powershell
# robust single-checkpoint bar chart
python scripts\plot_eval_results.py

# aggregate full-experiment plots (reads results/full_experiment.csv)
python scripts\plot_full_experiment.py

# run the degradation analysis and hypothesis tests
python scripts\analyze_degradation.py
```

---

## 8) Common git commands to push this repo (if needed)

```powershell
# initialize (if not a git repo)
git init
git add -A
git commit -m "Initial commit"
# create remote (use gh CLI or web UI). Using gh (recommended):
gh auth login
gh repo create <OWNER>/<REPO> --public --source=. --remote=origin --push
# or add remote manually and push
# git remote add origin https://github.com/<USER>/<REPO>.git
# git push -u origin main
```

---

## 9) Troubleshooting notes

- PyBullet disconnects: the dashboard occasionally raises `pybullet.error: Not connected to physics server` on repeated resets. If this happens, close the PyBullet window and re-run the dashboard. For long demos, run without `--pybullet` or increase time between episodes.
- PowerShell redirection: when redirecting output to files, PowerShell may create UTF-16 files with null bytes. The `scripts/plot_eval_results.py` and other parsing scripts already handle UTF-16 and strip nulls.
- GPU `torch` install: if you want CUDA support, install `torch` via the official instructions (https://pytorch.org) and skip the `torch` line in `requirements.txt`.

---


# System A (replace path if your checkpoint differs)
python dashboard.py --checkpoint checkpoints\system_A_gru_seed1\step_1001472.pt --pybullet

# System B
python dashboard.py --checkpoint checkpoints\system_B_gru_seed1\step_1001472.pt --pybullet


# System C
python dashboard.py --checkpoint checkpoints\system_C_gru_seed1\step_1001472.pt --pybullet --capture-mode sustained --sustained-steps 1



python dashboard.py --checkpoint checkpoints\system_C_gru_seed1\step_1001472.pt --pybullet --intruder-profile passive


python dashboard.py --checkpoint checkpoints\system_C_gru_seed1\step_1001472.pt --pybullet