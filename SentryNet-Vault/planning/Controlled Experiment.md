# Controlled Experiment (Stage 0.2)

The experiment that finally answers **"does trust help?"** honestly. The current comparison is confounded beyond repair (see [[Does Trust Actually Help]]); this fixes it.

## The invariant
The comparison is only valid if A/B/C are identical in **everything except the variable under test**.

| Held identical | Varies (the only difference) |
|---|---|
| GRU policy 128×128 · entropy 0.005 · `sustained` capture · 1M steps · **no curriculum** · same eval protocol · same seeds *(revised — see [[ADR-006 - GRU Sustained Known-Solvable Config]])* | **A:** clean training, no trust · **B:** adversarial training (drop 0.2, spoof 0.1), no trust · **C:** adversarial training + EMA trust |

> The old code violated this: System C used a **GRU + sustained capture + different entropy** while A/B used MLP + team. Any C-vs-A/B gap was uninterpretable. See [[Known Bugs and Confounds]].

## Protocol
- **8 seeds** per system (the docs themselves admit 3 is too few).
- **Eval:** 200 episodes × drop ∈ {0.0…0.8} × spoof 0.1 with a compromised drone, identical for all systems.
- **Stats:** bootstrap 95% CIs on capture rate; Welch's t-test on **C−B** per drop rate; report effect sizes. Log **per-step trust traces** for real dynamics plots (not synthetic).

## The decision gate
- **C > B (significant):** EMA trust helps even at N=3 → keep as a Stage-1 baseline.
- **C ≤ B:** honest negative result → the motivation for [[Plausibility-Based Trust]]. (This is what current data predicts.)

## Why this is scientifically valuable either way
A rigorous negative result is more credible — and more useful — than a broken positive one. It also gives Stage 1 a clean baseline to beat.

## Related
- [[Does Trust Actually Help]] · [[Metrics]] · [[MAPPO Trainer]] · [[Roadmap]]
