# 🛰️ SentryNet — Knowledge Vault

Your single place to understand this project end to end: the **theory**, the **code**, the **plan**, and the **decisions** behind every choice. Open this folder in Obsidian (`Open folder as vault`) and use the graph view — everything is interlinked.

> **One-line project thesis:** **Cooperative integrity monitoring for GNSS-denied drone
> swarms** — detect and isolate a spoofed node from physics that cannot be forged,
> packaged as a reusable benchmark. *(Reframed 2026-09-04; see
> [[ADR-011 - Pivot to Cooperative Integrity Monitoring]].)*

---

## 🚦 Where the project stands right now

- **v3, phase 0** — the premise test. Canonical plan: `docs/DESIGN-v3.md`; summary in
  [[Roadmap]]. Branch `sentrinet-v3-integrity`; v2 frozen on `sentrinet-v2-stage0`.
- The original claim ("EMA trust beats no-trust under packet loss") **does not
  replicate** — see [[Does Trust Actually Help]]. v3 supplies the *structural* reason:
  at N=3 a perfectly consistent lie always exists, so the problem was unsolvable rather
  than badly implemented.
- **ADR-005 → 010 made the v2 testbed learn** (~44% capture, 3 seeds × 1M steps) — and
  in doing so proved the RL is not the path on this hardware. That work is frozen, not
  discarded.
- Phase 0 has already validated the core statistic: the per-node χ² test is exactly
  chi-squared (mean 8.054 vs 8, KS p = 0.691).

## 🧭 How to navigate

**Start with the concepts, then the architecture, then the plan.**

### 1. Technical foundations (the "why it works" layer)
- [[Multi-Agent Reinforcement Learning]] — the paradigm
- [[MAPPO]] — the specific algorithm we use
- [[PPO and GAE]] — the optimization underneath MAPPO
- [[Trust and Reputation]] — the security idea
- [[Adversarial Communication]] — the attack surface
- [[GPS Spoofing and GNSS Denial]] — the real threat we pivot toward
- [[Plausibility-Based Trust]] — our core innovation
- [[Robust Statistics and Consensus]] — why swarm size matters
- [[Sim-to-Real Transfer]] — the path to hardware

### 2. The codebase (the "how it's built" layer)
- [[System Architecture]] — the five layers + data flow (**start here**)
- [[Environment - BorderEnv]] · [[Adversarial Channel]] · [[Trust Module and Aggregator]]
- [[Networks and Rollout Buffer]] · [[MAPPO Trainer]] · [[Observation and Action Spaces]]

### 3. The plan (the "what next" layer)
- [[Roadmap]] — the four stages (canonical detail in `docs/DESIGN.md`)
- [[Controlled Experiment]] · [[Threat Scenarios]] · [[Metrics]]

### 4. The decisions (the "why we chose this" layer)
- [[Decision Log]] — every non-obvious call, with rationale
- [[Does Trust Actually Help]] — the empirical analysis that drove the pivot
- [[Known Bugs and Confounds]] — what's broken and why it matters

### 5. Learning the field (go deep)
- [[Learning Roadmap]] — a staged curriculum (RL → deep RL → MARL → drones → security) tied directly to this code

---

## 📌 The mental model in five sentences

1. Multiple drones cooperatively hunt an intruder in a bounded airspace using [[MAPPO]].
2. They share noisy intruder estimates over a channel an adversary can attack ([[Adversarial Communication]]).
3. A **trust** layer is supposed to let them ignore bad senders without cryptography ([[Trust and Reputation]]).
4. The current trust design fails because it can't tell an adversary from ordinary packet loss ([[Does Trust Actually Help]]).
5. The fix is to redesign trust around **physical plausibility** under a realistic **GPS-spoofing** threat, at a swarm size where consensus actually works ([[Plausibility-Based Trust]], [[Robust Statistics and Consensus]]).

---
*Vault maintained alongside the code. When a decision changes, update the relevant [[Decision Log]] entry and the note it affects.*
