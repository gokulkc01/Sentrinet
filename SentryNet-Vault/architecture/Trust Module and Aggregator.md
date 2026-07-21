# Trust Module and Aggregator

**Files:** `trust_module.py` (`TrustModule`), `trust_aggregator.py` (`TrustAggregator`). Layer 3 of the [[System Architecture]] — the research core. Concept: [[Trust and Reputation]].

## TrustModule — per-sender scoring
Each drone owns one `TrustModule` tracking trust `τ ∈ [0,1]` for each *other* drone.

**`update(received_pos, reference_pos, dropped_mask)`** per sender j:
- **Dropped** → `τ_j *= decay_on_drop` (0.95).
- **Received** → build an accuracy signal from up to three sources and EMA it in (`τ = α·acc + (1−α)·τ`, α=0.1):
  1. error vs the receiver's own local estimate (`reference_pos`),
  2. error vs the **weighted-median consensus** of other senders,
  3. temporal consistency vs the sender's previous message.
- **`decay_on_drops(mask)`** — drop-decay only, used when the receiver is blind (no reference).

Constants: `EMA_ALPHA=0.1`, `DECAY_ON_DROP=0.95`, `MAX_ERROR=5.0`.

## TrustAggregator — trust-weighted fusion
**`aggregate(messages, trust_scores, dropped_mask)`** → `Σ(τ_j·msg_j)/Σ(τ_j)` over non-dropped messages; returns zeros if everything dropped or all τ=0. The result is injected into dims [6:12] of each drone's observation.

## Why it underperforms (the whole story)
- The **reference is the receiver's own estimate**, available only when *not* blind — so spoof detection fails when it's needed most.
- With **2 senders per receiver** (N=3), the "consensus" is a median of 2 → statistically meaningless. See [[Robust Statistics and Consensus]].
- Empirically, τ just tracks the drop rate instead of isolating the adversary. See [[Does Trust Actually Help]].

## The redesign
[[Plausibility-Based Trust]]: replace error-vs-own-estimate with physical-plausibility residuals (kinematics, RF ranging, consensus, temporal) → a learned per-sender score. See [[ADR-003 - Supervised Plausibility Trust]].

## Related
- [[Trust and Reputation]] · [[Plausibility-Based Trust]] · [[Does Trust Actually Help]] · [[Robust Statistics and Consensus]]
