# Trust and Reputation Systems

**Definition:** Mechanisms that let agents estimate *how much to believe* other agents based on the history of their behavior, so bad actors can be down-weighted without a central authority or cryptography.

## Core idea
Each agent keeps a per-sender **trust score** τ ∈ [0,1]. When a sender's messages look good, τ rises; when they look bad or vanish, τ falls. Messages are then combined **weighted by trust**, so unreliable sources have less influence.

## The classic EMA trust rule (what SentryNet uses today)
- **Accuracy signal:** `acc = max(0, 1 − error / max_error)`, where `error` = distance between the received position and a reference.
- **On receive:** `τ = α·acc + (1−α)·τ` (exponential moving average; α≈0.1).
- **On drop:** `τ = decay·τ` (multiplicative decay; decay≈0.95).
- **Aggregate:** `agg = Σ(τ_j · msg_j) / Σ(τ_j)`.

## Why this design is fragile (the crux of the whole project)
Trust needs a **reference** to judge "good vs bad." In SentryNet that reference is the receiver's *own* sensor estimate — which only exists when the receiver already sees the target, i.e. when it *doesn't need* the messages. When blind, it can only apply drop-decay and **cannot detect spoofing at all**. Result: trust degenerates into a global packet-loss meter. See [[Does Trust Actually Help]].

## The fix
Replace "error vs my own estimate" with **[[Plausibility-Based Trust]]** — judge messages by physical consistency (kinematics, RF ranging, cross-sensor agreement) that works even when blind — under a real [[GPS Spoofing and GNSS Denial]] threat.

## Why "without cryptography" is NOT the selling point
Crypto authenticates *who sent* a message; it can't detect a node honestly reporting a *spoofed self-belief*. Trust and crypto are complementary, not alternatives. See [[ADR-001 - GPS-Spoofing Pivot]].

## Related
- [[Plausibility-Based Trust]] · [[Adversarial Communication]] · [[Trust Module and Aggregator]] · [[Robust Statistics and Consensus]]
