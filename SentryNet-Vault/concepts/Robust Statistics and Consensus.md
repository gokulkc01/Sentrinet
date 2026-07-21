# Robust Statistics and Consensus

**Definition:** Methods for combining many measurements so that a few corrupted ones (outliers / adversaries) don't dominate the result. The classical bar any trust mechanism must beat.

## Key tools
- **Median / trimmed mean:** ignore the extremes; robust to a minority of bad values. A median tolerates up to 50% corruption in theory.
- **Weighted median:** the median where each point carries a weight (here, trust). Used in the current [[Trust Module and Aggregator|TrustModule]] consensus check.
- **Breakdown point:** the fraction of corrupted data a method tolerates before its output can be made arbitrarily wrong.

## Why swarm SIZE is decisive
Robust statistics need **enough honest samples**. With **N=3 drones**, each receiver has only **2 senders** — a "median of 2" is meaningless and a single adversary is 50% of the data. Consensus simply cannot work. This is a core reason the current results don't support the trust claim (see [[Does Trust Actually Help]]).

At **N=9** each receiver has **8 senders**, so:
- The consensus residual becomes statistically meaningful.
- One or two spoofed nodes are a clear minority → robust fusion can isolate them.
- We can sweep *k* spoofed nodes (1…⌈N/2⌉) to find the **breaking point** — a real, publishable result.

## In SentryNet
- Scaling to N=9 is [[ADR-002 - Scale to 9 Drones]].
- The consensus residual is signal #3 in [[Plausibility-Based Trust]].
- **Trimmed-mean fusion** is one of the baselines the learned method must beat — see [[Metrics]].

## Related
- [[Plausibility-Based Trust]] · [[Trust and Reputation]] · [[Multi-Agent Reinforcement Learning]] · [[ADR-002 - Scale to 9 Drones]]
