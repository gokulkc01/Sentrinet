# Plausibility-Based Trust (the core innovation)

**Definition:** Instead of scoring a sender by "how close its message is to my own estimate" (the fragile EMA approach), score it by **how physically plausible its claim is** — using signals that remain available even when the receiver is blind.

## Why the old approach fails
EMA trust ([[Trust and Reputation]]) needs a reference, and its only reference is the receiver's own sensor reading — available precisely when messages *aren't* needed. When blind, it can't detect spoofing. Empirically this makes trust a packet-loss meter, not an adversary detector. See [[Does Trust Actually Help]].

## The four plausibility signals
For each received message, compute residual features:

1. **Kinematic feasibility** — is the claimed change in position achievable given the platform's max speed/acceleration? Teleporting = implausible.
2. **RF-ranging cross-check** — compare the distance *implied* by two drones' claimed positions against the distance the **radio measures** (UWB time-of-flight). **The killer signal:** GPS can be spoofed, radio physics cannot.
3. **Consensus residual** — deviation from the trust-weighted fused estimate of all other senders (meaningful only at swarm scale — see [[Robust Statistics and Consensus]]).
4. **Temporal self-consistency** — the innovation sequence of a single sender over time; catches *slow-drift* spoofing that instantaneous checks miss.

## The architecture
A small GRU over these residual features → a per-sender trust score, trained **supervised** on simulated attacks (the sim gives ground-truth spoof labels). This is modular, independently testable, and verifiable — the design a defense reviewer accepts. See [[ADR-003 - Supervised Plausibility Trust]].

## Why it's better on all three project goals
- **Realistic:** uses signals real hardware actually has (IMU limits, UWB ranging).
- **Innovative:** learned physical-plausibility trust for GPS-denied cooperative tracking is under-explored.
- **Useful:** works when blind — i.e. exactly when trust is supposed to matter.

## Related
- [[Trust and Reputation]] · [[GPS Spoofing and GNSS Denial]] · [[Robust Statistics and Consensus]] · [[ADR-003 - Supervised Plausibility Trust]]
