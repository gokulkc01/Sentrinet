# ADR-001 — Pivot to GPS-Spoofing + Plausibility Trust

**Status:** ✅ Accepted

## Context
The original thesis was "a lightweight EMA trust mechanism improves robustness to packet loss/spoofing **without cryptography**." Two problems: (1) the empirical result doesn't hold ([[Does Trust Actually Help]]), and (2) "no cryptography" reads as a *weakness* to a defense audience, not a feature — they can afford message authentication.

## Decision
Reframe the project around **resilient cooperative tracking under [[GPS Spoofing and GNSS Denial|GPS spoofing]]**, using **[[Plausibility-Based Trust]]** — trust as a *sensor-integrity* layer, complementary to (not a replacement for) cryptography.

## Rationale
- GPS spoofing is *the* dominant real drone threat and directly defense-relevant.
- **Cryptography structurally cannot stop it:** a spoofed GNSS signal corrupts a *sensor input*, not a *peer message*. A node honestly broadcasts a false self-belief. This is the exact gap a plausibility/trust layer legitimately fills.
- It converts a weak, crowded research angle into a sharp, under-explored, useful one.

## Consequences
- The [[Adversarial Channel]] Gaussian-spoof toy is replaced by named [[Threat Scenarios]].
- Trust is redesigned ([[ADR-003 - Supervised Plausibility Trust]]).
- Requires swarm scale-up ([[ADR-002 - Scale to 9 Drones]]) for consensus to work.
- "Without cryptography" is dropped from the framing.

## Related
- [[GPS Spoofing and GNSS Denial]] · [[Plausibility-Based Trust]] · [[Trust and Reputation]]
