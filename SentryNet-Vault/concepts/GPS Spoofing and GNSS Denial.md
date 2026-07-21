# GPS Spoofing and GNSS Denial

**Definition:** Attacks on a drone's satellite navigation. **Jamming** drowns out GNSS signals so the drone can't get a fix (denial); **spoofing** transmits counterfeit signals so the drone computes a *false but confident* position.

## Why this is the right threat to build around
- It is **the dominant real-world drone threat**, and a live issue on contested borders — directly relevant to a defense audience.
- **Cryptography can't stop it.** A spoofed GNSS signal isn't a forged *message* from a peer; it's a corrupted *sensor input*. Message authentication is irrelevant. This is exactly the gap a [[Trust and Reputation|trust]] / plausibility layer can fill — see [[ADR-001 - GPS-Spoofing Pivot]].
- It reframes "trust" from a weak "no-crypto message filter" into a legitimate **sensor-integrity** problem.

## The key insight for our design
A spoofed drone **honestly reports a false self-belief**. It's not lying maliciously at the protocol level — it genuinely thinks it's somewhere it isn't. So the detector can't look for "malformed" or "unauthenticated" messages; it must look for **physical implausibility**:
- Does the claimed motion violate the platform's speed/acceleration limits?
- Does the claimed position disagree with what the **radio itself measures** (RF time-of-flight ranging)? *GPS can be spoofed; the physics of the radio link cannot.*
- Does it disagree with the swarm consensus?

That's [[Plausibility-Based Trust]].

## In SentryNet (planned, Stage 1)
- Becomes the primary threat, replacing abstract Gaussian spoofing.
- Attack scenarios S1–S4 in [[Threat Scenarios]] are all GPS-spoof variants (constant offset, slow drift, coordinated, intermittent).
- The **RF-ranging cross-check** is the killer signal and maps to real hardware (Crazyflie UWB deck) — see [[Sim-to-Real Transfer]].

## Related
- [[Plausibility-Based Trust]] · [[Adversarial Communication]] · [[Threat Scenarios]] · [[ADR-001 - GPS-Spoofing Pivot]]
