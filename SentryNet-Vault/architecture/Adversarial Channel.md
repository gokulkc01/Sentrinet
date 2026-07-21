# Adversarial Channel

**File:** `adversarial_channel.py` · **Class:** `AdversarialChannel`

Simulates the unreliable/attacked link between drones. This is Layer 2 of the [[System Architecture]] and the mechanism behind [[Adversarial Communication]].

## What it does
`transmit(messages)` takes an array of outgoing messages `[x,y,z,vx,vy,vz]` and returns `(received, dropped_mask)`:
1. **Drop:** with prob `p_drop`, the message → zero vector, `dropped_mask=True`, skip spoof.
2. **Spoof:** otherwise, with prob `p_spoof`, add Gaussian noise `N(0, spoof_std)`.

Tracks stats (`empirical_drop_rate`, `empirical_spoof_rate`) and exposes `set_drop_rate` / `set_spoof_rate` for evaluation sweeps.

## How the environment uses it
In [[Environment - BorderEnv|_comms_pipeline]], per sender→receiver pair:
- Distance-dependent drop: `effective_drop = min(0.95, p_drop + 0.025·dist)`.
- The **compromised drone** (adversary) overrides its message with a mirror-image fake position `2·own_pos − intruder_pos` and negated velocity — a "targeted adversary" with intruder knowledge.

## Constants
- `MAX_SPOOF_ERROR = 5.0` — the normalization constant [[Trust Module and Aggregator|TrustModule]] uses for its accuracy signal.

## Limitations (drives the pivot)
This is a *statistical toy*: independent Bernoulli events + Gaussian noise. Real channels have path loss, latency, terrain shadowing, and structured attacks ([[GPS Spoofing and GNSS Denial]], replay, Sybil). Stage 1 replaces this with a realistic model + named [[Threat Scenarios]].

## Related
- [[Adversarial Communication]] · [[Trust Module and Aggregator]] · [[Threat Scenarios]] · [[GPS Spoofing and GNSS Denial]]
