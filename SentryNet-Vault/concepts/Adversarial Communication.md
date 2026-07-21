# Adversarial Communication

**Definition:** The threat model in which the messages agents exchange can be lost, delayed, or manipulated by an adversary — breaking cooperation that assumes a reliable channel.

## The two attacks modeled today
1. **Packet drop (jamming/loss):** with probability `p_drop`, a message never arrives. The receiver gets nothing.
2. **Spoofing (false data):** with probability `p_spoof`, additive Gaussian noise corrupts the message coordinates. A stronger variant: a **compromised drone** deliberately reports an adversarial position.

## Why it breaks cooperation
Cooperative control assumes teammates share honest state. If a teammate's "intruder is over here" is wrong, the swarm chases a ghost. Packet loss alone degrades coordination; targeted spoofing can actively mislead it.

## Realistic threats we should model (the pivot)
Bernoulli-drop + Gaussian-spoof is a toy. Real contested airspace involves:
- **[[GPS Spoofing and GNSS Denial]]** — the dominant real threat.
- RF **jamming** (barrage, reactive, protocol-aware).
- **Replay** attacks, **Sybil** nodes, coordinated multi-adversary.
- Distance/terrain-dependent loss (path loss, [[Sim-to-Real Transfer|LoS occlusion]]).

See [[Threat Scenarios]] for the named, reproducible scenarios (S0–S4).

## In SentryNet
- Implemented in [[Adversarial Channel]] (`AdversarialChannel`).
- The environment applies distance-dependent drop and a targeted "compromised drone" adversary in [[Environment - BorderEnv|border_env._comms_pipeline]].

## Related
- [[Trust and Reputation]] · [[GPS Spoofing and GNSS Denial]] · [[Adversarial Channel]] · [[Threat Scenarios]]
