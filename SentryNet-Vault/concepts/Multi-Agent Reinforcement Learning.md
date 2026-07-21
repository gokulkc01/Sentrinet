# Multi-Agent Reinforcement Learning (MARL)

**Definition:** Reinforcement learning where multiple agents learn policies simultaneously in a shared environment, so each agent's optimal behavior depends on what the others are doing.

## Why it's harder than single-agent RL
- **Non-stationarity:** from any one agent's view, the environment keeps changing because the *other* agents are also learning. The "rules" move under your feet.
- **Credit assignment:** when the team succeeds, which agent's actions deserve the reward? Hard to disentangle.
- **Partial observability:** each agent usually sees only its local slice of the world, not the full state.
- **Coordination:** agents must learn to cooperate (or compete) without a central controller at execution time.

## Key paradigm: CTDE
**Centralized Training, Decentralized Execution.** During training you may use global information (e.g. a critic that sees everyone's state); at execution each agent acts on only its **own local observation**. This gives stable learning *and* deployable decentralized policies — critical for real drone swarms with no central brain.

## In SentryNet
- 3 hunter drones (learning agents) + 1 intruder (scripted) + 1 sensor (currently hard-coded, see [[Observation and Action Spaces]]).
- Uses [[MAPPO]], a CTDE actor-critic method: a **shared policy** each drone runs locally, and a **centralized critic** used only in training.
- Partial observability is real here: a drone only "sees" the intruder inside a field-of-view cone, which is exactly why teammate communication — and [[Trust and Reputation]] — matters.

## Related
- [[MAPPO]] · [[PPO and GAE]] · [[Robust Statistics and Consensus]] · [[System Architecture]]
