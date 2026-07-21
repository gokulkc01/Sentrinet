# Sim-to-Real Transfer

**Definition:** Getting a policy trained in simulation to work on real hardware, despite the "reality gap" between the simulator and the physical world.

## The main techniques
- **Domain randomization:** randomize sim parameters (mass, wind, sensor noise, latency) during training so the policy learns a *robust* behavior that spans the real value too. SentryNet already does light domain randomization ([[Environment - BorderEnv]]).
- **System identification:** measure real hardware parameters and calibrate the sim to match.
- **Right abstraction level:** don't learn what the hardware already does well.
- **Hardware-in-the-loop (HITL):** run the real flight controller / real radios in the loop with the sim.

## The biggest gap in SentryNet today
The policy outputs **raw 3D thrust**. Real drones are commanded through a **cascaded controller** (position → velocity → attitude → motor). Learning raw thrust is both unrealistic and brittle to transfer.

**Fix (Stage 3):** the policy should output **high-level velocity / waypoint setpoints** fed to a real autopilot (PX4 / Crazyswarm), not motor forces. This is the single most important sim-to-real change.

## Why the RF-ranging trust signal matters here
The [[Plausibility-Based Trust]] design leans on **UWB time-of-flight ranging** — which is a real sensor on the **Crazyflie Loco deck**. So the trust mechanism isn't a sim-only trick; it maps directly onto affordable hardware, making the sim-to-real story coherent instead of decorative.

## The realistic hardware target
Not an outdoor defense swarm (multi-year, multi-person). A **Crazyflie indoor swarm** with motion capture, real UWB radios, and software-injected GPS spoofing. Small, real, filmed — proof of concept, not product. See [[Roadmap]] Stage 3.

## Related
- [[Environment - BorderEnv]] · [[Plausibility-Based Trust]] · [[Roadmap]] · [[ADR-004 - Terrain Occlusion Only]]
