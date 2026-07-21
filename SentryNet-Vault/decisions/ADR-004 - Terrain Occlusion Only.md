# ADR-004 — Environmental Realism: Occlusion Yes, Photoreal Terrain No

**Status:** ✅ Accepted

## Context
Desire to make the environment "realistic" (terrain, etc.). But realism can be a scope trap — the project's main risk is breadth without a verified result.

## Decision
Add the parts of terrain that **change the science**, modeled cheaply; **defer** the parts that only change the screenshot.

**In (Stage 1):** heightmap + raycast **line-of-sight occlusion** for both sensing and comms, **terrain RF shadowing** (extra path loss when LoS is blocked), and no-fly volumes.
**Deferred (Stage 3, maybe):** photorealistic terrain meshes / rendering.

## Rationale
- **Occlusion is load-bearing:** obstacles that block sensing *and* radio are the physical reason a drone goes blind and must depend on (possibly spoofed) peer data — the exact condition where [[Plausibility-Based Trust|trust]] must prove itself. It *strengthens the core experiment*, not just the visuals.
- **Photoreal terrain is a screenshot:** slow (taxes training throughput), weeks of work, and for credibility a **real Crazyflie flight video beats any simulated canyon** ([[Sim-to-Real Transfer]]).

## Consequences
- Adds a raycast LoS check and a dynamic, distance+terrain-dependent communication graph.
- Keeps simulation fast enough for large seed sweeps.
- Visual realism is explicitly a demo-stage concern, not a research-stage one.

## Related
- [[Sim-to-Real Transfer]] · [[Adversarial Communication]] · [[Roadmap]]
