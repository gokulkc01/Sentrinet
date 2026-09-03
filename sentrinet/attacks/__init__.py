"""Threat models, from naive offsets to constraint-aware spoofing."""
from sentrinet.attacks.adaptive import (
    attack_margin_curve,
    best_lie_at_displacement,
    consistency_cost,
)

__all__ = [
    "consistency_cost",
    "best_lie_at_displacement",
    "attack_margin_curve",
]
