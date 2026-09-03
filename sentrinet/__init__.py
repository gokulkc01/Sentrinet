"""
SentryNet — cooperative integrity monitoring for GNSS-denied drone swarms.

A node's GNSS-derived self-position can be spoofed; the time-of-flight of the
radio link between two nodes cannot. Every detector in this package is built on
that asymmetry: a node's *claimed* position must remain consistent with the
ranges its peers physically measure to it.

See docs/DESIGN-v3.md for the research plan and the vault's ADR-011 for why the
project pivoted here from trust-aware MARL.
"""

__version__ = "0.1.0.dev0"
