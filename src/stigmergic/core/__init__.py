"""Core substrate of Stigmergic: the shared field the swarm lives in.

This subpackage holds the Horizon 1 environment -- the :class:`PheromoneGround`
and its value types. It is intentionally free of any deep-learning dependency so
that the substrate stays fast and importable everywhere.

The cognitive modules (``consensus`` for Byzantine quorum, ``latent_transfer``
for tensor injection) are *not* re-exported here; import them explicitly when you
opt into the heavier horizons.
"""

from stigmergic.core.backends import create_ground
from stigmergic.core.environment import (
    AbstractGround,
    Entropy,
    EventSignal,
    GroundEvent,
    Pheromone,
    PheromoneGround,
    Status,
)
from stigmergic.core.observability import SwarmInspector

__all__ = [
    "AbstractGround",
    "PheromoneGround",
    "Pheromone",
    "Status",
    "Entropy",
    "GroundEvent",
    "EventSignal",
    "create_ground",
    "SwarmInspector",
]
