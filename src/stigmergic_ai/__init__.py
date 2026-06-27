"""StigmergicAI: a zero-dependency, mathematically secure multi-agent framework.

StigmergicAI models a multi-agent system as a physical ecosystem rather than an
org chart. Agents ("ants") never call one another; they coordinate solely by
reading and mutating a shared semantic field -- the :class:`PheromoneGround`.
Work is represented as entropy in that field, and the swarm drives it to zero.

This top-level package eagerly exposes only the lightweight Horizon 1 primitives
(the environment and the base ant castes). The deep-learning horizons -- Byzantine
Cognitive Consensus and Latent State Transfer -- live in submodules that are
imported on demand, so ``import stigmergic_ai`` never pays the torch import tax.
"""

from stigmergic_ai.agents.base_ant import (
    BaseAnt,
    ConsumerAnt,
    Mutation,
    ProducerAnt,
)
from stigmergic_ai.core.environment import (
    Entropy,
    Pheromone,
    PheromoneGround,
    Status,
)

__version__ = "0.1.0"

__all__ = [
    # Environment (the Pheromone Ground)
    "PheromoneGround",
    "Pheromone",
    "Status",
    "Entropy",
    # Agent castes
    "BaseAnt",
    "ConsumerAnt",
    "ProducerAnt",
    "Mutation",
    # Metadata
    "__version__",
]
