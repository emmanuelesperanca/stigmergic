"""Agent castes of StigmergicAI: autonomous ants that react to the environment.

This subpackage holds the threaded base castes every concrete ant inherits from:
:class:`ConsumerAnt` (sense, claim, resolve) and :class:`ProducerAnt` (secrete
chaos), both built on :class:`BaseAnt`.

The NLI-backed ``verifier_ant`` is *not* re-exported here, since it pulls in the
transformers stack; import it explicitly when running Byzantine consensus.
"""

from stigmergic_ai.agents.base_ant import (
    BaseAnt,
    ConsumerAnt,
    Mutation,
    ProducerAnt,
)

__all__ = ["BaseAnt", "ConsumerAnt", "ProducerAnt", "Mutation"]
