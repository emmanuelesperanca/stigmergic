"""The Verifier: the caste that enforces Byzantine consensus on the ground.

A :class:`VerifierAnt` is the immune system of the swarm. It smells the
``PENDING_CONSENSUS`` trail -- proposals a Solver has staged but not committed --
claims one, and submits it to the :class:`~stigmergic.core.consensus.SemanticRaft`
quorum. The quorum's ruling is final and binary:

* **Quorum passes** -> the proposal was logically entailed by the request. The
  verifier finalizes it: ``entropy -> 0.0``, ``status -> RESOLVED``.
* **Quorum fails** -> the proposal is a hallucination or injection. The verifier
  *slashes* it: ``entropy -> 0.0``, ``status -> SLASHED``. Entropy still drops to
  zero so the poisoned trail evaporates and no other ant ever follows it.

Like every ant, the verifier coordinates only through the environment. It never
calls the Solver that produced the proposal, and the Solver never waits on it.
The verifier simply reacts to a chemical trail and lays a terminal one.

The NLI model is loaded lazily and shared process-wide (see
:mod:`stigmergic.core.consensus`), so constructing a verifier -- or a whole
swarm of them -- is cheap and torch-free until the first real verdict.
"""

from __future__ import annotations

from stigmergic.agents.base_ant import ConsumerAnt, Mutation
from stigmergic.core.consensus import (
    DEFAULT_NLI_MODEL,
    NLIJudge,
    SemanticRaft,
)
from stigmergic.core.environment import (
    Entropy,
    Pheromone,
    PheromoneGround,
    Status,
)

__all__ = ["VerifierAnt"]


class VerifierAnt(ConsumerAnt):
    """A consumer caste that runs Semantic Raft over pending proposals.

    Args:
        env: The shared Pheromone Ground.
        name: Identifier / ``owner`` stamp. Defaults to ``"VerifierAnt"``.
        raft: A pre-built :class:`SemanticRaft`. If omitted, one is created from
            the remaining arguments (using a lazily-loaded real NLI model unless
            ``judge`` is supplied).
        judge: An :class:`NLIJudge` to back a default quorum -- pass a
            ``MockNLIJudge`` here to run entirely offline in tests.
        model_name: NLI model to load when neither ``raft`` nor ``judge`` is given.
        quorum_size: Number of juror nodes (when building the default raft).
        approval_threshold: ``ENTAILMENT`` votes required to pass.
        target_status: The trail this caste reacts to (default
            :attr:`Status.PENDING_CONSENSUS`).
        proposal_key: Metadata key under which the Solver stored its proposal.
        poll_interval: Seconds between heartbeats.
    """

    def __init__(
        self,
        env: PheromoneGround,
        name: str | None = None,
        *,
        raft: SemanticRaft | None = None,
        judge: NLIJudge | None = None,
        model_name: str = DEFAULT_NLI_MODEL,
        quorum_size: int = 3,
        approval_threshold: int = 2,
        target_status: Status | str = Status.PENDING_CONSENSUS,
        proposal_key: str = "proposal",
        poll_interval: float = 0.5,
    ) -> None:
        super().__init__(
            env,
            name,
            entropy_threshold=Entropy.MIN,
            target_status=target_status,
            claimed_status=Status.CLAIMED,
            poll_interval=poll_interval,
        )
        self.proposal_key = proposal_key
        self.raft = raft if raft is not None else SemanticRaft(
            judge=judge,
            model_name=model_name,
            quorum_size=quorum_size,
            approval_threshold=approval_threshold,
            proposal_key=proposal_key,
        )

    def metabolize(self, task: Pheromone) -> Mutation:
        """Submit the staged proposal to the quorum and finalize or slash it.

        The proposal the Solver staged on the pheromone is reconstructed into a
        :class:`Mutation`, judged by the :class:`SemanticRaft`, and turned into a
        terminal mutation. The original ``metadata`` is preserved and the verdict
        is folded in beside it, leaving a durable audit trail on the ground.
        """
        proposal = self._reconstruct_proposal(task)
        verdict = self.raft.evaluate(task, proposal)
        self.log.info(
            "Pheromone id=%s -> %s (%d/%d jurors entailed).",
            task.id,
            verdict.verdict_status.value,
            verdict.approvals,
            verdict.quorum_size,
        )
        # Preserve provenance: update_state replaces the metadata blob wholesale,
        # so we merge the verdict into the existing metadata ourselves.
        return self.raft.finalize(verdict, base_metadata=task.metadata)

    def _reconstruct_proposal(self, task: Pheromone) -> Mutation:
        """Rebuild the Solver's staged proposal as a :class:`Mutation`.

        The Solver left its proposed action in the pheromone's metadata (and any
        latent context in ``latent_blob``); the verifier rehydrates that into the
        ``(Pheromone, Mutation)`` pair the consensus API expects.
        """
        return Mutation(
            new_entropy=Entropy.ZERO,
            new_status=Status.RESOLVED,
            latent_blob=task.latent_blob,
            metadata=dict(task.metadata or {}),
        )
