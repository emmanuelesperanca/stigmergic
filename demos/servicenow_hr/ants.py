"""The HR/ServiceNow ant castes -- the swarm that turns tickets into knowledge.

Four new castes extend the generic ants to wire ServiceNow and the vector
knowledge ground into one stigmergic pipeline. As always, no caste calls another;
they coordinate only through the pheromone trails on the shared ground:

    New incident
        |  ServiceNowIntakeAnt          (secretes RAW pheromone, marks In Progress)
        v
    RAW --GovernanceAnt (reused)-->  HYGIENIZED
        |  KnowledgeSolverAnt           (embeds question, searches the soil,
        v                                proposes the nearest answer)
    PENDING_CONSENSUS
        |  ReviewingVerifierAnt         (Byzantine quorum: the KB's immune system)
        |
        +--[quorum fails]--> SLASHED            (injection/hallucination; nothing
        |                                        is ever written to the soil, and
        |                                        the incident is Canceled)
        |
        +--[quorum passes]--> PENDING_HUMAN
                |  GardenerAnt           (tends the soil on the expert's word)
                |
                +--[approved]--> RESOLVED   (LEARNING: the approved answer is
                |                            persisted -- the soil grows)
                |
                +--[rejected]--> RESOLVED   (CORRECTION: the wrong entry is torn
                                             out and replaced with the expert's
                                             answer -- the soil self-heals)

The load-bearing idea: a self-improving knowledge base is a prompt-injection
*amplifier* unless every writeback is gated. Here the ``SemanticRaft`` quorum --
the very one benchmarked in the injection suite -- slashes a malicious ticket
*before* :class:`GardenerAnt` can ever persist it, so poison never reaches the
ground.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Callable

from stigmergic_ai.agents.base_ant import ConsumerAnt, Mutation, ProducerAnt
from stigmergic_ai.agents.verifier_ant import VerifierAnt
from stigmergic_ai.core.consensus import MockNLIJudge, NLIJudge, SemanticRaft
from stigmergic_ai.core.environment import (
    Entropy,
    Pheromone,
    PheromoneGround,
    Status,
)

import pathlib

_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from knowledge_ground import KnowledgeGround, KnowledgeSource  # noqa: E402
from mock_servicenow import IncidentState, ServiceNowClient  # noqa: E402

__all__ = [
    "ReviewContext",
    "ExpertDecision",
    "ExpertOracle",
    "ScriptedExpertOracle",
    "approve_all_oracle",
    "ServiceNowIntakeAnt",
    "KnowledgeSolverAnt",
    "ReviewingVerifierAnt",
    "GardenerAnt",
]


# ---------------------------------------------------------------------------
# The human-in-the-loop contract
# ---------------------------------------------------------------------------


@dataclass
class ReviewContext:
    """What the expert (or the oracle standing in for them) gets to see."""

    question: str
    proposed_answer: str
    kb_id: int | None
    kb_score: float | None
    retrieved_source: str | None
    incident_number: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExpertDecision:
    """The expert's ruling on a machine-proposed resolution.

    * ``approved=True`` -> the proposal is correct; it will be *learned*.
    * ``approved=False`` -> the proposal is wrong; ``correct_answer`` will be
      persisted instead, and if ``wrong_kb_id`` is set, that poisoned/outdated
      entry is deleted first (the soil self-heals).
    """

    approved: bool
    correct_answer: str | None = None
    wrong_kb_id: int | None = None
    reviewer: str = "hr-expert"
    note: str = ""


#: A human reviewer, abstracted to a callable so the demo can script it and a
#: real deployment can wire it to a UI/approval queue.
ExpertOracle = Callable[[ReviewContext], ExpertDecision]


def approve_all_oracle(reviewer: str = "auto-approver") -> ExpertOracle:
    """An oracle that rubber-stamps everything (useful for the learning path)."""

    def _oracle(_ctx: ReviewContext) -> ExpertDecision:
        return ExpertDecision(approved=True, reviewer=reviewer)

    return _oracle


class ScriptedExpertOracle:
    """A deterministic oracle driven by per-incident ground truth.

    The demo knows each ticket's intended outcome, so it can script the expert:
    keyed by incident number, each entry is either ``("approve", None)`` or
    ``("reject", "<the authoritative answer>")``. On a rejection the oracle marks
    the *retrieved* KB entry (``ctx.kb_id``) as the one to tear out, which is
    exactly the planted-wrong row a correction ticket is designed to expose.
    Unknown incidents default to approval.
    """

    def __init__(
        self,
        script: dict[str, tuple[str, str | None]],
        *,
        reviewer: str = "hr-expert",
    ) -> None:
        self._script = script
        self._reviewer = reviewer

    def __call__(self, ctx: ReviewContext) -> ExpertDecision:
        verdict, answer = self._script.get(ctx.incident_number or "", ("approve", None))
        if verdict == "approve":
            return ExpertDecision(approved=True, reviewer=self._reviewer)
        return ExpertDecision(
            approved=False,
            correct_answer=answer,
            wrong_kb_id=ctx.kb_id,
            reviewer=self._reviewer,
            note="Auto-proposed answer rejected by expert.",
        )


# ---------------------------------------------------------------------------
# The castes
# ---------------------------------------------------------------------------


class ServiceNowIntakeAnt(ProducerAnt):
    """Polls ServiceNow for New incidents and drops them onto the ground.

    Each heartbeat it claims every ``New`` incident by moving it to
    ``In Progress`` and secreting a matching high-entropy ``RAW`` pheromone whose
    metadata carries the ``sys_id``/``number`` (so later castes can sync the
    incident) and the ``question`` (the short description, kept clean for
    retrieval). Assumes a single intake ant; the immediate state flip to
    ``In Progress`` is what prevents an incident being ingested twice.
    """

    def __init__(
        self,
        env: PheromoneGround,
        client: ServiceNowClient,
        name: str | None = None,
        *,
        poll_interval: float = 0.15,
    ) -> None:
        super().__init__(env, name, poll_interval=poll_interval)
        self.client = client

    def secrete(self) -> None:
        for incident in self.client.list_incidents(state=IncidentState.NEW):
            question = incident.short_description
            raw = question
            if incident.description:
                raw = f"{question}\n\n{incident.description}"
            self.env.inject_chaos(
                raw,
                entropy=Entropy.CHAOS,
                status=Status.RAW,
                metadata={
                    "channel": "servicenow",
                    "sys_id": incident.sys_id,
                    "number": incident.number,
                    "question": question,
                    "caller_id": incident.caller_id,
                    "origin": self.name,
                },
            )
            self.client.update(
                incident.sys_id,
                state=IncidentState.IN_PROGRESS,
                assigned_to=self.name,
            )
            self.client.add_work_note(
                incident.sys_id,
                f"Ingested by {self.name}; dispatched to the HR knowledge swarm.",
            )
            self.log.debug("Ingested %s: %r", incident.number, question)


class KnowledgeSolverAnt(ConsumerAnt):
    """Embeds the question, searches the soil, and proposes the nearest answer.

    This is the "the leaf lands next to the answer" step. It wakes on the clean
    ``HYGIENIZED`` trail, embeds the ticket's question, retrieves the nearest
    knowledge entry, and stages a proposal at ``PENDING_CONSENSUS`` for the
    quorum. Like the generic Solver, it carries the *original request* into the
    proposal text, so any injection that slipped through hygiene is exposed to the
    semantic jury (and thus slashed before it can be learned).
    """

    def __init__(
        self,
        env: PheromoneGround,
        kb: KnowledgeGround,
        name: str | None = None,
        *,
        top_k: int = 3,
        poll_interval: float = 0.15,
    ) -> None:
        super().__init__(
            env,
            name,
            entropy_threshold=Entropy.MIN,
            target_status=Status.HYGIENIZED,
            poll_interval=poll_interval,
        )
        self.kb = kb
        self.top_k = top_k

    def metabolize(self, task: Pheromone) -> Mutation:
        metadata = dict(task.metadata or {})
        question = metadata.get("question") or metadata.get("original") or task.raw_data
        original = metadata.get("original") or task.raw_data

        hits = self.kb.search(question, k=self.top_k)
        best = hits[0] if hits else None
        if best is not None:
            answer = best.entry.answer
            metadata["kb_id"] = best.entry.id
            metadata["kb_score"] = round(best.score, 4)
            metadata["retrieved_source"] = best.entry.source
        else:
            answer = "No knowledge-base match found; escalating to a human expert."
            metadata["kb_id"] = None
            metadata["kb_score"] = 0.0
            metadata["retrieved_source"] = None

        metadata["proposed_answer"] = answer
        metadata["solved_by"] = self.name
        # Carry the original request into the proposal so the quorum judges the
        # answer *and* re-inspects the request for injections.
        metadata["proposal"] = (
            f"In response to the request «{original}», resolve the ticket with "
            f"this knowledge-base answer: {answer}"
        )
        return Mutation(
            new_entropy=Entropy.LOW,
            new_status=Status.PENDING_CONSENSUS,
            metadata=metadata,
            release_owner=True,
        )


class ReviewingVerifierAnt(VerifierAnt):
    """A verifier that routes a *passing* quorum to human review, not straight to done.

    Same Byzantine ``SemanticRaft`` as the base :class:`VerifierAnt`, but the
    terminal step differs so a human stays in the loop:

    * **Quorum passes** -> ``PENDING_HUMAN`` (entropy drops to :attr:`Entropy.LOW`,
      *not* zero -- the trail is still live because the work isn't finished until
      an expert signs off).
    * **Quorum fails** -> ``SLASHED`` (entropy ``0.0``; the poisoned trail
      evaporates) and, if a ServiceNow client is wired in, the incident is
      Canceled with a security note.

    Defaults to an offline, deterministic :class:`MockNLIJudge` so tests and the
    committed demo need no model download; pass ``judge=`` or ``raft=`` to use a
    real NLI panel.
    """

    def __init__(
        self,
        env: PheromoneGround,
        name: str | None = None,
        *,
        client: ServiceNowClient | None = None,
        raft: SemanticRaft | None = None,
        judge: NLIJudge | None = None,
        quorum_size: int = 3,
        approval_threshold: int = 2,
        poll_interval: float = 0.15,
    ) -> None:
        if raft is None and judge is None:
            judge = MockNLIJudge()
        super().__init__(
            env,
            name,
            raft=raft,
            judge=judge,
            quorum_size=quorum_size,
            approval_threshold=approval_threshold,
            target_status=Status.PENDING_CONSENSUS,
            poll_interval=poll_interval,
        )
        self.client = client

    def metabolize(self, task: Pheromone) -> Mutation:
        proposal = self._reconstruct_proposal(task)
        verdict = self.raft.evaluate(task, proposal)
        mutation = self.raft.finalize(verdict, base_metadata=task.metadata)
        self.log.info(
            "Pheromone id=%s -> %s (%d/%d jurors entailed).",
            task.id,
            "PENDING_HUMAN" if verdict.passed else verdict.verdict_status.value,
            verdict.approvals,
            verdict.quorum_size,
        )
        if verdict.passed:
            # Park it for a human sign-off instead of finalizing it ourselves.
            mutation.new_status = Status.PENDING_HUMAN
            mutation.new_entropy = Entropy.LOW
            mutation.metadata["review_state"] = "awaiting-human"
        else:
            # Slashed: reflect the rejection in ServiceNow if we can.
            sys_id = (task.metadata or {}).get("sys_id")
            if self.client is not None and sys_id:
                self.client.cancel(
                    sys_id,
                    "Rejected by automated consensus quorum "
                    "(possible prompt injection or hallucination).",
                )
        mutation.release_owner = True
        return mutation


class GardenerAnt(ConsumerAnt):
    """Tends the knowledge garden -- plants approved answers, weeds out bad ones.

    Like a leafcutter tending its fungus garden, this caste cultivates the
    durable soil: it wakes on ``PENDING_HUMAN`` (proposals the quorum already
    cleared), consults the injected :data:`ExpertOracle` for the verdict, and
    performs the writeback that is the ticket's terminal act (the ground forbids
    claiming a terminal pheromone, so learning/correction *must* happen here, as
    the pheromone is driven to ``RESOLVED``):

    * **Approved** -> :meth:`KnowledgeGround.add` the answer as a
      ``resolved-ticket`` (the garden *grows*), and Resolve the incident.
    * **Rejected** -> optionally :meth:`KnowledgeGround.delete` the wrong entry
      the answer came from, then plant the expert's authoritative answer as an
      ``expert-correction`` (the garden *self-heals*), and Resolve the incident
      with the corrected answer.

    The human expert is the oracle (external judgment); this ant is only the
    gardener's hands that apply that judgment to the soil.
    """

    def __init__(
        self,
        env: PheromoneGround,
        kb: KnowledgeGround,
        oracle: ExpertOracle,
        name: str | None = None,
        *,
        client: ServiceNowClient | None = None,
        poll_interval: float = 0.15,
    ) -> None:
        super().__init__(
            env,
            name,
            entropy_threshold=Entropy.MIN,
            target_status=Status.PENDING_HUMAN,
            poll_interval=poll_interval,
        )
        self.kb = kb
        self.oracle = oracle
        self.client = client

    def metabolize(self, task: Pheromone) -> Mutation:
        metadata = dict(task.metadata or {})
        question = metadata.get("question") or metadata.get("original") or task.raw_data
        proposed = metadata.get("proposed_answer") or ""
        sys_id = metadata.get("sys_id")

        ctx = ReviewContext(
            question=question,
            proposed_answer=proposed,
            kb_id=metadata.get("kb_id"),
            kb_score=metadata.get("kb_score"),
            retrieved_source=metadata.get("retrieved_source"),
            incident_number=metadata.get("number"),
            metadata=metadata,
        )
        decision = self.oracle(ctx)
        metadata["reviewed_by"] = decision.reviewer

        if decision.approved:
            self._learn(question, proposed, metadata)
            final_answer = proposed
            if self.client is not None and sys_id:
                self.client.resolve(
                    sys_id,
                    f"Resolved by the HR swarm, approved by {decision.reviewer}: "
                    f"{final_answer}",
                )
        else:
            final_answer = self._correct(question, decision, metadata)
            if self.client is not None and sys_id:
                self.client.resolve(
                    sys_id,
                    f"Auto-proposal rejected; resolved by {decision.reviewer} "
                    f"with the corrected answer: {final_answer}",
                )

        metadata["final_answer"] = final_answer
        return Mutation(
            new_entropy=Entropy.ZERO,
            new_status=Status.RESOLVED,
            metadata=metadata,
            release_owner=True,
        )

    # -- writeback helpers ----------------------------------------------------

    def _learn(self, question: str, answer: str, metadata: dict[str, Any]) -> None:
        """Persist an approved answer: the soil grows a new blade of grass."""
        new_id = self.kb.add(
            question,
            answer,
            source=KnowledgeSource.RESOLVED_TICKET,
            metadata={"incident": metadata.get("number")},
        )
        metadata["review"] = "approved"
        metadata["kb_write"] = "add"
        metadata["kb_new_id"] = new_id

    def _correct(
        self,
        question: str,
        decision: ExpertDecision,
        metadata: dict[str, Any],
    ) -> str:
        """Tear out the wrong entry (if any) and plant the expert's answer."""
        metadata["review"] = "rejected"
        removed = False
        if decision.wrong_kb_id is not None:
            removed = self.kb.delete(decision.wrong_kb_id)
        metadata["kb_deleted"] = removed
        metadata["kb_deleted_id"] = decision.wrong_kb_id if removed else None

        expert_answer = decision.correct_answer or ""
        if expert_answer:
            new_id = self.kb.add(
                question,
                expert_answer,
                source=KnowledgeSource.EXPERT_CORRECTION,
                metadata={"incident": metadata.get("number")},
            )
            metadata["kb_write"] = "delete+add" if removed else "add"
            metadata["kb_new_id"] = new_id
        else:
            metadata["kb_write"] = "delete" if removed else "none"
        return expert_answer or metadata.get("proposed_answer", "")
