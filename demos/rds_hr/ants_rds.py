"""ABAC-aware castes for the RDS/pgvector HR demo.

Only two castes need to change versus the SQLite demo; everything else
(GovernanceAnt, ReviewingVerifierAnt, JanitorAnt) is reused unchanged, proving
the swarm is oblivious to the storage swap:

* :class:`AbacKnowledgeSolverAnt` -- carries the ticket opener's clearance into
  the vector search, so a proposal is only ever built from knowledge that opener
  is allowed to see.
* :class:`AbacGardenerAnt` -- on the human's word, plants the learned answer with
  an ACL derived from the requester (and marked sensitive if the ticket carried
  PII), then writes the resolution -- including the learned row's id and the
  Byzantine verdict -- straight back to the ``hr_tickets`` row.
"""

from __future__ import annotations

import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for _p in (_ROOT / "src", _ROOT / "demos" / "servicenow_hr"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from stigmergic.agents.base_ant import Mutation  # noqa: E402
from stigmergic.core.environment import Entropy, Pheromone, Status  # noqa: E402

from ants import (  # noqa: E402
    ExpertOracle,
    GardenerAnt,
    KnowledgeSolverAnt,
    ReviewContext,
)


class AbacKnowledgeSolverAnt(KnowledgeSolverAnt):
    """A solver whose retrieval is filtered by the requester's ABAC clearance."""

    def metabolize(self, task: Pheromone) -> Mutation:
        metadata = dict(task.metadata or {})
        question = metadata.get("question") or metadata.get("original") or task.raw_data
        original = metadata.get("original") or task.raw_data
        requester = metadata.get("requester")
        domain = metadata.get("knowledge_domain")

        hits = self.kb.search(question, k=self.top_k, requester=requester, domain=domain)
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
        # Carry the original request into the proposal so the quorum re-inspects
        # it for injections, not just the answer.
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


class AbacGardenerAnt(GardenerAnt):
    """A gardener that writes ACL-correct learned rows and closes the ticket.

    ``client`` here is an :class:`~hr_tickets_client.HrTicketsClient`; the write
    back is richer than ServiceNow's ``resolve(sys_id, note)`` so it records the
    learned row id, the KB action and the consensus verdict on the ticket.
    """

    def metabolize(self, task: Pheromone) -> Mutation:
        metadata = dict(task.metadata or {})
        question = metadata.get("question") or metadata.get("original") or task.raw_data
        proposed = metadata.get("proposed_answer") or ""
        ticket_id = metadata.get("ticket_id")
        number = metadata.get("number")
        requester = metadata.get("requester")
        pii = metadata.get("pii_redacted_at_intake") or []
        consensus = metadata.get("consensus")
        sensitive = bool(pii)

        ctx = ReviewContext(
            question=question,
            proposed_answer=proposed,
            kb_id=metadata.get("kb_id"),
            kb_score=metadata.get("kb_score"),
            retrieved_source=metadata.get("retrieved_source"),
            incident_number=number,
            metadata=metadata,
        )
        decision = self.oracle(ctx)
        metadata["reviewed_by"] = decision.reviewer

        if decision.approved:
            new_id = self.kb.add(
                question,
                proposed,
                source="resolved-ticket",
                owner=decision.reviewer,
                aprovador=decision.reviewer,
                requester=requester,
                sensitive=sensitive,
                source_uri=number,
                metadata={"incident": number, "confidence": 0.8},
            )
            final_answer = proposed
            kb_action = "add"
            metadata["kb_write"] = "add"
        else:
            quarantined = False
            wrong_id = decision.wrong_kb_id
            if wrong_id is not None:
                quarantined = self.kb.quarantine(
                    wrong_id, reason="expert-correction", operator=decision.reviewer
                )
            expert_answer = decision.correct_answer or ""
            new_id = None
            if expert_answer:
                new_id = self.kb.add(
                    question,
                    expert_answer,
                    source="expert-correction",
                    owner=decision.reviewer,
                    aprovador=decision.reviewer,
                    requester=requester,
                    sensitive=sensitive,
                    source_uri=number,
                    metadata={"incident": number, "confidence": 0.95},
                )
            final_answer = expert_answer or proposed
            if quarantined and new_id:
                kb_action = "quarantine+add"
            elif new_id:
                kb_action = "add"
            elif quarantined:
                kb_action = "quarantine"
            else:
                kb_action = "none"
            metadata["kb_write"] = kb_action
            metadata["kb_quarantined_id"] = wrong_id if quarantined else None

        metadata["final_answer"] = final_answer
        metadata["kb_new_id"] = new_id

        if self.client is not None and ticket_id:
            self.client.resolve(
                ticket_id,
                resolucao=final_answer,
                resolvido_por=self.name,
                aprovado_por=decision.reviewer,
                kb_entry_id=new_id,
                kb_action=kb_action,
                consensus=consensus,
                contem_pii=sensitive,
                pii=list(pii),
            )

        return Mutation(
            new_entropy=Entropy.ZERO,
            new_status=Status.RESOLVED,
            metadata=metadata,
            release_owner=True,
        )


__all__ = ["AbacKnowledgeSolverAnt", "AbacGardenerAnt", "ExpertOracle"]
