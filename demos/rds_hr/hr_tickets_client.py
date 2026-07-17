"""The RH tickets table as a swarm intake source (PostgreSQL-backed).

Mirrors the role the ServiceNow mock plays in the ``servicenow_hr`` demo, but
against a real ``hr_tickets`` table: an intake caste polls for ``new`` rows,
scrubs PII, and drops each onto the pheromone ground carrying the opener's ABAC
attributes so the solver can retrieve only knowledge that opener is cleared for.
Resolution/rejection are written straight back to the row, closing the loop with
``kb_entry_id`` when the GardenerAnt learns the answer.

Security: identifiers are validated; every value is a bound parameter.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys
from dataclasses import dataclass, field
from typing import Any

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for _p in (_ROOT / "src", _ROOT / "demos" / "servicenow_hr"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from stigmergic.agents.base_ant import ProducerAnt  # noqa: E402
from stigmergic.agents.concrete import redact_pii  # noqa: E402
from stigmergic.core.environment import Entropy, Status  # noqa: E402


def _require_psycopg():
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            'HrTicketsClient needs psycopg (v3): pip install "psycopg[binary]>=3.2".'
        ) from exc
    return psycopg, dict_row


_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _safe_ident(name: str) -> str:
    if not _IDENT.match(name):
        raise ValueError(f"Unsafe SQL identifier: {name!r}.")
    return name


@dataclass
class Ticket:
    """One row of ``hr_tickets`` (the fields the swarm exercises)."""

    id: str
    ticket_number: str
    assunto: str
    descricao: str
    knowledge_domain: str
    solicitante_id: str
    solicitante_area: str
    solicitante_nivel: int
    solicitante_geografia: str
    solicitante_projetos: list[str] = field(default_factory=lambda: ["all"])
    status: str = "new"
    kb_action: str | None = None
    kb_entry_id: str | None = None
    resolucao: str | None = None
    veredito: str | None = None

    @property
    def requester(self) -> dict[str, Any]:
        """The ABAC attributes carried onto the pheromone for retrieval."""
        return {
            "id": self.solicitante_id,
            "area": self.solicitante_area,
            "nivel": self.solicitante_nivel,
            "geografia": self.solicitante_geografia,
            "projetos": self.solicitante_projetos,
        }


class HrTicketsClient:
    """CRUD over ``hr_tickets`` for intake and writeback."""

    def __init__(self, dsn: str, *, table: str = "hr_tickets") -> None:
        psycopg, dict_row = _require_psycopg()
        self.table = _safe_ident(table)
        self._conn = psycopg.connect(dsn, autocommit=True, row_factory=dict_row)

    def _row_to_ticket(self, row: dict[str, Any]) -> Ticket:
        return Ticket(
            id=row["id"],
            ticket_number=row["ticket_number"],
            assunto=row["assunto"],
            descricao=row["descricao"] or "",
            knowledge_domain=row["knowledge_domain"],
            solicitante_id=row["solicitante_id"],
            solicitante_area=row["solicitante_area"],
            solicitante_nivel=int(row["solicitante_nivel_hierarquico"]),
            solicitante_geografia=row["solicitante_geografia"],
            solicitante_projetos=list(row["solicitante_projetos"] or ["all"]),
            status=row["status"],
            kb_action=row.get("kb_action"),
            kb_entry_id=str(row["kb_entry_id"]) if row.get("kb_entry_id") else None,
            resolucao=row.get("resolucao"),
            veredito=row.get("veredito"),
        )

    # -- create / read --------------------------------------------------------

    def create_ticket(
        self,
        assunto: str,
        descricao: str = "",
        *,
        solicitante_id: str,
        solicitante_email: str | None = None,
        area: str = "all",
        nivel: int = 1,
        geografia: str = "all",
        projetos: list[str] | None = None,
        knowledge_domain: str = "rh_beneficios",
        canal: str = "portal",
    ) -> Ticket:
        row = self._conn.execute(
            f"INSERT INTO {self.table} "
            "(assunto, descricao, knowledge_domain, canal, solicitante_id, "
            " solicitante_email, solicitante_area, solicitante_nivel_hierarquico, "
            " solicitante_geografia, solicitante_projetos, status) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'new') "
            "RETURNING *",
            (
                assunto,
                descricao,
                knowledge_domain,
                canal,
                solicitante_id,
                solicitante_email,
                area,
                int(nivel),
                geografia,
                projetos or ["all"],
            ),
        ).fetchone()
        return self._row_to_ticket(row)

    def list_new(self, *, limit: int = 100) -> list[Ticket]:
        rows = self._conn.execute(
            f"SELECT * FROM {self.table} WHERE status = 'new' "
            "ORDER BY created_at ASC LIMIT %s",
            (int(limit),),
        ).fetchall()
        return [self._row_to_ticket(r) for r in rows]

    def get(self, ticket_id: str) -> Ticket | None:
        row = self._conn.execute(
            f"SELECT * FROM {self.table} WHERE id = %s::uuid", (str(ticket_id),)
        ).fetchone()
        return self._row_to_ticket(row) if row else None

    # -- lifecycle writes -----------------------------------------------------

    def mark_in_progress(
        self, ticket_id: str, worker: str, *, pheromone_id: int | None = None
    ) -> None:
        self._conn.execute(
            f"UPDATE {self.table} SET status = 'in_progress', assigned_to = %s, "
            "pheromone_id = %s WHERE id = %s::uuid",
            (worker, pheromone_id, str(ticket_id)),
        )

    def resolve(
        self,
        ticket_id: str,
        *,
        resolucao: str,
        resolvido_por: str,
        aprovado_por: str | None,
        kb_entry_id: str | None = None,
        kb_action: str | None = None,
        consensus: dict[str, Any] | None = None,
        contem_pii: bool = False,
        pii: list[str] | None = None,
    ) -> None:
        self._conn.execute(
            f"UPDATE {self.table} SET status = 'resolved', resolucao = %s, "
            "resolvido_por = %s, aprovado_por = %s, kb_entry_id = %s, "
            "kb_action = %s, consensus_resultado = %s::jsonb, veredito = 'passed', "
            "contem_pii = %s, pii_redigido = %s::jsonb, data_resolucao = now() "
            "WHERE id = %s::uuid",
            (
                resolucao,
                resolvido_por,
                aprovado_por,
                kb_entry_id,
                kb_action,
                json.dumps(consensus) if consensus is not None else None,
                bool(contem_pii),
                json.dumps(pii or []),
                str(ticket_id),
            ),
        )

    def reject(
        self,
        ticket_id: str,
        *,
        reason: str,
        consensus: dict[str, Any] | None = None,
    ) -> None:
        self._conn.execute(
            f"UPDATE {self.table} SET status = 'rejected', resolucao = %s, "
            "consensus_resultado = %s::jsonb, veredito = 'slashed', "
            "data_resolucao = now() WHERE id = %s::uuid",
            (
                reason,
                json.dumps(consensus) if consensus is not None else None,
                str(ticket_id),
            ),
        )

    # -- ServiceNow-shaped alias so the reused ReviewingVerifierAnt can slash --

    def cancel(self, sys_id: str, reason: str) -> None:
        """Alias used by the reused verifier on a slashed quorum (sys_id == ticket id)."""
        self.reject(sys_id, reason=reason)

    def close(self) -> None:
        if getattr(self, "_conn", None) is not None:
            self._conn.close()


class HrTicketIntakeAnt(ProducerAnt):
    """Polls ``hr_tickets`` for ``new`` rows and seeds the pheromone ground.

    Each heartbeat: claim every new ticket by flipping it to ``in_progress`` and
    secreting a high-entropy ``RAW`` pheromone. PII is scrubbed *before* the
    durable write (via ``redact_pii``); the opener's ABAC attributes ride along
    in metadata so the solver's retrieval is access-controlled. ``sys_id`` mirrors
    the ticket id so the reused verifier can reject a slashed ticket.
    """

    def __init__(
        self,
        env: Any,
        client: HrTicketsClient,
        name: str | None = None,
        *,
        poll_interval: float = 0.2,
    ) -> None:
        super().__init__(env, name, poll_interval=poll_interval)
        self.client = client

    def secrete(self) -> None:
        for ticket in self.client.list_new():
            raw = ticket.assunto
            if ticket.descricao:
                raw = f"{ticket.assunto}\n\n{ticket.descricao}"
            pheromone_id = self.env.inject_chaos(
                raw,
                entropy=Entropy.CHAOS,
                status=Status.RAW,
                redactor=redact_pii,
                idempotency_key=str(ticket.id),
                metadata={
                    "channel": "hr_portal",
                    "ticket_id": str(ticket.id),
                    "sys_id": str(ticket.id),
                    "number": ticket.ticket_number,
                    "question": ticket.assunto,
                    "knowledge_domain": ticket.knowledge_domain,
                    "requester": ticket.requester,
                    "origin": self.name,
                },
            )
            self.client.mark_in_progress(
                ticket.id, self.name, pheromone_id=pheromone_id
            )
            self.log.debug("Ingested %s: %r", ticket.ticket_number, ticket.assunto)
