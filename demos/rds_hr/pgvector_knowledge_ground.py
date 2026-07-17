"""A KnowledgeGround backed by PostgreSQL + pgvector (your ``knowledge_corporate``).

This is the production analogue of the demo's tiny SQLite ``KnowledgeGround``: the
same four-method contract the ant castes depend on --

    search(query, k, *, requester, domain) -> [SearchHit(entry, score)]
    add(question, answer, ...)             -> new row id (uuid, as str)
    quarantine(entry_id, ...)              -> bool
    count()                                -> int

-- but every call hits a real pgvector table. Nothing about the swarm changes;
only the substrate does, exactly the "code against the contract, swap the
storage" split the SQLite ground and the ServiceNow mock already use.

Two things make this a *production* ground rather than a copy of the demo:

* **ABAC-aware retrieval.** ``search`` filters by the ticket opener's clearance
  (area / hierarchy level / geography) against the row-level ACL columns, so the
  swarm never proposes an answer built from knowledge the requester cannot see.
* **A chunk store, not a Q&A store.** ``knowledge_corporate`` holds document
  *chunks* with rich provenance, so a learned ticket resolution is written as a
  new ``chunk_type='ticket_resolution'`` row: the answer in ``conteudo_original``,
  the question in ``section_title``, and the vector built from the *question* so
  future look-alike tickets land next to it.

Security: the table name is validated as a safe identifier and every value is a
bound parameter -- no string-built SQL. ``psycopg`` (v3) is imported lazily.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any


def _require_psycopg():
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PgVectorKnowledgeGround needs psycopg (v3). Install it with: "
            'pip install "psycopg[binary]>=3.2".'
        ) from exc
    return psycopg, dict_row


_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _safe_ident(name: str) -> str:
    if not _IDENT.match(name):
        raise ValueError(f"Unsafe SQL identifier: {name!r}.")
    return name


class KnowledgeSource:
    """Provenance tags written into ``source_type`` / ``chunk_type``."""

    RESOLVED_TICKET = "resolved-ticket"
    EXPERT_CORRECTION = "expert-correction"


@dataclass
class PgKnowledgeEntry:
    """A row of ``knowledge_corporate`` seen through the swarm's lens.

    Duck-typed to match the demo's ``KnowledgeEntry`` where the castes touch it
    (``.id``, ``.question``, ``.answer``, ``.source``), but ``id`` is the row's
    ``uuid`` as a string, not an int.
    """

    id: str
    question: str
    answer: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PgSearchHit:
    """A knowledge entry paired with its cosine similarity to a query."""

    entry: PgKnowledgeEntry
    score: float


class PgVectorKnowledgeGround:
    """The self-improving "soil" on PostgreSQL + pgvector.

    Args:
        dsn: libpq/psycopg connection string (``postgresql://user:pw@host/db``).
        embedder: anything with ``.encode(text) -> list[float]`` producing a
            vector of the table's ``embedding_dimensions`` (1536 for
            ``text-embedding-3-small``). Must be the SAME model the table was
            populated with, or the vectors are not comparable.
        table: the knowledge table (validated as a safe identifier).
        knowledge_domain: the domain this ground serves and writes into.
        learned_chunk_type: ``chunk_type`` stamped on learned/corrected rows.
    """

    def __init__(
        self,
        dsn: str,
        embedder: Any,
        *,
        table: str = "knowledge_corporate",
        knowledge_domain: str = "rh_beneficios",
        learned_chunk_type: str = "ticket_resolution",
        probes: int = 100,
    ) -> None:
        psycopg, dict_row = _require_psycopg()
        self.embedder = embedder
        self.table = _safe_ident(table)
        self.knowledge_domain = knowledge_domain
        self.learned_chunk_type = learned_chunk_type
        self.embedding_model = getattr(embedder, "name", "unknown")
        self.embedding_dim = int(getattr(embedder, "dim", 1536))
        self._conn = psycopg.connect(dsn, autocommit=True, row_factory=dict_row)
        # pgvector's ivfflat ANN index defaults to probing a single list. On a
        # small (or unevenly clustered) table that silently returns almost no
        # rows, so lift probes for reliable recall. `probes` is a validated int,
        # never user text; SET does not accept bound parameters.
        try:
            self._conn.execute(f"SET ivfflat.probes = {int(probes)}")
        except Exception:  # pragma: no cover - older pgvector without ivfflat
            pass

    # -- helpers --------------------------------------------------------------

    def _vector_literal(self, text: str) -> str:
        """Encode ``text`` and render it as a pgvector text literal ``[a,b,...]``."""
        vec = self.embedder.encode(text)
        if len(vec) != self.embedding_dim:
            raise ValueError(
                f"Embedder produced {len(vec)} dims but the table expects "
                f"{self.embedding_dim}. Are you using the same model the KB was "
                "populated with?"
            )
        return "[" + ",".join(f"{float(x):.8f}" for x in vec) + "]"

    # -- read side ------------------------------------------------------------

    def search(
        self,
        query: str,
        k: int = 3,
        *,
        requester: dict[str, Any] | None = None,
        domain: str | None = None,
    ) -> list[PgSearchHit]:
        """Return the ``k`` nearest *active* rows, cosine-best first.

        When ``requester`` is given (``{"area", "nivel", "geografia"}``) the scan
        is ABAC-filtered: a row is visible only if its ACL admits the opener.
        Quarantined, soft-deleted and expired rows are never served.
        """
        if k <= 0:
            return []
        qv = self._vector_literal(query)
        dom = domain or self.knowledge_domain
        where = [
            "is_active = true",
            "soft_deleted_at IS NULL",
            "knowledge_domain = %s",
            "(data_validade IS NULL OR data_validade >= CURRENT_DATE)",
        ]
        params: list[Any] = [dom]
        if requester:
            where.append("('all' = ANY(areas_liberadas) OR %s = ANY(areas_liberadas))")
            params.append(str(requester.get("area", "all")))
            where.append("nivel_hierarquico_minimo <= %s")
            params.append(int(requester.get("nivel", 1)))
            where.append("('all' = ANY(geografias_liberadas) OR %s = ANY(geografias_liberadas))")
            params.append(str(requester.get("geografia", "all")))
        sql = (
            f"SELECT id::text AS id, section_title, conteudo_original, source_type, "
            f"fonte_documento, 1 - (vetor <=> %s::vector) AS score "
            f"FROM {self.table} "
            f"WHERE {' AND '.join(where)} "
            f"ORDER BY vetor <=> %s::vector "
            f"LIMIT %s"
        )
        args = [qv, *params, qv, int(k)]
        rows = self._conn.execute(sql, args).fetchall()
        hits: list[PgSearchHit] = []
        for row in rows:
            entry = PgKnowledgeEntry(
                id=row["id"],
                question=row["section_title"] or "",
                answer=row["conteudo_original"],
                source=row["source_type"] or row["fonte_documento"] or "kb",
            )
            hits.append(PgSearchHit(entry=entry, score=float(row["score"])))
        return hits

    def best_match(self, query: str, **kw: Any) -> PgSearchHit | None:
        hits = self.search(query, k=1, **kw)
        return hits[0] if hits else None

    # -- write side -----------------------------------------------------------

    def add(
        self,
        question: str,
        answer: str,
        *,
        source: str = KnowledgeSource.RESOLVED_TICKET,
        owner: str = "system",
        aprovador: str | None = None,
        requester: dict[str, Any] | None = None,
        sensitive: bool = False,
        source_uri: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Plant a learned ``(question, answer)`` as a new row; return its uuid.

        The vector is built from the *question* so future look-alike tickets
        retrieve it. Visibility (ACL) defaults from the requester's clearance:
        the answer is shared with the opener's area/geography, and marked
        ``dado_sensivel`` when the ticket carried PII.
        """
        meta = dict(metadata or {})
        area = str((requester or {}).get("area", "all"))
        geo = str((requester or {}).get("geografia", "all"))
        areas = ["all"] if area == "all" else [area]
        geos = ["all"] if geo == "all" else [geo]
        vec = self._vector_literal(question)
        content_hash = hashlib.sha256((question + "\n" + answer).encode("utf-8")).hexdigest()
        tags = {"learned_from": "stigmergic-swarm", **meta}
        row = self._conn.execute(
            f"INSERT INTO {self.table} "
            "(conteudo_original, section_title, knowledge_domain, source_type, "
            " source_uri, fonte_documento, content_hash, chunk_type, responsavel, "
            " aprovador, dado_sensivel, areas_liberadas, nivel_hierarquico_minimo, "
            " geografias_liberadas, embedding_model, embedding_dimensions, vetor, "
            " tags, idioma, data_ingestao) "
            "VALUES (%s, %s, %s, 'form', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
            "        %s, %s::vector, %s::jsonb, %s, now()) "
            "RETURNING id::text AS id",
            (
                answer,
                question,
                self.knowledge_domain,
                source_uri,
                source_uri,
                content_hash,
                self.learned_chunk_type,
                owner,
                aprovador,
                bool(sensitive),
                areas,
                1,
                geos,
                self.embedding_model[:50],
                self.embedding_dim,
                vec,
                json.dumps(tags, ensure_ascii=False),
                "pt-BR",
            ),
        ).fetchone()
        return row["id"]

    def quarantine(
        self,
        entry_id: str,
        *,
        reason: str = "expert-correction",
        operator: str = "system",
    ) -> bool:
        """Soft-delete a wrong row (the self-healing half). Returns success."""
        row = self._conn.execute(
            f"UPDATE {self.table} SET is_active = false, soft_deleted_at = now() "
            "WHERE id = %s::uuid AND is_active = true "
            "RETURNING id",
            (str(entry_id),),
        ).fetchone()
        return row is not None

    # -- introspection --------------------------------------------------------

    def count(self, *, include_quarantined: bool = False) -> int:
        """Active-row count within this ground's domain."""
        if include_quarantined:
            sql = f"SELECT count(*) AS n FROM {self.table} WHERE knowledge_domain = %s"
        else:
            sql = (
                f"SELECT count(*) AS n FROM {self.table} "
                "WHERE knowledge_domain = %s AND is_active = true "
                "AND soft_deleted_at IS NULL"
            )
        return int(self._conn.execute(sql, (self.knowledge_domain,)).fetchone()["n"])

    def close(self) -> None:
        if getattr(self, "_conn", None) is not None:
            self._conn.close()
