"""The knowledge base as a literal "forest floor" -- a searchable vector soil.

This is the substrate the whole demo revolves around. HR knowledge (seed FAQ
documents, and later the answers to resolved tickets) is stored as
``(question, answer, embedding)`` rows. A new ticket's question is embedded and
dropped onto this floor; nearest-neighbour cosine search finds where it "lands",
right next to the answer that already lies there -- the stigmergic idea that a
mark left in the environment guides the next agent to the right spot.

Crucially, the soil is *mutable and self-improving*:

* :meth:`add` -- an approved ticket resolution is persisted, so the ground grows
  richer over time (the "learning" half of the story).
* :meth:`quarantine` / :meth:`replace` -- a wrong entry that a human expert
  corrected is set aside (kept on disk for audit and rollback, no longer served)
  and replaced with the authoritative answer (the "self-healing" half).

Because those writebacks *improve* the base that future answers are drawn from, a
poisoned write would compound. That is exactly why, in this demo, nothing reaches
:meth:`add` until a Byzantine consensus quorum has cleared it -- the same
``SemanticRaft`` benchmarked in the injection suite. The vector store itself is
deliberately simple (SQLite + pure-Python cosine, no numpy/faiss) so the moving
parts on show are the swarm and the consensus gate, not a vector index.
"""

from __future__ import annotations

import array
import json
import pathlib
import sqlite3
import sys
import threading
import time
from typing import Any

from pydantic import BaseModel, Field

_HERE = pathlib.Path(__file__).resolve().parent
_SRC = _HERE.parents[1] / "src"
for _p in (_HERE, _SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from embeddings import Embedder, cosine_similarity  # noqa: E402

__all__ = [
    "KnowledgeEntry",
    "SearchHit",
    "KnowledgeGround",
    "KnowledgeSource",
    "KnowledgeStatus",
]


class KnowledgeSource:
    """Provenance tags for a knowledge row (why it is on the floor)."""

    SEED_DOC = "seed-doc"
    RESOLVED_TICKET = "resolved-ticket"
    EXPERT_CORRECTION = "expert-correction"


class KnowledgeStatus:
    """Lifecycle state of a knowledge row.

    A wrong entry is never silently erased: it is moved to ``QUARANTINE`` so it
    stops being served by retrieval while remaining on disk for audit and
    rollback. Only :meth:`KnowledgeGround.delete` (a test/reset primitive) is a
    hard removal.
    """

    ACTIVE = "active"
    QUARANTINE = "quarantine"


class KnowledgeEntry(BaseModel):
    """One stored piece of HR knowledge."""

    id: int
    question: str
    answer: str
    source: str = KnowledgeSource.SEED_DOC
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: float = 0.0
    # Governance / provenance.
    confidence: float = 0.5
    expiry_at: float | None = None
    version: int = 1
    superseded_by: int | None = None
    owner: str = "system"
    status: str = KnowledgeStatus.ACTIVE


class SearchHit(BaseModel):
    """A knowledge entry paired with its cosine similarity to a query."""

    entry: KnowledgeEntry
    score: float


def _vec_to_blob(vec: list[float]) -> bytes:
    return array.array("f", vec).tobytes()


def _blob_to_vec(blob: bytes) -> list[float]:
    out = array.array("f")
    out.frombytes(blob)
    return list(out)


class KnowledgeGround:
    """A tiny embedding-backed vector store: the searchable, mutable soil.

    Bound to a single :class:`~embeddings.Embedder` -- every row and every query
    is encoded by the *same* embedder, so their vectors are comparable. Backed by
    SQLite (``:memory:`` by default; pass a path to persist the floor between
    runs). Nearest-neighbour search is a brute-force pure-Python cosine scan,
    which is more than fast enough for a demo-sized knowledge base and keeps the
    dependency surface at zero.
    """

    def __init__(self, embedder: Embedder, *, db_path: str = ":memory:") -> None:
        self.embedder = embedder
        self.db_path = db_path
        # check_same_thread=False + a lock: the ant castes run on separate
        # threads, exactly like the PheromoneGround, so the connection is shared
        # and every access is serialized through the lock.
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._ensure_schema()

    # -- lifecycle ------------------------------------------------------------

    def _ensure_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    question      TEXT NOT NULL,
                    answer        TEXT NOT NULL,
                    embedding     BLOB NOT NULL,
                    source        TEXT NOT NULL,
                    embedder      TEXT NOT NULL,
                    metadata      TEXT NOT NULL DEFAULT '{}',
                    created_at    REAL NOT NULL,
                    confidence    REAL NOT NULL DEFAULT 0.5,
                    expiry_at     REAL,
                    version       INTEGER NOT NULL DEFAULT 1,
                    superseded_by INTEGER,
                    owner         TEXT NOT NULL DEFAULT 'system',
                    status        TEXT NOT NULL DEFAULT 'active'
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_audit (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    kb_id      INTEGER NOT NULL,
                    operation  TEXT NOT NULL,
                    operator   TEXT NOT NULL,
                    reason     TEXT NOT NULL DEFAULT '',
                    metadata   TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL
                )
                """
            )
            self._migrate_governance_columns()
            self._conn.commit()

    def _migrate_governance_columns(self) -> None:
        """Add governance columns to a knowledge table created by an older version."""
        existing = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(knowledge)").fetchall()
        }
        additions = {
            "confidence": "REAL NOT NULL DEFAULT 0.5",
            "expiry_at": "REAL",
            "version": "INTEGER NOT NULL DEFAULT 1",
            "superseded_by": "INTEGER",
            "owner": "TEXT NOT NULL DEFAULT 'system'",
            "status": "TEXT NOT NULL DEFAULT 'active'",
        }
        for col, ddl in additions.items():
            if col not in existing:
                self._conn.execute(f"ALTER TABLE knowledge ADD COLUMN {col} {ddl}")

    def _audit(
        self,
        kb_id: int,
        operation: str,
        *,
        operator: str = "system",
        reason: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Append one immutable row to the knowledge audit trail (caller holds lock)."""
        self._conn.execute(
            "INSERT INTO knowledge_audit "
            "(kb_id, operation, operator, reason, metadata, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (kb_id, operation, operator, reason, json.dumps(metadata or {}), time.time()),
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "KnowledgeGround":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- writes (the soil grows and heals) ------------------------------------

    def add(
        self,
        question: str,
        answer: str,
        *,
        source: str = KnowledgeSource.SEED_DOC,
        metadata: dict[str, Any] | None = None,
        confidence: float = 0.5,
        expiry_at: float | None = None,
        owner: str = "system",
    ) -> int:
        """Persist a new ``(question, answer)`` pair and return its row id."""
        embedding = self.embedder.encode(question)
        created = time.time()
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO knowledge
                    (question, answer, embedding, source, embedder, metadata,
                     created_at, confidence, expiry_at, owner)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    question,
                    answer,
                    _vec_to_blob(embedding),
                    source,
                    self.embedder.name,
                    json.dumps(metadata or {}),
                    created,
                    confidence,
                    expiry_at,
                    owner,
                ),
            )
            new_id = int(cur.lastrowid)
            self._audit(
                new_id,
                "add",
                operator=owner,
                reason=source,
                metadata={"confidence": confidence},
            )
            self._conn.commit()
            return new_id

    def replace(
        self,
        entry_id: int,
        new_answer: str,
        *,
        source: str = KnowledgeSource.EXPERT_CORRECTION,
        metadata: dict[str, Any] | None = None,
        operator: str = "system",
    ) -> bool:
        """Overwrite an existing row's answer in place (keeps its id/question).

        The previous answer is recorded in the audit trail so :meth:`rollback`
        can restore it, and the row's ``version`` is bumped.
        """
        row = self.get(entry_id)
        if row is None:
            return False
        merged = dict(row.metadata)
        if metadata:
            merged.update(metadata)
        with self._lock:
            self._conn.execute(
                "UPDATE knowledge SET answer = ?, source = ?, metadata = ?, "
                "version = version + 1 WHERE id = ?",
                (new_answer, source, json.dumps(merged), entry_id),
            )
            self._audit(
                entry_id,
                "update",
                operator=operator,
                reason=source,
                metadata={"old_answer": row.answer, "old_version": row.version},
            )
            self._conn.commit()
        return True

    def quarantine(self, entry_id: int, *, reason: str, operator: str = "system") -> bool:
        """Soft-delete: stop serving an entry (retrieval skips it) but keep it on
        disk, logging who quarantined it and why. Returns ``True`` if it existed.
        """
        with self._lock:
            cur = self._conn.execute(
                "UPDATE knowledge SET status = ? WHERE id = ?",
                (KnowledgeStatus.QUARANTINE, entry_id),
            )
            changed = cur.rowcount > 0
            if changed:
                self._audit(entry_id, "quarantine", operator=operator, reason=reason)
            self._conn.commit()
        return changed

    def restore(self, entry_id: int, *, operator: str = "system") -> bool:
        """Return a quarantined entry to active service. Returns ``True`` if it existed."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE knowledge SET status = ? WHERE id = ?",
                (KnowledgeStatus.ACTIVE, entry_id),
            )
            changed = cur.rowcount > 0
            if changed:
                self._audit(entry_id, "restore", operator=operator)
            self._conn.commit()
        return changed

    def rollback(
        self, entry_id: int, *, to_version: int | None = None, operator: str = "system"
    ) -> bool:
        """Restore a prior answer recorded by :meth:`replace` in the audit trail.

        With ``to_version=None`` the most recent prior answer is restored;
        otherwise the answer captured when the row held ``to_version``. Returns
        ``False`` when there is no such history.
        """
        history = [
            a
            for a in self.get_audit_log(entry_id)
            if a["operation"] == "update" and "old_answer" in a["metadata"]
        ]
        if not history:
            return False
        if to_version is None:
            target = history[-1]
        else:
            target = next(
                (a for a in history if a["metadata"].get("old_version") == to_version),
                None,
            )
        if target is None:
            return False
        old_answer = target["metadata"]["old_answer"]
        with self._lock:
            cur = self._conn.execute(
                "UPDATE knowledge SET answer = ?, version = version + 1 WHERE id = ?",
                (old_answer, entry_id),
            )
            changed = cur.rowcount > 0
            if changed:
                self._audit(
                    entry_id,
                    "rollback",
                    operator=operator,
                    metadata={"restored_answer": old_answer},
                )
            self._conn.commit()
        return changed

    def get_audit_log(self, entry_id: int | None = None) -> list[dict[str, Any]]:
        """Return the audit trail, for one entry or (``None``) the whole floor."""
        with self._lock:
            if entry_id is None:
                rows = self._conn.execute(
                    "SELECT * FROM knowledge_audit ORDER BY id"
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM knowledge_audit WHERE kb_id = ? ORDER BY id",
                    (entry_id,),
                ).fetchall()
        return [
            {
                "id": r["id"],
                "kb_id": r["kb_id"],
                "operation": r["operation"],
                "operator": r["operator"],
                "reason": r["reason"],
                "metadata": json.loads(r["metadata"]),
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def delete(self, entry_id: int) -> bool:
        """Tear a (wrong) entry out of the soil. Returns ``True`` if removed."""
        with self._lock:
            cur = self._conn.execute("DELETE FROM knowledge WHERE id = ?", (entry_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def reset(self) -> None:
        """Empty the entire floor (mainly for tests)."""
        with self._lock:
            self._conn.execute("DELETE FROM knowledge")
            self._conn.commit()

    # -- reads (where does the leaf land?) ------------------------------------

    def _row_to_entry(self, row: sqlite3.Row) -> KnowledgeEntry:
        return KnowledgeEntry(
            id=row["id"],
            question=row["question"],
            answer=row["answer"],
            source=row["source"],
            metadata=json.loads(row["metadata"]),
            created_at=row["created_at"],
            confidence=row["confidence"],
            expiry_at=row["expiry_at"],
            version=row["version"],
            superseded_by=row["superseded_by"],
            owner=row["owner"],
            status=row["status"],
        )

    def get(self, entry_id: int) -> KnowledgeEntry | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM knowledge WHERE id = ?", (entry_id,)
            ).fetchone()
        return self._row_to_entry(row) if row is not None else None

    def all(self, *, include_quarantined: bool = False) -> list[KnowledgeEntry]:
        """Return stored entries -- active only by default (quarantined excluded)."""
        with self._lock:
            if include_quarantined:
                rows = self._conn.execute(
                    "SELECT * FROM knowledge ORDER BY id"
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM knowledge WHERE status = ? ORDER BY id",
                    (KnowledgeStatus.ACTIVE,),
                ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def count(self, *, include_quarantined: bool = False) -> int:
        """Count stored entries -- active only by default (quarantined excluded)."""
        with self._lock:
            if include_quarantined:
                return int(
                    self._conn.execute("SELECT COUNT(*) FROM knowledge").fetchone()[0]
                )
            return int(
                self._conn.execute(
                    "SELECT COUNT(*) FROM knowledge WHERE status = ?",
                    (KnowledgeStatus.ACTIVE,),
                ).fetchone()[0]
            )

    def search(self, query: str, k: int = 3) -> list[SearchHit]:
        """Embed ``query`` and return the ``k`` nearest rows, best score first.

        This is the moment the ticket "falls onto the floor near the answer":
        the query vector is compared against every stored vector by cosine
        similarity and the closest matches float to the top. Only *active*,
        unexpired entries are served -- quarantined or expired knowledge is
        never retrieved.
        """
        if k <= 0:
            return []
        query_vec = self.embedder.encode(query)
        now = time.time()
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM knowledge WHERE status = ? "
                "AND (expiry_at IS NULL OR expiry_at > ?)",
                (KnowledgeStatus.ACTIVE, now),
            ).fetchall()
        scored: list[SearchHit] = []
        for row in rows:
            score = cosine_similarity(query_vec, _blob_to_vec(row["embedding"]))
            scored.append(SearchHit(entry=self._row_to_entry(row), score=score))
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:k]

    def best_match(self, query: str) -> SearchHit | None:
        """Convenience: the single closest hit, or ``None`` on an empty floor."""
        hits = self.search(query, k=1)
        return hits[0] if hits else None
