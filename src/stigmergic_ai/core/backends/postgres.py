"""A production Pheromone Ground backed by PostgreSQL.

Why Postgres? Two reasons the SQLite reference ground can't match:

* **Lock-free concurrent claims.** ``SELECT ... FOR UPDATE SKIP LOCKED`` lets an
  arbitrary number of worker ants grab *different* pheromones simultaneously
  without ever blocking each other -- the database hands each claimant a distinct
  unclaimed row and silently skips the locked ones. This is the canonical
  work-queue pattern, done by the engine instead of an application mutex.
* **True push, zero polling.** ``LISTEN/NOTIFY`` turns the ground into an
  event stream: the instant any process lays a fresh trail, every waiting ant
  -- in *any* process, on *any* machine -- is woken by the server. No tight poll
  loop, no latency/CPU trade-off to tune.

``psycopg`` (v3) is imported lazily, so importing this module costs nothing until
you actually construct a :class:`PostgresGround`. Requires ``psycopg>=3.2`` for
the ``notifies(timeout=..., stop_after=...)`` API.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import TYPE_CHECKING, Any, Callable

from stigmergic_ai.core.environment import (
    AbstractGround,
    Entropy,
    EventSignal,
    GroundEvent,
    Pheromone,
    Status,
    TERMINAL_STATUSES,
    _coerce_status,
    _validate_entropy,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import psycopg

logger = logging.getLogger("stigmergic_ai.backends.postgres")

#: Identifiers (table/channel) are interpolated into SQL -- they cannot be bound
#: as parameters -- so they are validated against this allow-list to slam the
#: door on SQL injection (OWASP A03). Everything else is parameterized.
_SAFE_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _require_psycopg() -> "psycopg":
    """Import psycopg on demand with a friendly, actionable error if missing."""
    try:
        import psycopg  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised without the driver
        raise ImportError(
            "PostgresGround requires the psycopg driver. Install it with:\n"
            "    pip install 'psycopg[binary]>=3.2'\n"
            "(or add the 'postgres' extra: pip install 'stigmergic-ai[postgres]')."
        ) from exc
    return psycopg


def _safe_ident(name: str, *, kind: str) -> str:
    """Return ``name`` if it is a safe SQL identifier, else raise ``ValueError``."""
    if not _SAFE_IDENT.match(name):
        raise ValueError(
            f"Unsafe {kind} name {name!r}: must match {_SAFE_IDENT.pattern} "
            "(letters, digits, underscores; not starting with a digit)."
        )
    return name


class PostgresGround(AbstractGround):
    """A Pheromone Ground on PostgreSQL with ``SKIP LOCKED`` claims and ``NOTIFY``.

    The agent-facing surface is byte-for-byte identical to the SQLite
    :class:`~stigmergic_ai.core.environment.PheromoneGround`; only the substrate
    changes. A dedicated background connection holds a ``LISTEN`` and rebroadcasts
    every server notification onto the local :class:`EventSignal`, so both
    event-driven ants (:meth:`wait_for_change`) and the Swarm Inspector work
    across process and machine boundaries with no extra wiring.
    """

    def __init__(
        self,
        dsn: str,
        *,
        table: str = "pheromones",
        channel: str = "pheromone_events",
    ) -> None:
        """Connect, ensure the schema, and start listening for notifications.

        Args:
            dsn: A libpq/psycopg connection string (``postgresql://user:pw@host/db``).
            table: Table name for the pheromone field (validated as a safe ident).
            channel: ``LISTEN/NOTIFY`` channel name (validated as a safe ident).
        """
        psycopg = _require_psycopg()

        self.dsn = dsn
        self.table = _safe_ident(table, kind="table")
        self.channel = _safe_ident(channel, kind="channel")
        self.events = EventSignal()

        self._seq = 0
        self._seq_lock = threading.Lock()
        self._closed = False

        from psycopg.rows import dict_row

        # Main connection: every statement autocommits. The atomic claim is a
        # single CTE statement, so it needs no explicit transaction.
        self._conn = psycopg.connect(dsn, autocommit=True, row_factory=dict_row)
        # Dedicated connection parked on LISTEN -- never used for queries.
        self._listen_conn = psycopg.connect(dsn, autocommit=True)

        self._ensure_schema()

        self._listen_conn.execute(f"LISTEN {self.channel}")
        self._stop_listen = threading.Event()
        self._listener = threading.Thread(
            target=self._listen_loop,
            name=f"pgground-listen-{self.channel}",
            daemon=True,
        )
        self._listener.start()
        logger.debug("PostgresGround ready (table=%s channel=%s).", self.table, self.channel)

    # -- schema ----------------------------------------------------------------

    def _ensure_schema(self) -> None:
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.table} (
                id          BIGSERIAL PRIMARY KEY,
                raw_data    TEXT NOT NULL,
                latent_blob BYTEA,
                entropy     DOUBLE PRECISION NOT NULL,
                status      TEXT NOT NULL,
                owner       TEXT,
                created_at  DOUBLE PRECISION NOT NULL,
                updated_at  DOUBLE PRECISION NOT NULL,
                metadata    JSONB NOT NULL DEFAULT '{{}}'::jsonb
            )
            """
        )
        # Index the claim's sort/filter so SKIP LOCKED scans stay cheap at scale.
        self._conn.execute(
            f"CREATE INDEX IF NOT EXISTS {self.table}_claim_idx "
            f"ON {self.table} (status, entropy DESC, id ASC)"
        )

    # -- write side ------------------------------------------------------------

    def inject_chaos(
        self,
        raw_data: str,
        *,
        entropy: float = Entropy.CHAOS,
        status: Status | str = Status.RAW,
        metadata: dict[str, Any] | None = None,
        redactor: "Callable[[str], tuple[str, list[str]]] | None" = None,
    ) -> int:
        """Deposit a new pheromone and return its server-assigned id.

        ``redactor`` (``text -> (clean_text, categories)``) is applied to
        ``raw_data`` before it is written, so sensitive data never reaches the
        durable store; any categories are recorded under ``pii_redacted_at_intake``.
        """
        from psycopg.types.json import Jsonb

        self._ensure_open()
        entropy = _validate_entropy(entropy)
        status = _coerce_status(status)
        now = time.time()
        meta = dict(metadata or {})
        if redactor is not None:
            raw_data, redacted = redactor(raw_data)
            if redacted:
                meta.setdefault("pii_redacted_at_intake", redacted)
        row = self._conn.execute(
            f"""
            INSERT INTO {self.table}
                (raw_data, latent_blob, entropy, status, owner, created_at, updated_at, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (raw_data, None, entropy, status.value, None, now, now, Jsonb(meta)),
        ).fetchone()
        task_id = int(row["id"])
        self._notify("INJECT", task_id)
        return task_id

    def claim(
        self,
        owner: str,
        *,
        min_entropy: float = Entropy.MIN,
        status: Status | str | None = None,
        new_status: Status | str = Status.CLAIMED,
    ) -> Pheromone | None:
        """Atomically grab the most urgent matching pheromone (lock-free).

        Uses ``FOR UPDATE SKIP LOCKED`` so concurrent claimants never collide:
        the server hands each caller a different unclaimed row and skips any that
        a peer already holds. Terminal pheromones are always excluded.
        """
        self._ensure_open()
        min_entropy = _validate_entropy(min_entropy)
        new_status = _coerce_status(new_status)
        terminal = [s.value for s in TERMINAL_STATUSES]

        where = ["entropy >= %s", "status <> ALL(%s)"]
        params: list[Any] = [min_entropy, terminal]
        if status is not None:
            where.append("status = %s")
            params.append(_coerce_status(status).value)

        now = time.time()
        sql = f"""
            WITH candidate AS (
                SELECT id FROM {self.table}
                WHERE {' AND '.join(where)}
                ORDER BY entropy DESC, id ASC
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            UPDATE {self.table} AS t
               SET status = %s, owner = %s, updated_at = %s
              FROM candidate
             WHERE t.id = candidate.id
            RETURNING t.*
        """
        row = self._conn.execute(sql, (*params, new_status.value, owner, now)).fetchone()
        if row is None:
            return None
        claimed = self._row_to_pheromone(row)
        self._notify("CLAIM", claimed.id)
        logger.debug("Claimed id=%s by owner=%s -> %s", claimed.id, owner, new_status.value)
        return claimed

    def update_state(
        self,
        task_id: int,
        *,
        entropy: float | None = None,
        status: Status | str | None = None,
        raw_data: str | None = None,
        latent_blob: bytes | None = None,
        owner: str | None = None,
        clear_owner: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Mutate a pheromone in place; only non-None fields are written.

        ``metadata`` replaces the stored blob wholesale (matching the SQLite
        ground). ``clear_owner=True`` releases the trail back to the colony.
        """
        from psycopg.types.json import Jsonb

        self._ensure_open()
        assignments: list[str] = []
        params: list[Any] = []
        if entropy is not None:
            assignments.append("entropy = %s")
            params.append(_validate_entropy(entropy))
        if status is not None:
            assignments.append("status = %s")
            params.append(_coerce_status(status).value)
        if raw_data is not None:
            assignments.append("raw_data = %s")
            params.append(raw_data)
        if latent_blob is not None:
            assignments.append("latent_blob = %s")
            params.append(latent_blob)
        if clear_owner:
            assignments.append("owner = NULL")
        elif owner is not None:
            assignments.append("owner = %s")
            params.append(owner)
        if metadata is not None:
            assignments.append("metadata = %s")
            params.append(Jsonb(metadata))

        assignments.append("updated_at = %s")
        params.append(time.time())
        params.append(int(task_id))

        sql = f"UPDATE {self.table} SET {', '.join(assignments)} WHERE id = %s"
        cur = self._conn.execute(sql, params)
        changed = cur.rowcount > 0
        if not changed:
            logger.warning("update_state() found no pheromone with id=%s", task_id)
        else:
            self._notify("MUTATE", int(task_id))
        return changed

    # -- read side -------------------------------------------------------------

    def sense(
        self,
        *,
        min_entropy: float = Entropy.MIN,
        status: Status | str | None = None,
        limit: int = 10,
    ) -> list[Pheromone]:
        """Read the most urgent matching pheromones without mutating anything."""
        self._ensure_open()
        min_entropy = _validate_entropy(min_entropy)
        where = ["entropy >= %s"]
        params: list[Any] = [min_entropy]
        if status is not None:
            where.append("status = %s")
            params.append(_coerce_status(status).value)
        params.append(int(limit))
        rows = self._conn.execute(
            f"SELECT * FROM {self.table} WHERE {' AND '.join(where)} "
            f"ORDER BY entropy DESC, id ASC LIMIT %s",
            params,
        ).fetchall()
        return [self._row_to_pheromone(r) for r in rows]

    def get(self, task_id: int) -> Pheromone | None:
        """Fetch a single pheromone by id, or None."""
        self._ensure_open()
        row = self._conn.execute(
            f"SELECT * FROM {self.table} WHERE id = %s", (int(task_id),)
        ).fetchone()
        return self._row_to_pheromone(row) if row is not None else None

    def global_entropy(self) -> float:
        """Total residual entropy across all non-terminal pheromones."""
        self._ensure_open()
        terminal = [s.value for s in TERMINAL_STATUSES]
        row = self._conn.execute(
            f"SELECT COALESCE(SUM(entropy), 0.0) AS total FROM {self.table} "
            f"WHERE status <> ALL(%s)",
            (terminal,),
        ).fetchone()
        return float(row["total"])

    def stats(self) -> dict[str, int]:
        """Census of pheromones grouped by status."""
        self._ensure_open()
        rows = self._conn.execute(
            f"SELECT status, COUNT(*) AS n FROM {self.table} GROUP BY status"
        ).fetchall()
        return {r["status"]: int(r["n"]) for r in rows}

    def count(self, status: Status | str | None = None) -> int:
        """Count pheromones, optionally filtered to a single status."""
        self._ensure_open()
        if status is None:
            row = self._conn.execute(f"SELECT COUNT(*) AS n FROM {self.table}").fetchone()
        else:
            row = self._conn.execute(
                f"SELECT COUNT(*) AS n FROM {self.table} WHERE status = %s",
                (_coerce_status(status).value,),
            ).fetchone()
        return int(row["n"])

    def reset(self) -> None:
        """Wipe every pheromone and restart the id sequence."""
        self._ensure_open()
        self._conn.execute(f"TRUNCATE TABLE {self.table} RESTART IDENTITY")
        self.events.signal()

    # -- event plumbing --------------------------------------------------------

    def wait_for_change(
        self, timeout: float, stop_event: threading.Event | None = None
    ) -> bool:
        """Block until the ground changes or ``timeout`` elapses.

        The background listener rebroadcasts every server ``NOTIFY`` onto the
        local :class:`EventSignal`, so this rides the same in-process bus as the
        SQLite ground -- yet wakes on mutations from *any* process.
        """
        return self.events.wait(timeout, stop_event)

    def _next_seq(self) -> int:
        with self._seq_lock:
            self._seq += 1
            return self._seq

    def _notify(self, event_type: str, task_id: int) -> None:
        """Emit a server-side ``NOTIFY`` carrying the post-mutation snapshot.

        The payload is JSON so any listener -- in this process or another -- can
        reconstruct a :class:`GroundEvent` without a follow-up query.
        """
        snapshot = self.get(task_id)
        if snapshot is None:
            payload = {
                "event_type": event_type, "task_id": task_id,
                "status": Status.RESOLVED.value, "entropy": 0.0,
                "owner": None, "global_entropy": self.global_entropy(),
            }
        else:
            payload = {
                "event_type": event_type, "task_id": snapshot.id,
                "status": snapshot.status.value, "entropy": snapshot.entropy,
                "owner": snapshot.owner, "global_entropy": self.global_entropy(),
            }
        # pg_notify() parameterizes the payload safely (NOTIFY cannot bind args).
        self._conn.execute("SELECT pg_notify(%s, %s)", (self.channel, json.dumps(payload)))

    def _listen_loop(self) -> None:
        """Drain server notifications and rebroadcast them onto the local bus."""
        try:
            while not self._stop_listen.is_set():
                try:
                    for note in self._listen_conn.notifies(timeout=0.5):
                        self._dispatch_notify(note.payload)
                except Exception:  # noqa: BLE001 -- a transient listen error must not kill the thread
                    if self._stop_listen.is_set():
                        break
                    logger.exception("LISTEN loop error; retrying.")
                    time.sleep(0.5)
        finally:
            logger.debug("PostgresGround listener stopped.")

    def _dispatch_notify(self, payload: str) -> None:
        """Turn a NOTIFY payload into a GroundEvent and fan it out locally."""
        try:
            data = json.loads(payload)
            event = GroundEvent(
                seq=self._next_seq(),
                ts=time.time(),
                event_type=str(data.get("event_type", "MUTATE")),
                task_id=int(data.get("task_id", 0)),
                status=_coerce_status(data.get("status", Status.RESOLVED)),
                entropy=float(data.get("entropy", 0.0)),
                owner=data.get("owner"),
                global_entropy=float(data.get("global_entropy", 0.0)),
            )
        except Exception:  # noqa: BLE001 -- a malformed payload still wakes waiters
            self.events.signal()
            return
        self.events.signal(lambda: event)

    # -- lifecycle -------------------------------------------------------------

    def close(self) -> None:
        """Stop the listener and close both connections (idempotent)."""
        if self._closed:
            return
        self._closed = True
        self._stop_listen.set()
        # Wake the local bus so any ant blocked in wait_for_change returns.
        self.events.signal()
        if self._listener.is_alive():
            self._listener.join(timeout=2.0)
        for conn in (self._listen_conn, self._conn):
            try:
                conn.close()
            except Exception:  # noqa: BLE001 -- closing is best-effort
                pass
        logger.debug("PostgresGround closed.")

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("PostgresGround is closed.")

    @staticmethod
    def _row_to_pheromone(row: dict[str, Any]) -> Pheromone:
        """Map a psycopg ``dict_row`` into a :class:`Pheromone`."""
        blob = row.get("latent_blob")
        if isinstance(blob, memoryview):  # psycopg may hand back a memoryview
            blob = bytes(blob)
        return Pheromone(
            id=int(row["id"]),
            raw_data=row["raw_data"],
            latent_blob=blob,
            entropy=float(row["entropy"]),
            status=row["status"],
            owner=row.get("owner"),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            metadata=row.get("metadata") or {},
        )
