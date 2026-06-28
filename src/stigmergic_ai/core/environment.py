"""The Pheromone Ground: the shared semantic field of the StigmergicAI swarm.

In stigmergy, organisms do not communicate directly. They modify a shared
environment, and *the environment itself* coordinates their collective
behaviour. An ant deposits a pheromone; another ant, much later and with no
knowledge of the first, senses that chemical trail and reacts to it.

``PheromoneGround`` is the digital substrate for exactly that. It is a thin,
thread-safe SQLite wrapper that stores **pheromones** -- units of work carrying
a chaotic ``raw_data`` payload, an optional latent tensor blob, an ``entropy``
score (how unresolved/chaotic the task still is), and a ``status`` trail. Agents
never call each other; they only:

* :meth:`PheromoneGround.inject_chaos`  -- raise entropy (deposit new work),
* :meth:`PheromoneGround.sense`         -- smell the field (pure read),
* :meth:`PheromoneGround.claim`         -- atomically grab one task, and
* :meth:`PheromoneGround.update_state`  -- lower entropy / lay a fresh trail.

This module deliberately depends only on the Python standard library and
``pydantic``. No torch, no transformers, no orchestration framework. The heavy
cognitive horizons (Byzantine consensus, latent transfer) build *on top* of this
substrate -- they never leak into it.
"""

from __future__ import annotations

import abc
import itertools
import json
import logging
import sqlite3
import threading
import time
from enum import Enum
from typing import Any, Callable

from pydantic import BaseModel, Field, field_validator

__all__ = [
    "Status",
    "Entropy",
    "Pheromone",
    "GroundEvent",
    "EventSignal",
    "AbstractGround",
    "PheromoneGround",
]

logger = logging.getLogger("stigmergic_ai.environment")


class Status(str, Enum):
    """The chemical trail a pheromone currently carries.

    Statuses describe *which kind of ant* should react next, not which ant
    produced them. This is what keeps the swarm decoupled: a Solver does not
    know a Governance ant exists, it only knows it reacts to ``HYGIENIZED``.
    """

    RAW = "RAW"
    """Freshly injected chaos. Unsanitized, untrusted, high entropy."""

    CLAIMED = "CLAIMED"
    """An ant has atomically grabbed this task and is metabolizing it."""

    HYGIENIZED = "HYGIENIZED"
    """Sanitized and safe. A clean trail downstream solvers can trust."""

    PENDING_CONSENSUS = "PENDING_CONSENSUS"
    """A Solver has proposed a resolution; it awaits the Byzantine quorum.

    The proposal is *not* yet committed. A Verifier caste must smell this trail,
    run the Semantic Raft quorum over the proposed logic, and only then drive the
    pheromone to :attr:`RESOLVED` (quorum passed) or :attr:`SLASHED` (rejected).
    """

    LATENT_READY = "LATENT_READY"
    """A latent tensor (Horizon 3) has been deposited in ``latent_blob``.

    A Reader caste has distilled a heavy payload into a hidden-state tensor and
    parked it here. A downstream caste consumes the tensor directly -- injecting
    it into its own residual stream -- without ever re-reading the original text.
    This is Latent State Transfer: pure mathematical context, zero String Tax.
    """

    RESOLVED = "RESOLVED"
    """Terminal success. Entropy has been driven to zero."""

    SLASHED = "SLASHED"
    """Terminal failure. Rejected by Byzantine quorum (hallucination/injection)."""


#: Statuses from which no further work should be picked up.
TERMINAL_STATUSES: frozenset[Status] = frozenset({Status.RESOLVED, Status.SLASHED})


class Entropy:
    """Named entropy bands, in the closed interval ``[0.0, 1.0]``.

    Entropy is the swarm's only global signal. High entropy means "chaotic,
    unresolved work is here"; zero entropy means "settled". Ants wake on
    thresholds rather than on messages, which is what makes the system
    eventually consistent and crash-resilient.
    """

    CHAOS = 1.0
    """Maximal disorder -- freshly injected, completely unprocessed."""

    HIGH = 0.7
    """The classic Governance wake threshold (README: ``entropy > 0.7``)."""

    MEDIUM = 0.4

    LOW = 0.2
    """A hygienized trail: low residual entropy, safe to solve."""

    ZERO = 0.0
    """Fully resolved. The pheromone has evaporated."""

    MIN = 0.0
    MAX = 1.0


def _validate_entropy(value: float) -> float:
    """Coerce ``value`` to ``float`` and assert it lies within ``[0.0, 1.0]``."""
    try:
        entropy = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Entropy must be a real number, got {value!r}.") from exc
    if not (Entropy.MIN <= entropy <= Entropy.MAX):
        raise ValueError(
            f"Entropy must lie within [{Entropy.MIN}, {Entropy.MAX}], got {entropy}."
        )
    return entropy


def _coerce_status(value: Status | str) -> Status:
    """Normalize a status given as a :class:`Status` or raw string."""
    if isinstance(value, Status):
        return value
    try:
        return Status(value)
    except ValueError as exc:
        valid = ", ".join(s.value for s in Status)
        raise ValueError(
            f"Unknown status {value!r}. Expected one of: {valid}."
        ) from exc


class Pheromone(BaseModel):
    """A single unit of work resting on the Pheromone Ground (one DB row).

    A pheromone bundles the raw request, an optional latent tensor blob (for
    Horizon 3 / Latent State Transfer), its current entropy, and the chemical
    trail (``status``) that tells the swarm what to do with it next.
    """

    id: int
    raw_data: str
    latent_blob: bytes | None = None
    entropy: float = Field(ge=Entropy.MIN, le=Entropy.MAX)
    status: Status
    owner: str | None = None
    created_at: float
    updated_at: float
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata", mode="before")
    @classmethod
    def _parse_metadata(cls, value: Any) -> dict[str, Any]:
        """Accept a JSON string (as stored in SQLite) or an existing mapping."""
        if value is None or value == "":
            return {}
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                logger.warning("Corrupt metadata JSON; defaulting to empty dict.")
                return {}
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        if isinstance(value, dict):
            return value
        return {"value": value}


class GroundEvent(BaseModel):
    """A single observable transition on the ground (for the Swarm Inspector).

    The ground emits one of these after every successful mutation. Because a
    stigmergic system has no central orchestrator to log a call graph, this
    event stream *is* the swarm's flight recorder: replay it to reconstruct the
    full life of any pheromone, or chart the field's entropy over time.
    """

    seq: int
    ts: float
    event_type: str
    task_id: int
    status: Status
    entropy: float
    owner: str | None = None
    global_entropy: float = 0.0


class EventSignal:
    """An in-process wake-and-fan-out bus for ground mutations.

    It plays two roles at once:

    * **Wake:** ants can *wait* on it instead of busy-polling -- a consumer
      sleeps until the ground actually changes (or a timeout elapses), turning
      the swarm event-driven and killing the CPU cost of tight poll loops.
    * **Fan-out:** observers (the ``SwarmInspector``) subscribe to receive every
      :class:`GroundEvent`. Listener exceptions are swallowed, so a buggy
      observer can never stall or crash the colony.

    The event object is built lazily via a factory, so when nobody is listening
    the ground pays nothing beyond bumping a counter and notifying waiters.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._token = 0
        self._listeners: list[Callable[[GroundEvent], None]] = []

    def subscribe(self, listener: Callable[[GroundEvent], None]) -> Callable[[], None]:
        """Register ``listener``; returns a zero-arg unsubscribe handle."""
        with self._cond:
            self._listeners.append(listener)
        return lambda: self.unsubscribe(listener)

    def unsubscribe(self, listener: Callable[[GroundEvent], None]) -> None:
        """Remove a previously-subscribed listener (idempotent)."""
        with self._cond:
            try:
                self._listeners.remove(listener)
            except ValueError:
                pass

    @property
    def listening(self) -> bool:
        """``True`` while at least one observer is subscribed."""
        return bool(self._listeners)

    def signal(self, event_factory: Callable[[], GroundEvent] | None = None) -> None:
        """Wake all waiters and, if anyone is listening, fan out an event.

        ``event_factory`` is invoked only when there is at least one subscriber,
        so the (potentially non-trivial) event construction is skipped entirely
        on the hot path of an unobserved swarm.
        """
        with self._cond:
            self._token += 1
            self._cond.notify_all()
            listeners = list(self._listeners)
        if event_factory is None or not listeners:
            return
        event = event_factory()
        for listener in listeners:
            try:
                listener(event)
            except Exception:  # noqa: BLE001 -- an observer must never break the swarm
                logger.exception("Ground event listener failed; ignored.")

    def wait(self, timeout: float, stop_event: threading.Event | None = None) -> bool:
        """Block up to ``timeout`` seconds for the next signal.

        Returns ``True`` if the ground changed, ``False`` on timeout. If
        ``stop_event`` is already set, returns immediately so a stopping ant
        never sleeps.
        """
        with self._cond:
            if stop_event is not None and stop_event.is_set():
                return False
            start = self._token
            self._cond.wait(timeout)
            return self._token != start


class AbstractGround(abc.ABC):
    """The pluggable contract every Pheromone Ground backend must satisfy.

    A *ground* is the shared semantic field the swarm coordinates through. The
    reference backend is SQLite (:class:`PheromoneGround`); production backends
    (Postgres with ``LISTEN/NOTIFY``, Redis, DynamoDB) implement this same small
    surface so the agents -- which only ever touch these ten methods -- stay
    completely storage-agnostic. Swap the substrate, keep the swarm.

    Every concrete ground owns an :class:`EventSignal` (``self.events``) and is
    expected to ``signal`` it after each successful mutation. That single
    convention is what lets both event-driven ants and the Swarm Inspector work
    uniformly across every backend.
    """

    events: EventSignal

    @abc.abstractmethod
    def inject_chaos(self, raw_data: str, *, entropy: float = Entropy.CHAOS,
                     status: Status | str = Status.RAW,
                     metadata: dict[str, Any] | None = None) -> int:
        """Deposit a new pheromone, raising the field's entropy. Returns its id."""

    @abc.abstractmethod
    def sense(self, *, min_entropy: float = Entropy.MIN,
              status: Status | str | None = None, limit: int = 10) -> list[Pheromone]:
        """Read the most urgent matching pheromones without mutating anything."""

    @abc.abstractmethod
    def claim(self, owner: str, *, min_entropy: float = Entropy.MIN,
              status: Status | str | None = None,
              new_status: Status | str = Status.CLAIMED) -> Pheromone | None:
        """Atomically grab the single most urgent matching pheromone, or None."""

    @abc.abstractmethod
    def update_state(self, task_id: int, *, entropy: float | None = None,
                     status: Status | str | None = None, raw_data: str | None = None,
                     latent_blob: bytes | None = None, owner: str | None = None,
                     clear_owner: bool = False,
                     metadata: dict[str, Any] | None = None) -> bool:
        """Mutate a pheromone (lower entropy / lay a fresh trail). Returns success."""

    @abc.abstractmethod
    def get(self, task_id: int) -> Pheromone | None:
        """Fetch a single pheromone by id, or None."""

    @abc.abstractmethod
    def global_entropy(self) -> float:
        """Total residual entropy across all non-terminal pheromones."""

    @abc.abstractmethod
    def stats(self) -> dict[str, int]:
        """Census of pheromones grouped by status."""

    @abc.abstractmethod
    def count(self, status: Status | str | None = None) -> int:
        """Count pheromones, optionally filtered to a single status."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Wipe every pheromone from the ground."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release the backend's resources (idempotent)."""

    def wait_for_change(self, timeout: float, stop_event: threading.Event | None = None) -> bool:
        """Block until the ground changes or ``timeout`` elapses (event-driven).

        Backends with a native push channel (Postgres ``LISTEN/NOTIFY``) override
        this; the default rides on the in-process :class:`EventSignal`, which is
        already enough to make a single-process swarm fully event-driven.
        """
        return self.events.wait(timeout, stop_event)

    def __enter__(self) -> "AbstractGround":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class PheromoneGround(AbstractGround):
    """A thread-safe SQLite substrate the swarm reads from and writes to.

    A single connection is shared across every ant thread (guarded by a
    re-entrant lock) so that an in-memory database -- which is private to one
    connection -- still works for tests and demos. For file-backed databases,
    WAL journaling and a busy timeout additionally keep things sane under
    multi-process access.

    The object is a context manager::

        with PheromoneGround("swarm.db") as ground:
            task_id = ground.inject_chaos("update salary to 15k urgent")
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS pheromones (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        raw_data    TEXT    NOT NULL,
        latent_blob BLOB,
        entropy     REAL    NOT NULL,
        status      TEXT    NOT NULL,
        owner       TEXT,
        created_at  REAL    NOT NULL,
        updated_at  REAL    NOT NULL,
        metadata    TEXT
    );
    """

    _INDEXES = (
        "CREATE INDEX IF NOT EXISTS idx_pheromones_status "
        "ON pheromones (status);",
        "CREATE INDEX IF NOT EXISTS idx_pheromones_sense "
        "ON pheromones (status, entropy DESC, id ASC);",
    )

    _COLUMNS = (
        "id, raw_data, latent_blob, entropy, status, "
        "owner, created_at, updated_at, metadata"
    )

    def __init__(self, db_path: str = ":memory:", *, busy_timeout_ms: int = 5000) -> None:
        """Open (and initialize) the Pheromone Ground.

        Args:
            db_path: Path to the SQLite file, or ``":memory:"`` (default) for an
                ephemeral in-process ground.
            busy_timeout_ms: How long a writer waits on a locked database before
                raising, in milliseconds. Relevant for file-backed grounds under
                concurrent access.
        """
        self.db_path = db_path
        self.events = EventSignal()
        self._seq = itertools.count(1)
        self._lock = threading.RLock()
        # check_same_thread=False: the connection is created here but used by
        # background ant threads. We serialize every access through _lock, so
        # this is safe. isolation_level=None puts us in autocommit mode, giving
        # us explicit control over the BEGIN IMMEDIATE transaction in claim().
        self._conn = sqlite3.connect(
            db_path,
            check_same_thread=False,
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._closed = False
        self._configure(busy_timeout_ms)
        self._init_schema()
        logger.info("PheromoneGround initialized at %s", db_path)

    # -- setup ----------------------------------------------------------------

    def _configure(self, busy_timeout_ms: int) -> None:
        """Apply pragmas that make concurrent ant access robust."""
        with self._lock:
            cur = self._conn.cursor()
            # WAL allows concurrent readers alongside a writer (no-op / 'memory'
            # for :memory: databases, which is harmless).
            cur.execute("PRAGMA journal_mode=WAL;")
            cur.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)};")
            cur.execute("PRAGMA foreign_keys=ON;")
            cur.close()

    def _init_schema(self) -> None:
        """Create the ``pheromones`` table and its indexes if absent."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(self._SCHEMA)
            for statement in self._INDEXES:
                cur.execute(statement)
            cur.close()

    # -- deposit (raise entropy) ---------------------------------------------

    def inject_chaos(
        self,
        raw_data: str,
        *,
        entropy: float = Entropy.CHAOS,
        status: Status | str = Status.RAW,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Deposit a new pheromone, *raising* the field's entropy.

        This is the Forager's primitive: it floods the ground with chaotic,
        unsanitized work and walks away. No ant is notified; the work simply
        exists now, waiting to be smelled.

        Args:
            raw_data: The chaotic request payload (stored verbatim, parameterized).
            entropy: Initial entropy in ``[0.0, 1.0]`` (defaults to full chaos).
            status: Initial chemical trail (defaults to :attr:`Status.RAW`).
            metadata: Optional JSON-serializable side-channel data.

        Returns:
            The autoincrement id of the freshly deposited pheromone.
        """
        entropy = _validate_entropy(entropy)
        status = _coerce_status(status)
        now = time.time()
        payload = json.dumps(metadata or {})
        with self._lock:
            self._ensure_open()
            cur = self._conn.execute(
                "INSERT INTO pheromones "
                "(raw_data, latent_blob, entropy, status, owner, "
                "created_at, updated_at, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                (raw_data, None, entropy, status.value, None, now, now, payload),
            )
            task_id = int(cur.lastrowid)
        self._emit("INJECT", task_id)
        logger.debug(
            "Chaos injected: id=%s entropy=%.3f status=%s", task_id, entropy, status.value
        )
        return task_id

    # -- sense (pure read) ----------------------------------------------------

    def sense(
        self,
        *,
        min_entropy: float = Entropy.MIN,
        status: Status | str | None = None,
        limit: int = 10,
    ) -> list[Pheromone]:
        """Smell the field without mutating it.

        Returns the most urgent matching pheromones first (highest entropy, then
        oldest). This is a read-only perception primitive -- it never claims or
        changes anything, so several ants may sense the same task.

        Args:
            min_entropy: Only return pheromones with ``entropy >= min_entropy``.
            status: If given, restrict to this chemical trail.
            limit: Maximum number of pheromones to return.

        Returns:
            A list of :class:`Pheromone` ordered by urgency.
        """
        min_entropy = _validate_entropy(min_entropy)
        clauses = ["entropy >= ?"]
        params: list[Any] = [min_entropy]
        if status is not None:
            clauses.append("status = ?")
            params.append(_coerce_status(status).value)
        params.append(int(limit))
        sql = (
            f"SELECT {self._COLUMNS} FROM pheromones "
            f"WHERE {' AND '.join(clauses)} "
            "ORDER BY entropy DESC, id ASC LIMIT ?;"
        )
        with self._lock:
            self._ensure_open()
            rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_pheromone(row) for row in rows]

    # -- claim (atomic grab) --------------------------------------------------

    def claim(
        self,
        owner: str,
        *,
        min_entropy: float = Entropy.MIN,
        status: Status | str | None = None,
        new_status: Status | str = Status.CLAIMED,
    ) -> Pheromone | None:
        """Atomically grab the single most urgent matching pheromone.

        This is what makes a multi-ant swarm safe: two ants of the same caste
        polling simultaneously will never both grab the same task. The select +
        update happens inside a ``BEGIN IMMEDIATE`` transaction (and under the
        in-process lock), so exactly one wins; the loser gets the next task or
        ``None``.

        Args:
            owner: Identifier of the claiming ant (stamped onto the row).
            min_entropy: Only consider pheromones with ``entropy >= min_entropy``.
            status: If given, only claim pheromones currently on this trail.
            new_status: The trail to stamp on success (defaults to
                :attr:`Status.CLAIMED`).

        Returns:
            The claimed :class:`Pheromone` (already reflecting ``new_status`` and
            ``owner``), or ``None`` if nothing matched.
        """
        min_entropy = _validate_entropy(min_entropy)
        new_status = _coerce_status(new_status)
        terminal = tuple(s.value for s in TERMINAL_STATUSES)

        clauses = ["entropy >= ?", f"status NOT IN ({','.join('?' * len(terminal))})"]
        params: list[Any] = [min_entropy, *terminal]
        if status is not None:
            clauses.append("status = ?")
            params.append(_coerce_status(status).value)
        select_sql = (
            f"SELECT {self._COLUMNS} FROM pheromones "
            f"WHERE {' AND '.join(clauses)} "
            "ORDER BY entropy DESC, id ASC LIMIT 1;"
        )

        with self._lock:
            self._ensure_open()
            cur = self._conn.cursor()
            try:
                cur.execute("BEGIN IMMEDIATE;")
                row = cur.execute(select_sql, params).fetchone()
                if row is None:
                    cur.execute("COMMIT;")
                    return None
                now = time.time()
                cur.execute(
                    "UPDATE pheromones SET status = ?, owner = ?, updated_at = ? "
                    "WHERE id = ?;",
                    (new_status.value, owner, now, row["id"]),
                )
                cur.execute("COMMIT;")
            except sqlite3.Error:
                cur.execute("ROLLBACK;")
                logger.exception("claim() failed; rolled back.")
                raise
            finally:
                cur.close()

        claimed = self.get(int(row["id"]))
        if claimed is not None:
            self._emit("CLAIM", claimed.id)
            logger.debug("Claimed id=%s by owner=%s -> %s", claimed.id, owner, new_status.value)
        return claimed

    # -- mutate (lower entropy / lay trail) -----------------------------------

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
        """Mutate an existing pheromone -- the universal "act" primitive.

        Lowering ``entropy`` and stamping a new ``status`` is how an ant lays a
        fresh chemical trail for the next caste (or evaporates the pheromone
        entirely by driving entropy to zero). Only the fields you pass are
        touched; ``updated_at`` is always refreshed.

        Args:
            task_id: The pheromone to mutate.
            entropy: New entropy in ``[0.0, 1.0]``, if changing.
            status: New chemical trail, if changing.
            raw_data: Replacement raw payload (e.g. a sanitized rewrite), if any.
            latent_blob: Latent tensor bytes to attach (Horizon 3), if any.
            owner: New owner to stamp, if any.
            clear_owner: If ``True``, reset ``owner`` to ``NULL`` (mutually
                exclusive with ``owner``).
            metadata: Replacement metadata mapping, if changing.

        Returns:
            ``True`` if a row was updated, ``False`` if ``task_id`` did not exist.
        """
        if owner is not None and clear_owner:
            raise ValueError("Pass either owner or clear_owner, not both.")

        assignments: list[str] = []
        params: list[Any] = []
        if entropy is not None:
            assignments.append("entropy = ?")
            params.append(_validate_entropy(entropy))
        if status is not None:
            assignments.append("status = ?")
            params.append(_coerce_status(status).value)
        if raw_data is not None:
            assignments.append("raw_data = ?")
            params.append(raw_data)
        if latent_blob is not None:
            assignments.append("latent_blob = ?")
            params.append(latent_blob)
        if clear_owner:
            assignments.append("owner = NULL")
        elif owner is not None:
            assignments.append("owner = ?")
            params.append(owner)
        if metadata is not None:
            assignments.append("metadata = ?")
            params.append(json.dumps(metadata))

        # Always refresh the heartbeat, even for an otherwise empty mutation.
        assignments.append("updated_at = ?")
        params.append(time.time())
        params.append(int(task_id))

        sql = f"UPDATE pheromones SET {', '.join(assignments)} WHERE id = ?;"
        with self._lock:
            self._ensure_open()
            cur = self._conn.execute(sql, params)
            changed = cur.rowcount > 0
        if not changed:
            logger.warning("update_state() found no pheromone with id=%s", task_id)
        else:
            self._emit("MUTATE", int(task_id))
            logger.debug("Mutated id=%s (%s)", task_id, ", ".join(assignments[:-1]) or "touch")
        return changed

    # -- introspection --------------------------------------------------------

    def get(self, task_id: int) -> Pheromone | None:
        """Fetch a single pheromone by id, or ``None`` if it does not exist."""
        with self._lock:
            self._ensure_open()
            row = self._conn.execute(
                f"SELECT {self._COLUMNS} FROM pheromones WHERE id = ?;",
                (int(task_id),),
            ).fetchone()
        return self._row_to_pheromone(row) if row is not None else None

    def global_entropy(self) -> float:
        """Total residual entropy across all non-terminal pheromones.

        This is the swarm's single scalar health signal: a high number means the
        colony has chaotic work outstanding; ``0.0`` means the ground has fully
        settled.
        """
        terminal = tuple(s.value for s in TERMINAL_STATUSES)
        placeholders = ",".join("?" * len(terminal))
        with self._lock:
            self._ensure_open()
            row = self._conn.execute(
                "SELECT COALESCE(SUM(entropy), 0.0) AS total FROM pheromones "
                f"WHERE status NOT IN ({placeholders});",
                terminal,
            ).fetchone()
        return float(row["total"])

    def stats(self) -> dict[str, int]:
        """Return a count of pheromones grouped by status."""
        with self._lock:
            self._ensure_open()
            rows = self._conn.execute(
                "SELECT status, COUNT(*) AS n FROM pheromones GROUP BY status;"
            ).fetchall()
        return {row["status"]: int(row["n"]) for row in rows}

    def count(self, status: Status | str | None = None) -> int:
        """Count pheromones, optionally filtered to a single ``status``."""
        with self._lock:
            self._ensure_open()
            if status is None:
                row = self._conn.execute(
                    "SELECT COUNT(*) AS n FROM pheromones;"
                ).fetchone()
            else:
                row = self._conn.execute(
                    "SELECT COUNT(*) AS n FROM pheromones WHERE status = ?;",
                    (_coerce_status(status).value,),
                ).fetchone()
        return int(row["n"])

    # -- lifecycle ------------------------------------------------------------

    def reset(self) -> None:
        """Wipe every pheromone from the ground (primarily for tests/demos)."""
        with self._lock:
            self._ensure_open()
            self._conn.execute("DELETE FROM pheromones;")
        logger.info("PheromoneGround reset: all pheromones evaporated.")

    def close(self) -> None:
        """Close the underlying SQLite connection (idempotent)."""
        with self._lock:
            if self._closed:
                return
            self._conn.close()
            self._closed = True
        logger.info("PheromoneGround at %s closed.", self.db_path)

    def __enter__(self) -> "PheromoneGround":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- internals ------------------------------------------------------------

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("PheromoneGround is closed.")

    def _emit(self, event_type: str, task_id: int) -> None:
        """Wake any event-driven waiters and (if observed) fan out an event."""
        self.events.signal(lambda: self._build_event(event_type, task_id))

    def _build_event(self, event_type: str, task_id: int) -> GroundEvent:
        """Snapshot the post-mutation state of ``task_id`` into a GroundEvent.

        Built lazily (only when an observer is attached), so the extra reads
        here never burden an unobserved swarm.
        """
        snapshot = self.get(task_id)
        if snapshot is None:
            return GroundEvent(
                seq=next(self._seq), ts=time.time(), event_type=event_type,
                task_id=task_id, status=Status.RESOLVED, entropy=0.0,
                owner=None, global_entropy=self.global_entropy(),
            )
        return GroundEvent(
            seq=next(self._seq), ts=time.time(), event_type=event_type,
            task_id=snapshot.id, status=snapshot.status, entropy=snapshot.entropy,
            owner=snapshot.owner, global_entropy=self.global_entropy(),
        )

    @staticmethod
    def _row_to_pheromone(row: sqlite3.Row) -> Pheromone:
        """Build a validated :class:`Pheromone` from a raw SQLite row."""
        return Pheromone(
            id=row["id"],
            raw_data=row["raw_data"],
            latent_blob=row["latent_blob"],
            entropy=row["entropy"],
            status=row["status"],
            owner=row["owner"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=row["metadata"],
        )
