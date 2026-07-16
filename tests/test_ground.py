"""Pytest suite for the pluggable Pheromone Ground backends and event bus.

Everything here is dependency-free: the SQLite ground needs no driver, and the
PostgreSQL backend is exercised only along the paths that do *not* require a live
server (lazy import hygiene, the missing-driver error, identifier validation and
abstract-contract conformance). Spinning up a real Postgres is out of scope for a
unit run, so those server-dependent behaviours are covered by design review, not
by mocking a database.

Covered:

* event emission on inject / claim / mutate, and unsubscribe;
* a faulty listener never breaks the colony;
* :meth:`wait_for_change` -- timeout, wake-on-mutation, and stop-event short-cut;
* :class:`AbstractGround` cannot be instantiated; the SQLite ground fully
  implements it;
* the :func:`create_ground` factory routing and its error branches;
* :class:`PostgresGround` lazy-imports psycopg, raises a clean install error
  without it, and rejects unsafe table/channel identifiers (SQL-injection guard).
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import threading
import time

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from stigmergic.core.backends import create_ground  # noqa: E402
from stigmergic.core.environment import (  # noqa: E402
    AbstractGround,
    Entropy,
    GroundEvent,
    PheromoneGround,
    Status,
    TERMINAL_STATUSES,
)
from stigmergic.agents.base_ant import ConsumerAnt, Mutation  # noqa: E402

_HAS_PSYCOPG = importlib.util.find_spec("psycopg") is not None

#: A live PostgreSQL DSN (e.g. ``postgresql://user:pw@localhost/db``). When set --
#: which is the case in the dedicated CI ``postgres`` job -- the conformance suite
#: at the bottom of this module runs against a real server. Unset locally, so the
#: default (torch-free, driverless) run simply skips those tests.
_PG_DSN = os.environ.get("STIG_TEST_PG_DSN")


@pytest.fixture()
def ground():
    g = PheromoneGround()
    try:
        yield g
    finally:
        g.close()


# -- reliability core: idempotency, optimistic concurrency, leases, DLQ -------


class _AlwaysFailsAnt(ConsumerAnt):
    """A caste whose metabolize always raises -- to exercise the dead-letter path."""

    def metabolize(self, task):  # noqa: ANN001 - test double
        raise RuntimeError("boom")


class _ResolverAnt(ConsumerAnt):
    """A healthy caste that drives any task it claims straight to RESOLVED."""

    def metabolize(self, task):  # noqa: ANN001 - test double
        return Mutation(new_status=Status.RESOLVED, new_entropy=Entropy.ZERO)


def test_inject_chaos_deduplicates_on_idempotency_key(ground):
    first = ground.inject_chaos("ticket A", idempotency_key="sys-1")
    dup = ground.inject_chaos("ticket A re-delivered", idempotency_key="sys-1")
    assert dup == first  # same key -> same task; no duplicate work is created
    other = ground.inject_chaos("ticket B", idempotency_key="sys-2")
    assert other != first
    assert ground.count() == 2  # only the two distinct tasks exist

    # Once the task reaches a terminal state, its key is free to be reused.
    ground.update_state(first, status=Status.RESOLVED)
    reused = ground.inject_chaos("ticket A resubmitted", idempotency_key="sys-1")
    assert reused != first


def test_update_state_optimistic_concurrency_rejects_stale_write(ground):
    task_id = ground.inject_chaos("x", status=Status.RAW)
    version0 = ground.get(task_id).version

    # A write carrying a stale expected_version is rejected and changes nothing.
    assert (
        ground.update_state(
            task_id, status=Status.HYGIENIZED, expected_version=version0 + 5
        )
        is False
    )
    assert ground.get(task_id).status is Status.RAW

    # The current version is accepted and the version is bumped on success.
    assert (
        ground.update_state(
            task_id, status=Status.HYGIENIZED, expected_version=version0
        )
        is True
    )
    after = ground.get(task_id)
    assert after.status is Status.HYGIENIZED
    assert after.version == version0 + 1


def test_claim_sets_a_lease_and_reclaim_returns_expired_work(ground):
    task_id = ground.inject_chaos("work", status=Status.HYGIENIZED)
    claimed = ground.claim("worker", status=Status.HYGIENIZED, lease_seconds=0.0)
    assert claimed is not None and claimed.owner == "worker"
    assert claimed.lease_expires_at is not None

    time.sleep(0.01)  # let the (0-second) lease elapse
    reclaimed = ground.reclaim_expired_leases()
    assert reclaimed == [task_id]

    task = ground.get(task_id)
    assert task.owner is None  # ownership released
    assert task.status is Status.HYGIENIZED  # reverted to the claimed-from trail
    assert task.lease_expires_at is None
    assert task.version > claimed.version  # bumped -> the dead worker's write is stale


def test_reclaim_ignores_unexpired_leases(ground):
    ground.inject_chaos("work", status=Status.HYGIENIZED)
    claimed = ground.claim("worker", status=Status.HYGIENIZED, lease_seconds=60.0)
    assert claimed is not None
    # A healthy, un-expired lease is left alone.
    assert ground.reclaim_expired_leases() == []
    assert ground.get(claimed.id).owner == "worker"


def test_poison_pill_is_dead_lettered_after_max_retries(ground):
    task_id = ground.inject_chaos("bad task", status=Status.RAW)
    ant = _AlwaysFailsAnt(ground, "faildozer", target_status=Status.RAW, max_retries=2)

    # Each tick claims (bumping retry_count), metabolize raises, and the task is
    # released back to RAW -- until the retry budget is exceeded and it is parked.
    for _ in range(6):
        ant.tick()

    task = ground.get(task_id)
    assert task.status is Status.DEAD_LETTER
    assert task.owner is None
    assert task.dlq_reason and "max_retries" in task.dlq_reason
    # DEAD_LETTER is terminal: no ant will ever pick it up again.
    assert ground.claim("anyone", status=Status.RAW) is None
    assert Status.DEAD_LETTER in TERMINAL_STATUSES


def test_reclaimed_task_is_completed_by_a_healthy_worker(ground):
    task_id = ground.inject_chaos("recover me", status=Status.RAW)
    # A worker claims it with a 0-second lease, then "crashes" (never commits).
    dead = ground.claim("crashed-worker", status=Status.RAW, lease_seconds=0.0)
    assert dead is not None and dead.owner == "crashed-worker"

    time.sleep(0.01)
    # The janitor sweep returns the abandoned task to the RAW pool...
    assert ground.reclaim_expired_leases() == [task_id]
    # ...and a healthy worker claims it and drives it to RESOLVED.
    _ResolverAnt(ground, "healthy", target_status=Status.RAW).tick()

    task = ground.get(task_id)
    assert task.status is Status.RESOLVED
    assert task.owner is None


def test_enforce_transitions_rejects_an_illegal_hop():
    ground = PheromoneGround(enforce_transitions=True)
    try:
        task_id = ground.inject_chaos("x", status=Status.RAW)
        # RAW may only advance to CLAIMED / HYGIENIZED / DEAD_LETTER.
        with pytest.raises(ValueError, match="Illegal transition"):
            ground.update_state(task_id, status=Status.RESOLVED)
        # A permitted hop still goes through.
        assert ground.update_state(task_id, status=Status.HYGIENIZED) is True
    finally:
        ground.close()


def test_transitions_unenforced_by_default(ground):
    # The default ground does not police the lifecycle (backward compatible).
    task_id = ground.inject_chaos("x", status=Status.RAW)
    assert ground.update_state(task_id, status=Status.RESOLVED) is True


# -- event emission -----------------------------------------------------------


def test_inject_claim_mutate_emit_events(ground: PheromoneGround) -> None:
    seen: list[GroundEvent] = []
    ground.events.subscribe(seen.append)

    tid = ground.inject_chaos("summarize report", metadata={"k": "v"})
    ground.claim("w", min_entropy=Entropy.MIN)
    ground.update_state(tid, status=Status.HYGIENIZED, entropy=0.2, clear_owner=True)

    assert [e.event_type for e in seen] == ["INJECT", "CLAIM", "MUTATE"]
    assert all(e.task_id == tid for e in seen)
    assert seen[0].status is Status.RAW
    assert seen[-1].status is Status.HYGIENIZED


def test_event_carries_global_entropy(ground: PheromoneGround) -> None:
    seen: list[GroundEvent] = []
    ground.events.subscribe(seen.append)
    ground.inject_chaos("a", entropy=Entropy.CHAOS)
    assert seen[-1].global_entropy == pytest.approx(Entropy.CHAOS)


def test_unsubscribe_stops_delivery(ground: PheromoneGround) -> None:
    seen: list[GroundEvent] = []
    unsub = ground.events.subscribe(seen.append)
    ground.inject_chaos("a")
    unsub()
    ground.inject_chaos("b")
    assert len(seen) == 1


def test_faulty_listener_never_breaks_the_colony(ground: PheromoneGround) -> None:
    good: list[GroundEvent] = []

    def boom(_event: GroundEvent) -> None:
        raise RuntimeError("observer blew up")

    ground.events.subscribe(boom)
    ground.events.subscribe(good.append)

    tid = ground.inject_chaos("still works")  # must not raise
    assert tid > 0
    assert len(good) == 1


# -- event-driven waiting -----------------------------------------------------


def test_wait_for_change_times_out_without_mutation(ground: PheromoneGround) -> None:
    # The contract is the *return value*: with no mutation, the wait reports
    # "no change" (False). We deliberately do not assert a strict lower bound on
    # elapsed time -- a condition-variable wait is allowed to wake spuriously and
    # return early, which still correctly yields False here. We do bound it from
    # above so a regression that hangs forever is still caught.
    start = time.monotonic()
    assert ground.wait_for_change(0.05) is False
    assert time.monotonic() - start < 1.0


def test_wait_for_change_returns_false_when_stop_event_set(ground: PheromoneGround) -> None:
    stop = threading.Event()
    stop.set()
    assert ground.wait_for_change(1.0, stop) is False


def test_wait_for_change_wakes_on_mutation(ground: PheromoneGround) -> None:
    result: dict[str, object] = {}

    def waiter() -> None:
        t0 = time.monotonic()
        result["woke"] = ground.wait_for_change(2.0)
        result["elapsed"] = time.monotonic() - t0

    thread = threading.Thread(target=waiter)
    thread.start()
    time.sleep(0.1)  # let the waiter enter the wait before we mutate
    ground.inject_chaos("wake up")
    thread.join(timeout=3.0)

    assert result["woke"] is True
    assert result["elapsed"] < 1.5  # woke promptly, not via the 2s timeout


# -- AbstractGround contract --------------------------------------------------


def test_abstract_ground_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        AbstractGround()  # type: ignore[abstract]


def test_sqlite_ground_fully_implements_the_contract() -> None:
    assert issubclass(PheromoneGround, AbstractGround)
    assert PheromoneGround.__abstractmethods__ == frozenset()


# -- the create_ground factory ------------------------------------------------


@pytest.mark.parametrize("dsn", ["sqlite://:memory:", "", ":memory:"])
def test_factory_routes_sqlite_memory(dsn: str) -> None:
    g = create_ground(dsn)
    try:
        assert isinstance(g, PheromoneGround)
        assert isinstance(g, AbstractGround)
    finally:
        g.close()


def test_factory_routes_sqlite_file(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    g = create_ground("swarm.db")
    try:
        g.inject_chaos("persisted")
    finally:
        g.close()
    assert (tmp_path / "swarm.db").exists()


def test_factory_default_is_in_memory_sqlite() -> None:
    g = create_ground()
    try:
        assert isinstance(g, PheromoneGround)
    finally:
        g.close()


@pytest.mark.parametrize("scheme", ["redis", "dynamodb"])
def test_factory_roadmap_schemes_raise_not_implemented(scheme: str) -> None:
    with pytest.raises(NotImplementedError):
        create_ground(f"{scheme}://localhost")


def test_factory_unknown_scheme_raises_value_error() -> None:
    with pytest.raises(ValueError):
        create_ground("mongodb://localhost")


# -- PostgresGround (driverless paths only) -----------------------------------


def test_postgres_module_imports_without_psycopg() -> None:
    # Importing the backend module must not drag the driver into the process.
    import stigmergic.core.backends.postgres as pg  # noqa: F401

    if not _HAS_PSYCOPG:
        assert "psycopg" not in sys.modules


def test_postgres_ground_conforms_to_abstract_contract() -> None:
    from stigmergic.core.backends.postgres import PostgresGround

    assert issubclass(PostgresGround, AbstractGround)
    assert PostgresGround.__abstractmethods__ == frozenset()


@pytest.mark.skipif(_HAS_PSYCOPG, reason="psycopg is installed; the driver import would succeed")
def test_postgres_ground_raises_clean_error_without_driver() -> None:
    from stigmergic.core.backends.postgres import PostgresGround

    with pytest.raises(ImportError) as excinfo:
        PostgresGround("postgresql://user:pw@localhost/db")
    assert "pip install" in str(excinfo.value)


@pytest.mark.parametrize("bad", ["pheromones; DROP TABLE x", "1nvalid", "a-b", "has space", ""])
def test_postgres_rejects_unsafe_identifiers(bad: str) -> None:
    from stigmergic.core.backends.postgres import _safe_ident

    with pytest.raises(ValueError):
        _safe_ident(bad, kind="table")


@pytest.mark.parametrize("ok", ["pheromones", "_p", "events_v2", "Channel1"])
def test_postgres_accepts_safe_identifiers(ok: str) -> None:
    from stigmergic.core.backends.postgres import _safe_ident

    assert _safe_ident(ok, kind="table") == ok


# -- PostgresGround (live server) ---------------------------------------------
#
# These only run when STIG_TEST_PG_DSN points at a real PostgreSQL (the CI
# `postgres` job wires up a `services: postgres` container). They are the first
# and only place PostgresGround touches an actual database, so they exercise the
# whole AbstractGround contract end to end: CRUD, the lock-free SKIP LOCKED
# claim, global entropy accounting, and the LISTEN/NOTIFY wakeup path.

requires_pg = pytest.mark.skipif(
    not (_HAS_PSYCOPG and _PG_DSN),
    reason="set STIG_TEST_PG_DSN (and install psycopg) to run live PostgresGround tests",
)


@pytest.fixture()
def pg_ground():
    from stigmergic.core.backends.postgres import PostgresGround

    g = PostgresGround(
        _PG_DSN,
        table="pheromones_pytest",
        channel="pheromone_events_pytest",
    )
    g.reset()  # start from a clean, id-reset field even if a prior run left rows
    try:
        yield g
    finally:
        try:
            g.reset()
        except Exception:  # noqa: BLE001 -- teardown is best-effort
            pass
        g.close()


@requires_pg
def test_pg_inject_and_get_roundtrip(pg_ground) -> None:
    tid = pg_ground.inject_chaos("summarize the Q3 report", metadata={"k": "v"})
    assert tid > 0
    ph = pg_ground.get(tid)
    assert ph is not None
    assert ph.raw_data == "summarize the Q3 report"
    assert ph.status is Status.RAW
    assert ph.entropy == pytest.approx(Entropy.CHAOS)
    assert ph.metadata == {"k": "v"}
    assert pg_ground.count() == 1


@requires_pg
def test_pg_claim_is_ordered_and_exhausts(pg_ground) -> None:
    # Higher entropy is more urgent, so it must be handed out first.
    a = pg_ground.inject_chaos("alpha", entropy=Entropy.CHAOS)
    b = pg_ground.inject_chaos("beta", entropy=Entropy.HIGH)

    first = pg_ground.claim("w1", status=Status.RAW)
    assert first is not None and first.id == a
    assert first.status is Status.CLAIMED and first.owner == "w1"

    second = pg_ground.claim("w2", status=Status.RAW)
    assert second is not None and second.id == b

    # Nothing RAW is left to grab -> the lock-free claim yields None.
    assert pg_ground.claim("w3", status=Status.RAW) is None


@requires_pg
def test_pg_update_state_drives_entropy_to_zero(pg_ground) -> None:
    tid = pg_ground.inject_chaos("resolve me", entropy=Entropy.CHAOS)
    assert pg_ground.global_entropy() > 0.0

    changed = pg_ground.update_state(
        tid, status=Status.RESOLVED, entropy=Entropy.MIN, clear_owner=True
    )
    assert changed is True

    ph = pg_ground.get(tid)
    assert ph is not None and ph.status is Status.RESOLVED and ph.owner is None
    assert pg_ground.global_entropy() == pytest.approx(0.0)
    # A terminal pheromone is excluded from the claimable set.
    assert pg_ground.claim("w") is None


@requires_pg
def test_pg_reliability_idempotency_cas_and_reclaim(pg_ground) -> None:
    # Idempotency: the same key returns the same task, never a duplicate.
    first = pg_ground.inject_chaos("ticket", idempotency_key="k1")
    assert pg_ground.inject_chaos("ticket re-delivered", idempotency_key="k1") == first
    assert pg_ground.count() == 1

    # Optimistic concurrency: a stale expected_version is rejected, current wins.
    version0 = pg_ground.get(first).version
    assert (
        pg_ground.update_state(
            first, status=Status.HYGIENIZED, expected_version=version0 + 9
        )
        is False
    )
    assert (
        pg_ground.update_state(
            first, status=Status.HYGIENIZED, expected_version=version0
        )
        is True
    )
    assert pg_ground.get(first).version == version0 + 1

    # Lease + reclaim: a 0-second lease is swept back to the claimed-from trail.
    claimed = pg_ground.claim("dead", status=Status.HYGIENIZED, lease_seconds=0.0)
    assert claimed is not None and claimed.owner == "dead"
    time.sleep(0.01)
    assert pg_ground.reclaim_expired_leases() == [first]
    recovered = pg_ground.get(first)
    assert recovered.owner is None and recovered.status is Status.HYGIENIZED


@requires_pg
def test_pg_sense_reads_without_mutating(pg_ground) -> None:
    pg_ground.inject_chaos("observe only", entropy=Entropy.CHAOS)
    first = pg_ground.sense(limit=5)
    assert len(first) == 1 and first[0].status is Status.RAW
    # Sensing again returns the same untouched row (no claim happened).
    assert pg_ground.sense()[0].status is Status.RAW
    assert pg_ground.count(Status.RAW) == 1


@requires_pg
def test_pg_reset_wipes_the_field(pg_ground) -> None:
    pg_ground.inject_chaos("gone soon")
    assert pg_ground.count() >= 1
    pg_ground.reset()
    assert pg_ground.count() == 0
    assert pg_ground.global_entropy() == pytest.approx(0.0)


@requires_pg
def test_pg_notify_wakes_wait_for_change(pg_ground) -> None:
    # The real payoff of Postgres: a mutation fires a server NOTIFY that the
    # background listener rebroadcasts onto the local bus, waking any waiter.
    result: dict[str, object] = {}

    def waiter() -> None:
        result["woke"] = pg_ground.wait_for_change(3.0)

    thread = threading.Thread(target=waiter)
    thread.start()
    time.sleep(0.3)  # let the waiter enter the wait and the listener settle
    pg_ground.inject_chaos("wake the colony")
    thread.join(timeout=4.0)

    assert result.get("woke") is True
