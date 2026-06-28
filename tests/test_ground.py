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
import pathlib
import sys
import threading
import time

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from stigmergic_ai.core.backends import create_ground  # noqa: E402
from stigmergic_ai.core.environment import (  # noqa: E402
    AbstractGround,
    Entropy,
    GroundEvent,
    PheromoneGround,
    Status,
)

_HAS_PSYCOPG = importlib.util.find_spec("psycopg") is not None


@pytest.fixture()
def ground():
    g = PheromoneGround()
    try:
        yield g
    finally:
        g.close()


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
    import stigmergic_ai.core.backends.postgres as pg  # noqa: F401

    if not _HAS_PSYCOPG:
        assert "psycopg" not in sys.modules


def test_postgres_ground_conforms_to_abstract_contract() -> None:
    from stigmergic_ai.core.backends.postgres import PostgresGround

    assert issubclass(PostgresGround, AbstractGround)
    assert PostgresGround.__abstractmethods__ == frozenset()


@pytest.mark.skipif(_HAS_PSYCOPG, reason="psycopg is installed; the driver import would succeed")
def test_postgres_ground_raises_clean_error_without_driver() -> None:
    from stigmergic_ai.core.backends.postgres import PostgresGround

    with pytest.raises(ImportError) as excinfo:
        PostgresGround("postgresql://user:pw@localhost/db")
    assert "pip install" in str(excinfo.value)


@pytest.mark.parametrize("bad", ["pheromones; DROP TABLE x", "1nvalid", "a-b", "has space", ""])
def test_postgres_rejects_unsafe_identifiers(bad: str) -> None:
    from stigmergic_ai.core.backends.postgres import _safe_ident

    with pytest.raises(ValueError):
        _safe_ident(bad, kind="table")


@pytest.mark.parametrize("ok", ["pheromones", "_p", "events_v2", "Channel1"])
def test_postgres_accepts_safe_identifiers(ok: str) -> None:
    from stigmergic_ai.core.backends.postgres import _safe_ident

    assert _safe_ident(ok, kind="table") == ok
