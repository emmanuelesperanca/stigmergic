"""Pytest suite for the Swarm Inspector observability layer.

Dependency-free and fast: it drives a real in-memory
:class:`~stigmergic.core.environment.PheromoneGround`, attaches a
:class:`~stigmergic.core.observability.SwarmInspector`, and asserts the
recorder's queries -- lifecycle, entropy series, sparkline, stats, replay -- as
well as its sinks and context-manager hygiene.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from stigmergic.core.environment import Entropy, PheromoneGround, Status  # noqa: E402
from stigmergic.core.observability import SwarmInspector  # noqa: E402


@pytest.fixture()
def ground():
    g = PheromoneGround()
    try:
        yield g
    finally:
        g.close()


def _resolve_one(ground: PheromoneGround, text: str = "do the thing") -> int:
    """Drive a single pheromone RAW -> CLAIMED -> HYGIENIZED -> RESOLVED."""
    tid = ground.inject_chaos(text, entropy=Entropy.CHAOS)
    ground.claim("worker", min_entropy=Entropy.MIN)
    ground.update_state(tid, status=Status.HYGIENIZED, entropy=Entropy.LOW, clear_owner=True)
    ground.claim("solver", min_entropy=Entropy.MIN, status=Status.HYGIENIZED)
    ground.update_state(tid, status=Status.RESOLVED, entropy=Entropy.ZERO, clear_owner=True)
    return tid


def test_attach_records_events(ground: PheromoneGround) -> None:
    insp = SwarmInspector().attach(ground)
    tid = _resolve_one(ground)
    assert len(insp.events()) == 5
    assert all(e.task_id == tid for e in insp.events())


def test_attach_returns_self_and_is_idempotent(ground: PheromoneGround) -> None:
    insp = SwarmInspector()
    assert insp.attach(ground) is insp
    insp.attach(ground)  # re-attach must not double-subscribe
    ground.inject_chaos("once")
    assert len(insp.events()) == 1


def test_lifecycle_collapses_consecutive_statuses(ground: PheromoneGround) -> None:
    insp = SwarmInspector().attach(ground)
    tid = _resolve_one(ground)
    assert insp.lifecycle(tid) == "RAW -> CLAIMED -> HYGIENIZED -> CLAIMED -> RESOLVED"


def test_timeline_is_scoped_per_task(ground: PheromoneGround) -> None:
    insp = SwarmInspector().attach(ground)
    a = ground.inject_chaos("a")
    b = ground.inject_chaos("b")
    assert [e.task_id for e in insp.timeline(a)] == [a]
    assert [e.task_id for e in insp.timeline(b)] == [b]
    assert insp.timeline(999) == []


def test_entropy_series_is_chronological(ground: PheromoneGround) -> None:
    insp = SwarmInspector().attach(ground)
    _resolve_one(ground)
    series = insp.entropy_series()
    assert len(series) == 5
    timestamps = [ts for ts, _ in series]
    assert timestamps == sorted(timestamps)
    assert series[-1][1] == pytest.approx(0.0)  # field fully resolved


def test_sparkline_glyphs_and_width(ground: PheromoneGround) -> None:
    insp = SwarmInspector().attach(ground)
    for _ in range(100):
        ground.inject_chaos("x")
    bars = insp.sparkline(width=40)
    assert len(bars) == 40
    assert set(bars) <= set("▁▂▃▄▅▆▇█")


def test_sparkline_empty_history_is_blank() -> None:
    assert SwarmInspector().sparkline() == ""


def test_sparkline_flat_history_renders(ground: PheromoneGround) -> None:
    insp = SwarmInspector().attach(ground)
    # Identical entropy each time -> a flat (non-crashing) sparkline.
    ground.inject_chaos("a", entropy=Entropy.ZERO)
    ground.inject_chaos("b", entropy=Entropy.ZERO)
    assert insp.sparkline() != ""


def test_stats_counts_events_and_terminals(ground: PheromoneGround) -> None:
    insp = SwarmInspector().attach(ground)
    _resolve_one(ground)
    stats = insp.stats()
    assert stats["events"] == 5
    assert stats["tasks"] == 1
    assert stats["resolved"] == 1
    assert stats["terminal"] == 1
    assert stats["by_event_type"] == {"INJECT": 1, "CLAIM": 2, "MUTATE": 2}


def test_report_is_a_human_readable_summary(ground: PheromoneGround) -> None:
    insp = SwarmInspector().attach(ground)
    _resolve_one(ground)
    report = insp.report()
    assert "Swarm Inspector" in report
    assert "resolved=1" in report


def test_replay_visits_events_in_seq_order(ground: PheromoneGround) -> None:
    insp = SwarmInspector().attach(ground)
    _resolve_one(ground)
    order: list[str] = []
    insp.replay(lambda e: order.append(e.event_type))
    assert order == ["INJECT", "CLAIM", "MUTATE", "CLAIM", "MUTATE"]


def test_detach_stops_recording(ground: PheromoneGround) -> None:
    insp = SwarmInspector().attach(ground)
    ground.inject_chaos("before")
    insp.detach()
    ground.inject_chaos("after")
    assert len(insp.events()) == 1


def test_context_manager_detaches_on_exit(ground: PheromoneGround) -> None:
    with SwarmInspector() as insp:
        insp.attach(ground)
        ground.inject_chaos("inside")
    ground.inject_chaos("outside")
    assert len(insp.events()) == 1


def test_callable_sink_receives_events(ground: PheromoneGround) -> None:
    captured: list = []
    SwarmInspector(sink=captured.append).attach(ground)
    ground.inject_chaos("routed")
    assert len(captured) == 1
    assert captured[0].event_type == "INJECT"


def test_jsonl_file_sink_persists_events(ground: PheromoneGround, tmp_path) -> None:
    sink_path = tmp_path / "events.jsonl"
    insp = SwarmInspector(sink=str(sink_path)).attach(ground)
    _resolve_one(ground)
    insp.close()

    lines = sink_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 5
    first = json.loads(lines[0])
    assert first["event_type"] == "INJECT"
    assert "global_entropy" in first


def test_observability_is_torch_free(ground: PheromoneGround) -> None:
    SwarmInspector().attach(ground)
    _resolve_one(ground)
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules
