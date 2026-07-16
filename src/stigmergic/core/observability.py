"""The Swarm Inspector: a flight recorder for a leaderless colony.

A stigmergic system has no orchestrator and therefore no call graph to trace --
coordination lives entirely in the shared field. That is a feature (no single
point of failure) and an observability problem (no obvious place to watch). The
:class:`SwarmInspector` solves it by subscribing to the ground's event bus and
turning the raw :class:`~stigmergic_ai.core.environment.GroundEvent` stream into
the three things an operator actually wants:

* **Lifecycle** -- replay the full life of any single pheromone, from ``RAW``
  chaos to ``RESOLVED`` (or ``SLASHED``) order.
* **Entropy over time** -- chart the field's total disorder as the swarm grinds
  it down; a healthy colony's curve falls monotonically toward zero.
* **Replay** -- re-run the recorded history, optionally in real time, to debug a
  past run deterministically.

It is a passive observer: listener exceptions are swallowed by the bus, so the
inspector can never slow down or crash the colony. It is also dependency-free
(stdlib only) and works identically over any
:class:`~stigmergic_ai.core.environment.AbstractGround` backend -- SQLite today,
Postgres tomorrow.
"""

from __future__ import annotations

import threading
import time
from collections import Counter, defaultdict, deque
from typing import IO, Callable

from stigmergic_ai.core.environment import (
    AbstractGround,
    GroundEvent,
    Status,
)

__all__ = ["SwarmInspector"]

#: Eight block glyphs, low to high, for the entropy sparkline.
_BARS = "▁▂▃▄▅▆▇█"


class SwarmInspector:
    """Subscribes to a ground's event stream and answers questions about it.

    Args:
        max_events: Ring-buffer capacity. The newest ``max_events`` are retained;
            older events age out so a long run cannot exhaust memory.
        sink: Optional durable JSONL sink. May be a filesystem path (opened for
            append and owned by the inspector), an already-open writable file
            object (written to, not closed), or a ``callable(GroundEvent)`` for
            custom routing. Each event is appended as one JSON line.
    """

    def __init__(
        self,
        *,
        max_events: int = 10_000,
        sink: str | IO[str] | Callable[[GroundEvent], None] | None = None,
    ) -> None:
        self._events: deque[GroundEvent] = deque(maxlen=max_events)
        self._by_task: dict[int, list[GroundEvent]] = defaultdict(list)
        self._lock = threading.RLock()
        self._unsub: Callable[[], None] | None = None

        self._sink_call: Callable[[GroundEvent], None] | None = None
        self._sink_file: IO[str] | None = None
        self._owns_sink = False
        if callable(sink):
            self._sink_call = sink
        elif isinstance(sink, str):
            self._sink_file = open(sink, "a", encoding="utf-8")
            self._owns_sink = True
        elif sink is not None:
            self._sink_file = sink  # an already-open file-like; we do not own it

    # -- wiring ---------------------------------------------------------------

    def attach(self, ground: AbstractGround) -> "SwarmInspector":
        """Start recording ``ground``'s events. Returns ``self`` for chaining.

        Idempotent per inspector: attaching again first detaches the previous
        subscription so events are never double-counted.
        """
        self.detach()
        self._unsub = ground.events.subscribe(self._on_event)
        return self

    def detach(self) -> None:
        """Stop recording (idempotent)."""
        if self._unsub is not None:
            self._unsub()
            self._unsub = None

    def close(self) -> None:
        """Detach and close an owned sink file (idempotent)."""
        self.detach()
        if self._sink_file is not None and self._owns_sink:
            try:
                self._sink_file.close()
            finally:
                self._sink_file = None
                self._owns_sink = False

    def __enter__(self) -> "SwarmInspector":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- ingest ---------------------------------------------------------------

    def _on_event(self, event: GroundEvent) -> None:
        with self._lock:
            self._events.append(event)
            self._by_task[event.task_id].append(event)
        if self._sink_call is not None:
            self._sink_call(event)
        elif self._sink_file is not None:
            self._sink_file.write(event.model_dump_json() + "\n")
            self._sink_file.flush()

    # -- queries --------------------------------------------------------------

    def events(self) -> list[GroundEvent]:
        """A snapshot copy of every recorded event, in arrival (seq) order."""
        with self._lock:
            return list(self._events)

    def timeline(self, task_id: int) -> list[GroundEvent]:
        """Every recorded event for a single pheromone, in order."""
        with self._lock:
            return list(self._by_task.get(task_id, ()))

    def lifecycle(self, task_id: int) -> str:
        """The collapsed status trail of one pheromone (e.g. ``RAW -> CLAIMED -> RESOLVED``).

        Consecutive duplicate statuses are merged, so this shows transitions, not
        every event.
        """
        statuses: list[str] = []
        for event in self.timeline(task_id):
            name = event.status.value
            if not statuses or statuses[-1] != name:
                statuses.append(name)
        return " -> ".join(statuses)

    def entropy_series(self) -> list[tuple[float, float]]:
        """``(timestamp, global_entropy)`` samples in chronological order."""
        with self._lock:
            ordered = sorted(self._events, key=lambda e: e.seq)
        return [(e.ts, e.global_entropy) for e in ordered]

    def sparkline(self, width: int = 40) -> str:
        """An ASCII chart of global entropy over time, at most ``width`` glyphs.

        Longer histories are bucket-averaged down to ``width`` points so the
        shape of the decay stays readable at a glance.
        """
        series = [v for _, v in self.entropy_series()]
        if not series:
            return ""
        if width >= 1 and len(series) > width:
            step = len(series) / width
            reduced: list[float] = []
            for i in range(width):
                lo = int(i * step)
                hi = max(int((i + 1) * step), lo + 1)
                chunk = series[lo:hi]
                reduced.append(sum(chunk) / len(chunk))
            series = reduced
        low, high = min(series), max(series)
        span = high - low
        if span <= 0.0:
            return _BARS[0] * len(series)
        out = []
        top = len(_BARS) - 1
        for value in series:
            frac = (value - low) / span
            out.append(_BARS[min(top, int(frac * top + 0.5))])
        return "".join(out)

    def stats(self) -> dict[str, object]:
        """A compact dashboard: event/type counts, status census, throughput."""
        with self._lock:
            events = list(self._events)
        type_counts = Counter(e.event_type for e in events)
        latest: dict[int, Status] = {}
        for event in events:
            latest[event.task_id] = event.status
        status_counts = Counter(s.value for s in latest.values())
        resolved = status_counts.get(Status.RESOLVED.value, 0)
        slashed = status_counts.get(Status.SLASHED.value, 0)
        span = (events[-1].ts - events[0].ts) if len(events) >= 2 else 0.0
        throughput = (resolved + slashed) / span if span > 0 else 0.0
        return {
            "events": len(events),
            "tasks": len(latest),
            "by_event_type": dict(type_counts),
            "by_status": dict(status_counts),
            "resolved": resolved,
            "slashed": slashed,
            "terminal": resolved + slashed,
            "throughput_per_s": round(throughput, 3),
            "global_entropy": events[-1].global_entropy if events else 0.0,
        }

    def report(self) -> str:
        """A human-readable multi-line summary, sparkline included."""
        s = self.stats()
        lines = [
            "=== Swarm Inspector ===",
            f"events={s['events']}  tasks={s['tasks']}  "
            f"resolved={s['resolved']}  slashed={s['slashed']}",
            f"throughput={s['throughput_per_s']}/s  "
            f"global_entropy={s['global_entropy']:.3f}",
            f"by_event_type={s['by_event_type']}",
            f"by_status={s['by_status']}",
            f"entropy {self.sparkline()}",
        ]
        return "\n".join(lines)

    def replay(
        self,
        callback: Callable[[GroundEvent], None],
        *,
        speed: float = 0.0,
    ) -> None:
        """Re-emit the recorded history to ``callback`` in seq order.

        Args:
            callback: Invoked once per event, oldest first.
            speed: ``0.0`` (default) replays as fast as possible. A positive
                value sleeps the real inter-event gap divided by ``speed`` --
                ``1.0`` is wall-clock real time, ``2.0`` is double speed.
        """
        with self._lock:
            ordered = sorted(self._events, key=lambda e: e.seq)
        prev_ts: float | None = None
        for event in ordered:
            if speed > 0 and prev_ts is not None:
                delay = (event.ts - prev_ts) / speed
                if delay > 0:
                    time.sleep(delay)
            callback(event)
            prev_ts = event.ts
