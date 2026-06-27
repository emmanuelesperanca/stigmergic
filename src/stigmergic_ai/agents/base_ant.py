"""The Ant: an autonomous worker that lives or dies by the Pheromone Ground.

An ant in StigmergicAI is not an object other code calls. It is a *process* --
a background thread with a heartbeat. It wakes, smells the shared field, reacts
to entropy, mutates the field, and goes back to sleep. It has no inbox, no
caller, and no knowledge of any sibling ant. The only thing connecting one ant
to another is the environment they both touch.

Two castes inherit from :class:`BaseAnt`:

* :class:`ConsumerAnt` -- *senses and resolves*. It atomically claims the most
  urgent matching pheromone, metabolizes it (the cognitive step subclasses
  implement), and lays a fresh trail by lowering entropy. Governance and Solver
  ants are consumers.
* :class:`ProducerAnt` -- *secretes*. It periodically injects chaos into the
  ground without ever sensing anything. The Forager is a producer.

The split is deliberate: a producer's loop has no "claim" phase, and forcing
both behaviours through one hook would blur the two fundamentally different
roles in a stigmergic system (raising entropy vs. lowering it).

Crucially, the loop is crash-resilient. A failure inside one ``tick`` is logged
and swallowed; the thread keeps breathing. Kill the process entirely and the
work still sits in the database, waiting for the swarm to resume -- eventual
consistency with no orchestrator.
"""

from __future__ import annotations

import abc
import logging
import threading

from pydantic import BaseModel, Field

from stigmergic_ai.core.environment import (
    Entropy,
    Pheromone,
    PheromoneGround,
    Status,
)

__all__ = ["Mutation", "BaseAnt", "ConsumerAnt", "ProducerAnt"]

logger = logging.getLogger("stigmergic_ai.agents")


class Mutation(BaseModel):
    """The result a :class:`ConsumerAnt` proposes after metabolizing a task.

    A mutation is a *description* of how the pheromone should change, not the
    change itself -- which keeps the cognitive step (``metabolize``) pure and,
    later, makes it trivial to route a proposed mutation through the Byzantine
    quorum before it is ever committed. Every field is optional; only what you
    set is applied.
    """

    new_entropy: float | None = Field(default=None, ge=Entropy.MIN, le=Entropy.MAX)
    """The entropy to settle the pheromone to (e.g. ``0.2`` for a clean trail)."""

    new_status: Status | None = None
    """The chemical trail to stamp (e.g. ``HYGIENIZED``, ``RESOLVED``, ``SLASHED``)."""

    new_raw_data: str | None = None
    """Optional rewritten raw payload (e.g. a Governance ant's sanitized text)."""

    latent_blob: bytes | None = None
    """Optional latent tensor bytes to attach (Horizon 3 / Latent State Transfer)."""

    metadata: dict | None = None
    """Optional replacement metadata mapping."""

    release_owner: bool = True
    """If ``True``, drop ownership so the next caste can freely claim the trail."""


class BaseAnt(abc.ABC):
    """An autonomous worker running a poll loop on a background thread.

    ``BaseAnt`` owns only the *lifecycle*: a daemon thread, a stop event, and a
    resilient heartbeat that repeatedly calls :meth:`tick`. It does not know how
    to do any work -- that is the caste's job. Composition over
    :class:`threading.Thread` (rather than subclassing it) keeps the threading
    machinery out of the cognitive API and lets an ant be stopped and restarted.

    Subclasses must implement :meth:`tick`, one unit of perception-and-action.
    """

    def __init__(
        self,
        env: PheromoneGround,
        name: str | None = None,
        *,
        poll_interval: float = 0.5,
    ) -> None:
        """Create an ant bound to a Pheromone Ground.

        Args:
            env: The shared environment this ant senses and mutates.
            name: A human-readable identifier, also stamped as the ``owner`` on
                claimed pheromones. Defaults to the class name.
            poll_interval: Seconds the ant sleeps between heartbeats. The sleep
                is interruptible, so :meth:`stop` takes effect promptly.
        """
        if poll_interval < 0:
            raise ValueError("poll_interval must be non-negative.")
        self.env = env
        self.name = name or type(self).__name__
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.log = logger.getChild(self.name)

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        """Spawn the background thread and begin the heartbeat.

        Safe to call again after :meth:`stop`: a fresh thread is created and the
        stop signal is cleared, so an ant can be revived to vacuum a backlog.
        """
        if self.is_alive():
            raise RuntimeError(f"Ant {self.name!r} is already running.")
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name=self.name, daemon=True
        )
        self._thread.start()
        self.log.info("Ant %r awakened (poll_interval=%ss).", self.name, self.poll_interval)

    def stop(self) -> None:
        """Signal the ant to finish its current tick and then sleep forever.

        Non-blocking: this only raises the stop flag. Use :meth:`join` to wait
        for the thread to actually wind down.
        """
        self._stop_event.set()
        self.log.info("Ant %r signalled to stop.", self.name)

    def join(self, timeout: float | None = None) -> None:
        """Block until the ant's thread terminates (or ``timeout`` elapses)."""
        if self._thread is not None:
            self._thread.join(timeout)

    def is_alive(self) -> bool:
        """Return ``True`` while the background thread is running."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def stopping(self) -> bool:
        """``True`` once :meth:`stop` has been requested."""
        return self._stop_event.is_set()

    # -- the heartbeat --------------------------------------------------------

    def _run(self) -> None:
        """The poll loop. One failed tick must never kill the colony."""
        self.log.debug("Heartbeat started.")
        while not self._stop_event.is_set():
            try:
                self.tick()
            except Exception:  # noqa: BLE001 -- resilience is the whole point
                # Swallow and survive: the unfinished work remains durable in
                # the database, exactly as the Chaos Monkey test demands.
                self.log.exception("tick() raised; ant survives and retries.")
            # Interruptible sleep: stop() wakes us immediately.
            self._stop_event.wait(self.poll_interval)
        self.log.debug("Heartbeat stopped.")

    @abc.abstractmethod
    def tick(self) -> None:
        """Perform exactly one unit of work. Implemented by each caste."""
        raise NotImplementedError

    def __repr__(self) -> str:
        state = "alive" if self.is_alive() else "dormant"
        return f"<{type(self).__name__} name={self.name!r} {state}>"


class ConsumerAnt(BaseAnt):
    """An ant that senses entropy, claims a task, resolves it, and lowers entropy.

    This is the workhorse caste. Each heartbeat it atomically claims the single
    most urgent pheromone matching its trigger (an entropy threshold and,
    optionally, a specific incoming ``status``), metabolizes it, and commits the
    resulting :class:`Mutation` back to the ground. Because the claim is atomic,
    any number of identical consumers can forage the same ground without ever
    treading on each other.

    Subclasses implement :meth:`metabolize` -- the actual cognition.
    """

    def __init__(
        self,
        env: PheromoneGround,
        name: str | None = None,
        *,
        entropy_threshold: float = Entropy.MIN,
        target_status: Status | str | None = None,
        claimed_status: Status | str = Status.CLAIMED,
        poll_interval: float = 0.5,
    ) -> None:
        """Configure what this consumer reacts to.

        Args:
            env: The shared Pheromone Ground.
            name: Identifier / ``owner`` stamp. Defaults to the class name.
            entropy_threshold: Minimum entropy required to wake on a pheromone
                (e.g. ``Entropy.HIGH`` for a Governance ant).
            target_status: If set, only claim pheromones currently on this trail
                (e.g. ``Status.HYGIENIZED`` for a Solver ant).
            claimed_status: The intermediate trail stamped while the task is being
                metabolized, preventing sibling ants from double-claiming it.
            poll_interval: Seconds between heartbeats.
        """
        super().__init__(env, name, poll_interval=poll_interval)
        self.entropy_threshold = entropy_threshold
        self.target_status = target_status
        self.claimed_status = claimed_status

    def tick(self) -> None:
        """Claim one matching pheromone, metabolize it, and commit the result."""
        task = self.env.claim(
            owner=self.name,
            min_entropy=self.entropy_threshold,
            status=self.target_status,
            new_status=self.claimed_status,
        )
        if task is None:
            return  # Nothing urgent enough on the ground; sleep again.

        self.log.debug("Claimed pheromone id=%s; metabolizing.", task.id)
        mutation = self.metabolize(task)
        if mutation is None:
            # The ant declined to act. Leave the claim in place; a later tick (or
            # operator) can decide what to do. We never silently lose the task.
            self.log.warning("metabolize() returned None for id=%s.", task.id)
            return

        self.env.update_state(
            task.id,
            entropy=mutation.new_entropy,
            status=mutation.new_status,
            raw_data=mutation.new_raw_data,
            latent_blob=mutation.latent_blob,
            metadata=mutation.metadata,
            clear_owner=mutation.release_owner,
        )
        self.log.debug(
            "Committed mutation for id=%s (entropy=%s status=%s).",
            task.id,
            mutation.new_entropy,
            mutation.new_status.value if mutation.new_status else None,
        )

    @abc.abstractmethod
    def metabolize(self, task: Pheromone) -> Mutation | None:
        """Process a claimed pheromone and propose how to mutate the ground.

        This is the cognitive heart of a consumer caste. Return a
        :class:`Mutation` (typically lowering entropy and stamping the next
        trail), or ``None`` to decline and leave the task claimed.
        """
        raise NotImplementedError


class ProducerAnt(BaseAnt):
    """An ant that only secretes chaos, never sensing or claiming anything.

    A producer is the entropy *source* of the colony. Its heartbeat does one
    thing: call :meth:`secrete`, which typically deposits fresh pheromones via
    :meth:`PheromoneGround.inject_chaos`. The Forager is the canonical producer.

    Subclasses implement :meth:`secrete`.
    """

    def tick(self) -> None:
        """Emit one burst of chaos onto the ground."""
        self.secrete()

    @abc.abstractmethod
    def secrete(self) -> None:
        """Deposit new work onto the Pheromone Ground (raising entropy).

        Implementations usually call :meth:`PheromoneGround.inject_chaos`. The
        cadence is governed by ``poll_interval``.
        """
        raise NotImplementedError
