"""Hello, Stigmergy: the Chaos Monkey test (Horizon 1, zero deep-learning deps).

Run it directly::

    python examples/01_hello_stigmergy.py

This is the README's "Chaos Monkey" demo -- the smallest possible proof that a
StigmergicAI swarm coordinates purely through a shared environment, with *no*
agent ever calling another, and survives having a worker thread killed
mid-flight.

Three castes share one in-memory Pheromone Ground:

* **Forager** (producer) floods the ground with high-entropy ``RAW`` requests.
* **Governance** (consumer) smells entropy ``>= 0.7``, sanitizes the payload,
  and lays a low-entropy ``HYGIENIZED`` trail.
* **Solver** (consumer) smells the clean trail, executes the (mock) business
  logic, and drives entropy to zero with a terminal ``RESOLVED`` stamp.

The Chaos Monkey: midway we *kill the Governance thread*. The Forager keeps
piling on work and the Solver idles (nothing is hygienized for it), but nothing
crashes -- the backlog simply sits durably in the database. We then *revive*
Governance and watch it vacuum the backlog at full speed. Eventual consistency
with no orchestrator, no retry queue, no Step Functions.

NOTE: this horizon is about coordination and resilience, not security, so the
Forager emits only legitimate work (``injection_rate=0.0``). Slashing prompt
injections through the Byzantine quorum is the job of ``02_byzantine_fault.py``.
"""

from __future__ import annotations

import pathlib
import random
import sys
import time

# Make the src-layout package importable when running this file directly,
# without first running `pip install -e .`.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from stigmergic_ai.agents.base_ant import ConsumerAnt, Mutation  # noqa: E402
from stigmergic_ai.agents.concrete import ForagerAnt, GovernanceAnt  # noqa: E402
from stigmergic_ai.core.environment import (  # noqa: E402
    Entropy,
    Pheromone,
    PheromoneGround,
    Status,
)


class SolverAnt(ConsumerAnt):
    """The Horizon-1 *terminal* solver: it resolves a clean trail to zero entropy.

    It wakes on the ``HYGIENIZED`` trail (any entropy), mocks the business logic,
    and drives the pheromone straight to :attr:`Status.RESOLVED` at zero entropy.

    This is deliberately simpler than the Byzantine-pipeline
    :class:`stigmergic_ai.agents.concrete.SolverAnt`, which instead *stages* its
    proposal at ``PENDING_CONSENSUS`` for a Verifier quorum to adjudicate. Here
    there is no quorum -- this is the pure-stigmergy demo -- so the Solver simply
    finalizes its own work.
    """

    def __init__(
        self,
        env: PheromoneGround,
        name: str | None = None,
        *,
        poll_interval: float = 0.1,
    ) -> None:
        super().__init__(
            env,
            name,
            entropy_threshold=Entropy.MIN,
            target_status=Status.HYGIENIZED,
            poll_interval=poll_interval,
        )

    def metabolize(self, task: Pheromone) -> Mutation:
        # "Execute" the safe business logic. In a real colony this is where the
        # solver would call a tool, run a query, or invoke an LLM. The clean
        # trail it follows was laid by Governance; the two never spoke.
        metadata = dict(task.metadata or {})
        metadata["resolution"] = "Approved"
        metadata["solved_by"] = self.name
        return Mutation(
            new_entropy=Entropy.ZERO,
            new_status=Status.RESOLVED,
            metadata=metadata,
            release_owner=True,
        )


def snapshot(ground: PheromoneGround) -> str:
    """A one-line census of entropy and pheromone counts per chemical trail."""
    stats = ground.stats()
    cells = "  ".join(
        f"{s.value}={stats.get(s.value, 0)}"
        for s in (Status.RAW, Status.CLAIMED, Status.HYGIENIZED, Status.RESOLVED)
    )
    return f"entropy={ground.global_entropy():6.2f} | {cells}"


def run_for(seconds: float, ground: PheromoneGround, *, every: float = 0.5, tag: str = "") -> None:
    """Let the swarm breathe for ``seconds``, printing a census periodically."""
    end = time.time() + seconds
    while time.time() < end:
        time.sleep(every)
        print(f"  {tag:<9} {snapshot(ground)}")


def main() -> None:
    rng = random.Random(7)  # reproducible chaos
    ground = PheromoneGround(":memory:")

    forager = ForagerAnt(
        ground, "forager", injection_rate=0.0, poll_interval=0.3, rng=rng
    )
    governance = GovernanceAnt(ground, "governance", poll_interval=0.1)
    solver = SolverAnt(ground, "solver", poll_interval=0.1)

    print("=" * 74)
    print("  StigmergicAI -- Hello, Stigmergy (the Chaos Monkey test)")
    print("  Forager (produces) -> Governance (sanitizes) -> Solver (resolves)")
    print("  No ant calls another. They share only the Pheromone Ground.")
    print("=" * 74)

    # -- Phase 1: a healthy colony -------------------------------------------
    print("\n[1] HEALTHY COLONY  (all three castes alive)")
    forager.start()
    governance.start()
    solver.start()
    run_for(3.0, ground, every=0.5, tag="working")

    # -- Phase 2: the Chaos Monkey strikes -----------------------------------
    print("\n[2] CHAOS MONKEY  (kill the Governance thread mid-flight)")
    governance.stop()
    governance.join(2)
    print(f"  governance.is_alive() = {governance.is_alive()}  -- yet the swarm does NOT crash.")
    print("  Forager keeps secreting; Solver idles (nothing HYGIENIZED to claim);")
    print("  the RAW backlog just piles up, durable in the database:")
    run_for(3.0, ground, every=0.5, tag="degraded")
    stranded = ground.count(Status.RAW)
    print(f"  --> {stranded} RAW tasks stranded, entropy climbing. Not one was lost.")

    # -- Phase 3: revive Governance and watch it vacuum the backlog ----------
    print("\n[3] SELF-HEALING  (revive Governance -- same object, brand-new thread)")
    governance.start()  # BaseAnt.start() is safe to call again after stop().
    run_for(3.0, ground, every=0.5, tag="healing")

    # -- Phase 4: stop the chaos source and let the swarm settle -------------
    print("\n[4] DRAIN  (stop the Forager; let the swarm settle to zero entropy)")
    forager.stop()
    forager.join(2)
    deadline = time.time() + 6
    while ground.global_entropy() > 0.0 and time.time() < deadline:
        time.sleep(0.1)
    governance.stop()
    solver.stop()
    governance.join(2)
    solver.join(2)

    # -- Final report --------------------------------------------------------
    stats = ground.stats()
    resolved = stats.get(Status.RESOLVED.value, 0)
    total = ground.count()
    print("-" * 74)
    print("  FINAL CENSUS")
    print("-" * 74)
    print(f"  {snapshot(ground)}")
    print(f"  total injected   : {total}")
    print(f"  resolved         : {resolved}")
    print(f"  still in-flight  : {total - resolved}")
    print(f"  global entropy   : {ground.global_entropy():.2f}")

    sample = ground.sense(min_entropy=Entropy.MIN, status=Status.RESOLVED, limit=1)
    if sample:
        p = sample[0]
        print("-" * 74)
        print("  SAMPLE RESOLVED PHEROMONE (note the Governance sanitation tag):")
        print(f"    raw_data   : {p.raw_data!r}")
        print(f"    status     : {p.status.value}")
        print(f"    hygienized : {p.metadata.get('hygienized_by')}")
        print(f"    solved_by  : {p.metadata.get('solved_by')}")
        print(f"    resolution : {p.metadata.get('resolution')}")

    print("=" * 74)
    print("  The Governance thread died and was reborn; not one task was lost.")
    print("  That is stigmergic resilience -- the database is the orchestrator.")
    print("=" * 74)

    ground.close()


if __name__ == "__main__":
    main()
