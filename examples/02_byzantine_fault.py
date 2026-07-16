"""End-to-end Byzantine Fault demo: a live swarm with a hallucination-slashing jury.

Run it directly::

    python examples/02_byzantine_fault.py

A Forager floods a shared in-memory "Pheromone Ground" with a mix of legitimate
requests and blatant prompt injections. A Governance ant sanitizes them, two
Solver ants stage resolutions, and a Verifier ant runs a Byzantine NLI quorum
(Semantic Raft) that finalizes honest work and *slashes* the poisoned trails.

No ant ever calls another -- they coordinate solely by reading and mutating the
shared environment. The Byzantine judge runs on a deterministic ``MockNLIJudge``
so the demo is fast and reproducible without torch/transformers.
"""

from __future__ import annotations

import pathlib
import random
import sys
import time

# Make the src-layout package importable when running this file directly,
# without first running `pip install -e .`.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from stigmergic.agents.concrete import (  # noqa: E402
    ForagerAnt,
    GovernanceAnt,
    SolverAnt,
)
from stigmergic.agents.verifier_ant import VerifierAnt  # noqa: E402
from stigmergic.core.consensus import MockNLIJudge  # noqa: E402
from stigmergic.core.environment import (  # noqa: E402
    PheromoneGround,
    Status,
)

RUN_SECONDS = 15
REPORT_EVERY = 2
DRAIN_TIMEOUT = 6


def snapshot(ground: PheromoneGround) -> str:
    """A one-line view of entropy and the per-status pheromone census."""
    stats = ground.stats()
    cells = "  ".join(f"{s.value}={stats.get(s.value, 0)}" for s in Status)
    return f"entropy={ground.global_entropy():5.2f} | {cells}"


def main() -> None:
    rng = random.Random(1234)  # reproducible chaos
    ground = PheromoneGround(":memory:")

    forager = ForagerAnt(
        ground, "forager", injection_rate=0.4, poll_interval=0.5, rng=rng
    )
    governance = GovernanceAnt(ground, "governance", poll_interval=0.12)
    solver_a = SolverAnt(ground, "solver-A", poll_interval=0.12)
    solver_b = SolverAnt(ground, "solver-B", poll_interval=0.12)
    verifier = VerifierAnt(
        ground,
        "verifier",
        judge=MockNLIJudge(),
        quorum_size=3,
        approval_threshold=2,
        poll_interval=0.12,
    )
    swarm = [forager, governance, solver_a, solver_b, verifier]

    print("=" * 74)
    print("  Stigmergic -- Byzantine Fault demo")
    print("  swarm: 1 Forager, 1 Governance, 2 Solvers, 1 Verifier (Mock NLI quorum)")
    print("=" * 74)

    for ant in swarm:
        ant.start()

    try:
        start = time.time()
        while time.time() - start < RUN_SECONDS:
            time.sleep(REPORT_EVERY)
            print(f"[t+{time.time() - start:4.1f}s] {snapshot(ground)}")

        # Graceful drain: cut off the chaos source and let the consumers vacuum
        # the backlog to a settled state -- eventual consistency in action.
        print("-" * 74)
        print("Cutting off the Forager; draining the backlog...")
        forager.stop()
        forager.join(2)

        drain_start = time.time()
        while ground.global_entropy() > 0.0 and (time.time() - drain_start) < DRAIN_TIMEOUT:
            time.sleep(0.4)
        print(f"[drained ] {snapshot(ground)}")
    finally:
        for ant in swarm:
            ant.stop()
        for ant in swarm:
            ant.join(2)

    resolved = ground.count(Status.RESOLVED)
    slashed = ground.count(Status.SLASHED)
    total = ground.count()
    in_flight = total - resolved - slashed

    print("=" * 74)
    print("  FINAL REPORT")
    print("-" * 74)
    print(f"  Total tasks injected : {total}")
    print(f"  RESOLVED (passed)    : {resolved}")
    print(f"  SLASHED  (rejected)  : {slashed}")
    print(f"  Still in-flight      : {in_flight}")
    print(f"  Residual entropy     : {ground.global_entropy():.2f}")
    print(f"  All threads stopped  : {not any(a.is_alive() for a in swarm)}")
    print("=" * 74)
    if slashed:
        print("  The Byzantine quorum swung its axe: prompt injections were slashed,")
        print("  legitimate work was resolved -- with zero direct calls between ants.")
    print()

    ground.close()


if __name__ == "__main__":
    main()
