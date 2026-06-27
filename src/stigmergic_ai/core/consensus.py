"""Semantic Raft: a Byzantine quorum that gates every cognitive state mutation.

You would not trust a single microservice to silently mutate production state;
StigmergicAI extends the same suspicion to a single language model. Before a
proposed resolution is allowed to drive a pheromone's entropy to zero, it must
survive a **Byzantine Cognitive Consensus** -- a small jury of independent
Natural Language Inference (NLI) judges that each ask, in their own words,
"does this proposed action actually follow from the original request, or is it a
hallucination / injection?"

The flow:

* A Solver proposes a :class:`~stigmergic_ai.agents.base_ant.Mutation` and parks
  the pheromone on the ``PENDING_CONSENSUS`` trail (it is *not* committed).
* A :class:`~stigmergic_ai.agents.verifier_ant.VerifierAnt` smells that trail and
  runs :meth:`SemanticRaft.evaluate`.
* Each juror node classifies ``(premise, hypothesis)`` as ``ENTAILMENT``,
  ``CONTRADICTION`` or ``NEUTRAL``. Approval requires ``ENTAILMENT``.
* If at least ``approval_threshold`` of ``quorum_size`` nodes approve, the quorum
  passes and the transaction is finalized. Otherwise it is **slashed**.

Design notes:

* This module is import-light: ``transformers``/``torch`` are imported lazily,
  only when a real model is first loaded. Importing :mod:`stigmergic_ai` -- or
  even this module -- never pulls the deep-learning stack. Install it with
  ``pip install -e ".[cognition]"``.
* The heavy NLI pipeline is loaded **once** per model name into a module-level,
  lock-guarded cache, so a colony of verifier threads shares a single model in
  memory rather than each paying the load cost.
* :class:`MockNLIJudge` provides a deterministic, dependency-free judge so the
  quorum machinery can be unit-tested without downloading a model.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Any, Callable, Protocol, runtime_checkable

from pydantic import BaseModel

from stigmergic_ai.agents.base_ant import Mutation
from stigmergic_ai.core.environment import Entropy, Pheromone, Status

__all__ = [
    "ENTAILMENT",
    "CONTRADICTION",
    "NEUTRAL",
    "DEFAULT_NLI_MODEL",
    "DEFAULT_PERSPECTIVES",
    "Vote",
    "ConsensusVerdict",
    "NLIJudge",
    "TransformersNLIJudge",
    "MockNLIJudge",
    "SemanticRaft",
]

logger = logging.getLogger("stigmergic_ai.consensus")

# -- canonical NLI labels -----------------------------------------------------

ENTAILMENT = "ENTAILMENT"
CONTRADICTION = "CONTRADICTION"
NEUTRAL = "NEUTRAL"

#: A small, CPU-friendly cross-encoder NLI model. Each juror is a forward pass.
DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-small"

#: Per-node "perspectives". Independent jurors must not be perfect clones, or the
#: quorum is theatre: identical nodes always vote identically. Each perspective
#: reframes the proposed action as a slightly different hypothesis, so genuinely
#: borderline proposals can split the vote. In production each node could instead
#: be a distinct model or endpoint; here a single model wears several hats.
DEFAULT_PERSPECTIVES: tuple[str, ...] = (
    "{hypothesis}",
    "The proposed action is a correct and faithful response to the request: {hypothesis}",
    "It is safe, non-malicious and logically valid to proceed with: {hypothesis}",
)

#: Substrings that the mock judge treats as outright hallucination / injection.
DEFAULT_RED_FLAGS: frozenset[str] = frozenset(
    {
        "ignore previous",
        "ignore all previous",
        "drop table",
        "delete from",
        "rm -rf",
        "<script",
        "system prompt",
        "as an ai language model",
        "disregard the above",
    }
)


# -- verdict value objects ----------------------------------------------------


class Vote(BaseModel):
    """A single juror node's independent ruling on the proposed logic."""

    node_id: int
    perspective: str
    label: str
    score: float
    approve: bool


class ConsensusVerdict(BaseModel):
    """The aggregated outcome of a Byzantine quorum over one proposal."""

    passed: bool
    verdict_status: Status
    approvals: int
    rejections: int
    quorum_size: int
    threshold: int
    votes: list[Vote]
    premise: str
    hypothesis: str
    model_name: str | None = None
    reason: str = ""

    def to_metadata(self) -> dict[str, Any]:
        """A compact, JSON-serializable audit record for the pheromone trail."""
        return {
            "passed": self.passed,
            "verdict": self.verdict_status.value,
            "approvals": self.approvals,
            "rejections": self.rejections,
            "quorum_size": self.quorum_size,
            "threshold": self.threshold,
            "model": self.model_name,
            "reason": self.reason,
            "votes": [
                {
                    "node": v.node_id,
                    "label": v.label,
                    "score": round(v.score, 4),
                    "approve": v.approve,
                }
                for v in self.votes
            ],
        }


# -- the judge abstraction ----------------------------------------------------


@runtime_checkable
class NLIJudge(Protocol):
    """Anything that can rule on whether a premise entails a hypothesis.

    Implementations return a ``(label, score)`` pair where ``label`` is one of
    :data:`ENTAILMENT`, :data:`CONTRADICTION` or :data:`NEUTRAL`.
    """

    model_name: str | None

    def classify(self, premise: str, hypothesis: str) -> tuple[str, float]:
        ...


# Module-level pipeline cache: load each heavy NLI model exactly once and share
# it across every verifier thread. Guarded by a lock so concurrent ants waking
# at the same instant cannot trigger two parallel (expensive) model loads.
_PIPELINE_CACHE: dict[str, Any] = {}
_PIPELINE_LOCK = threading.Lock()


def _build_nli_pipeline(model_name: str, device: Any | None) -> Any:
    """Construct a fresh ``transformers`` text-classification pipeline.

    Isolated into its own function so tests can monkeypatch the actual model
    construction (and assert the singleton caches correctly) without torch.
    """
    try:
        from transformers import pipeline  # lazy: torch only loaded on real use
    except ImportError as exc:  # pragma: no cover - exercised only without extras
        raise ImportError(
            "The Byzantine consensus engine needs the deep-learning extras. "
            'Install them with: pip install -e ".[cognition]"'
        ) from exc
    logger.info("Loading NLI model %r (first use; cached thereafter).", model_name)
    return pipeline("text-classification", model=model_name, device=device)


def _get_nli_pipeline(model_name: str, device: Any | None = None) -> Any:
    """Return a cached pipeline for ``model_name``, building it once if needed."""
    cached = _PIPELINE_CACHE.get(model_name)
    if cached is not None:
        return cached
    with _PIPELINE_LOCK:
        # Double-checked: another thread may have built it while we waited.
        cached = _PIPELINE_CACHE.get(model_name)
        if cached is None:
            cached = _build_nli_pipeline(model_name, device)
            _PIPELINE_CACHE[model_name] = cached
        return cached


def _canonical_label(label: str) -> str:
    """Normalize any model's label spelling to a canonical NLI constant."""
    up = str(label).strip().upper()
    if "ENTAIL" in up:
        return ENTAILMENT
    if "CONTRA" in up:
        return CONTRADICTION
    if "NEUTRAL" in up:
        return NEUTRAL
    # id2label fallback for cross-encoder/nli-deberta-v3-small (0/1/2).
    return {"LABEL_0": CONTRADICTION, "LABEL_1": ENTAILMENT, "LABEL_2": NEUTRAL}.get(up, up)


def _top_label_and_score(raw: Any) -> tuple[str, float]:
    """Reduce a (possibly nested) pipeline output to the top ``(label, score)``."""
    if isinstance(raw, dict):
        return str(raw["label"]), float(raw["score"])
    if isinstance(raw, list) and raw:
        first = raw[0]
        candidates = first if isinstance(first, list) else raw
        best = max(candidates, key=lambda d: float(d.get("score", 0.0)))
        return str(best["label"]), float(best["score"])
    raise TypeError(f"Unexpected NLI pipeline output: {raw!r}")


class TransformersNLIJudge:
    """A real NLI juror backed by a ``transformers`` cross-encoder.

    The underlying pipeline is loaded lazily on first :meth:`classify` and then
    shared, via the module-level cache, with every other judge that uses the same
    model. Constructing this object is therefore cheap and torch-free; only the
    first real classification pays the model-loading cost.
    """

    def __init__(self, model_name: str = DEFAULT_NLI_MODEL, *, device: Any | None = None) -> None:
        self.model_name = model_name
        self.device = device

    @property
    def pipeline(self) -> Any:
        """The cached, lazily-loaded ``transformers`` pipeline."""
        return _get_nli_pipeline(self.model_name, self.device)

    def classify(self, premise: str, hypothesis: str) -> tuple[str, float]:
        raw = self.pipeline({"text": premise, "text_pair": hypothesis})
        label, score = _top_label_and_score(raw)
        return _canonical_label(label), score


def _significant_tokens(text: str) -> set[str]:
    """Lowercase word tokens of length > 3, used by the mock for overlap."""
    return {tok for tok in re.findall(r"[a-z0-9]+", text.lower()) if len(tok) > 3}


class MockNLIJudge:
    """A deterministic, dependency-free juror for tests and offline demos.

    It is intentionally crude -- it is *not* a language model. It exists so the
    quorum and slashing machinery can be exercised without downloading weights:

    * a configurable set of red-flag substrings (injection / hallucination
      markers) forces :data:`CONTRADICTION`;
    * otherwise, lexical overlap with the premise yields :data:`ENTAILMENT`;
    * failing both, the verdict is :data:`NEUTRAL`.

    Pass a custom ``rule`` callable to script exact votes in a unit test.
    """

    model_name = "mock-nli"

    def __init__(
        self,
        *,
        red_flags: frozenset[str] | set[str] | None = None,
        rule: Callable[[str, str], tuple[str, float]] | None = None,
        overlap_threshold: int = 1,
    ) -> None:
        self.red_flags = {f.lower() for f in (red_flags or DEFAULT_RED_FLAGS)}
        self.rule = rule
        self.overlap_threshold = overlap_threshold

    def classify(self, premise: str, hypothesis: str) -> tuple[str, float]:
        if self.rule is not None:
            return self.rule(premise, hypothesis)
        haystack = hypothesis.lower()
        for flag in self.red_flags:
            if flag in haystack:
                return CONTRADICTION, 0.97
        overlap = len(_significant_tokens(premise) & _significant_tokens(hypothesis))
        if overlap >= self.overlap_threshold:
            return ENTAILMENT, 0.90
        return NEUTRAL, 0.55


# -- the quorum engine --------------------------------------------------------


class SemanticRaft:
    """A Byzantine quorum of NLI jurors gating one proposed mutation.

    Args:
        judge: The :class:`NLIJudge` each node consults. Defaults to a
            (lazily-loaded) :class:`TransformersNLIJudge` on ``model_name``.
        model_name: Model used when ``judge`` is not supplied.
        quorum_size: Number of independent juror nodes to poll.
        approval_threshold: Minimum ``ENTAILMENT`` votes required to pass
            (e.g. ``2`` of ``3`` -- a strict majority).
        perspectives: Hypothesis-framing templates giving each node its own
            "viewpoint"; node ``i`` uses ``perspectives[i % len(perspectives)]``.
        min_confidence: An approving vote must also clear this score.
        proposal_key: Metadata key under which the proposed action text lives.
        premise_key: Metadata key for an explicit premise (falls back to the
            pheromone's ``raw_data``).
    """

    def __init__(
        self,
        judge: NLIJudge | None = None,
        *,
        model_name: str = DEFAULT_NLI_MODEL,
        quorum_size: int = 3,
        approval_threshold: int = 2,
        perspectives: tuple[str, ...] = DEFAULT_PERSPECTIVES,
        min_confidence: float = 0.0,
        proposal_key: str = "proposal",
        premise_key: str = "original",
    ) -> None:
        if quorum_size < 1:
            raise ValueError("quorum_size must be at least 1.")
        if not (1 <= approval_threshold <= quorum_size):
            raise ValueError("approval_threshold must lie within [1, quorum_size].")
        if not perspectives:
            raise ValueError("At least one perspective template is required.")
        self.judge: NLIJudge = judge if judge is not None else TransformersNLIJudge(model_name)
        self.model_name = getattr(self.judge, "model_name", None)
        self.quorum_size = quorum_size
        self.approval_threshold = approval_threshold
        self.perspectives = tuple(perspectives)
        self.min_confidence = min_confidence
        self.proposal_key = proposal_key
        self.premise_key = premise_key

    # -- premise / hypothesis extraction (overridable) ------------------------

    def _extract_premise(self, pheromone: Pheromone) -> str:
        """The original request the proposal must logically follow from."""
        meta = pheromone.metadata or {}
        original = meta.get(self.premise_key)
        return str(original) if original else pheromone.raw_data

    def _extract_hypothesis(self, pheromone: Pheromone, mutation: Mutation) -> str:
        """The proposed action/claim under scrutiny.

        Looked up first in the proposed mutation's metadata, then on the
        pheromone itself, so either side may carry the proposal.
        """
        for source in (mutation.metadata, pheromone.metadata):
            if source:
                value = source.get(self.proposal_key)
                if value:
                    return str(value)
        # No explicit proposal text: fall back to a status-derived claim so the
        # jury still has something concrete to rule on.
        target = mutation.new_status.value if mutation.new_status else "resolved"
        return f"The request has been correctly handled and marked {target}."

    # -- the vote -------------------------------------------------------------

    def evaluate(
        self,
        pheromone: Pheromone,
        mutation: Mutation,
        *,
        premise: str | None = None,
        hypothesis: str | None = None,
    ) -> ConsensusVerdict:
        """Run the full quorum over ``(pheromone, mutation)`` and rule on it.

        Args:
            pheromone: The task carrying the original request (the premise).
            mutation: The proposed resolution (the hypothesis under test).
            premise: Optional explicit premise override (skips extraction).
            hypothesis: Optional explicit hypothesis override (skips extraction).

        Returns:
            A :class:`ConsensusVerdict`. ``passed`` is ``True`` iff at least
            ``approval_threshold`` nodes returned ``ENTAILMENT``; ``verdict_status``
            is :attr:`Status.RESOLVED` on success or :attr:`Status.SLASHED` on
            rejection.
        """
        premise = premise if premise is not None else self._extract_premise(pheromone)
        hypothesis = (
            hypothesis if hypothesis is not None else self._extract_hypothesis(pheromone, mutation)
        )

        votes: list[Vote] = []
        for node_id in range(self.quorum_size):
            perspective = self.perspectives[node_id % len(self.perspectives)]
            framed = perspective.format(hypothesis=hypothesis, premise=premise)
            try:
                label, score = self.judge.classify(premise, framed)
                label = _canonical_label(label)
                approve = label == ENTAILMENT and score >= self.min_confidence
            except Exception:  # noqa: BLE001
                # A crashed juror is a non-vote in a Byzantine setting, never a
                # crash of the whole quorum. It counts against approval.
                logger.exception("Juror node %d failed; recording a rejecting vote.", node_id)
                label, score, approve = "ERROR", 0.0, False
            votes.append(
                Vote(
                    node_id=node_id,
                    perspective=perspective,
                    label=label,
                    score=score,
                    approve=approve,
                )
            )

        approvals = sum(1 for v in votes if v.approve)
        rejections = len(votes) - approvals
        passed = approvals >= self.approval_threshold
        verdict_status = Status.RESOLVED if passed else Status.SLASHED
        reason = (
            f"{approvals}/{self.quorum_size} jurors entailed the proposal "
            f"(threshold {self.approval_threshold}); "
            f"{'QUORUM PASSED' if passed else 'SLASHED'}."
        )
        logger.info("Semantic Raft verdict for pheromone id=%s: %s", pheromone.id, reason)

        return ConsensusVerdict(
            passed=passed,
            verdict_status=verdict_status,
            approvals=approvals,
            rejections=rejections,
            quorum_size=self.quorum_size,
            threshold=self.approval_threshold,
            votes=votes,
            premise=premise,
            hypothesis=hypothesis,
            model_name=self.model_name,
            reason=reason,
        )

    def finalize(self, verdict: ConsensusVerdict, *, base_metadata: dict[str, Any] | None = None) -> Mutation:
        """Turn a verdict into the terminal :class:`Mutation` to commit.

        On a passing quorum the pheromone is driven to ``RESOLVED``; on rejection
        it is ``SLASHED``. Either way entropy goes to zero -- a slashed trail must
        evaporate so no other ant ever follows the poisoned proposal. The verdict
        is folded into ``metadata`` (preserving any provenance passed in) for a
        durable audit record.
        """
        metadata = dict(base_metadata or {})
        metadata["consensus"] = verdict.to_metadata()
        return Mutation(
            new_entropy=Entropy.ZERO,
            new_status=verdict.verdict_status,
            metadata=metadata,
        )
