"""Latent Telepathy: two ants share *understanding* as a tensor, never as text.

Run it directly::

    python examples/03_latent_telepathy.py

A 2-stage stigmergic pipeline on the LOCAL engine:

* **Ant A (Reader)** wakes on a heavy document, runs a forward pass to distill it
  into a last-layer hidden-state tensor ``[1, seq_len, hidden_dim]``, serializes
  that tensor into the pheromone's ``latent_blob``, and -- crucially -- *erases the
  text* from ``raw_data`` (replacing it with a marker). It stamps ``LATENT_READY``.
* **Ant B (Decider)** wakes on ``LATENT_READY``. It never reads the document (the
  text is gone). It deserializes the tensor, injects it directly into its own
  ``inputs_embeds`` residual stream, and generates a decision from pure latent
  context. Zero tokens crossed between the two ants.

Mocking note: so this runs without downloading a multi-GB model (or even needing
torch), the model below is a lightweight ``MockLatentLM`` that fakes the tensor
shapes and the embedding concatenation. Every mock is annotated with the real
PyTorch / transformers calls it stands in for. The Reader is a real
:class:`HybridSolverAnt` in ``engine="local"`` mode, driven by a mock encoder.
"""

from __future__ import annotations

import hashlib
import pathlib
import pickle
import sys
import time
from dataclasses import dataclass

# Make the src-layout package importable when running this file directly.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from stigmergic_ai.agents.base_ant import ConsumerAnt, Mutation  # noqa: E402
from stigmergic_ai.agents.hybrid_ant import HybridSolverAnt  # noqa: E402
from stigmergic_ai.core.environment import (  # noqa: E402
    Entropy,
    Pheromone,
    PheromoneGround,
    Status,
)

HIDDEN_DIM = 64
MAX_SEQ = 32


# ---------------------------------------------------------------------------
# Mock model + tensor (stand-ins for torch / transformers)
# ---------------------------------------------------------------------------


@dataclass
class MockTensor:
    """A stand-in for a ``torch.Tensor`` of shape ``[batch, seq_len, hidden_dim]``.

    Real code would use an actual ``torch.Tensor``; here we only carry the shape
    and a tiny deterministic payload so the demo runs with no dependencies.
    """

    shape: tuple[int, int, int]
    data: list[float]

    def numel(self) -> int:
        return self.shape[0] * self.shape[1] * self.shape[2]

    def mean(self) -> float:
        return sum(self.data) / len(self.data) if self.data else 0.0


class MockLatentLM:
    """A featherweight fake of a causal LM exposing hidden states + embedding gen.

    Stands in for, e.g.::

        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    """

    def __init__(self, hidden_dim: int = HIDDEN_DIM) -> None:
        self.hidden_dim = hidden_dim

    def encode(self, text: str) -> MockTensor:
        """Forward pass -> final-layer hidden states ``[1, seq_len, hidden_dim]``.

        Real path::

            inputs = tokenizer(text, return_tensors="pt")
            out = model(**inputs, output_hidden_states=True)
            return out.hidden_states[-1]          # [1, seq_len, hidden_dim]
        """
        seq_len = max(1, min(len(text.split()), MAX_SEQ))
        # Deterministic pseudo-activations derived from the text, so the demo is
        # reproducible. (A real model would compute these via attention layers.)
        seed = int(hashlib.sha256(text.encode()).hexdigest(), 16)
        data = [((seed >> (i % 53)) & 0xFF) / 255.0 for i in range(seq_len * self.hidden_dim)]
        return MockTensor(shape=(1, seq_len, self.hidden_dim), data=data)

    def decide_from_latent(self, latent: MockTensor, prompt: str) -> tuple[str, tuple[int, int, int]]:
        """Inject a latent tensor into the residual stream and generate a decision.

        Real path (the heart of Latent State Transfer)::

            prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
            prompt_embeds = model.get_input_embeddings()(prompt_ids)   # [1, P, H]
            # `latent` are the Reader's hidden states: [1, S, H]
            combined = torch.cat([latent, prompt_embeds], dim=1)       # [1, S+P, H]
            generated = model.generate(inputs_embeds=combined, max_new_tokens=32)
            return tokenizer.decode(generated[0]), tuple(combined.shape)

        Here we mock the concatenation (shapes only) and derive a deterministic
        verdict from the injected activations -- without ever seeing the document.
        """
        prompt_len = max(1, len(prompt.split()))
        combined_shape = (1, latent.shape[1] + prompt_len, latent.shape[2])
        verdict = "APPROVE" if latent.mean() >= 0.5 else "ESCALATE"
        decision = (
            f"{verdict}: decision generated from {latent.shape[1]} latent positions "
            f"injected into a {combined_shape[1]}-position residual stream "
            f"(no document tokens were read)."
        )
        return decision, combined_shape


# A pickle codec for the MockTensor. In production this is exactly where you call
# `latent_transfer.serialize_tensor(real_tensor)` / `deserialize_tensor(blob)`,
# which use torch.save / torch.load under the hood.
def mock_serialize(tensor: MockTensor) -> bytes:
    return pickle.dumps(tensor)


def mock_deserialize(blob: bytes) -> MockTensor:
    return pickle.loads(blob)  # noqa: S301 - trusted, demo-local mock blob


MODEL = MockLatentLM()


# ---------------------------------------------------------------------------
# Ant B: the Decider (consumes latent only; never reads the document)
# ---------------------------------------------------------------------------


class DeciderAnt(ConsumerAnt):
    """Wakes on ``LATENT_READY``, injects the tensor, and generates a decision."""

    def __init__(self, env: PheromoneGround, name: str = "decider", **kw) -> None:
        super().__init__(
            env,
            name,
            entropy_threshold=Entropy.MIN,
            target_status=Status.LATENT_READY,
            poll_interval=0.1,
            **kw,
        )

    def metabolize(self, task: Pheromone) -> Mutation:
        # The document text is gone from raw_data; all we have is the tensor.
        latent = mock_deserialize(task.latent_blob)
        decision, combined_shape = MODEL.decide_from_latent(
            latent, prompt="Given the report, should we proceed?"
        )
        self.log.info("Decided from latent %s -> %s", latent.shape, combined_shape)
        metadata = dict(task.metadata or {})
        metadata.update(
            {
                "decision": decision,
                "decided_by": self.name,
                "recovered_latent_shape": list(latent.shape),
                "residual_stream_shape": list(combined_shape),
            }
        )
        return Mutation(
            new_entropy=Entropy.ZERO,
            new_status=Status.RESOLVED,
            metadata=metadata,
            release_owner=True,
        )


HEAVY_DOCUMENT = (
    "QUARTERLY RISK REPORT. " + (
        "The regional portfolio shows resilient margins despite supply volatility; "
        "liquidity buffers exceed policy thresholds and counterparty exposure is "
        "well diversified across investment-grade names. "
    ) * 12
)


def main() -> None:
    ground = PheromoneGround(":memory:")

    # Ant A (Reader): a real HybridSolverAnt on the LOCAL engine, driven by the
    # mock encoder. It extracts the latent, parks it in latent_blob, and WITHHOLDS
    # the text by replacing raw_data with a marker.
    reader = HybridSolverAnt(
        ground,
        "reader",
        engine="local",
        latent_encoder=lambda text: mock_serialize(MODEL.encode(text)),
        target_status=Status.RAW,
        result_status=Status.LATENT_READY,
        latent_marker="<<latent encoded — original text withheld>>",
        poll_interval=0.1,
    )
    decider = DeciderAnt(ground)

    print("=" * 74)
    print("  StigmergicAI -- Latent Telepathy (Latent State Transfer)")
    print("  Reader (HybridSolverAnt, engine=local)  ->  Decider")
    print("=" * 74)

    doc_id = ground.inject_chaos(HEAVY_DOCUMENT, entropy=Entropy.CHAOS,
                                 metadata={"doc_tokens": len(HEAVY_DOCUMENT.split())})
    print(f"Injected heavy document id={doc_id}: {len(HEAVY_DOCUMENT)} chars, "
          f"{len(HEAVY_DOCUMENT.split())} tokens.")

    reader.start()
    decider.start()
    try:
        deadline = time.time() + 10
        while ground.count(Status.RESOLVED) < 1 and time.time() < deadline:
            time.sleep(0.1)
    finally:
        reader.stop()
        decider.stop()
        reader.join(2)
        decider.join(2)

    final = ground.get(doc_id)
    print("-" * 74)
    print("  PROOF OF TELEPATHY")
    print("-" * 74)
    print(f"  raw_data now      : {final.raw_data!r}")
    print(f"  (text withheld)   : {'<<latent' in (final.raw_data or '')}")
    print(f"  latent_blob bytes : {final.metadata.get('latent_nbytes')}")
    print(f"  transfer mode     : {final.metadata.get('transfer')}")
    print(f"  recovered shape   : {final.metadata.get('recovered_latent_shape')}")
    print(f"  residual stream   : {final.metadata.get('residual_stream_shape')}")
    print(f"  final status      : {final.status.value}")
    print(f"  global entropy    : {ground.global_entropy():.2f}")
    print("-" * 74)
    print(f"  DECISION: {final.metadata.get('decision')}")
    print("=" * 74)
    print("  Ant B decided using only the tensor in latent_blob -- the document")
    print("  text never crossed between the ants. That is Latent State Transfer.")
    print()

    ground.close()


if __name__ == "__main__":
    main()
