# Benchmark Comparison - All Modes

> Regenerated on 2026-06-28 20:18 UTC from `results/*.json`. Each row is one config
> from one run mode; re-run any mode to refresh its rows.

| Mode | Config | Capture rate | FPR | Precision | F1 | Accuracy |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| byzantine | `honest-single` | 97.6% | 33.3% | 80.0% | 0.879 | 84.5% |
| byzantine | `compromised-single` | 0.0% | 0.0% | 0.0% | 0.000 | 42.2% |
| byzantine | `honest-quorum-3` | 97.6% | 33.3% | 80.0% | 0.879 | 84.5% |
| byzantine | `byzantine-quorum-3` | 97.6% | 33.3% | 80.0% | 0.879 | 84.5% |
| byzantine | `paranoid-single` | 100.0% | 100.0% | 57.8% | 0.732 | 57.8% |
| byzantine | `paranoid-quorum-3` | 97.6% | 33.3% | 80.0% | 0.879 | 84.5% |
| real-NLI | `keyword-only` | 61.0% | 33.3% | 71.4% | 0.658 | 63.4% |
| real-NLI | `single-judge` | 100.0% | 86.7% | 61.2% | 0.759 | 63.4% |
| real-NLI | `quorum-3` | 100.0% | 96.7% | 58.6% | 0.739 | 59.2% |
| real-NLI | `heterogeneous-panel` | 97.6% | 33.3% | 80.0% | 0.879 | 84.5% |
| torch-free | `keyword-only` | 61.0% | 33.3% | 71.4% | 0.658 | 63.4% |
| torch-free | `single-judge` | 97.6% | 33.3% | 80.0% | 0.879 | 84.5% |
| torch-free | `quorum-3` | 97.6% | 33.3% | 80.0% | 0.879 | 84.5% |
| torch-free | `heterogeneous-panel` | 97.6% | 33.3% | 80.0% | 0.879 | 84.5% |

## Key findings

- **Relational check beats a denylist (torch-free, reproducible).** Capture rises from **61.0%** (`keyword-only`) to **97.6%** (`single-judge`) at the *same* false-positive rate -- the committed headline.
- **A real NLI model is not trustworthy *alone* (model-dependent evidence).** The raw cross-encoder juror catches **100.0%** of attacks but over-blocks legitimate work at a **86.7%** false-positive rate -- unusable on its own. Inside the heterogeneous panel its over-eagerness is outvoted: capture **97.6%** at a **33.3%** false-positive rate. *Consensus across uncorrelated jurors tames a single over-confident model.*
- **The quorum survives a faulty juror (torch-free, deterministic).** One bribed juror drops a single judge to **0.0%** capture, yet the same traitor inside a 3-node quorum holds at **97.6%**. A broken juror drives a lone judge to a **100.0%** false-positive rate; the quorum outvotes it back to **33.3%**. See `BYZANTINE.md`.

## How to read this

- **Capture rate** - recall on attacks (OWASP LLM01). Higher is better.
- **FPR** - benign work wrongly blocked. Lower is better.
- The `torch-free` rows are the committed, reproducible headline. The
  `real-nli` / `real-llm` rows depend on the model/provider used and are
  evidence, not a reproducibility guarantee. The `byzantine` rows isolate
  the value of consensus under a faulty juror (see `BYZANTINE.md`).
