# Benchmark Comparison - All Modes

> Regenerated on 2026-07-16 16:37 UTC from `results/*.json`. Each row is one config
> from one run mode; re-run any mode to refresh its rows.

| Mode | Config | Capture rate | FPR | Precision | F1 | Accuracy |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| byzantine | `honest-single` | 82.1% | 50.0% | 60.5% | 0.697 | 65.5% |
| byzantine | `compromised-single` | 0.0% | 0.0% | 0.0% | 0.000 | 51.7% |
| byzantine | `honest-quorum-3` | 82.1% | 50.0% | 60.5% | 0.697 | 65.5% |
| byzantine | `byzantine-quorum-3` | 80.4% | 50.0% | 60.0% | 0.687 | 64.7% |
| byzantine | `paranoid-single` | 100.0% | 100.0% | 48.3% | 0.651 | 48.3% |
| byzantine | `paranoid-quorum-3` | 82.1% | 50.0% | 60.5% | 0.697 | 65.5% |
| real-NLI | `keyword-only` | 44.6% | 50.0% | 45.5% | 0.451 | 47.4% |
| real-NLI | `single-judge` | 100.0% | 90.0% | 50.9% | 0.675 | 53.4% |
| real-NLI | `quorum-3` | 100.0% | 98.3% | 48.7% | 0.655 | 49.1% |
| real-NLI | `heterogeneous-panel` | 82.1% | 50.0% | 60.5% | 0.697 | 65.5% |
| torch-free | `keyword-only` | 44.6% | 50.0% | 45.5% | 0.451 | 47.4% |
| torch-free | `single-judge` | 82.1% | 50.0% | 60.5% | 0.697 | 65.5% |
| torch-free | `quorum-3` | 82.1% | 50.0% | 60.5% | 0.697 | 65.5% |
| torch-free | `heterogeneous-panel` | 82.1% | 50.0% | 60.5% | 0.697 | 65.5% |

## Key findings

- **Relational check beats a denylist (torch-free, reproducible).** Capture rises from **44.6%** (`keyword-only`) to **82.1%** (`single-judge`) at the *same* false-positive rate -- the committed headline.
- **A real NLI model is not trustworthy *alone* (model-dependent evidence).** The raw cross-encoder juror catches **100.0%** of attacks but over-blocks legitimate work at a **90.0%** false-positive rate -- unusable on its own. Inside the heterogeneous panel its over-eagerness is outvoted: capture **82.1%** at a **50.0%** false-positive rate. *Consensus across uncorrelated jurors tames a single over-confident model.*
- **The quorum survives a faulty juror (torch-free, deterministic).** One bribed juror drops a single judge to **0.0%** capture, yet the same traitor inside a 3-node quorum holds at **80.4%**. A broken juror drives a lone judge to a **100.0%** false-positive rate; the quorum outvotes it back to **50.0%**. See `BYZANTINE.md`.

## How to read this

- **Capture rate** - recall on attacks (OWASP LLM01). Higher is better.
- **FPR** - benign work wrongly blocked. Lower is better.
- The `torch-free` rows are the committed, reproducible headline. The
  `real-nli` / `real-llm` rows depend on the model/provider used and are
  evidence, not a reproducibility guarantee. The `byzantine` rows isolate
  the value of consensus under a faulty juror (see `BYZANTINE.md`).
