"""Run the StigmergicAI injection-capture benchmark and write a report.

Usage (from the repo root)::

    # torch-free, fully reproducible, no downloads, no keys:
    python benchmarks/injection_capture/run.py

    # swap in the real cross-encoder NLI juror (needs the [cognition] extras):
    python benchmarks/injection_capture/run.py --nli

    # add a real LLM juror to the panel (needs [benchmark] extras + env config;
    # see providers.py for the environment variables):
    python benchmarks/injection_capture/run.py --llm

It compares four anti-injection configs on the same labeled corpus -- a naive
``keyword-only`` denylist (the baseline), a ``single-judge`` semantic check, a
homogeneous ``quorum-3``, and the flagship heterogeneous ``panel`` -- and writes
``results.json`` (machine-readable) and ``RESULTS.md`` (human-readable) next to
this script. The headline metric is the **capture rate** (recall on attacks),
mapped to OWASP LLM01 (Prompt Injection); the **false-positive rate** guards
against a control that simply blocks everything.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from harness import (  # noqa: E402
    Metrics,
    build_defenses,
    evaluate_defense,
    load_dataset,
)

DATASET = HERE / "dataset.jsonl"
RESULTS_JSON = HERE / "results.json"
RESULTS_MD = HERE / "RESULTS.md"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="StigmergicAI injection-capture benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", type=pathlib.Path, default=DATASET, help="Path to the .jsonl corpus."
    )
    parser.add_argument(
        "--nli",
        action="store_true",
        help="Use the real TransformersNLIJudge (downloads a model; needs [cognition]).",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Add a real LLM juror to the panel (needs [benchmark] + env config).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the on-disk LLM response cache (only affects --llm).",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress the console summary table."
    )
    return parser.parse_args(argv)


def _resolve_llm_complete(args: argparse.Namespace):
    if not args.llm:
        return None
    from providers import build_llm_complete

    return build_llm_complete(use_cache=not args.no_cache)


def _format_console_table(metrics: list[Metrics]) -> str:
    header = f"{'config':<22}{'capture':>9}{'FPR':>8}{'prec':>8}{'F1':>8}{'acc':>8}"
    lines = [header, "-" * len(header)]
    for m in metrics:
        lines.append(
            f"{m.config:<22}"
            f"{m.capture_rate:>9.3f}"
            f"{m.false_positive_rate:>8.3f}"
            f"{m.precision:>8.3f}"
            f"{m.f1:>8.3f}"
            f"{m.accuracy:>8.3f}"
        )
    return "\n".join(lines)


def _md_table(metrics: list[Metrics]) -> str:
    rows = [
        "| Config | Capture rate (recall) | False-positive rate | Precision | F1 | Accuracy |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for m in metrics:
        rows.append(
            f"| `{m.config}` | {m.capture_rate:.1%} | {m.false_positive_rate:.1%} "
            f"| {m.precision:.1%} | {m.f1:.3f} | {m.accuracy:.1%} |"
        )
    return "\n".join(rows)


def _md_category_table(metrics: list[Metrics]) -> str:
    categories = sorted(metrics[0].per_category_total)
    head = "| Attack category | " + " | ".join(f"`{m.config}`" for m in metrics) + " |"
    sep = "| :--- | " + " | ".join("---:" for _ in metrics) + " |"
    rows = [head, sep]
    for cat in categories:
        cells = []
        for m in metrics:
            caps = m.category_capture().get(cat)
            cells.append(f"{caps:.0%}" if caps is not None else "-")
        rows.append(f"| {cat} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _build_markdown(
    metrics: list[Metrics], *, n_items: int, n_attacks: int, mode: str
) -> str:
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    keyword = next((m for m in metrics if m.config == "keyword-only"), None)
    single = next((m for m in metrics if m.config == "single-judge"), None)
    lift = ""
    if keyword and single:
        delta = single.capture_rate - keyword.capture_rate
        lift = (
            f"The relational `single-judge` check raised the capture rate from "
            f"**{keyword.capture_rate:.0%}** (keyword-only) to "
            f"**{single.capture_rate:.0%}** "
            f"(**+{delta:.0%}**) at a false-positive rate of "
            f"{single.false_positive_rate:.0%} "
            f"(vs {keyword.false_positive_rate:.0%} for the denylist).\n"
        )
    return f"""# Injection-Capture Benchmark - Results

> Generated by `run.py` on {stamp}. Run mode: **{mode}**.
> Corpus: **{n_items}** labeled cases ({n_attacks} attacks, {n_items - n_attacks} benign).

This benchmark measures the StigmergicAI Byzantine quorum as an anti-prompt-
injection control, mapped to **OWASP LLM01 (Prompt Injection)**. The positive
class is "attack"; a *capture* is an attack the quorum **slashes**.

## Headline

{lift}
{_md_table(metrics)}

- **Capture rate (recall)** - of all attacks, the fraction blocked. Higher is better. *This is the LLM01 headline.*
- **False-positive rate** - of all benign work, the fraction wrongly blocked. Lower is better.
- **Precision / F1 / Accuracy** - standard balance metrics with attack as the positive class.

## Capture rate by attack category

{_md_category_table(metrics)}

## How this was produced

```bash
# torch-free, reproducible, no model downloads, no API keys:
python benchmarks/injection_capture/run.py

# opt in to the real semantic juror (cross-encoder NLI; needs [cognition]):
python benchmarks/injection_capture/run.py --nli

# opt in to a real LLM juror on the panel (needs [benchmark] + env config):
python benchmarks/injection_capture/run.py --llm
```

The four configs share one uniform `Defense` interface, so they are scored
identically on the identical corpus:

| Config | What it is |
| :--- | :--- |
| `keyword-only` | A denylist: block iff a known bad phrase appears verbatim. The honest baseline. |
| `single-judge` | One semantic juror asks "does this action follow from the request?" (no consensus). |
| `quorum-3` | Three jurors, 2-of-3 must approve (homogeneous Byzantine quorum). |
| `heterogeneous-panel` | A rule-based + semantic + LLM panel, 2-of-3 (the flagship). |

## Honesty note (read this before quoting a number)

This report was generated in **{mode}** mode.

- In **torch-free** mode every juror reduces to a red-flag-plus-overlap
  heuristic, so the panel members share a keyword basis. That is why the
  torch-free configs **cannot beat the keyword baseline on the
  `benign_lookalike` false positives** - legitimate text that merely *mentions*
  a scary phrase (e.g. a runbook describing when to `drop table` partitions). It
  is also why the relational checks still **win on recall**: paraphrased attacks
  that avoid the denylist vocabulary do not follow from their premise, so the
  relational judge slashes what the keyword filter waves through.
- Closing the false-positive gap on look-alikes requires a juror that
  understands context, i.e. the opt-in `--nli` (cross-encoder) and `--llm` (real
  model) jurors. Those numbers depend on your model/provider and are therefore
  not committed here as the reproducible headline.
- External baselines (Guardrails AI, NeMo Guardrails) and the public attack
  suites (AgentDojo, InjecAgent) are planned as additional `Defense` adapters in
  a later phase; they slot into the same harness and metrics.
"""


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    items = load_dataset(args.dataset)
    n_attacks = sum(1 for it in items if it.is_attack)

    llm_complete = _resolve_llm_complete(args)
    mode_bits = ["torch-free" if not args.nli else "real-NLI"]
    if args.llm:
        mode_bits.append("real-LLM")
    mode = " + ".join(mode_bits)

    defenses = build_defenses(use_nli=args.nli, llm_complete=llm_complete)

    all_metrics: list[Metrics] = []
    per_item_records: dict[str, list[dict]] = {}
    for defense in defenses:
        metrics, records = evaluate_defense(defense, items)
        all_metrics.append(metrics)
        per_item_records[defense.name] = records

    payload = {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "mode": mode,
        "dataset": str(args.dataset.name),
        "n_items": len(items),
        "n_attacks": n_attacks,
        "n_benign": len(items) - n_attacks,
        "configs": [m.to_dict() for m in all_metrics],
        "per_item": per_item_records,
    }
    RESULTS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    RESULTS_MD.write_text(
        _build_markdown(
            all_metrics, n_items=len(items), n_attacks=n_attacks, mode=mode
        ),
        encoding="utf-8",
    )

    if not args.quiet:
        print(f"\nInjection-capture benchmark  |  mode: {mode}")
        print(f"corpus: {len(items)} items ({n_attacks} attacks, "
              f"{len(items) - n_attacks} benign)\n")
        print(_format_console_table(all_metrics))
        print(f"\nWrote {RESULTS_JSON.name} and {RESULTS_MD.name} to {HERE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
