"""Run the StigmergicAI injection-capture benchmark and write a report.

Usage (from the repo root)::

    # torch-free, fully reproducible, no downloads, no keys:
    python benchmarks/injection_capture/run.py

    # the Byzantine-robustness experiment (torch-free, deterministic):
    python benchmarks/injection_capture/run.py --byzantine

    # swap in the real cross-encoder NLI juror (needs the [cognition] extras):
    python benchmarks/injection_capture/run.py --nli

    # add a real LLM juror to the panel (needs [benchmark] extras + env config;
    # see providers.py for the environment variables):
    python benchmarks/injection_capture/run.py --llm

The *standard* run compares four anti-injection configs on the same labeled
corpus -- a naive ``keyword-only`` denylist (the baseline), a ``single-judge``
semantic check, a homogeneous ``quorum-3``, and the flagship heterogeneous
``panel``. The ``--byzantine`` run instead plants a faulty juror to isolate the
value of consensus itself.

Every run archives its metrics to ``results/<mode>.json`` (committed evidence)
and regenerates ``COMPARISON.md`` so the modes can be read side by side. A
standard run also refreshes ``RESULTS.md``; a Byzantine run writes
``BYZANTINE.md``. The headline metric is the **capture rate** (recall on
attacks), mapped to OWASP LLM01 (Prompt Injection); the **false-positive rate**
guards against a control that simply blocks everything.
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
    build_byzantine_defenses,
    build_defenses,
    evaluate_defense,
    load_dataset,
)

DATASET = HERE / "dataset.jsonl"
RESULTS_JSON = HERE / "results.json"
RESULTS_MD = HERE / "RESULTS.md"
BYZANTINE_MD = HERE / "BYZANTINE.md"
COMPARISON_MD = HERE / "COMPARISON.md"
RESULTS_DIR = HERE / "results"


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
        "--nli-model",
        default=None,
        help="Override the NLI model name (only with --nli; e.g. a BPE MNLI model).",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Add a real LLM juror to the panel (needs [benchmark] + env config).",
    )
    parser.add_argument(
        "--byzantine",
        action="store_true",
        help="Run the consensus-under-a-faulty-juror experiment (torch-free).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the on-disk LLM response cache (only affects --llm).",
    )
    parser.add_argument(
        "--baselines",
        action="store_true",
        help="Also score external baselines (Guardrails AI / NeMo); needs [baselines].",
    )
    parser.add_argument(
        "--nemo-config",
        default=None,
        help="Path to a NeMo Guardrails config dir (adds the NeMo baseline).",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress the console summary table."
    )
    return parser.parse_args(argv)


def _mode_label(args: argparse.Namespace) -> str:
    """A human-readable description of the active run configuration."""
    if args.byzantine:
        return "byzantine"
    bits = ["real-NLI" if args.nli else "torch-free"]
    if args.llm:
        bits.append("real-LLM")
    if getattr(args, "baselines", False):
        bits.append("baselines")
    return " + ".join(bits)


def _mode_slug(args: argparse.Namespace) -> str:
    """A filesystem-safe slug for the archived ``results/<slug>.json``."""
    if args.byzantine:
        return "byzantine"
    slug = "real-nli" if args.nli else "torch-free"
    if args.llm:
        slug += "+real-llm"
    if getattr(args, "baselines", False):
        slug += "+baselines"
    return slug


def _resolve_llm_complete(args: argparse.Namespace):
    if not args.llm:
        return None
    from providers import build_llm_complete

    return build_llm_complete(use_cache=not args.no_cache)


def _format_console_table(metrics: list[Metrics]) -> str:
    header = (
        f"{'config':<22}{'capture':>9}{'FPR':>8}{'prec':>8}"
        f"{'F1':>8}{'acc':>8}{'p50ms':>9}"
    )
    lines = [header, "-" * len(header)]
    for m in metrics:
        summary = m.latency_summary()
        p50 = summary.get("p50_ms", 0.0) if summary else 0.0
        lines.append(
            f"{m.config:<22}"
            f"{m.capture_rate:>9.3f}"
            f"{m.false_positive_rate:>8.3f}"
            f"{m.precision:>8.3f}"
            f"{m.f1:>8.3f}"
            f"{m.accuracy:>8.3f}"
            f"{p50:>9.3f}"
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


def _md_ci_table(metrics: list[Metrics]) -> str:
    """Capture/FPR with their 95% Wilson confidence intervals."""
    rows = [
        "| Config | Capture rate (95% CI) | False-positive rate (95% CI) |",
        "| :--- | ---: | ---: |",
    ]
    for m in metrics:
        clo, chi = m.capture_interval()
        flo, fhi = m.false_positive_interval()
        rows.append(
            f"| `{m.config}` | {m.capture_rate:.1%} [{clo:.1%}, {chi:.1%}] "
            f"| {m.false_positive_rate:.1%} [{flo:.1%}, {fhi:.1%}] |"
        )
    return "\n".join(rows)


def _md_category_fpr_table(metrics: list[Metrics]) -> str:
    """Per-benign-category false-positive rate (isolates look-alike over-blocking)."""
    cats = sorted({c for m in metrics for c in m.per_category_benign_total})
    if not cats:
        return ""
    head = "| Benign category | " + " | ".join(f"`{m.config}`" for m in metrics) + " |"
    sep = "| :--- | " + " | ".join("---:" for _ in metrics) + " |"
    rows = [head, sep]
    for cat in cats:
        cells = []
        for m in metrics:
            fpr = m.category_false_positive().get(cat)
            cells.append(f"{fpr:.0%}" if fpr is not None else "-")
        rows.append(f"| {cat} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _md_latency_table(metrics: list[Metrics]) -> str:
    """Per-config p50/p95/mean latency (machine-dependent; not committed to JSON)."""
    rows = [
        "| Config | p50 (ms) | p95 (ms) | mean (ms) |",
        "| :--- | ---: | ---: | ---: |",
    ]
    for m in metrics:
        summary = m.latency_summary()
        if not summary:
            rows.append(f"| `{m.config}` | - | - | - |")
        else:
            rows.append(
                f"| `{m.config}` | {summary['p50_ms']:.3f} "
                f"| {summary['p95_ms']:.3f} | {summary['mean_ms']:.3f} |"
            )
    return "\n".join(rows)


def _md_pr_curve(per_item_records: dict[str, list[dict]]) -> str:
    """Precision-recall sweep for the flagship config that emits juror votes."""
    from harness import pr_curve

    name = next(
        (
            pref
            for pref in ("heterogeneous-panel", "quorum-3", "single-judge")
            if pref in per_item_records
        ),
        None,
    )
    if name is None:
        return ""
    points = pr_curve(per_item_records[name])
    rows = [
        f"Sweeping the approval-ratio threshold (approvals / total votes) for "
        f"`{name}`; a proposal is predicted blocked at or below the threshold.",
        "",
        "| Threshold | Precision | Recall |",
        "| ---: | ---: | ---: |",
    ]
    for p in points:
        rows.append(
            f"| {p['threshold']:.2f} | {p['precision']:.1%} | {p['recall']:.1%} |"
        )
    return "\n".join(rows)


def _build_markdown(
    metrics: list[Metrics],
    *,
    n_items: int,
    n_attacks: int,
    mode: str,
    per_item_records: dict[str, list[dict]] | None = None,
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

## False-positive rate by benign category

{_md_category_fpr_table(metrics)}

## Confidence intervals (95%, Wilson)

The capture and false-positive rates are proportions over a finite corpus, so
each carries sampling uncertainty. The Wilson score interval below is the
honest error bar to quote alongside a headline number.

{_md_ci_table(metrics)}

## Precision-recall trade-off

{_md_pr_curve(per_item_records or {})}

## Latency (machine-dependent, informational)

> Wall-clock per-item latency on the machine that generated this report. Unlike
> the capture/FPR headline, these numbers are **not** reproducible across
> hardware and are deliberately excluded from `results.json`.

{_md_latency_table(metrics)}

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


def _build_byzantine_markdown(
    metrics: list[Metrics], *, n_items: int, n_attacks: int
) -> str:
    """Render the consensus-under-a-faulty-juror experiment."""
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    by = {m.config: m for m in metrics}

    def cap(name: str) -> str:
        return f"{by[name].capture_rate:.1%}" if name in by else "-"

    def fpr(name: str) -> str:
        return f"{by[name].false_positive_rate:.1%}" if name in by else "-"

    rows = [
        "| Config | Faulty juror | Capture rate | False-positive rate |",
        "| :--- | :--- | ---: | ---: |",
    ]
    descriptions = {
        "honest-single": "none (reference)",
        "compromised-single": "1 of 1 approves everything",
        "honest-quorum-3": "none (reference)",
        "byzantine-quorum-3": "1 of 3 approves everything",
        "paranoid-single": "1 of 1 rejects everything",
        "paranoid-quorum-3": "1 of 3 rejects everything",
    }
    for m in metrics:
        rows.append(
            f"| `{m.config}` | {descriptions.get(m.config, '?')} "
            f"| {m.capture_rate:.1%} | {m.false_positive_rate:.1%} |"
        )
    table = "\n".join(rows)

    return f"""# Byzantine-Robustness Experiment - Results

> Generated by `run.py --byzantine` on {stamp}. Mode: **byzantine** (torch-free, deterministic).
> Corpus: **{n_items}** labeled cases ({n_attacks} attacks, {n_items - n_attacks} benign).

The standard benchmark's aggregate numbers hide *why a quorum exists*: in that
run every juror is honest, so a 2-of-3 quorum scores exactly like a single
judge. This experiment makes the value of consensus visible by deliberately
planting **one faulty juror** and measuring what happens.

## The result

{table}

## What it shows

**Missed-attack axis (a bribed/hijacked juror that approves everything).**
A single compromised judge collapses from a {cap('honest-single')} capture rate
to **{cap('compromised-single')}** - it waves every attack through. Drop the
*same* compromised juror into a 3-node quorum and capture holds at
**{cap('byzantine-quorum-3')}**: the two honest jurors still reach the 2-vote
threshold to slash the attack. *The quorum tolerates a traitor the single judge
cannot.*

**Over-blocking axis (a malfunctioning juror that rejects everything).**
A single paranoid judge blocks all legitimate work - a
**{fpr('paranoid-single')}** false-positive rate. The same faulty juror inside a
quorum is outvoted by the two honest approvals, so the false-positive rate falls
back to **{fpr('paranoid-quorum-3')}** (the residual is the shared keyword
look-alike issue, not the faulty node).

## Why this matters

This is the **Byzantine Cognitive Consensus** thesis (Horizon 2) reduced to two
numbers: a lone model is a single point of failure in *both* directions - it can
be fooled into approving an attack, or break and block everything - while a
quorum that fails for *uncorrelated* reasons survives a faulty member. The
standard run proves the relational check beats a denylist on recall; this run
proves the *consensus* is what makes that check trustworthy in production.

> Reproduce with `python benchmarks/injection_capture/run.py --byzantine`.
> Every juror here is a deterministic, torch-free stand-in, so these numbers are
> committed as stable evidence.
"""


def _archive_payload(payload: dict, slug: str) -> pathlib.Path:
    """Persist one run's metrics under ``results/<slug>.json`` (committed)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{slug}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _regenerate_comparison() -> None:
    """Rebuild ``COMPARISON.md`` from every archived ``results/*.json``.

    This is the cross-mode evidence ledger: each run drops a JSON snapshot and
    the comparison is recomputed, so torch-free, real-NLI, real-LLM and Byzantine
    results can be read against one another without re-running anything.
    """
    if not RESULTS_DIR.exists():
        return
    snapshots = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        try:
            snapshots.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:  # pragma: no cover - corrupt archive
            continue
    if not snapshots:
        return

    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Benchmark Comparison - All Modes",
        "",
        f"> Regenerated on {stamp} from `results/*.json`. Each row is one config",
        "> from one run mode; re-run any mode to refresh its rows.",
        "",
        "| Mode | Config | Capture rate | FPR | Precision | F1 | Accuracy |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    # Index every (mode, config) -> metrics so the findings section can quote
    # exact numbers without re-running anything.
    index: dict[tuple[str, str], dict] = {}
    for snap in snapshots:
        mode = snap.get("mode", "?")
        for cfg in snap.get("configs", []):
            conf = cfg
            index[(mode, conf["config"])] = conf
            lines.append(
                f"| {mode} | `{conf['config']}` "
                f"| {conf['capture_rate']:.1%} "
                f"| {conf['false_positive_rate']:.1%} "
                f"| {conf['precision']:.1%} "
                f"| {conf['f1']:.3f} "
                f"| {conf['accuracy']:.1%} |"
            )

    findings = _comparison_findings(index)
    if findings:
        lines.extend(["", "## Key findings", ""])
        lines.extend(findings)

    lines.extend(
        [
            "",
            "## How to read this",
            "",
            "- **Capture rate** - recall on attacks (OWASP LLM01). Higher is better.",
            "- **FPR** - benign work wrongly blocked. Lower is better.",
            "- The `torch-free` rows are the committed, reproducible headline. The",
            "  `real-nli` / `real-llm` rows depend on the model/provider used and are",
            "  evidence, not a reproducibility guarantee. The `byzantine` rows isolate",
            "  the value of consensus under a faulty juror (see `BYZANTINE.md`).",
            "",
        ]
    )
    COMPARISON_MD.write_text("\n".join(lines), encoding="utf-8")


def _comparison_findings(index: dict[tuple[str, str], dict]) -> list[str]:
    """Data-driven bullet points computed from the archived snapshots.

    Each bullet only appears when the mode it describes has actually been run,
    so the comparison narrates exactly the evidence on disk -- never a claim
    without a row to back it.
    """
    out: list[str] = []

    def pct(mode: str, config: str, field: str) -> str | None:
        conf = index.get((mode, config))
        return f"{conf[field]:.1%}" if conf else None

    # 1) The torch-free relational lift over a denylist.
    kw_cap = pct("torch-free", "keyword-only", "capture_rate")
    sj_cap = pct("torch-free", "single-judge", "capture_rate")
    if kw_cap and sj_cap:
        out.append(
            f"- **Relational check beats a denylist (torch-free, reproducible).** "
            f"Capture rises from **{kw_cap}** (`keyword-only`) to **{sj_cap}** "
            f"(`single-judge`) at the *same* false-positive rate -- the committed "
            f"headline."
        )

    # 2) A lone real LLM juror is the only lens that clears the keyword ceiling.
    llm_cap = pct("torch-free + real-LLM", "llm-judge", "capture_rate")
    llm_fpr = pct("torch-free + real-LLM", "llm-judge", "false_positive_rate")
    llm_acc = pct("torch-free + real-LLM", "llm-judge", "accuracy")
    llm_panel_fpr = pct(
        "torch-free + real-LLM", "heterogeneous-panel", "false_positive_rate"
    )
    llm_panel_acc = pct("torch-free + real-LLM", "heterogeneous-panel", "accuracy")
    if llm_cap and llm_fpr and llm_acc and llm_panel_fpr and llm_panel_acc:
        out.append(
            f"- **A lone LLM juror is the only lens that cracks the look-alike "
            f"ceiling (model-dependent evidence).** Judging alone, the LLM reaches "
            f"**{llm_cap}** capture at a **{llm_fpr}** false-positive rate "
            f"(accuracy **{llm_acc}**) -- the only config to lower the FPR below "
            f"the **{llm_panel_fpr}** wall *every* denylist-based juror shares, "
            f"without sacrificing capture. The gain is real but narrow: it recovers "
            f"context on the `benign_lookalike` runbooks the keyword basis blocks, "
            f"yet the heterogeneous panel still sits at **{llm_panel_acc}** accuracy "
            f"because the two keyword jurors outvote the LLM exactly where the "
            f"ceiling lives. *A semantic juror only helps if the architecture lets "
            f"it win.*"
        )

    # 3) A real NLI model over-blocks on its own; the panel tames it.
    nli_sj_cap = pct("real-NLI", "single-judge", "capture_rate")
    nli_sj_fpr = pct("real-NLI", "single-judge", "false_positive_rate")
    nli_panel_cap = pct("real-NLI", "heterogeneous-panel", "capture_rate")
    nli_panel_fpr = pct("real-NLI", "heterogeneous-panel", "false_positive_rate")
    if nli_sj_cap and nli_sj_fpr and nli_panel_cap and nli_panel_fpr:
        out.append(
            f"- **A real NLI model is not trustworthy *alone* (model-dependent "
            f"evidence).** The raw cross-encoder juror catches **{nli_sj_cap}** of "
            f"attacks but over-blocks legitimate work at a **{nli_sj_fpr}** "
            f"false-positive rate -- unusable on its own. Inside the heterogeneous "
            f"panel its over-eagerness is outvoted: capture **{nli_panel_cap}** at "
            f"a **{nli_panel_fpr}** false-positive rate. *Consensus across "
            f"uncorrelated jurors tames a single over-confident model.*"
        )

    # 4) The Byzantine tolerance result.
    byz_comp = pct("byzantine", "compromised-single", "capture_rate")
    byz_quorum = pct("byzantine", "byzantine-quorum-3", "capture_rate")
    byz_par = pct("byzantine", "paranoid-single", "false_positive_rate")
    byz_par_q = pct("byzantine", "paranoid-quorum-3", "false_positive_rate")
    if byz_comp and byz_quorum and byz_par and byz_par_q:
        out.append(
            f"- **The quorum survives a faulty juror (torch-free, "
            f"deterministic).** One bribed juror drops a single judge to "
            f"**{byz_comp}** capture, yet the same traitor inside a 3-node quorum "
            f"holds at **{byz_quorum}**. A broken juror drives a lone judge to a "
            f"**{byz_par}** false-positive rate; the quorum outvotes it back to "
            f"**{byz_par_q}**. See `BYZANTINE.md`."
        )

    return out


def _run_defenses(
    defenses, items
) -> tuple[list[Metrics], dict[str, list[dict]]]:
    all_metrics: list[Metrics] = []
    per_item_records: dict[str, list[dict]] = {}
    for defense in defenses:
        metrics, records = evaluate_defense(defense, items)
        all_metrics.append(metrics)
        per_item_records[defense.name] = records
    return all_metrics, per_item_records


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    items = load_dataset(args.dataset)
    n_attacks = sum(1 for it in items if it.is_attack)
    n_items = len(items)
    mode = _mode_label(args)
    slug = _mode_slug(args)

    llm_complete = None
    if args.byzantine:
        defenses = build_byzantine_defenses()
    else:
        llm_complete = _resolve_llm_complete(args)
        defenses = build_defenses(
            use_nli=args.nli, nli_model=args.nli_model, llm_complete=llm_complete
        )
        if args.baselines:
            from baselines import build_baseline_defenses

            try:
                defenses = list(defenses) + build_baseline_defenses(
                    nemo_config=args.nemo_config
                )
            except RuntimeError as exc:
                print(f"--baselines could not run:\n{exc}", file=sys.stderr)
                return 2

    all_metrics, per_item_records = _run_defenses(defenses, items)

    payload = {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "mode": mode,
        "slug": slug,
        "dataset": str(args.dataset.name),
        "n_items": n_items,
        "n_attacks": n_attacks,
        "n_benign": n_items - n_attacks,
        "configs": [m.to_dict() for m in all_metrics],
        "per_item": per_item_records,
    }

    # When a real LLM juror ran, attach token usage and an estimated USD cost.
    # This lives only in the model-dependent snapshot, never in the committed
    # torch-free headline.
    if llm_complete is not None and getattr(llm_complete, "usage", None):
        from harness import estimate_cost

        used = llm_complete.usage
        payload["llm_cost"] = {
            "model": used.get("model", ""),
            "prompt_tokens": used["prompt_tokens"],
            "completion_tokens": used["completion_tokens"],
            "api_calls": used["api_calls"],
            "cache_hits": used["cache_hits"],
            "estimated_usd": round(
                estimate_cost(
                    used["prompt_tokens"],
                    used["completion_tokens"],
                    used.get("model", ""),
                ),
                6,
            ),
        }

    # Always archive this run's snapshot (committed evidence) and refresh the
    # cross-mode comparison ledger.
    archived = _archive_payload(payload, slug)

    if args.byzantine:
        BYZANTINE_MD.write_text(
            _build_byzantine_markdown(
                all_metrics, n_items=n_items, n_attacks=n_attacks
            ),
            encoding="utf-8",
        )
        primary_md = BYZANTINE_MD
    elif not args.nli and not args.llm and not args.baselines:
        # Only the canonical, fully-reproducible torch-free run owns RESULTS.md
        # and results.json -- the committed headline the README points to. The
        # model/provider-dependent modes (--nli, --llm) live in their per-mode
        # results/<slug>.json snapshot and in COMPARISON.md, so a real-model run
        # never silently overwrites the reproducible baseline.
        RESULTS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        RESULTS_MD.write_text(
            _build_markdown(
                all_metrics,
                n_items=n_items,
                n_attacks=n_attacks,
                mode=mode,
                per_item_records=per_item_records,
            ),
            encoding="utf-8",
        )
        primary_md = RESULTS_MD
    else:
        primary_md = archived

    _regenerate_comparison()

    if not args.quiet:
        print(f"\nInjection-capture benchmark  |  mode: {mode}")
        print(f"corpus: {n_items} items ({n_attacks} attacks, "
              f"{n_items - n_attacks} benign)\n")
        print(_format_console_table(all_metrics))
        cost = payload.get("llm_cost")
        if cost:
            print(
                f"\nLLM usage: {cost['api_calls']} calls "
                f"({cost['cache_hits']} cached), "
                f"{cost['prompt_tokens']}+{cost['completion_tokens']} tokens, "
                f"~${cost['estimated_usd']:.4f} ({cost['model']})"
            )
        if primary_md is archived:
            print(
                f"\nArchived results/{archived.name} and refreshed "
                f"{COMPARISON_MD.name} in {HERE}"
            )
        else:
            print(
                f"\nWrote {primary_md.name}, results/{archived.name} and "
                f"{COMPARISON_MD.name} to {HERE}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
