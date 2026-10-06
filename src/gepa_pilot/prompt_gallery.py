"""Write artifacts/prompt_examples/README.md: seed vs. improved prompts for every experiment,
and the edits a perplexity filter would keep or drop.

Inputs: artifacts/fluency_analysis/{agnews,emotion}.json (from fluency_analysis.py) and the
standalone-ADMM pilot result. Every number comes from those files.
"""

from __future__ import annotations

import difflib
import html
import json
import statistics
from pathlib import Path

ARMS = {
    "agnews": [
        ("E2", "GEPA + Euclidean ADMM proposer, minibatch acceptance (ρ = 0.3)", "agnews/p1_admm_euclid_rho0.3_minibatch"),
        ("E4", "GEPA + Euclidean ADMM proposer, held-out acceptance (ρ = 0.3)", "agnews/p1_admm_euclid_rho0.3_heldout"),
        ("E7", "Reflective GEPA (12 proposals)", "agnews/reflective_p12"),
        ("E7", "Reflect-then-polish (greedy top-p polish)", "agnews/reflect_polish"),
        ("E7", "GEPA + top-p ADMM proposer", "agnews/admm_topp"),
        ("E8", "GEPA + beam ADMM proposer, λ = 0.3", "agnews/admm_beam_lam0.3"),
        ("E8", "GEPA + beam ADMM proposer, λ = 1.0", "agnews/admm_beam_lam1.0"),
        ("E9", "Reflective GEPA, 40 proposals", "agnews/reflective_p40"),
        ("E9", "Reflective GEPA, held-out acceptance, 30 proposals (matched baseline)", "agnews/reflective_heldout_p30"),
        ("E11", "GEPA + beam ADMM proposer + per-class duals", "agnews/admm_beam_lam0.3_duals"),
        ("E12", "GEPA + beam ADMM proposer, independent acceptance (fair)", "agnews/admm_beam_lam0.3_indep_accept"),
    ],
    "emotion": [
        ("E10", "Reflective GEPA (12 proposals)", "emotion/reflective_p12"),
        ("E10", "Reflective GEPA, held-out acceptance, 30 proposals (matched baseline)", "emotion/reflective_heldout_p30"),
        ("E10", "GEPA + top-p ADMM proposer", "emotion/admm_topp"),
        ("E10", "GEPA + beam ADMM proposer, λ = 0.3", "emotion/admm_beam_lam0.3"),
        ("E11", "GEPA + beam ADMM proposer + per-class duals", "emotion/admm_beam_lam0.3_duals"),
        ("E12", "GEPA + beam ADMM proposer, independent acceptance (fair)", "emotion/admm_beam_lam0.3_indep_accept"),
    ],
}
FILTER_NATS = 12.0


def _words(text: str) -> list[str]:
    return text.replace("\n", " ⏎ ").split()


def _escape(text: str) -> str:
    return html.escape(text, quote=False).replace("|", "\\|")


def highlight(seed: str, text: str) -> str:
    """Prompt text with runs of words that differ from the seed in bold (word-level diff).

    Newlines are shown as ⏎. Deleted seed words are not shown; the seed is printed above.
    """
    a, b = _words(seed), _words(text)
    out: list[str] = []
    for op, _, _, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if j1 == j2:
            continue
        span = _escape(" ".join(b[j1:j2]))
        out.append(span if op == "equal" else f"**{span}**")
    return " ".join(out) if out else "*(empty)*"


def show_changes(changes: list[list[str]]) -> str:
    return ", ".join(f"`{p!r}` → `{c!r}`" for p, c in changes)


def task_section(task: str, data: dict) -> list[str]:
    seed = data["seed"]
    name = "AG News" if task == "agnews" else "Emotion"
    lines = [f"## {name}", "",
             f"**Seed instruction** (report-half accuracy {SEED_ACCURACY[task]:.2f}%, "
             f"per-token perplexity {seed['per_token_perplexity']:.0f}):", "",
             f"> {seed['text']}", ""]
    if task == "agnews":
        pilot = json.loads(Path("artifacts/admm_rho1/admm_result.json").read_text(encoding="utf-8"))
        best = pilot["best_checkpoint"]
        text = best["roundtrip"]["reconstructed_instruction"].strip()
        lines += ["### Pilot (Stage III): standalone ADMM, ρ = 1 (outside GEPA, one run)", "",
                  f"Best checkpoint: iteration {best['iteration']}, full-validation accuracy "
                  f"{100 * best['discrete_validation']['accuracy']:.2f}% (pilot protocol, not report half).", "",
                  f"> {highlight(seed['text'], text)}", ""]
    for experiment, title, arm in ARMS[task]:
        rows = sorted(data["arms"][arm], key=lambda r: r["report_accuracy"])
        median = rows[(len(rows) - 1) // 2]
        best = rows[-1]
        mean = statistics.mean(r["report_accuracy"] for r in rows)
        on_seed = sum(r["text"] == seed["text"] for r in rows)
        lines += [f"### {experiment} — {title}", "",
                  f"{len(rows)} runs, mean report-half accuracy {mean:.2f}%; "
                  f"{on_seed} run(s) ended on the seed instruction unchanged.", "",
                  "| Run | Report-half acc. | Perplexity / token | Selected prompt (changes vs. seed in bold) |",
                  "|---|---:|---:|---|"]
        for label, row in (("best", best), ("median", median)):
            lines.append(f"| {label} | {row['report_accuracy']:.2f} | {row['per_token_perplexity']:.0f} | "
                         f"{highlight(seed['text'], row['text'])} |")
        lines.append("")
    return lines


def filter_section(data: dict, task: str) -> list[str]:
    name = "AG News" if task == "agnews" else "Emotion"
    lines = [f"### {name}: the most helpful edits per projection", "",
             f"Validation gain is the full-validation accuracy change from parent to child. "
             f"A filter at {FILTER_NATS:.0f} nats keeps an edit only if every inserted token has NLL ≤ "
             f"{FILTER_NATS:.0f} in context.", ""]
    for projection, edits in data["proposals"].items():
        scored = [e for e in edits if e["validation_gain_pp"] is not None]
        if not scored:
            continue
        helpful = [e for e in scored if e["validation_gain_pp"] > 0]
        kept = [e for e in helpful if e["edit_nll"] <= FILTER_NATS]
        lines += [f"**{projection}**: {len(helpful)} of {len(scored)} edits helped; a {FILTER_NATS:.0f}-nat "
                  f"filter would keep {len(kept)} of those {len(helpful)}.", "",
                  "| Gain (pp) | Edit NLL (nats) | Filter | Token changes | Child prompt |",
                  "|---:|---:|---|---|---|"]
        for e in sorted(scored, key=lambda e: -e["validation_gain_pp"])[:5]:
            verdict = "kept" if e["edit_nll"] <= FILTER_NATS else "**dropped**"
            child = _escape(e["child_text"].replace("\n", " ⏎ "))
            lines.append(f"| {e['validation_gain_pp']:+.2f} | {e['edit_nll']:.1f} | {verdict} | "
                         f"{show_changes(e['changes'])} | {child} |")
        lines.append("")
    return lines


SEED_ACCURACY: dict[str, float] = {}


def main() -> None:
    summary = json.loads(Path("artifacts/classification_results/summary.json").read_text(encoding="utf-8"))
    SEED_ACCURACY["agnews"] = summary["arms"]["agnews/reflective_p12"]["seed_instruction_report_accuracy"]
    SEED_ACCURACY["emotion"] = summary["arms"]["emotion/reflective_p12"]["seed_instruction_report_accuracy"]
    data = {task: json.loads(Path(f"artifacts/fluency_analysis/{task}.json").read_text(encoding="utf-8"))
            for task in ("agnews", "emotion")}
    lines = [
        "# Prompt gallery: seed vs. improved prompts, and what a perplexity filter does",
        "",
        "Generated by [`src/gepa_pilot/prompt_gallery.py`](../../src/gepa_pilot/prompt_gallery.py) from "
        "[`artifacts/fluency_analysis/`](../fluency_analysis/); see "
        "[`CLASSIFICATION_RESULTS.md`](../../CLASSIFICATION_RESULTS.md) for the experiments (E-numbers).",
        "",
        "- **Best** is the run with the highest report-half accuracy; **median** is the middle run. "
        "Words that differ from the seed are in **bold** (word-level diff; deleted seed words are not shown, "
        "and ⏎ marks a newline).",
        "- **Perplexity / token** = exp(mean NLL of the instruction tokens) under the frozen model, given the "
        "fixed prompt prefix. The label mapping comes *after* the instruction, so task words such as "
        "\"capital letter\" are surprising; compare against the seed rather than reading it as absolute fluency.",
        "- Rows titled \"ADMM proposer\" are GEPA with ADMM replacing reflection; \"standalone ADMM\" is the "
        "pilot only.",
        "",
    ]
    for task in ("agnews", "emotion"):
        lines += task_section(task, data[task])
    lines += ["## What a perplexity filter does", "",
              "The three projections are three strengths of fluency constraint: **Euclidean** = none, "
              "**top-p** = a hard per-token constraint (only tokens in the LM's 99% nucleus), **beam λ** = a "
              "soft NLL penalty on the whole slot. A *post-hoc* perplexity filter instead lets the search run "
              "unconstrained and discards disfluent results afterwards. The tables show why that fails for "
              "Euclidean ADMM: its helpful edits are the disfluent ones, so the filter throws away the gains. "
              "The constrained projections find gains among fluent edits instead.",
              "",
              "![Filter keep rate](../figures/fig3_filter_keep_rate.png)",
              "",
              "![Edit fluency vs gain](../figures/fig5_fluency_vs_gain.png)",
              ""]
    for task in ("agnews", "emotion"):
        lines += filter_section(data[task], task)
    out = Path("artifacts/prompt_examples/README.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"WROTE {out}")


if __name__ == "__main__":
    main()
