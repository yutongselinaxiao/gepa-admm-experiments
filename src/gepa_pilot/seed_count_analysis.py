"""How results change with the number of seeds (3, 5, 10), on both report measurements.

For every 10-seed arm, the first n seeds (n = 3, 5, 10; the order in which seeds were actually
run) give mean, standard deviation and a 95% t-interval, measured on (i) the 256-example report
half and (ii) the 2,000-example report sets (report_set.py). Key comparisons are recomputed at
each n with Welch's t-test.

Writes artifacts/seed_count/seed_count.json and artifacts/seed_count/tables.md.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from scipy import stats

ARMS = {
    "agnews": ["reflective_p12", "reflective_heldout_p30", "reflective_p40", "admm_topp",
               "admm_beam_lam0.3", "admm_beam_lam0.3_indep_accept"],
    "emotion": ["reflective_p12", "reflective_heldout_p30", "admm_topp",
                "admm_beam_lam0.3", "admm_beam_lam0.3_indep_accept"],
}
LABELS = {
    "reflective_p12": "Reflective GEPA (12 proposals)",
    "reflective_heldout_p30": "Matched reflective (30 proposals, held-out)",
    "reflective_p40": "Reflective GEPA (40 proposals)",
    "admm_topp": "GEPA + top-p ADMM proposer",
    "admm_beam_lam0.3": "GEPA + beam ADMM proposer (S used twice)",
    "admm_beam_lam0.3_indep_accept": "GEPA + beam ADMM proposer (fair S')",
}
COMPARISONS = [
    ("admm_beam_lam0.3_indep_accept", "reflective_heldout_p30"),
    ("admm_beam_lam0.3_indep_accept", "reflective_p12"),
    ("admm_beam_lam0.3", "reflective_p12"),
]
SEED_COUNTS = (3, 5, 10)


def interval(values: list[float]) -> tuple[float, float, float]:
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half = stats.t.ppf(0.975, len(values) - 1) * sd / math.sqrt(len(values))
    return mean, sd, half


def main() -> None:
    summary = json.loads(Path("artifacts/classification_results/summary.json").read_text(encoding="utf-8"))
    report2k = {t: json.loads(Path(f"artifacts/report2k/{t}.json").read_text(encoding="utf-8"))
                for t in ARMS}
    scores: dict = {}
    for task, arms in ARMS.items():
        for arm in arms:
            runs = summary["arms"][f"{task}/{arm}"]["runs"]
            scores[(task, arm, "256")] = [r["report_accuracy"] for r in runs]
            scores[(task, arm, "2k")] = [100 * report2k[task]["by_run"][r["run"]] for r in runs]
    out: dict = {"arms": {}, "comparisons": {}}
    lines = ["# Seed-count comparison", ""]
    for task, arms in ARMS.items():
        name = "AG News" if task == "agnews" else "Emotion"
        seed_256 = summary["arms"][f"{task}/reflective_p12"]["seed_instruction_report_accuracy"]
        seed_2k = 100 * report2k[task]["seed_accuracy"]
        lines += [f"## {name}", "",
                  f"Seed instruction: {seed_256:.2f}% on the 256-example half, {seed_2k:.2f}% on the "
                  f"{report2k[task]['report_size']:,}-example set.", "",
                  "Mean ± sd [95% CI half-width] using the first n seeds:", "",
                  "| Arm | Measure | n = 3 | n = 5 | n = 10 |", "|---|---|---|---|---|"]
        for arm in arms:
            for measure in ("256", "2k"):
                cells = []
                for n in SEED_COUNTS:
                    mean, sd, half = interval(scores[(task, arm, measure)][:n])
                    out["arms"].setdefault(f"{task}/{arm}/{measure}", {})[n] = {"mean": mean, "sd": sd, "ci_half": half}
                    cells.append(f"{mean:.2f} ± {sd:.2f} [±{half:.2f}]")
                label = LABELS[arm] if measure == "256" else ""
                lines.append(f"| {label} | {'256 ex.' if measure == '256' else '2k ex.'} | " + " | ".join(cells) + " |")
        lines += ["", "Differences (Welch p) using the first n seeds:", "",
                  "| Comparison | Measure | n = 3 | n = 5 | n = 10 |", "|---|---|---|---|---|"]
        for a, b in COMPARISONS:
            for measure in ("256", "2k"):
                cells = []
                for n in SEED_COUNTS:
                    x, y = scores[(task, a, measure)][:n], scores[(task, b, measure)][:n]
                    diff = statistics.mean(x) - statistics.mean(y)
                    p = float(stats.ttest_ind(x, y, equal_var=False).pvalue)
                    out["comparisons"].setdefault(f"{task}/{a} - {b}/{measure}", {})[n] = {"diff": diff, "p": p}
                    cells.append(f"{diff:+.2f} (p = {p:.3f})")
                label = f"{LABELS[a]} − {LABELS[b]}" if measure == "256" else ""
                lines.append(f"| {label} | {'256 ex.' if measure == '256' else '2k ex.'} | " + " | ".join(cells) + " |")
        lines.append("")
    path = Path("artifacts/seed_count")
    path.mkdir(parents=True, exist_ok=True)
    (path / "seed_count.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (path / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
