"""Collect every classification arm into one table (report-half accuracy, budgets, tests).

Arms run with --selection-split report the GEPA-selected candidate's accuracy on the
256-example report half. Phase 1 arms (run before the split existed) are re-selected on the
selection half and scored on the report half, as in rescore_split.

Writes artifacts/classification_results/summary.json and prints a compact table.
"""

from __future__ import annotations

import json
import random
import statistics
from pathlib import Path
from typing import Any

from .data import load_examples_json, split_selection_report
from .rescore_split import rescore_run

SPLIT_SEED = 20261002

# name -> (task, run directories, protocol note)
ARMS: dict[str, tuple[str, list[str], str]] = {
    # --- AG News, Phase 1 (full-validation selection; re-scored on the report half) ---
    "agnews/p1_reflective": ("agnews", ["artifacts/gepa_confirmation_mb8", "artifacts/phase1/reflective_s1", "artifacts/phase1/reflective_s2"], "rescored"),
    "agnews/p1_admm_euclid_rho0.3_minibatch": ("agnews", [f"artifacts/phase1/admm_rho0.3_s{i}" for i in range(3)], "rescored"),
    "agnews/p1_admm_euclid_rho1_minibatch": ("agnews", [f"artifacts/phase1/admm_rho1_s{i}" for i in range(3)], "rescored"),
    "agnews/p1_admm_euclid_rho0.3_heldout": ("agnews", [f"artifacts/phase1_heldout/admm_rho0.3_s{i}" for i in range(3)], "rescored"),
    "agnews/p1_admm_euclid_rho1_heldout": ("agnews", [f"artifacts/phase1_heldout/admm_rho1_s{i}" for i in range(3)], "rescored"),
    # --- AG News, Phases 2-5 (selection split) ---
    "agnews/reflective_p12": ("agnews", [f"artifacts/phase2/reflective_s{i}" for i in range(10)], "split"),
    "agnews/reflect_polish": ("agnews", [f"artifacts/phase2/polish_s{i}" for i in range(3)], "split"),
    "agnews/admm_topp": ("agnews", [f"artifacts/phase2/admm_topp_s{i}" for i in range(10)], "split"),
    "agnews/admm_beam_lam0.3": ("agnews", [f"artifacts/phase2/beam_lam0.3_s{i}" for i in range(10)], "split"),
    "agnews/admm_beam_lam1.0": ("agnews", [f"artifacts/phase2/beam_lam1.0_s{i}" for i in range(3)], "split"),
    "agnews/reflective_p30": ("agnews", [f"artifacts/phase3/reflective_p30_s{i}" for i in range(10)], "split"),
    "agnews/reflective_p40": ("agnews", [f"artifacts/phase3/reflective_p40_s{i}" for i in range(10)], "split"),
    "agnews/reflective_heldout_p12": ("agnews", [f"artifacts/phase3/reflective_heldout_p12_s{i}" for i in range(10)], "split"),
    "agnews/reflective_heldout_p30": ("agnews", [f"artifacts/phase3/reflective_heldout_p30_s{i}" for i in range(10)], "split"),
    "agnews/admm_beam_lam0.3_indep_accept": ("agnews", [f"artifacts/phase6_independent_acceptance/agnews_beam_indep_s{i}" for i in range(10)], "split"),
    "agnews/admm_beam_lam0.3_duals": ("agnews", [f"artifacts/phase5_duals/agnews_beam_dual_s{i}" for i in range(10)], "split"),
    # --- Emotion ---
    "emotion/reflective_p12": ("emotion", [f"artifacts/phase4_emotion/reflective_p12_s{i}" for i in range(10)], "split"),
    "emotion/reflective_heldout_p30": ("emotion", [f"artifacts/phase4_emotion/reflective_heldout_p30_s{i}" for i in range(10)], "split"),
    "emotion/admm_topp": ("emotion", [f"artifacts/phase4_emotion/admm_topp_s{i}" for i in range(10)], "split"),
    "emotion/admm_beam_lam0.3": ("emotion", [f"artifacts/phase4_emotion/beam_lam0.3_s{i}" for i in range(10)], "split"),
    "emotion/admm_beam_lam0.3_indep_accept": ("emotion", [f"artifacts/phase6_independent_acceptance/emotion_beam_indep_s{i}" for i in range(10)], "split"),
    "emotion/admm_beam_lam0.3_duals": ("emotion", [f"artifacts/phase5_duals/emotion_beam_dual_s{i}" for i in range(10)], "split"),
}

COMPARISONS = [
    ("agnews/admm_beam_lam0.3", "agnews/reflective_p12"),
    ("agnews/admm_beam_lam0.3", "agnews/reflective_p30"),
    ("agnews/admm_beam_lam0.3", "agnews/reflective_p40"),
    ("agnews/admm_beam_lam0.3", "agnews/reflective_heldout_p12"),
    ("agnews/admm_beam_lam0.3", "agnews/reflective_heldout_p30"),
    ("agnews/admm_beam_lam0.3", "agnews/admm_topp"),
    ("agnews/admm_topp", "agnews/reflective_p12"),
    ("agnews/admm_beam_lam0.3_duals", "agnews/admm_beam_lam0.3"),
    ("agnews/admm_beam_lam0.3_indep_accept", "agnews/admm_beam_lam0.3"),
    ("agnews/admm_beam_lam0.3_indep_accept", "agnews/reflective_heldout_p30"),
    ("agnews/admm_beam_lam0.3_indep_accept", "agnews/reflective_p40"),
    ("agnews/admm_beam_lam0.3_indep_accept", "agnews/reflective_p12"),
    ("emotion/admm_beam_lam0.3", "emotion/reflective_p12"),
    ("emotion/admm_beam_lam0.3", "emotion/reflective_heldout_p30"),
    ("emotion/admm_topp", "emotion/reflective_p12"),
    ("emotion/admm_topp", "emotion/reflective_heldout_p30"),
    ("emotion/admm_beam_lam0.3", "emotion/admm_topp"),
    ("emotion/admm_beam_lam0.3_duals", "emotion/admm_beam_lam0.3"),
    ("emotion/admm_beam_lam0.3_indep_accept", "emotion/admm_beam_lam0.3"),
    ("emotion/admm_beam_lam0.3_indep_accept", "emotion/reflective_heldout_p30"),
    ("emotion/admm_beam_lam0.3_indep_accept", "emotion/reflective_p12"),
]


def _report_ids(task: str) -> tuple[list[str], list[str]]:
    path = "artifacts/data/validation.json" if task == "agnews" else "artifacts/data_emotion/validation.json"
    selection, report = split_selection_report(load_examples_json(path), seed=SPLIT_SEED)
    return [e.example_id for e in selection], [e.example_id for e in report]


def _arm(task: str, runs: list[str], protocol: str) -> dict[str, Any]:
    selection_ids, report_ids = _report_ids(task)
    rows = []
    for run in runs:
        result = json.loads(Path(run, "gepa_result.json").read_text(encoding="utf-8"))
        if protocol == "split":
            report = 100 * result["report_half"]["best_accuracy"]
            seed_report = 100 * result["report_half"]["seed_accuracy"]
            instruction = result["best_instruction"]
        else:
            rescored = rescore_run(Path(run), selection_ids, report_ids)
            report = 100 * rescored["chosen"]["report_accuracy"]
            seed_report = 100 * rescored["seed_report_accuracy"]
            instruction = rescored["chosen"]["instruction"]
        admm = result.get("admm_proposer") or {}
        rows.append({
            "run": run,
            "report_accuracy": report,
            "seed_report_accuracy": seed_report,
            "selected_instruction": instruction,
            "accepted": result["candidate_count_accepted_including_seed"] - 1,
            "proposals": result["proposal_count"],
            "wall_time_seconds": result["optimization_wall_time_seconds"],
            "metric_calls": result["total_metric_calls"],
            "admm_backward_calls": admm.get("backward_calls", 0),
        })
    scores = [row["report_accuracy"] for row in rows]
    return {
        "task": task,
        "protocol": protocol,
        "seeds": len(rows),
        "mean": statistics.mean(scores),
        "sd": statistics.stdev(scores) if len(scores) > 1 else 0.0,
        "scores": scores,
        "seed_instruction_report_accuracy": rows[0]["seed_report_accuracy"],
        "improved_over_seed": sum(s > rows[0]["seed_report_accuracy"] + 1e-9 for s in scores),
        "mean_accepted": statistics.mean(row["accepted"] for row in rows),
        "mean_wall_time_seconds": statistics.mean(row["wall_time_seconds"] for row in rows),
        "mean_admm_backward_calls": statistics.mean(row["admm_backward_calls"] for row in rows),
        "runs": rows,
    }


def _compare(a: list[float], b: list[float]) -> dict[str, float]:
    from scipy import stats

    rng = random.Random(0)
    boot = sorted(
        statistics.mean(rng.choices(a, k=len(a))) - statistics.mean(rng.choices(b, k=len(b)))
        for _ in range(20000)
    )
    return {
        "difference": statistics.mean(a) - statistics.mean(b),
        "welch_p": float(stats.ttest_ind(a, b, equal_var=False).pvalue),
        "mann_whitney_p": float(stats.mannwhitneyu(a, b).pvalue),
        "bootstrap_ci95": [boot[500], boot[19499]],
    }


def main() -> None:
    arms = {name: _arm(*spec) for name, spec in ARMS.items()}
    comparisons = {
        f"{a} vs {b}": _compare(arms[a]["scores"], arms[b]["scores"]) for a, b in COMPARISONS
    }
    output = Path("artifacts/classification_results/summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"arms": arms, "comparisons": comparisons}, indent=2), encoding="utf-8")
    for name, arm in arms.items():
        print(f"{name:42s} n={arm['seeds']:2d} {arm['mean']:6.2f} ± {arm['sd']:4.2f} "
              f"improved {arm['improved_over_seed']:2d}/{arm['seeds']:<2d} accepted {arm['mean_accepted']:4.1f} "
              f"wall {arm['mean_wall_time_seconds']:4.0f}s bwd {arm['mean_admm_backward_calls']:4.0f} "
              f"seed {arm['seed_instruction_report_accuracy']:.2f}")
    for name, c in comparisons.items():
        print(f"{name:72s} {c['difference']:+5.2f}  welch p={c['welch_p']:.4f}  MW p={c['mann_whitney_p']:.4f}  "
              f"CI [{c['bootstrap_ci95'][0]:+.2f}, {c['bootstrap_ci95'][1]:+.2f}]")
    print(f"WROTE {output}")


if __name__ == "__main__":
    main()
