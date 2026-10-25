"""Letters vs. label names: proposal-level bench summary (length_bench.py rows).

For each task setting and ADMM method: how often the child differs from its parent, how often S'
accepts it, and its net on the 2k report set (fixes - breaks, in percentage points). Children of the
same parent are not independent, so the 95% CI averages each parent's children first and uses a t
interval over parents. Writes artifacts/name_bench/summary.md.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

from scipy import stats

ROOT = Path("artifacts/name_bench")
SETTINGS = [("yahoo", "Yahoo, letters"), ("yahoo_names", "Yahoo, names"),
            ("emotion", "Emotion, letters"), ("emotion_names", "Emotion, names")]
METHODS = [("fixed", "C: fixed-length"), ("ml_grad", "F: gradient insertion"), ("gated", "G: gated"),
           ("multistart", "I: multi-start")]


def rows(setting: str, method: str) -> list[dict]:
    path = ROOT / f"{setting}_{method}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def parent_cluster_ci(values_by_parent: dict[int, list[float]]) -> tuple[float, float]:
    means = [statistics.fmean(v) for v in values_by_parent.values()]
    if len(means) < 2:
        return statistics.fmean(means) if means else float("nan"), float("nan")
    half = stats.t.ppf(0.975, len(means) - 1) * statistics.stdev(means) / len(means) ** 0.5
    return statistics.fmean(means), half


def main() -> None:
    lines = [
        "| Setting | Method | Children | Changed | Accepted on S′ | Mean net, pp (95% CI) | Better (net > 0) "
        "| Mean net of accepted, pp |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    summary = {}
    for setting, setting_label in SETTINGS:
        for method, method_label in METHODS:
            rs = rows(setting, method)
            if not rs:
                continue
            n_scored = 2000 if "yahoo" in setting else 2004  # report-set sizes
            net = [100 * (r["fixes"] - r["breaks"]) / n_scored for r in rs]
            by_parent = defaultdict(list)
            for r, value in zip(rs, net):
                by_parent[r["parent_index"]].append(value)
            mean, half = parent_cluster_ci(by_parent)
            accepted = [value for r, value in zip(rs, net) if r["accepted_on_S_prime"]]
            changed = sum(r["child"] != r["parent"] for r in rs)
            summary[f"{setting}/{method}"] = {
                "children": len(rs), "changed": changed, "accepted": len(accepted), "mean_net_pp": mean,
                "ci_half_pp": half, "better": sum(v > 0 for v in net),
                "mean_net_accepted_pp": statistics.fmean(accepted) if accepted else None,
            }
            lines.append(
                f"| {setting_label} | {method_label} | {len(rs)} | {changed} | {len(accepted)} | "
                f"{mean:+.2f} (± {half:.2f}) | {sum(v > 0 for v in net)} | "
                f"{statistics.fmean(accepted):+.2f} |" if accepted else
                f"| {setting_label} | {method_label} | {len(rs)} | {changed} | 0 | {mean:+.2f} (± {half:.2f}) | "
                f"{sum(v > 0 for v in net)} | — |"
            )
    (ROOT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
