from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from .config import PilotConfig


def build_admm_report(
    run_specs: list[str],
    output_dir: str | Path,
    *,
    config_path: str | Path = "configs/pilot.yaml",
) -> dict[str, Any]:
    config = PilotConfig.from_yaml(config_path)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.resolved_name_or_path,
        revision=None if config.model.local_snapshot_path else config.model.revision,
        local_files_only=config.model.local_files_only,
        use_fast=True,
    )
    runs: list[dict[str, Any]] = []
    raw_results: dict[str, dict[str, Any]] = {}
    for spec in run_specs:
        if "=" not in spec:
            raise ValueError(f"Run specification must be LABEL=PATH, received {spec!r}.")
        label, path_text = spec.split("=", 1)
        path = Path(path_text)
        result = json.loads((path / "admm_result.json").read_text(encoding="utf-8"))
        manifest = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
        raw_results[label] = result
        best = result["best_checkpoint"]
        seed_ids = result["seed_slot_token_ids"]
        best_ids = best["slot_token_ids"]
        token_changes = []
        for position, (seed_id, best_id) in enumerate(zip(seed_ids, best_ids)):
            if seed_id == best_id:
                continue
            token_changes.append(
                {
                    "position": position,
                    "seed_token_id": seed_id,
                    "seed_token_text": tokenizer.decode(
                        [seed_id], clean_up_tokenization_spaces=False
                    ),
                    "selected_token_id": best_id,
                    "selected_token_text": tokenizer.decode(
                        [best_id], clean_up_tokenization_spaces=False
                    ),
                }
            )
        runs.append(
            {
                "run": label,
                "rho": result["hyperparameters"]["rho"],
                "best_iteration": best["iteration"],
                "best_discrete_accuracy": best["discrete_validation"]["accuracy"],
                "best_discrete_mean_loss": best["discrete_validation"]["mean_loss"],
                "continuous_accuracy_at_best": best["continuous_validation"]["accuracy"],
                "continuous_mean_loss_at_best": best["continuous_validation"]["mean_loss"],
                "primal_residual_norm_at_best": best["primal_residual_norm"],
                "dual_residual_norm_at_best": best["dual_residual_norm"],
                "changed_token_count": best["changed_token_count_from_seed"],
                "roundtrip_exact": best["roundtrip"]["roundtrip_exact"],
                "selected_instruction": best["roundtrip"]["reconstructed_instruction"],
                "token_changes": token_changes,
                "forward_calls": result["forward_calls"],
                "backward_calls": result["backward_calls"],
                "wall_time_seconds": result["optimization_wall_time_seconds"],
                "official_test_examples_present": manifest[
                    "official_test_examples_present"
                ],
            }
        )
    best_run = max(
        runs,
        key=lambda row: (
            row["best_discrete_accuracy"],
            -row["best_discrete_mean_loss"],
        ),
    )
    report = {
        "runs": runs,
        "selected_run": best_run["run"],
        "selected_rho": best_run["rho"],
        "selected_iteration": best_run["best_iteration"],
        "selected_discrete_accuracy": best_run["best_discrete_accuracy"],
        "selected_discrete_mean_loss": best_run["best_discrete_mean_loss"],
        "selected_instruction": best_run["selected_instruction"],
        "selected_token_changes": best_run["token_changes"],
        "official_test_unused_in_all_runs": not any(
            row["official_test_examples_present"] for row in runs
        ),
        "diagnosis": (
            "The continuous relaxation improves substantially, but projection usually "
            "returns the seed. The low-rho run crosses token boundaries with a modest "
            "discrete gain and a large primal residual, so projection/feasibility is the "
            "dominant bottleneck."
        ),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "admm_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    fields = [
        "run",
        "rho",
        "best_iteration",
        "best_discrete_accuracy",
        "best_discrete_mean_loss",
        "continuous_accuracy_at_best",
        "continuous_mean_loss_at_best",
        "primal_residual_norm_at_best",
        "dual_residual_norm_at_best",
        "changed_token_count",
        "roundtrip_exact",
        "forward_calls",
        "backward_calls",
        "wall_time_seconds",
        "selected_instruction",
    ]
    with (output_dir / "admm_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(runs)

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/gepa-pilot-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for label, result in raw_results.items():
        checkpoints = result["checkpoints"]
        iterations = [row["iteration"] for row in checkpoints]
        axes[0, 0].plot(
            iterations,
            [row["continuous_validation"]["mean_loss"] for row in checkpoints],
            marker="o",
            label=label,
        )
        axes[0, 1].plot(
            iterations,
            [row["discrete_validation"]["mean_loss"] for row in checkpoints],
            marker="o",
            label=label,
        )
        axes[1, 0].plot(
            iterations,
            [row["discrete_validation"]["accuracy"] for row in checkpoints],
            marker="o",
            label=label,
        )
        axes[1, 1].plot(
            iterations,
            [row["primal_residual_norm"] for row in checkpoints],
            marker="o",
            label=label,
        )
    titles = [
        "Continuous validation loss",
        "Projected token-ID validation loss",
        "Projected token-ID validation accuracy",
        "Primal residual norm",
    ]
    for axis, title in zip(axes.flat, titles):
        axis.set_title(title)
        axis.set_xlabel("Outer iteration")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "admm_curves.png", dpi=180)
    plt.close(figure)

    table = [
        "| Rho | Best iteration | Discrete accuracy | Discrete loss | Changed tokens | Round-trip |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for row in runs:
        table.append(
            f"| {row['rho']:g} | {row['best_iteration']} | "
            f"{row['best_discrete_accuracy']:.4f} | {row['best_discrete_mean_loss']:.4f} | "
            f"{row['changed_token_count']} | {row['roundtrip_exact']} |"
        )
    changes = "\n".join(
        f"- Position {row['position']}: `{row['seed_token_text']}` "
        f"({row['seed_token_id']}) → `{row['selected_token_text']}` "
        f"({row['selected_token_id']})"
        for row in best_run["token_changes"]
    ) or "- No token changes."
    markdown = "\n".join(
        [
            "# ADMM pilot summary",
            "",
            *table,
            "",
            f"Selected configuration: rho={best_run['rho']:g}, iteration "
            f"{best_run['best_iteration']}, validation accuracy "
            f"{best_run['best_discrete_accuracy']:.4f}.",
            "",
            report["diagnosis"],
            "",
            "## Selected projected instruction",
            "",
            f"> {best_run['selected_instruction']}",
            "",
            "## Token changes",
            "",
            changes,
            "",
            f"Official test examples unused in every run: {report['official_test_unused_in_all_runs']}.",
            "",
        ]
    )
    (output_dir / "admm_summary.md").write_text(markdown, encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize an ADMM rho sweep.")
    parser.add_argument("--run", action="append", default=None)
    parser.add_argument("--output-dir", default="artifacts/admm_comparison")
    parser.add_argument("--config", default="configs/pilot.yaml")
    args = parser.parse_args()
    specs = args.run or [
        "rho1=artifacts/admm_rho1",
        "rho10=artifacts/admm_rho10",
        "rho100=artifacts/admm_rho100",
    ]
    report = build_admm_report(specs, args.output_dir, config_path=args.config)
    print(
        f"SELECTED rho={report['selected_rho']:g} iteration={report['selected_iteration']} "
        f"accuracy={report['selected_discrete_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
