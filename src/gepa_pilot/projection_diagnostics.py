"""Diagnostics for the ADMM Z-step (nearest-token projection).

1. Projection gap vs. Lipschitz bound: for saved standalone-ADMM iterates, compare
   F(E(z)) - F(e) with L_seg * ||e - E(z)||, where L_seg is the largest gradient norm
   sampled on the segment from e to E(z).
2. Off-manifold drift and hubness: how far projection targets sit from real tokens,
   and which tokens absorb random displacements of the seed embeddings.
3. Alternative projections of the same targets: Euclidean, ASCII-only Euclidean,
   cosine, whitened, and all-but-the-top (mean and top PCA directions removed).
4. Token statistics of the edits the ADMM proposer made inside GEPA.

All losses use a fixed 32-example training batch; validation data is not used.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

from .admm import build_allowed_token_ids, gold_indices
from .config import PilotConfig
from .data import load_examples_json
from .evaluator import CommonEvaluator


def _is_ascii(text: str) -> bool:
    return bool(text) and all(ord(character) < 128 for character in text)


class Projector:
    """Nearest-token projection under a fixed linear transform of embedding space."""

    def __init__(
        self,
        name: str,
        vocabulary: torch.Tensor,
        ids: torch.Tensor,
        transform: Callable[[torch.Tensor], torch.Tensor],
        cosine: bool,
    ) -> None:
        self.name, self.ids, self.transform, self.cosine = name, ids, transform, cosine
        transformed = transform(vocabulary)
        self.matrix = F.normalize(transformed, dim=1) if cosine else transformed
        self.squared_norms = None if cosine else transformed.square().sum(dim=1)

    def __call__(self, targets: torch.Tensor, chunk: int = 32768) -> torch.Tensor:
        query = self.transform(targets)
        if self.cosine:
            query = F.normalize(query, dim=1)
        best = torch.full((len(query),), -float("inf"), device=query.device)
        best_ids = torch.zeros(len(query), dtype=torch.long, device=query.device)
        for start in range(0, len(self.matrix), chunk):
            block = self.matrix[start : start + chunk]
            scores = query @ block.T
            if not self.cosine:
                scores = 2.0 * scores - self.squared_norms[start : start + chunk]
            values, indices = scores.max(dim=1)
            replace = values > best
            best = torch.where(replace, values, best)
            best_ids = torch.where(replace, self.ids[start + indices], best_ids)
        return best_ids


def build_projectors(weight: torch.Tensor, allowed: torch.Tensor, ascii_ids: torch.Tensor,
                     top_directions: int) -> list[Projector]:
    vocabulary = weight.index_select(0, allowed).float()
    mean = vocabulary.mean(dim=0)
    covariance = torch.cov((vocabulary - mean).T)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    whitening = eigenvectors / torch.sqrt(eigenvalues.clamp_min(1e-6))
    top = eigenvectors[:, -top_directions:]

    def identity(x: torch.Tensor) -> torch.Tensor:
        return x.float()

    def whiten(x: torch.Tensor) -> torch.Tensor:
        return (x.float() - mean) @ whitening

    def all_but_the_top(x: torch.Tensor) -> torch.Tensor:
        centered = x.float() - mean
        return centered - (centered @ top) @ top.T

    ascii_vocabulary = weight.index_select(0, ascii_ids).float()
    return [
        Projector("euclidean", vocabulary, allowed, identity, cosine=False),
        Projector("euclidean_ascii", ascii_vocabulary, ascii_ids, identity, cosine=False),
        Projector("cosine", vocabulary, allowed, identity, cosine=True),
        Projector("whitened", vocabulary, allowed, whiten, cosine=False),
        Projector(f"all_but_top{top_directions}", vocabulary, allowed, all_but_the_top, cosine=True),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--train", default="artifacts/data/train.json")
    parser.add_argument(
        "--checkpoint-runs",
        nargs="+",
        default=[
            "artifacts/admm_rho1",
            "artifacts/repeat_original_admm_rho1",
            "artifacts/parity_refactor_admm_rho1",
        ],
    )
    parser.add_argument(
        "--proposal-logs",
        nargs="+",
        default=sorted(str(p) for p in Path("artifacts").glob("phase1*/admm_*/admm_proposals.jsonl")),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--segment-points", type=int, default=9)
    parser.add_argument("--hub-samples", type=int, default=200)
    parser.add_argument("--top-directions", type=int, default=20)
    parser.add_argument("--output", default="artifacts/projection_diagnostics/report.json")
    args = parser.parse_args()

    config = PilotConfig.from_yaml(args.config)
    evaluator = CommonEvaluator.from_pretrained(
        name_or_path=config.model.resolved_name_or_path,
        revision=None if config.model.local_snapshot_path else config.model.revision,
        local_files_only=config.model.local_files_only,
        dtype=config.model.dtype,
        device=config.model.device,
        system_prompt=config.prompt.system_prompt,
        label_names=config.prompt.label_names,
        enable_thinking=config.prompt.enable_thinking,
    )
    tokenizer = evaluator.tokenizer
    instruction = config.prompt.seed_instruction
    train = load_examples_json(args.train)
    batch = train[: args.batch_size]
    targets = gold_indices(batch, evaluator.labels, evaluator.device)
    weight = evaluator.model.get_input_embeddings().weight.detach()
    allowed = build_allowed_token_ids(tokenizer, weight.shape[0]).to(evaluator.device)
    decoded = {int(i): tokenizer.decode([int(i)]) for i in allowed.tolist()}
    ascii_ids = torch.tensor(
        [i for i in allowed.tolist() if _is_ascii(decoded[i])], device=evaluator.device
    )
    norms = weight.index_select(0, allowed).float().norm(dim=1)
    tag_like = [
        decoded[i] for i in allowed.tolist()
        if decoded[i].startswith("<") and decoded[i].endswith(">") and len(decoded[i]) > 2
    ]
    sorted_norms = norms.sort().values

    def norm_percentile(token_id: int) -> float:
        value = weight[token_id].float().norm()
        return float((sorted_norms < value).float().mean() * 100)

    def soft_loss_and_grad(x: torch.Tensor) -> tuple[float, float]:
        x = x.detach().float().clone().requires_grad_(True)
        loss = F.cross_entropy(
            evaluator.score_slot_embeddings(batch, instruction, x).float(), targets
        )
        loss.backward()
        return float(loss.detach()), float(x.grad.norm())

    def discrete(ids: torch.Tensor) -> dict[str, float]:
        scores = evaluator.score_fixed_slot_ids(batch, instruction, ids).float().to(targets.device)
        return {
            "loss": float(F.cross_entropy(scores, targets)),
            "accuracy": float(scores.argmax(dim=-1).eq(targets).float().mean()),
        }

    seed_ids = torch.tensor(
        evaluator.prompt_builder.encode(batch[0].article, instruction).slot_ids,
        device=evaluator.device,
    )
    seed_embeddings = weight.index_select(0, seed_ids).float()
    projectors = build_projectors(weight, allowed, ascii_ids, args.top_directions)
    report: dict[str, Any] = {
        "batch_size": len(batch),
        "allowed_vocabulary": len(allowed),
        "ascii_vocabulary": len(ascii_ids),
        "tied_input_output_embeddings": bool(
            getattr(evaluator.model.config, "tie_word_embeddings", False)
        ),
        "seed_discrete": discrete(seed_ids),
        "allowed_tag_like_tokens": tag_like,
    }

    # Reference scale: distance from each allowed token to its nearest other token.
    positions = torch.randperm(len(allowed), generator=torch.Generator().manual_seed(0))[:2000]
    positions = positions.to(allowed.device)
    sample_vectors = weight.index_select(0, allowed[positions]).float()
    vocabulary = weight.index_select(0, allowed).float()
    distances = torch.cdist(sample_vectors, vocabulary, compute_mode="donot_use_mm_for_euclid_dist")
    distances[torch.arange(len(positions), device=allowed.device), positions] = float("inf")
    nn_distance = distances.min(dim=1).values
    report["token_nearest_neighbor_distance"] = {
        "median": float(nn_distance.median()),
        "p10": float(nn_distance.quantile(0.1)),
        "p90": float(nn_distance.quantile(0.9)),
    }
    report["token_norm"] = {
        "median": float(norms.median()), "p10": float(norms.quantile(0.1)), "p90": float(norms.quantile(0.9))
    }

    # 1 + 3: saved standalone-ADMM iterates.
    iterates = []
    for run in args.checkpoint_runs:
        for path in sorted(Path(run).glob("checkpoint_*.pt")):
            checkpoint = torch.load(path, map_location=evaluator.device)
            if checkpoint["iteration"] == 0:
                continue
            e = checkpoint["e"].float().to(evaluator.device)
            z = checkpoint["z"].to(evaluator.device)
            ez = weight.index_select(0, z).float()
            # The saved u is u_k = u_{k-1} + e_k - E(z_k); z_k projected e_k + u_{k-1} = u_k + E(z_k).
            target = checkpoint["u"].float().to(evaluator.device) + ez
            residual = float((e - ez).norm())
            soft_loss, _ = soft_loss_and_grad(e)
            gradient_norms = [
                soft_loss_and_grad((1 - t) * e + t * ez)[1]
                for t in torch.linspace(0, 1, args.segment_points).tolist()
            ]
            discrete_z = discrete(z)
            per_position_target_distance = (target - ez).norm(dim=1)
            row: dict[str, Any] = {
                "run": run,
                "iteration": checkpoint["iteration"],
                "soft_loss": soft_loss,
                "discrete_loss": discrete_z["loss"],
                "discrete_accuracy": discrete_z["accuracy"],
                "projection_gap": discrete_z["loss"] - soft_loss,
                "primal_residual": residual,
                "segment_lipschitz": max(gradient_norms),
                "gradient_norm_at_e": gradient_norms[0],
                "gradient_norm_at_Ez": gradient_norms[-1],
                "lipschitz_bound": max(gradient_norms) * residual,
                "target_to_token_distance_median": float(per_position_target_distance.median()),
                "target_to_token_distance_max": float(per_position_target_distance.max()),
                "euclidean_matches_saved_z": None,
                "projections": {},
            }
            for projector in projectors:
                ids = projector(target)
                if projector.name == "euclidean":
                    row["euclidean_matches_saved_z"] = bool(ids.eq(z).all())
                changed = [
                    {"position": p, "seed": decoded.get(int(seed_ids[p]), tokenizer.decode([int(seed_ids[p])])),
                     "projected": decoded.get(int(ids[p]), tokenizer.decode([int(ids[p])]))}
                    for p in range(len(ids)) if int(ids[p]) != int(seed_ids[p])
                ]
                row["projections"][projector.name] = {
                    **discrete(ids),
                    "changed_from_seed": len(changed),
                    "changed_tokens": changed,
                    "ascii_fraction_of_changes": (
                        sum(_is_ascii(c["projected"]) for c in changed) / len(changed) if changed else None
                    ),
                }
            iterates.append(row)
            print(f"ITERATE {run} it={row['iteration']} gap={row['projection_gap']:.3f} "
                  f"bound={row['lipschitz_bound']:.1f} r={residual:.2f}")
    report["iterates"] = iterates

    # 2: random displacements of the seed embeddings.
    generator = torch.Generator(device="cpu").manual_seed(1)
    euclidean = projectors[0]
    hub_rows = []
    for radius in (1.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0):
        directions = torch.randn(args.hub_samples, *seed_embeddings.shape, generator=generator)
        directions = F.normalize(directions.to(evaluator.device), dim=-1)
        queries = (seed_embeddings.unsqueeze(0) + radius * directions).reshape(-1, seed_embeddings.shape[1])
        ids = euclidean(queries).reshape(args.hub_samples, -1)
        kept = ids.eq(seed_ids.unsqueeze(0)).float().mean()
        changed_ids = ids[ids.ne(seed_ids.unsqueeze(0))].tolist()
        counts = Counter(changed_ids)
        hub_rows.append({
            "radius": radius,
            "fraction_kept_seed_token": float(kept),
            "changed_count": len(changed_ids),
            "distinct_replacement_tokens": len(counts),
            "ascii_fraction_of_replacements": (
                sum(_is_ascii(decoded[i]) for i in changed_ids) / len(changed_ids) if changed_ids else None
            ),
            "median_norm_percentile_of_replacements": (
                float(torch.tensor([norm_percentile(i) for i in changed_ids]).median()) if changed_ids else None
            ),
            "top_replacements": [
                {"token": decoded[i], "count": c, "norm_percentile": norm_percentile(i)}
                for i, c in counts.most_common(10)
            ],
        })
        print(f"HUB radius={radius} kept={float(kept):.3f} distinct={len(counts)}")
    report["random_displacement"] = hub_rows

    # 4: edits the ADMM proposer made inside GEPA.
    edits = []
    for log in args.proposal_logs:
        for line in Path(log).read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            chosen = record.get("chosen")
            if not chosen:
                continue
            for parent_id, child_id in zip(record["parent_slot_ids"], chosen["slot_ids"]):
                if parent_id != child_id:
                    edits.append({
                        "log": log,
                        "parent": tokenizer.decode([parent_id]),
                        "child": tokenizer.decode([child_id]),
                        "child_ascii": _is_ascii(tokenizer.decode([child_id])),
                        "child_norm_percentile": norm_percentile(child_id),
                    })
    report["proposer_edits"] = {
        "count": len(edits),
        "ascii_fraction": sum(e["child_ascii"] for e in edits) / len(edits) if edits else None,
        "median_child_norm_percentile": (
            float(torch.tensor([e["child_norm_percentile"] for e in edits]).median()) if edits else None
        ),
        "edits": edits,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"WROTE {output}")


if __name__ == "__main__":
    main()
