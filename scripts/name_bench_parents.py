#!/usr/bin/env python3
"""Parent prompts for the letters-vs-names proposal bench (length_bench.py --parents-file).

Per task setting: the seed instruction, the config's manual instructions, then the reflection LM's
rewrites from short GEPA runs (A arm, seeds 0 and 1, in call order), deduplicated, up to --count.
Usage: name_bench_parents.py CONFIG OUT RUN_DIR [RUN_DIR ...]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml


def reflection_outputs(run_dir: str) -> list[str]:
    def walk(node):
        if isinstance(node, dict):
            if "completion_token_count" in node and "raw_output" in node:
                yield node["raw_output"]
            for value in node.values():
                yield from walk(value)
        elif isinstance(node, list):
            for value in node:
                yield from walk(value)

    texts = []
    for raw in walk(json.loads(Path(run_dir, "gepa_result.json").read_text(encoding="utf-8"))):
        fenced = re.findall(r"```(?:[a-z]*\n)?(.*?)```", raw, flags=re.S)
        text = (fenced[-1] if fenced else raw).strip()
        if text:
            texts.append(" ".join(text.split()))
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config")
    parser.add_argument("out")
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    pool = [" ".join(config["prompt"]["seed_instruction"].split())]
    pool += [" ".join(text.split()) for text in config.get("manual_instructions", [])]
    rewrites = [reflection_outputs(run) for run in args.runs]
    for index in range(max(map(len, rewrites))):  # alternate between runs
        for texts in rewrites:
            if index < len(texts) and texts[index] not in pool:
                pool.append(texts[index])
    pool = pool[: args.count]
    Path(args.out).write_text(json.dumps(pool, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{args.out}: {len(pool)} parents")


if __name__ == "__main__":
    main()
