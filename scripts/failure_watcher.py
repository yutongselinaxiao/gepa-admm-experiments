#!/usr/bin/env python3
"""Watch a running gpu_queue batch and rerun jobs that failed with a GPU error.

For queues already running (which can't pick up gpu_queue.py changes). Polls the batch's
<jobs>.status.json. When a job is recorded as failed and its stdout log or GEPA run_log.txt shows
a GPU error (see gpu_queue.GPU_ERRORS), the failed outputs are moved to --failed-dir and the job
is rerun with its run directory and log rewritten into --rerun-dir, so the main pipeline never
sees a half-finished run. Reruns use the same memory-aware GPU choice (GPUs 4-7 only) and are
themselves retried up to --retries times. Exits once the main queue has finished and no rerun is
pending or running. Progress goes to <rerun-dir>/watcher.status.json.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gpu_queue import classify, free_memory_mb, gpu_error  # noqa: E402


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def move_aside(job: dict, failed_dir: Path, tag: str) -> None:
    failed_dir.mkdir(parents=True, exist_ok=True)
    for source in (job.get("run_dir"), job["log"]):
        if source and Path(source).exists():
            target = failed_dir / f"{Path(source).name}.{tag}"
            shutil.rmtree(target, ignore_errors=True) if target.is_dir() else target.unlink(missing_ok=True)
            shutil.move(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("jobs")
    parser.add_argument("--rerun-dir", required=True)
    parser.add_argument("--failed-dir", required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[4, 5, 6, 7])
    parser.add_argument("--min-free-mb", type=int, default=18000)
    parser.add_argument("--max-per-gpu", type=int, default=1)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--settle", type=int, default=60)
    parser.add_argument("--poll", type=int, default=30)
    args = parser.parse_args()
    if any(g < 4 for g in args.gpus):
        raise SystemExit("Only GPUs 4-7 may be used for this project.")

    jobs = {j["name"]: j for j in map(json.loads, Path(args.jobs).read_text().splitlines()) if j}
    main_status = Path(args.jobs + ".status.json")
    rerun_dir, failed_dir = Path(args.rerun_dir), Path(args.failed_dir)
    rerun_dir.mkdir(parents=True, exist_ok=True)
    handled: set[str] = set()
    attempts: dict[str, int] = {}
    pending: list[dict] = []
    running: dict[str, tuple[subprocess.Popen, int, dict]] = {}
    results: dict[str, str] = {}
    hold_until: dict[int, float] = {}

    def log(message: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

    def rerun_job(job: dict) -> dict:
        new = dict(job)
        for key in ("run_dir", "log"):
            new[key] = str(rerun_dir / Path(job[key]).name)
        new["cmd"] = job["cmd"].replace(job["run_dir"], new["run_dir"])
        return new

    def write_status() -> None:
        (rerun_dir / "watcher.status.json").write_text(json.dumps({
            "pending": [j["name"] for j in pending], "running": {n: g for n, (_, g, _) in running.items()},
            "results": results, "attempts": attempts, "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, indent=2))

    while True:
        status = json.loads(read(main_status) or "{}")
        for name, outcome in status.get("results", {}).items():
            if name in handled or outcome == "done":
                continue
            handled.add(name)
            job = jobs[name]
            texts = [read(Path(job["log"])), read(Path(job.get("run_dir", ""), "run_log.txt")),
                     read(failed_dir / Path(job["log"]).name), read(failed_dir / name / "run_log.txt")]
            if not gpu_error(*texts):
                log(f"NOT RETRYING {name}: '{outcome}' is not a GPU error")
                results[name] = f"not retried ({outcome})"
                continue
            move_aside(job, failed_dir, "main")
            attempts[name] = 0
            pending.append(rerun_job(job))
            log(f"GPU ERROR in main batch: {name}; queued for rerun in {rerun_dir}")
        for name, (proc, gpu, job) in list(running.items()):
            if proc.poll() is None:
                continue
            del running[name]
            outcome = classify(job, proc.returncode)
            if outcome.startswith("gpu error") and attempts[name] <= args.retries:
                move_aside(job, failed_dir, f"rerun{attempts[name]}")
                pending.append(job)
                log(f"GPU ERROR again: {name} (GPU {gpu}); requeue")
            else:
                results[name] = outcome
                log(f"{outcome.upper()} {name} (GPU {gpu})")
        if pending:
            now = time.time()
            free = free_memory_mb(args.gpus)
            ours = {g: sum(1 for _, gg, _ in running.values() if gg == g) for g in args.gpus}
            candidates = [g for g in args.gpus if free.get(g, 0) >= args.min_free_mb
                          and ours[g] < args.max_per_gpu and hold_until.get(g, 0) <= now]
            if candidates:
                gpu = max(candidates, key=lambda g: free[g])
                job = pending.pop(0)
                attempts[job["name"]] += 1
                with open(job["log"], "w") as out:
                    proc = subprocess.Popen(job["cmd"], shell=True, stdout=out, stderr=subprocess.STDOUT,
                                            env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)))
                running[job["name"]] = (proc, gpu, job)
                hold_until[gpu] = now + args.settle
                log(f"START rerun {job['name']} on GPU {gpu} (free {free[gpu]} MB)")
        write_status()
        main_finished = bool(status) and not status.get("pending") and not status.get("running")
        if main_finished and not pending and not running and set(status["results"]) <= handled | {
                n for n, o in status["results"].items() if o == "done"}:
            break
        time.sleep(args.poll)
    log(f"WATCHER DONE: {sum(v == 'done' for v in results.values())} rerun(s) succeeded; {results}")


if __name__ == "__main__":
    main()
