#!/usr/bin/env python3
"""Memory-aware job queue for shared GPUs.

Runs shell commands from a JSONL job file on the allowed GPUs (default 4-7 only). A job is
launched on the allowed GPU with the most free memory, provided that GPU has at least
--min-free-mb free and fewer than --max-per-gpu of *our* jobs. After a launch, that GPU is held
for --settle seconds so the new job's model load shows up in nvidia-smi before the next
decision. A job that hits a GPU error (CUDA out of memory, or a CUDA/cuDNN/cuBLAS failure, which
on shared GPUs usually means memory pressure) is retried, whether the error killed the process
or GEPA caught it and logged it to run_log.txt; its run directory is removed first. Any other
failure is recorded and not retried. Progress goes to <jobs>.status.json.

Job line: {"name": ..., "cmd": ..., "log": ..., "done_marker": "DONE", "run_dir": ...}
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


def free_memory_mb(gpus: list[int]) -> dict[int, int]:
    output = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    free = {}
    for line in output.strip().splitlines():
        index, mb = (int(part.strip()) for part in line.split(","))
        if index in gpus:
            free[index] = mb
    return free


SWALLOWED = ("reflection failed", "Traceback")  # GEPA logs proposer exceptions and keeps going
GPU_ERRORS = ("out of memory", "outofmemoryerror", "cuda error", "cudnn", "mha_graph", "cublas_status",
              "cuda failure", "device-side assert")


def gpu_error(*texts: str) -> bool:
    return any(marker in text.lower() for text in texts for marker in GPU_ERRORS)


def classify(job: dict, returncode: int) -> str:
    log = Path(job["log"]).read_text(encoding="utf-8", errors="replace") if Path(job["log"]).exists() else ""
    run_log = Path(job.get("run_dir", ""), "run_log.txt")
    gepa_log = (run_log.read_text(encoding="utf-8", errors="replace")
                if job.get("run_dir") and run_log.exists() else "")
    if gpu_error(log, gepa_log):
        return "gpu error (retryable)"
    if any(marker in gepa_log for marker in SWALLOWED):
        return "failed (proposal errors swallowed by GEPA; see run_log.txt)"
    if returncode == 0 and job.get("done_marker", "") in log:
        return "done"
    return "failed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("jobs")
    parser.add_argument("--gpus", type=int, nargs="+", default=[4, 5, 6, 7])
    parser.add_argument("--min-free-mb", type=int, default=20000)
    parser.add_argument("--max-per-gpu", type=int, default=3)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--settle", type=int, default=90)
    parser.add_argument("--poll", type=int, default=15)
    args = parser.parse_args()
    if any(g < 4 for g in args.gpus):
        raise SystemExit("Only GPUs 4-7 may be used for this project.")

    jobs = [json.loads(line) for line in Path(args.jobs).read_text().splitlines() if line.strip()]
    pending = list(jobs)
    attempts = {job["name"]: 0 for job in jobs}
    running: dict[str, tuple[subprocess.Popen, int, dict]] = {}
    results: dict[str, str] = {}
    hold_until: dict[int, float] = {}
    status_path = Path(args.jobs + ".status.json")

    def write_status() -> None:
        status_path.write_text(json.dumps({
            "pending": [j["name"] for j in pending],
            "running": {name: gpu for name, (_, gpu, _) in running.items()},
            "results": results, "attempts": attempts, "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, indent=2))

    while pending or running:
        for name, (proc, gpu, job) in list(running.items()):
            if proc.poll() is None:
                continue
            del running[name]
            outcome = classify(job, proc.returncode)
            if outcome.startswith("gpu error") and attempts[name] <= args.retries:
                print(f"[{time.strftime('%H:%M:%S')}] GPU ERROR {name} on GPU {gpu}; requeue", flush=True)
                if job.get("run_dir"):
                    shutil.rmtree(job["run_dir"], ignore_errors=True)
                pending.append(job)
            else:
                results[name] = outcome
                print(f"[{time.strftime('%H:%M:%S')}] {outcome.upper()} {name} (GPU {gpu})", flush=True)
        if pending:
            now = time.time()
            free = free_memory_mb(args.gpus)
            ours = {g: sum(1 for _, gg, _ in running.values() if gg == g) for g in args.gpus}
            candidates = [g for g in args.gpus
                          if free.get(g, 0) >= args.min_free_mb and ours[g] < args.max_per_gpu
                          and hold_until.get(g, 0) <= now]
            if candidates:
                gpu = max(candidates, key=lambda g: free[g])
                job = pending.pop(0)
                attempts[job["name"]] += 1
                Path(job["log"]).parent.mkdir(parents=True, exist_ok=True)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
                with open(job["log"], "w") as log:
                    proc = subprocess.Popen(job["cmd"], shell=True, stdout=log, stderr=subprocess.STDOUT, env=env)
                running[job["name"]] = (proc, gpu, job)
                hold_until[gpu] = now + args.settle
                print(f"[{time.strftime('%H:%M:%S')}] START {job['name']} on GPU {gpu} "
                      f"(free {free[gpu]} MB, ours {ours[gpu] + 1})", flush=True)
                write_status()
                continue  # re-check immediately for other GPUs
        write_status()
        time.sleep(args.poll)
    write_status()
    print(f"ALL DONE: {sum(v == 'done' for v in results.values())}/{len(jobs)} succeeded", flush=True)


if __name__ == "__main__":
    main()
