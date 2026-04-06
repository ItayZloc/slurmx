#!/usr/bin/env python3
"""Submit Claude Code as a persistent SLURM CPU job.

Two modes:
  - Interactive (remote-control): `claude remote-control --name NAME`
  - One-shot (send and forget): `claude -p "PROMPT" --model MODEL --effort EFFORT`

Usage:
    claude-job [name] [--days N] [--workdir PATH] [--permission-mode MODE]
    claude-job --prompt "do something" --model opus --effort max

Importable:
    from claude_job import submit_claude_job
    job_id, log_path = submit_claude_job(name="my-session")
    job_id, log_path = submit_claude_job(prompt="fix the tests", model="opus")
"""

import argparse
import os
import re
import subprocess
import time

from config import CPU_PARTITION, CPU_QOS, CPU_CPUS, CPU_MEM, CLAUDE_LOG_DIR
from maintenance import cap_time_limit

VALID_PERMISSION_MODES = {"default", "acceptEdits", "bypassPermissions", "dontAsk", "plan"}
VALID_MODELS = {"opus", "sonnet", "haiku"}
VALID_EFFORTS = {"low", "medium", "high", "max"}


def submit_claude_job(
    name=None,
    days=7,
    workdir=None,
    permission_mode="bypassPermissions",
    prompt=None,
    model=None,
    effort=None,
):
    """Submit a Claude Code job as a SLURM CPU job.

    Args:
        name: Session name (default: auto-generated).
        days: Job duration in days (default: 7).
        workdir: Working directory (default: cwd).
        permission_mode: Permission mode (default: bypassPermissions).
        prompt: If set, runs one-shot task (`claude -p`). If None, launches remote-control.
        model: Model to use (opus, sonnet, haiku). Only for one-shot mode.
        effort: Effort level (low, medium, high, max). Only for one-shot mode.

    Returns:
        (job_id, log_path) tuple.
    """
    name = name or f"claude-{int(time.time()) % 10000}"
    workdir = workdir or os.getcwd()
    time_str = cap_time_limit(f"{days}-0:00:00")
    log_dir = CLAUDE_LOG_DIR or "logs"
    log_path = os.path.join(log_dir, f"claude-{name}-%J.out")

    if prompt:
        # One-shot mode: claude -p "prompt"
        # Escape single quotes in prompt
        escaped = prompt.replace("'", "'\\''")
        cmd = f"claude -p '{escaped}' --permission-mode {permission_mode}"
        if model and model in VALID_MODELS:
            cmd += f" --model {model}"
        if effort and effort in VALID_EFFORTS:
            cmd += f" --effort {effort}"
    else:
        # Interactive mode: claude remote-control
        cmd = f"echo y | claude remote-control --name '{name}'"
        if permission_mode and permission_mode in VALID_PERMISSION_MODES:
            cmd += f" --permission-mode {permission_mode}"

    os.makedirs(log_dir, exist_ok=True)

    result = subprocess.run([
        "sbatch",
        f"--job-name=claude-{name}",
        f"--partition={CPU_PARTITION}", f"--qos={CPU_QOS}",
        f"--cpus-per-task={CPU_CPUS}", f"--mem={CPU_MEM}",
        f"--time={time_str}",
        f"--output={log_path}",
        f"--wrap=cd {workdir} && {cmd}",
    ], capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed: {result.stderr.strip()}")

    m = re.search(r"(\d+)", result.stdout)
    job_id = int(m.group(1)) if m else None
    actual_log = log_path.replace("%J", str(job_id)) if job_id else log_path

    return job_id, actual_log


def main():
    parser = argparse.ArgumentParser(description="Submit Claude as a SLURM CPU job")
    parser.add_argument("name", nargs="?", help="Session name")
    parser.add_argument("--days", type=int, default=7, help="Job duration in days (default: 7)")
    parser.add_argument("--workdir", help="Working directory (default: cwd)")
    parser.add_argument("--permission-mode", default="bypassPermissions",
                        choices=sorted(VALID_PERMISSION_MODES),
                        help="Permission mode (default: bypassPermissions)")
    parser.add_argument("--prompt", help="Run one-shot task instead of remote-control")
    parser.add_argument("--model", choices=sorted(VALID_MODELS), help="Model (one-shot only)")
    parser.add_argument("--effort", choices=sorted(VALID_EFFORTS), help="Effort level (one-shot only)")
    args = parser.parse_args()

    job_id, log_path = submit_claude_job(
        name=args.name, days=args.days,
        workdir=args.workdir, permission_mode=args.permission_mode,
        prompt=args.prompt, model=args.model, effort=args.effort,
    )
    mode = "one-shot task" if args.prompt else "remote-control"
    print(f"Submitted {mode} job {job_id}")
    print(f"Watch log: tail -f {log_path}")


if __name__ == "__main__":
    main()
