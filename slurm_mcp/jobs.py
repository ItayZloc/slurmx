"""Listing and cancelling user jobs."""

from __future__ import annotations

import os
import subprocess
from typing import Optional

from . import shell


def squeue_me() -> str:
    """Raw `squeue --me` text (default columns), rstripped.

    Returns an error string (not a raise) on failure so callers can print it
    verbatim. Shared by the one-shot `slurmx status` and the live TUI.
    """
    try:
        r = subprocess.run(["squeue", "--me"], capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            return f"squeue failed: {r.stderr.strip()}"
        return r.stdout.rstrip()
    except Exception as e:  # noqa: BLE001 — surface the message, don't crash
        return f"squeue error: {e}"


def my_jobs(qos: Optional[str] = None) -> list[dict]:
    """
    Return current user's jobs as a list of dicts.

    Each dict has: job_id, name, state, qos, gpu_gres, runtime, node, partition,
    reason. `reason` is the scheduler's pending reason (e.g. "Resources",
    "QOSMaxGRESPerAccount", "Priority") for PENDING jobs, "None" for RUNNING ones.
    """
    user = os.environ.get("USER", "")
    cmd = ["squeue", "-u", user, "-h", "-O",
           "JobId:12,Name:40,State:12,QOS:12,tres-per-node:50,TimeUsed:12,NodeList:25,Partition:20,Reason:30"]
    if qos:
        cmd.extend(["--qos", qos])

    raw = shell._run_quiet(cmd)
    jobs = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        job = {
            "job_id": line[0:12].strip(),
            "name": line[12:52].strip(),
            "state": line[52:64].strip(),
            "qos": line[64:76].strip(),
            "gpu_gres": line[76:126].strip(),
            "runtime": line[126:138].strip(),
            "node": line[138:163].strip(),
            "partition": line[163:183].strip(),
            "reason": line[183:].strip(),
        }
        jobs.append(job)
    return jobs


def cancel_jobs(
    job_ids: Optional[list[int]] = None,
    all_jobs: bool = False,
    pending_only: bool = False,
) -> int:
    """
    Cancel SLURM jobs.

    Args:
        job_ids: Specific job IDs to cancel.
        all_jobs: Cancel all jobs for current user.
        pending_only: Only cancel pending jobs.

    Returns:
        Number of jobs cancelled.
    """
    user = os.environ.get("USER", "")

    if job_ids:
        for jid in job_ids:
            shell._run_quiet(["scancel", str(jid)])
        return len(job_ids)

    if all_jobs:
        cmd = ["squeue", "-u", user, "-h"]
        if pending_only:
            cmd.extend(["-t", "PENDING"])
        raw = shell._run_quiet(cmd)
        count = len([l for l in raw.splitlines() if l.strip()])

        cancel_cmd = ["scancel", "-u", user]
        if pending_only:
            cancel_cmd.extend(["-t", "PENDING"])
        shell._run_quiet(cancel_cmd)
        return count

    return 0
