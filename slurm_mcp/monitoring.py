"""Job monitoring — squeue/sacct status, log reading, polling for completion."""

from __future__ import annotations

import glob as _glob
import os
import time
from typing import Optional

from . import shell
from .types import JobStatus


_FINISHED_STATES = {
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT",
    "NODE_FAIL", "PREEMPTED", "OUT_OF_MEMORY",
    "CANCELLED+",
}

_UNRECOVERABLE_REASONS = {
    "DependencyNeverSatisfied",
    "InvalidAccount",
    "InvalidQOS",
    "BadConstraints",
    "PartitionDown",
    "PartitionInactive",
}

_QOS_QUOTA_REASONS = {
    "QOSMaxGRESPerAccount",
    "QOSMaxResourceLimit",
    "AssocMaxGRESPerAccount",
    "AssocMaxJobsLimit",
    "PartitionNodeLimit",
}

_USER_QUOTA_REASONS = {
    "QOSMaxGRESPerUser",
    "QOSMaxJobsPerUserLimit",
}

_QUOTA_REASONS = _QOS_QUOTA_REASONS | _USER_QUOTA_REASONS


def get_job_status(job_id: int) -> JobStatus:
    """
    Get the current status of a SLURM job.

    Checks squeue first (fast, for running/pending jobs), then falls back
    to sacct (for completed/failed jobs).
    """
    raw = shell._run_quiet([
        "squeue", "-j", str(job_id), "-h", "-O",
        "JobId:12,State:20,NodeList:25,TimeUsed:15,Reason:40"
    ])
    for line in raw.splitlines():
        if not line.strip():
            continue
        state = line[12:32].strip()
        node = line[32:57].strip()
        elapsed = line[57:72].strip()
        reason = line[72:].strip()
        if reason == "None":
            reason = ""
        return JobStatus(
            job_id=job_id,
            state=state,
            node=node,
            elapsed=elapsed,
            reason=reason,
            finished=False,
        )

    raw = shell._run_quiet([
        "sacct", "-j", str(job_id), "--format=JobID,State,ExitCode,NodeList,Elapsed",
        "-n", "-P", "--noconvert",
    ])
    for line in raw.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 5:
            continue
        if parts[0] != str(job_id):
            continue
        state = parts[1].split()[0]
        exit_str = parts[2].split(":")[0]
        exit_code = int(exit_str) if exit_str.isdigit() else 0
        return JobStatus(
            job_id=job_id,
            state=state,
            exit_code=exit_code,
            node=parts[3],
            elapsed=parts[4],
            finished=state in _FINISHED_STATES,
        )

    return JobStatus(job_id=job_id, state="UNKNOWN", finished=False)


def read_job_log(
    job_id: int,
    output_dir: str = "logs",
    job_name: Optional[str] = None,
    tail: int = 0,
) -> Optional[str]:
    """
    Read the SLURM log file for a job.

    Searches for files matching slurm-*-{job_id}.out or slurm-{job_id}.out
    in output_dir.
    """
    patterns = [
        os.path.join(output_dir, f"slurm-*-{job_id}.out"),
        os.path.join(output_dir, f"slurm-{job_id}.out"),
        # Generic prefix — matches e.g. claude-<name>-<id>.out written by
        # launch_remote_session. The trailing `.out` anchor keeps short
        # job IDs from matching longer ones (12345 vs 123456789).
        os.path.join(output_dir, f"*-{job_id}.out"),
    ]
    if job_name:
        patterns.insert(0, os.path.join(output_dir, f"slurm-{job_name}-{job_id}.out"))

    for pattern in patterns:
        matches = _glob.glob(pattern)
        if matches:
            path = matches[0]
            with open(path) as f:
                content = f.read()
            if tail > 0:
                lines = content.splitlines()
                content = "\n".join(lines[-tail:])
            return content

    return None


def wait_for_job(
    job_id: int,
    poll_interval: int = 30,
    timeout: int = 0,
) -> JobStatus:
    """Block until a SLURM job finishes."""
    start = time.time()
    while True:
        status = get_job_status(job_id)
        if status.finished or status.state in _FINISHED_STATES:
            status.finished = True
            return status
        if timeout > 0 and (time.time() - start) >= timeout:
            return status
        time.sleep(poll_interval)
