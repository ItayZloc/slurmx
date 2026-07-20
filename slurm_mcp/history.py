"""Recent-job history from SLURM accounting (sacct).

Shared by the `job_history` MCP tool (server.py) and the `slurmx history` CLI.
Unlike my_jobs() (running/pending only), this shows finished jobs too.
"""

from __future__ import annotations

import os
import re
import subprocess
from typing import Optional


def job_history(days: int = 3, state: Optional[str] = None, limit: int = 30) -> str:
    """Return a formatted table of recent completed/failed jobs from sacct.

    Args:
        days: Number of days of history (default: 3).
        state: Filter by state: COMPLETED, FAILED, TIMEOUT, OOM, CANCELLED, or
            None for all.
        limit: Max jobs to return, most recent first (default: 30).
    """
    user = os.environ.get("USER", "")
    cmd = [
        "sacct",
        "--starttime", f"now-{days}days",
        "--format=JobID,JobName%30,State%20,ExitCode,Elapsed,AllocTRES%50,NodeList%20,Start",
        "-n", "-P", "--noconvert",
        "--user", user,
    ]
    if state:
        state_map = {"OOM": "OUT_OF_MEMORY"}
        cmd.extend(["--state", state_map.get(state.upper(), state.upper())])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return f"sacct query failed: {result.stderr.strip()}"
        raw = result.stdout
    except Exception as e:
        return f"sacct query failed: {e}"

    # Parse — only main job lines (plain integer JobID, not .batch/.extern)
    rows = []
    for line in raw.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 8:
            continue
        if not parts[0].isdigit():
            continue
        gpu = ""
        m = re.search(r"gres/gpu:([^:,]+:\d+)", parts[5])
        if m:
            gpu = m.group(1)
        rows.append({
            "job_id": parts[0],
            "name": parts[1],
            "state": parts[2].split()[0],
            "exit": parts[3],
            "elapsed": parts[4],
            "gpu": gpu,
            "node": parts[6],
        })

    if not rows:
        return f"No jobs found in the last {days} day(s)." + (f" (filter: {state})" if state else "")

    # Most recent first (rows come sorted by start time ascending)
    rows.reverse()
    rows = rows[:limit]

    header = f"{'JOB_ID':<12} {'NAME':<30} {'STATE':<16} {'EXIT':<6} {'ELAPSED':<12} {'GPU':<20} {'NODE'}"
    lines = [f"Recent jobs (last {days} day(s)):", header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r['job_id']:<12} {r['name']:<30} {r['state']:<16} {r['exit']:<6} "
            f"{r['elapsed']:<12} {r['gpu']:<20} {r['node']}"
        )
    lines.append(f"\n{len(rows)} job(s) shown.")
    return "\n".join(lines)
