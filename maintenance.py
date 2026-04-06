"""Cluster maintenance window enforcement.

Add (start, end) datetime tuples to WINDOWS to cap job time limits.
Set WINDOWS = [] when no maintenance is scheduled.
"""

import re
import subprocess
from datetime import datetime, timedelta
from typing import Optional


# --- Scheduled maintenance windows ---
# Each entry: (start_datetime, end_datetime)
WINDOWS = [
    # Add maintenance windows as (start, end) datetime tuples. Example:
    # (datetime(2026, 5, 1, 8, 0), datetime(2026, 5, 1, 20, 0)),
]

BUFFER = timedelta(minutes=15)  # jobs must finish this long before a window


class MaintenanceWindowError(Exception):
    """Raised when a job cannot be submitted due to cluster maintenance."""
    pass


def _parse_slurm_time(s: str) -> timedelta:
    """Parse 'D-HH:MM:SS' into timedelta."""
    days, hhmmss = s.split("-")
    h, m, sec = hhmmss.split(":")
    return timedelta(days=int(days), hours=int(h), minutes=int(m), seconds=int(sec))


def _format_slurm_time(td: timedelta) -> str:
    """Format timedelta as 'D-HH:MM:SS'."""
    total = int(td.total_seconds())
    d = total // 86400
    r = total % 86400
    return f"{d}-{r // 3600:02d}:{(r % 3600) // 60:02d}:{r % 60:02d}"


def _get_job_end_time(dependency: str) -> Optional[datetime]:
    """Query SLURM for the latest expected end time among parent jobs.

    Parses dependency strings like 'afterok:12345', 'afterany:111:222',
    'afterok:123,afterok:456'. Returns the latest EndTime, or None on failure.
    """
    job_ids = re.findall(r"\d+", dependency)
    if not job_ids:
        return None

    latest = None
    for jid in job_ids:
        try:
            result = subprocess.run(
                ["squeue", "-j", jid, "-h", "-o", "%e"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0 or not result.stdout.strip():
                continue
            end_str = result.stdout.strip()
            if end_str in ("N/A", "Unknown"):
                continue
            end_dt = datetime.strptime(end_str, "%Y-%m-%dT%H:%M:%S")
            if latest is None or end_dt > latest:
                latest = end_dt
        except (subprocess.TimeoutExpired, ValueError):
            continue

    return latest


def cap_time_limit(requested: str, dependency: Optional[str] = None) -> str:
    """Cap a SLURM time-limit string to fit before the next maintenance window.

    Args:
        requested: SLURM time string in D-HH:MM:SS format.
        dependency: SLURM dependency string (e.g. 'afterok:12345'). If set,
            the parent job's end time is queried to estimate when this job
            will actually start running.

    Returns:
        Possibly shortened SLURM time string.

    Raises:
        MaintenanceWindowError: If the cluster is currently in maintenance
            or less than 5 minutes remain before a window.
    """
    if not WINDOWS:
        return requested

    now = datetime.now()
    limit = _parse_slurm_time(requested)

    # Estimate when the job will start running
    earliest_start = now
    if dependency:
        parent_end = _get_job_end_time(dependency)
        if parent_end and parent_end > now:
            earliest_start = parent_end

    job_end = earliest_start + limit

    for start, end in WINDOWS:
        # Currently inside a maintenance window
        if start <= now < end:
            raise MaintenanceWindowError(
                f"Cluster maintenance in progress ({start:%b %d, %H:%M}–{end:%H:%M}). "
                "No jobs can be submitted."
            )

        # Job would overlap this window — cap it (with safety buffer)
        if earliest_start < start < job_end:
            available = start - BUFFER - earliest_start
            if available < timedelta(minutes=5):
                raise MaintenanceWindowError(
                    f"Less than 5 minutes of runtime before maintenance "
                    f"({start:%b %d, %H:%M}–{end:%H:%M}). No jobs can be submitted."
                )
            return _format_slurm_time(available)

    return requested
