"""slurm_mcp — smart SLURM job submission for GPU clusters.

Usage:
    from slurm_mcp import submit_job, check_availability, select_gpu
"""

from __future__ import annotations

from config import (
    MAIL_USER, MAX_MEM_GB, TIME_LIMIT, START_TIMEOUT,
    CPU_PARTITION, CPU_QOS, CPU_MEM, CPU_CPUS,
    GOLDEN_QOS, EXCLUDE_NODES,
)

from .types import (
    GPUType, GPUAvailability, Availability, JobResult, JobStatus,
)
from .gpu_catalog import (
    GPU_TYPES, GPU_BY_NAME, GPU_TYPES_BY_QOS, PRIMARY_QOS,
)
from .shell import _run, _run_quiet
from .availability import check_availability, golden_queue, golden_queues
from .selection import select_gpu
from .submission import (
    submit_job, _build_sbatch_script, _wait_for_running,
)
from .jobs import my_jobs, cancel_jobs, squeue_me
from .monitoring import (
    get_job_status, read_job_log, wait_for_job,
    _FINISHED_STATES, _UNRECOVERABLE_REASONS,
    _QOS_QUOTA_REASONS, _USER_QUOTA_REASONS, _QUOTA_REASONS,
)
from .diagnostics import diagnose_job
from .history import job_history
