"""Dataclasses for slurm_mcp — separate from logic so any module can import
without pulling SLURM dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GPUType:
    name: str
    display_name: str
    vram_gb: int
    golden_quota: int = 0
    golden_partition: Optional[str] = None


@dataclass
class GPUAvailability:
    gpu_type: str
    total: int
    used: int                                       # alias: running (kept for back-compat)
    free: int                                       # quota - running (pending is informational)
    users: dict = field(default_factory=dict)       # alias: running_users (kept for back-compat)
    running: int = 0
    pending: int = 0
    running_users: dict = field(default_factory=dict)
    pending_users: dict = field(default_factory=dict)


@dataclass
class Availability:
    golden: dict = field(default_factory=dict)         # primary QoS: gpu_type -> GPUAvailability
    cluster: dict = field(default_factory=dict)        # gpu_type -> GPUAvailability
    golden_by_qos: dict = field(default_factory=dict)  # qos -> {gpu_type -> GPUAvailability}


@dataclass
class JobResult:
    success: bool
    job_id: Optional[int]
    gpu_type: str
    partition: str
    qos: str
    message: str
    sbatch_script: str


@dataclass
class JobStatus:
    job_id: int
    state: str          # PENDING, RUNNING, COMPLETED, FAILED, CANCELLED, TIMEOUT, etc.
    exit_code: int = 0
    node: str = ""
    elapsed: str = ""
    reason: str = ""    # Pending reason (e.g., "Resources", "QOSMaxGRESPerAccount")
    finished: bool = False

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "state": self.state,
            "exit_code": self.exit_code,
            "node": self.node,
            "elapsed": self.elapsed,
            "reason": self.reason,
            "finished": self.finished,
        }
