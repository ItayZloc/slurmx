"""
slurm_utils — Smart SLURM job submission for GPU clusters.

Usage from a project:
    from slurm_utils import submit_job, check_availability, select_gpu

    # Auto-select GPU based on VRAM requirement
    result = submit_job(
        cmd="python train.py --lr 1e-4",
        vram_gb=48,
        job_name="train-bert",
    )
    print(f"Submitted job {result.job_id} on {result.gpu_type}")

    # Check availability
    avail = check_availability()
    print(f"Golden rtx_pro_6000 free: {avail.golden['rtx_pro_6000'].free}")
"""

from __future__ import annotations

import glob as _glob
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Optional

from config import (
    MAIL_USER, GOLDEN_QOS, MAX_MEM_GB, TIME_LIMIT, START_TIMEOUT,
    CPU_PARTITION, CPU_QOS, CPU_MEM, CPU_CPUS, GPU_DEFINITIONS,
)
from maintenance import cap_time_limit


# ============================================================
# GPU Definitions
# ============================================================

@dataclass
class GPUType:
    name: str
    display_name: str
    vram_gb: int
    golden_quota: int = 0
    golden_partition: Optional[str] = None

GPU_TYPES = [
    GPUType(name=d[0], display_name=d[1], vram_gb=d[2],
            golden_quota=d[3], golden_partition=d[4])
    for d in GPU_DEFINITIONS
]

GPU_BY_NAME = {g.name: g for g in GPU_TYPES}


# ============================================================
# Data Structures
# ============================================================

@dataclass
class GPUAvailability:
    gpu_type: str
    total: int
    used: int
    free: int
    users: dict = field(default_factory=dict)


@dataclass
class Availability:
    golden: dict = field(default_factory=dict)   # gpu_type -> GPUAvailability
    cluster: dict = field(default_factory=dict)  # gpu_type -> GPUAvailability


@dataclass
class JobResult:
    success: bool
    job_id: Optional[int]
    gpu_type: str
    partition: str
    qos: str
    message: str
    sbatch_script: str


# ============================================================
# SLURM Command Helpers
# ============================================================

def _run(cmd: list[str]) -> str:
    """Run a command and return stdout. Raises on failure."""
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")
    return result.stdout


def _run_quiet(cmd: list[str]) -> str:
    """Run a command, return stdout. Returns empty string on failure."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.stdout if result.returncode == 0 else ""
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


# ============================================================
# Availability Checking
# ============================================================

def check_availability() -> Availability:
    """
    Query GPU availability — both golden tickets and cluster-wide.

    Returns an Availability object with .golden and .cluster dicts,
    each keyed by GPU type name.
    """
    avail = Availability()

    # --- Golden ticket availability (yisroel QoS) ---
    golden_used = {}   # gpu_type -> total GPUs used
    golden_users = {}  # gpu_type -> {user: count}

    raw = _run_quiet([
        "squeue", "--qos", GOLDEN_QOS, "-h", "-O",
        "UserName:20,tres-per-job:60,tres-per-node:60,State:12"
    ])

    for line in raw.splitlines():
        if not line.strip():
            continue

        user = line[0:20].strip()
        tres_job = line[20:80].strip()
        tres_node = line[80:140].strip()
        state = line[140:152].strip()

        if state != "RUNNING":
            continue

        # Find GPU info from whichever TRES field has it
        gres_field = ""
        for f in (tres_job, tres_node):
            if f != "N/A" and "gres/gpu:" in f:
                gres_field = f
                break
        if not gres_field:
            continue

        # Parse gres/gpu:TYPE:COUNT
        m = re.search(r"gres/gpu:([^:,]+):(\d+)", gres_field)
        if not m:
            continue

        gpu_type = m.group(1)
        gpu_count = int(m.group(2))

        golden_used[gpu_type] = golden_used.get(gpu_type, 0) + gpu_count
        if gpu_type not in golden_users:
            golden_users[gpu_type] = {}
        golden_users[gpu_type][user] = golden_users[gpu_type].get(user, 0) + gpu_count

    # Build golden availability
    for gpu in GPU_TYPES:
        if gpu.golden_quota > 0:
            used = golden_used.get(gpu.name, 0)
            avail.golden[gpu.name] = GPUAvailability(
                gpu_type=gpu.name,
                total=gpu.golden_quota,
                used=used,
                free=gpu.golden_quota - used,
                users=golden_users.get(gpu.name, {}),
            )

    # --- Cluster-wide availability (all nodes) ---
    cluster_total = {}  # gpu_type -> total count
    cluster_alloc = {}  # gpu_type -> allocated count

    raw = _run_quiet([
        "sinfo", "-N", "-h", "-O",
        "NodeHost:20,Gres:40,GresUsed:40,StateLong:20"
    ])

    seen_nodes = set()
    for line in raw.splitlines():
        if not line.strip():
            continue

        node = line[0:20].strip()
        gres_str = line[20:60].strip()
        gres_used_str = line[60:100].strip()
        state = line[100:].strip()

        # Deduplicate (nodes appear once per partition)
        if node in seen_nodes:
            continue
        seen_nodes.add(node)

        # Skip down/drained nodes
        if "down" in state or "drain" in state:
            continue

        # Total GPUs: gpu:TYPE:COUNT or gpu:TYPE:COUNT(S:0-1)
        if gres_str and gres_str != "(null)" and "gpu:" in gres_str:
            m = re.search(r"gpu:([^:]+):(\d+)", gres_str)
            if m:
                gtype = m.group(1)
                gcount = int(m.group(2))
                cluster_total[gtype] = cluster_total.get(gtype, 0) + gcount

        # Used GPUs: gpu:TYPE:COUNT(IDX:...) or gpu:TYPE:0
        if gres_used_str and gres_used_str != "(null)" and "gpu:" in gres_used_str:
            m = re.search(r"gpu:([^:]+):(\d+)", gres_used_str)
            if m:
                gtype = m.group(1)
                gcount = int(m.group(2))
                cluster_alloc[gtype] = cluster_alloc.get(gtype, 0) + gcount

    for gpu in GPU_TYPES:
        total = cluster_total.get(gpu.name, 0)
        alloc = cluster_alloc.get(gpu.name, 0)
        free = max(0, total - alloc)
        avail.cluster[gpu.name] = GPUAvailability(
            gpu_type=gpu.name,
            total=total,
            used=alloc,
            free=free,
        )

    return avail


# ============================================================
# GPU Selection
# ============================================================

def select_gpu(vram_gb: int) -> Optional[tuple[str, str, str]]:
    """
    Pick the best available GPU for the given VRAM requirement.

    Strategy:
      1. Filter GPU types where vram >= vram_gb, sort cheapest first
      2. Try golden tickets first (smallest fitting golden GPU)
      3. Fall back to normal QoS cluster-wide (smallest first)
      4. Return None if nothing available

    Returns:
        (gpu_type, partition, qos) or None
    """
    candidates = [g for g in GPU_TYPES if g.vram_gb >= vram_gb]
    candidates.sort(key=lambda g: (g.vram_gb, g.name))

    if not candidates:
        return None  # No GPU type has enough VRAM

    avail = check_availability()

    # Phase 1: Golden tickets (smallest fitting golden GPU first)
    golden_candidates = [g for g in candidates if g.golden_quota > 0]
    for gpu in golden_candidates:
        golden = avail.golden.get(gpu.name)
        if golden and golden.free > 0:
            return (gpu.name, gpu.golden_partition, GOLDEN_QOS)

    # Phase 2: Normal QoS cluster-wide (smallest first)
    for gpu in candidates:
        cluster = avail.cluster.get(gpu.name)
        if cluster and cluster.free > 0:
            return (gpu.name, "main", "normal")

    # Phase 3: Nothing available
    return None


# ============================================================
# Job Submission
# ============================================================

_CPU_PARTITION = CPU_PARTITION
_CPU_QOS = CPU_QOS
_CPU_MEM = CPU_MEM
_CPU_CPUS = CPU_CPUS


def _build_sbatch_script(
    cmd: str,
    partition: str,
    qos: str,
    gpu_type: str,
    num_gpus: int,
    job_name: str,
    output_path: str,
    workdir: Optional[str],
    dependency: Optional[str] = None,
) -> str:
    """Generate a sbatch script following the lab's conventions."""
    is_cpu = not gpu_type

    lines = [
        "#!/bin/bash",
        "",
        "### --- Slurm Job Configuration ---",
        "",
        f"#SBATCH --partition {partition}",
        f"#SBATCH --qos={qos}",
        f"#SBATCH --time {cap_time_limit(TIME_LIMIT, dependency)}",
        f"#SBATCH --job-name {job_name}",
        f"#SBATCH --output {output_path}",
    ]

    if is_cpu:
        lines.append(f"#SBATCH --cpus-per-task={_CPU_CPUS}")
        lines.append(f"#SBATCH --mem={_CPU_MEM}")
    else:
        lines.append(f"#SBATCH --gres=gpu:{gpu_type}:{num_gpus}")
        lines.append(f"#SBATCH --nodes=1")
        lines.append(f"#SBATCH --mem={MAX_MEM_GB}G")

    if dependency:
        lines.append(f"#SBATCH --dependency={dependency}")

    lines += [
        "",
        f"#SBATCH --mail-user={MAIL_USER}",
        "#SBATCH --mail-type=ALL",
        "",
        "################ Following lines will be executed by the compute node ################",
        "",
        "# --- Scratch directory (fallback to /tmp if /scratch unavailable) ---",
        "export SCRATCH_DIR=/scratch/$USER/$SLURM_JOB_ID",
        'mkdir -p "$SCRATCH_DIR" 2>/dev/null || { export SCRATCH_DIR=/tmp/$USER/slurm_$SLURM_JOB_ID; mkdir -p "$SCRATCH_DIR"; }',
        "trap 'rm -rf \"$SCRATCH_DIR\"' EXIT",
        "",
    ]

    # Ensure log output directory exists
    log_dir = os.path.dirname(output_path)
    if log_dir and log_dir != ".":
        lines.append(f'mkdir -p "{log_dir}"')
        lines.append("")

    if workdir:
        lines.append(f"cd {workdir}")
        lines.append("")

    lines.append(cmd)
    lines.append("")
    return "\n".join(lines)


def _wait_for_running(job_result: JobResult, timeout: int) -> tuple[JobResult, str]:
    """
    Poll a submitted job until it reaches RUNNING state or hits an issue.

    Returns:
        (job_result, outcome) where outcome is one of:
        - "running": job is running
        - "finished": job ended before running
        - "fatal": unrecoverable error, job cancelled
        - "quota": QoS/account quota hit, job cancelled (caller should fallback)
        - "user_quota": per-user GPU limit hit, job cancelled (no fallback possible)
        - "still_pending": timeout reached but job is still queued (NOT cancelled)
    """
    poll_interval = 5
    start = time.time()

    while True:
        status = get_job_status(job_result.job_id)

        if status.state == "RUNNING":
            job_result.message = (
                f"Job {job_result.job_id} is RUNNING on {status.node}"
            )
            return job_result, "running"

        if status.state in _FINISHED_STATES:
            job_result.success = False
            job_result.message = (
                f"Job {job_result.job_id} ended before running: "
                f"{status.state} (exit_code={status.exit_code})"
            )
            return job_result, "finished"

        if status.reason in _UNRECOVERABLE_REASONS:
            _run_quiet(["scancel", str(job_result.job_id)])
            job_result.success = False
            job_result.message = (
                f"Job {job_result.job_id} cancelled — fatal error: "
                f"{status.reason}"
            )
            return job_result, "fatal"

        if status.reason in _USER_QUOTA_REASONS:
            _run_quiet(["scancel", str(job_result.job_id)])
            job_result.success = False
            job_result.message = (
                f"Job {job_result.job_id} cancelled — per-user limit: "
                f"{status.reason}. You have too many GPUs allocated "
                f"across all partitions. Wait for running jobs to "
                f"finish or cancel some."
            )
            return job_result, "user_quota"

        if status.reason in _QOS_QUOTA_REASONS:
            _run_quiet(["scancel", str(job_result.job_id)])
            job_result.success = False
            job_result.message = (
                f"Job {job_result.job_id} cancelled — quota limit: "
                f"{status.reason}"
            )
            return job_result, "quota"

        elapsed = time.time() - start
        if timeout > 0 and elapsed >= timeout:
            # Job is still pending with a benign reason (e.g., Resources).
            # Do NOT cancel — it will eventually run.
            reason_str = status.reason or "unknown"
            job_result.message = (
                f"Job {job_result.job_id} still pending after {int(elapsed)}s "
                f"(reason: {reason_str}). Job remains queued."
            )
            return job_result, "still_pending"

        time.sleep(poll_interval)


def _do_submit(
    script: str,
    gpu_type: str,
    partition: str,
    qos: str,
) -> JobResult:
    """Write script to temp file, submit via sbatch, return JobResult."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".sh", prefix="slurm-submit-",
        dir="/tmp", delete=False
    ) as f:
        f.write(script)
        tmpfile = f.name

    try:
        os.chmod(tmpfile, 0o755)
        result = _run(["sbatch", tmpfile])
        m = re.search(r"(\d+)", result)
        job_id = int(m.group(1)) if m else None
        return JobResult(
            success=True,
            job_id=job_id,
            gpu_type=gpu_type,
            partition=partition,
            qos=qos,
            message=result.strip(),
            sbatch_script=script,
        )
    except RuntimeError as e:
        return JobResult(
            success=False,
            job_id=None,
            gpu_type=gpu_type,
            partition=partition,
            qos=qos,
            message=str(e),
            sbatch_script=script,
        )
    finally:
        os.unlink(tmpfile)


def submit_job(
    cmd: str,
    vram_gb: int,
    job_name: Optional[str] = None,
    num_gpus: int = 1,
    workdir: Optional[str] = None,
    output_dir: str = "logs",
    gpu_type: Optional[str] = None,
    qos: Optional[str] = None,
    dependency: Optional[str] = None,
    wait_until_running: bool = True,
    dry_run: bool = False,
) -> JobResult:
    """
    Smart job submission. Auto-selects GPU based on VRAM if gpu_type not specified.

    Every job automatically gets a scratch directory at /scratch/$USER/$SLURM_JOB_ID,
    exported as $SCRATCH_DIR. It is cleaned up when the job finishes.

    Args:
        cmd: Command to run on the compute node (e.g., "python train.py --lr 1e-4")
        vram_gb: GPU VRAM needed in GB
        job_name: Job name (default: first word of cmd)
        num_gpus: Number of GPUs (default: 1)
        workdir: Working directory on compute node
        output_dir: Directory for SLURM log files (default: "logs")
        gpu_type: Override automatic GPU selection
        qos: Override automatic QoS selection
        dependency: Job dependency expression (e.g., "afterok:12345", "afterany:111:222")
        wait_until_running: If True, poll until the job reaches RUNNING state.
            On quota limits (e.g., golden tickets full), auto-cancels and
            retries on normal QoS. On benign pending (e.g., Resources),
            returns without cancelling — job stays queued. Only cancels
            on truly unrecoverable errors. (default: True)
        dry_run: If True, return the script without submitting

    Returns:
        JobResult with success status, job_id, and the generated script.
    """
    # Default job name from command
    if not job_name:
        first_word = cmd.strip().split()[0] if cmd.strip() else "job"
        job_name = os.path.basename(first_word).replace(".", "-")

    # Hard cap: cluster policy limits to 2 GPUs per job
    if num_gpus > 2:
        return JobResult(False, None, "", "", "",
                         f"num_gpus={num_gpus} exceeds cluster limit of 2 GPUs per job.", "")

    # CPU-only job (vram_gb=0)
    if vram_gb == 0:
        selected_gpu = ""
        selected_partition = _CPU_PARTITION
        selected_qos = _CPU_QOS

        output_path = os.path.join(output_dir, f"slurm-{job_name}-%J.out")
        script = _build_sbatch_script(
            cmd=cmd,
            partition=selected_partition,
            qos=selected_qos,
            gpu_type=selected_gpu,
            num_gpus=0,
            job_name=job_name,
            output_path=output_path,
            workdir=workdir,
            dependency=dependency,
        )

        if dry_run:
            return JobResult(
                success=True, job_id=None,
                gpu_type="cpu", partition=selected_partition,
                qos=selected_qos,
                message=f"[DRY RUN] Would submit CPU job to {selected_partition}/{selected_qos}",
                sbatch_script=script,
            )

        job_result = _do_submit(script, "cpu", selected_partition, selected_qos)
        if not job_result.success:
            return job_result

        if wait_until_running and job_result.job_id is not None:
            job_result, outcome = _wait_for_running(job_result, START_TIMEOUT)

        return job_result

    # GPU selection
    if gpu_type:
        # Manual override — validate
        if gpu_type not in GPU_BY_NAME:
            return JobResult(False, None, gpu_type, "", "",
                             f"Unknown GPU type: {gpu_type}. "
                             f"Valid: {', '.join(GPU_BY_NAME.keys())}", "")

        gpu_info = GPU_BY_NAME[gpu_type]
        if gpu_info.vram_gb < vram_gb:
            return JobResult(False, None, gpu_type, "", "",
                             f"{gpu_type} has {gpu_info.vram_gb}GB VRAM, "
                             f"but {vram_gb}GB requested", "")

        # Determine partition/QoS for manual override
        if qos:
            selected_qos = qos
        elif gpu_info.golden_quota > 0:
            selected_qos = GOLDEN_QOS
        else:
            selected_qos = "normal"

        if selected_qos == GOLDEN_QOS and gpu_info.golden_partition:
            selected_partition = gpu_info.golden_partition
        else:
            selected_partition = "main"

        selected_gpu = gpu_type
    else:
        # Auto-select based on VRAM
        selection = select_gpu(vram_gb)
        if selection is None:
            # Build a helpful message
            capable_gpus = [g for g in GPU_TYPES if g.vram_gb >= vram_gb]
            if not capable_gpus:
                max_gpu = max(GPU_TYPES, key=lambda g: g.vram_gb)
                msg_parts = [
                    f"No GPU type has >= {vram_gb}GB VRAM.",
                    f"Maximum available: {max_gpu.vram_gb}GB ({max_gpu.name}).",
                ]
            else:
                avail = check_availability()
                msg_parts = [f"No GPU with >= {vram_gb}GB VRAM is currently free."]
                msg_parts.append("")
                msg_parts.append("Current availability:")
                for gpu in capable_gpus:
                    golden = avail.golden.get(gpu.name)
                    cluster = avail.cluster.get(gpu.name)
                    golden_str = f"golden: {golden.free}/{golden.total}" if golden else "no golden"
                    cluster_str = f"cluster: {cluster.free}/{cluster.total}" if cluster else "N/A"
                    msg_parts.append(f"  {gpu.name} ({gpu.vram_gb}GB): {golden_str}, {cluster_str}")

            return JobResult(False, None, "", "", "", "\n".join(msg_parts), "")

        selected_gpu, selected_partition, selected_qos = selection

    # Override QoS if explicitly set
    if qos and not gpu_type:
        selected_qos = qos
        gpu_info = GPU_BY_NAME[selected_gpu]
        if selected_qos == GOLDEN_QOS and gpu_info.golden_partition:
            selected_partition = gpu_info.golden_partition
        else:
            selected_partition = "main"

    # Build output path
    output_path = os.path.join(output_dir, f"slurm-{job_name}-%J.out")

    # Build sbatch script
    script = _build_sbatch_script(
        cmd=cmd,
        partition=selected_partition,
        qos=selected_qos,
        gpu_type=selected_gpu,
        num_gpus=num_gpus,
        job_name=job_name,
        output_path=output_path,
        workdir=workdir,
        dependency=dependency,
    )

    if dry_run:
        return JobResult(
            success=True,
            job_id=None,
            gpu_type=selected_gpu,
            partition=selected_partition,
            qos=selected_qos,
            message=f"[DRY RUN] Would submit to {selected_gpu} "
                    f"(partition={selected_partition}, qos={selected_qos})",
            sbatch_script=script,
        )

    # Submit and wait
    job_result = _do_submit(script, selected_gpu, selected_partition, selected_qos)
    if not job_result.success:
        return job_result

    if wait_until_running and job_result.job_id is not None:
        job_result, outcome = _wait_for_running(job_result, START_TIMEOUT)

        # Per-user GPU limit — no fallback possible, return immediately
        if outcome == "user_quota":
            return job_result

        # Quota hit on golden QoS — fallback to normal QoS on main partition
        if outcome == "quota" and selected_qos == GOLDEN_QOS:
            fallback_script = _build_sbatch_script(
                cmd=cmd, partition="main", qos="normal",
                gpu_type=selected_gpu, num_gpus=num_gpus,
                job_name=job_name, output_path=output_path,
                workdir=workdir, dependency=dependency,
            )
            job_result = _do_submit(
                fallback_script, selected_gpu, "main", "normal",
            )
            if not job_result.success:
                return job_result

            if job_result.job_id is not None:
                job_result, outcome = _wait_for_running(
                    job_result, START_TIMEOUT,
                )

    return job_result


# ============================================================
# Job Management
# ============================================================

def my_jobs(qos: Optional[str] = None) -> list[dict]:
    """
    Return current user's jobs as a list of dicts.

    Each dict has: job_id, name, state, qos, gpu_gres, runtime, node, partition.
    """
    user = os.environ.get("USER", "")
    cmd = ["squeue", "-u", user, "-h", "-O",
           "JobId:12,Name:40,State:12,QOS:12,tres-per-node:50,TimeUsed:12,NodeList:25,Partition:20"]
    if qos:
        cmd.extend(["--qos", qos])

    raw = _run_quiet(cmd)
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
            "partition": line[163:].strip(),
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
            _run_quiet(["scancel", str(jid)])
        return len(job_ids)

    if all_jobs:
        # Count first
        cmd = ["squeue", "-u", user, "-h"]
        if pending_only:
            cmd.extend(["-t", "PENDING"])
        raw = _run_quiet(cmd)
        count = len([l for l in raw.splitlines() if l.strip()])

        cancel_cmd = ["scancel", "-u", user]
        if pending_only:
            cancel_cmd.extend(["-t", "PENDING"])
        _run_quiet(cancel_cmd)
        return count

    return 0


# ============================================================
# Job Monitoring
# ============================================================

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


_FINISHED_STATES = {
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT",
    "NODE_FAIL", "PREEMPTED", "OUT_OF_MEMORY",
    "CANCELLED+",  # sacct sometimes reports this
}

# Truly unrecoverable — no retry or fallback can fix these.
_UNRECOVERABLE_REASONS = {
    "DependencyNeverSatisfied",
    "InvalidAccount",
    "InvalidQOS",
    "BadConstraints",
    "PartitionDown",
    "PartitionInactive",
}

# Quota/resource limits — job can't run on THIS QoS, but may work on another.
_QOS_QUOTA_REASONS = {
    "QOSMaxGRESPerAccount",
    "QOSMaxResourceLimit",
    "AssocMaxGRESPerAccount",
    "AssocMaxJobsLimit",
    "PartitionNodeLimit",
}

# Per-user limits — applies across ALL QoS levels, fallback won't help.
_USER_QUOTA_REASONS = {
    "QOSMaxGRESPerUser",
    "QOSMaxJobsPerUserLimit",
}

# Combined set for detection in _wait_for_running.
_QUOTA_REASONS = _QOS_QUOTA_REASONS | _USER_QUOTA_REASONS


def get_job_status(job_id: int) -> JobStatus:
    """
    Get the current status of a SLURM job.

    Checks squeue first (fast, for running/pending jobs), then falls back
    to sacct (for completed/failed jobs).
    """
    # Try squeue first (works for queued/running jobs)
    raw = _run_quiet([
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

    # Not in squeue — check sacct for finished jobs
    raw = _run_quiet([
        "sacct", "-j", str(job_id), "--format=JobID,State,ExitCode,NodeList,Elapsed",
        "-n", "-P", "--noconvert",
    ])
    for line in raw.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 5:
            continue
        # Only match the main job line (not .batch, .extern, etc.)
        if parts[0] != str(job_id):
            continue
        state = parts[1].split()[0]  # strip trailing modifiers
        exit_str = parts[2].split(":")[0]  # "0:0" -> "0"
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

    Searches for files matching the pattern slurm-*-{job_id}.out or slurm-{job_id}.out
    in output_dir.

    Args:
        job_id: The SLURM job ID.
        output_dir: Directory to search for log files (default: "logs").
        job_name: Optional job name to narrow the search.
        tail: If > 0, return only the last N lines.

    Returns:
        Log file contents as a string, or None if not found.
    """
    # Try common patterns
    patterns = [
        os.path.join(output_dir, f"slurm-*-{job_id}.out"),
        os.path.join(output_dir, f"slurm-{job_id}.out"),
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
    """
    Block until a SLURM job finishes.

    Args:
        job_id: The SLURM job ID to wait for.
        poll_interval: Seconds between status checks (default: 30).
        timeout: Maximum seconds to wait. 0 = no limit.

    Returns:
        Final JobStatus once the job has finished or timeout is reached.
    """
    start = time.time()
    while True:
        status = get_job_status(job_id)
        if status.finished or status.state in _FINISHED_STATES:
            status.finished = True
            return status
        if timeout > 0 and (time.time() - start) >= timeout:
            return status
        time.sleep(poll_interval)
