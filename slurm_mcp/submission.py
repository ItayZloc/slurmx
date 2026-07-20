"""Job submission — sbatch script generation, submit + poll, golden -> normal
QoS fallback when quota hits."""

from __future__ import annotations

import os
import re
import tempfile
import time
from typing import Optional

from config import (
    CPU_PARTITION, CPU_QOS, CPU_MEM, CPU_CPUS,
    EXCLUDE_NODES, MAIL_USER, MAX_MEM_GB, START_TIMEOUT, TIME_LIMIT,
)
from maintenance import cap_time_limit

from . import availability, monitoring, selection, shell
from .gpu_catalog import GPU_BY_NAME, GPU_TYPES, PRIMARY_QOS
from .monitoring import (
    _FINISHED_STATES, _QOS_QUOTA_REASONS,
    _UNRECOVERABLE_REASONS, _USER_QUOTA_REASONS,
)
from .types import JobResult


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

    if EXCLUDE_NODES:
        lines.append(f"#SBATCH --exclude={','.join(EXCLUDE_NODES)}")

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


def _wait_for_running(
    job_result: JobResult, timeout: int, golden_only: bool = False,
) -> tuple[JobResult, str]:
    """
    Poll a submitted job until it reaches RUNNING state or hits an issue.

    Outcomes: "running", "finished", "fatal", "quota", "user_quota", "still_pending".

    When golden_only=True, quota-limit reasons (golden partition full) do NOT
    cancel the job — it is left queued and reported as "still_pending" so it
    starts automatically when a golden slot frees. Unrecoverable reasons
    (e.g. InvalidQOS) still cancel as usual.
    """
    poll_interval = 5
    start = time.time()

    while True:
        status = monitoring.get_job_status(job_result.job_id)

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
            shell._run_quiet(["scancel", str(job_result.job_id)])
            job_result.success = False
            job_result.message = (
                f"Job {job_result.job_id} cancelled — fatal error: "
                f"{status.reason}"
            )
            return job_result, "fatal"

        if status.reason in _USER_QUOTA_REASONS and not golden_only:
            shell._run_quiet(["scancel", str(job_result.job_id)])
            job_result.success = False
            job_result.message = (
                f"Job {job_result.job_id} cancelled — per-user limit: "
                f"{status.reason}. You have too many GPUs allocated "
                f"across all partitions. Wait for running jobs to "
                f"finish or cancel some."
            )
            return job_result, "user_quota"

        if status.reason in _QOS_QUOTA_REASONS and not golden_only:
            shell._run_quiet(["scancel", str(job_result.job_id)])
            job_result.success = False
            job_result.message = (
                f"Job {job_result.job_id} cancelled — quota limit: "
                f"{status.reason}"
            )
            return job_result, "quota"

        elapsed = time.time() - start
        if timeout > 0 and elapsed >= timeout:
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
        result = shell._run(["sbatch", tmpfile])
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
    golden_only: bool = True,
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
        golden_only: If True, force the golden ticket — qos=yisroel on the card's
            dedicated per-GPU partition (preemption-immune) — and NEVER fall back
            to the preemptible main pool. The job is left queued if the golden
            partition is saturated (starts automatically as slots free) instead of
            being downgraded. Overrides `qos`. Ignored for CPU jobs (vram_gb=0).
            (default: True. Pass False for the golden-first-then-main fallback.)
        dependency: Job dependency expression (e.g., "afterok:12345")
        wait_until_running: If True, poll until the job reaches RUNNING state.
            On quota limits (e.g., golden tickets full), auto-cancels and
            retries on normal QoS. On benign pending (e.g., Resources),
            returns without cancelling — job stays queued. Only cancels
            on truly unrecoverable errors. (default: True)
        dry_run: If True, return the script without submitting

    Returns:
        JobResult with success status, job_id, and the generated script.
    """
    if not job_name:
        first_word = cmd.strip().split()[0] if cmd.strip() else "job"
        job_name = os.path.basename(first_word).replace(".", "-")

    if num_gpus > 2:
        return JobResult(False, None, "", "", "",
                         f"num_gpus={num_gpus} exceeds cluster limit of 2 GPUs per job.", "")

    # CPU-only job (vram_gb=0 and no explicit GPU requested). An explicit
    # gpu_type means the caller wants that card even if vram_gb was left at 0,
    # so fall through to the GPU path in that case.
    if vram_gb == 0 and not gpu_type:
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
        if gpu_type not in GPU_BY_NAME:
            return JobResult(False, None, gpu_type, "", "",
                             f"Unknown GPU type: {gpu_type}. "
                             f"Valid: {', '.join(GPU_BY_NAME.keys())}", "")

        gpu_info = GPU_BY_NAME[gpu_type]
        if gpu_info.vram_gb < vram_gb:
            return JobResult(False, None, gpu_type, "", "",
                             f"{gpu_type} has {gpu_info.vram_gb}GB VRAM, "
                             f"but {vram_gb}GB requested", "")

        if golden_only:
            if not gpu_info.golden_partition:
                return JobResult(False, None, gpu_type, "", "",
                                 f"golden_only=True but {gpu_type} has no golden "
                                 f"partition configured.", "")
            selected_qos = PRIMARY_QOS
            selected_partition = gpu_info.golden_partition
        else:
            if qos:
                selected_qos = qos
            elif gpu_info.golden_quota > 0:
                selected_qos = PRIMARY_QOS
            else:
                selected_qos = "normal"

            if selected_qos == PRIMARY_QOS and gpu_info.golden_partition:
                selected_partition = gpu_info.golden_partition
            else:
                selected_partition = "main"

        selected_gpu = gpu_type
    else:
        sel = selection.select_gpu(vram_gb, golden_only=golden_only)
        if sel is None:
            capable_gpus = [g for g in GPU_TYPES if g.vram_gb >= vram_gb]
            if not capable_gpus:
                max_gpu = max(GPU_TYPES, key=lambda g: g.vram_gb)
                msg_parts = [
                    f"No GPU type has >= {vram_gb}GB VRAM.",
                    f"Maximum available: {max_gpu.vram_gb}GB ({max_gpu.name}).",
                ]
            else:
                avail = availability.check_availability()
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

        selected_gpu, selected_partition, selected_qos = sel

    if qos and not gpu_type and not golden_only:
        selected_qos = qos
        gpu_info = GPU_BY_NAME[selected_gpu]
        if selected_qos == PRIMARY_QOS and gpu_info.golden_partition:
            selected_partition = gpu_info.golden_partition
        else:
            selected_partition = "main"

    output_path = os.path.join(output_dir, f"slurm-{job_name}-%J.out")

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

    job_result = _do_submit(script, selected_gpu, selected_partition, selected_qos)
    if not job_result.success:
        return job_result

    if wait_until_running and job_result.job_id is not None:
        job_result, outcome = _wait_for_running(
            job_result, START_TIMEOUT, golden_only=golden_only,
        )

        if outcome == "user_quota":
            return job_result

        if outcome == "quota" and selected_qos == PRIMARY_QOS and not golden_only:
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
