"""
MCP server exposing slurm_mcp tools to Claude Code.

Run: uv run --with "mcp[cli]" python ~/.claude/mcp-servers/slurmx/server.py
"""

from __future__ import annotations

import json
import sys
import os
from typing import Literal

# Ensure slurm_mcp, cli, config, maintenance are importable from same directory
sys.path.insert(0, os.path.dirname(__file__))

import slurm_mcp
from cli import render
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("slurmx", instructions="""\
## SLURM Job Submission Rules

When the user says "job" or "a job running", they mean a SLURM job — not a local process.
Check with MCP cluster_summary or squeue, not ps aux.

### Key Rules
- Always use dry_run=true first to preview the sbatch script before submitting.
- For submit_job: always specify vram_gb — never pick GPU types manually unless
  the user asks.
- Default to 1 GPU. Use num_gpus=2 only when the user explicitly requests multi-GPU
  or the workload requires it (e.g. model too large for a single card).
  Max 2 GPUs per cluster policy — requesting more raises an error.
- Golden-only is the DEFAULT (golden_only=true): jobs run preemption-immune on the
  golden partition and queue there if it's full, never dropping to the preemptible
  main pool. Pass golden_only=false only when the user explicitly wants to burst onto
  the shared main pool. Ignored for CPU jobs.
- For multi-GPU training, use `torchrun --nproc_per_node=2 train.py` as the command.
- Maintenance windows are enforced automatically — job time limits are capped to finish
  before scheduled maintenance. If a window is imminent (<5 min), submissions are blocked.
- Do NOT write raw sbatch scripts or Python code that imports slurm_mcp — always use MCP tools.
- Use cluster_summary as the single dashboard tool: it covers jobs AND GPU availability.
  Use view="jobs" or view="gpu" to narrow the output.
- Use diagnose_job to classify failures (OOM, timeout, missing module, code error).
""")


@mcp.tool()
def select_gpu(vram_gb: int) -> str:
    """Recommend the best GPU for a given VRAM requirement.

    Shows which GPU type, partition, and QoS to use, plus current availability.
    Useful for deciding before submitting a job.

    Args:
        vram_gb: GPU VRAM needed in GB (0 for CPU-only).
    """
    if vram_gb == 0:
        return "CPU-only job — no GPU needed. Use submit_job with vram_gb=0."

    selection = slurm_mcp.select_gpu(vram_gb)
    avail = slurm_mcp.check_availability()

    if selection is None:
        capable = [g for g in slurm_mcp.GPU_TYPES if g.vram_gb >= vram_gb]
        if not capable:
            max_gpu = max(slurm_mcp.GPU_TYPES, key=lambda g: g.vram_gb)
            return f"No GPU has >= {vram_gb}GB VRAM. Max available: {max_gpu.vram_gb}GB ({max_gpu.name})."
        lines = [f"No GPU with >= {vram_gb}GB VRAM is currently free.", ""]
        for g in capable:
            golden = avail.golden.get(g.name)
            cluster = avail.cluster.get(g.name)
            gstr = f"golden {golden.free}/{golden.total}" if golden else "no golden"
            cstr = f"cluster {cluster.free}/{cluster.total}" if cluster else "N/A"
            lines.append(f"  {g.name} ({g.vram_gb}GB): {gstr}, {cstr}")
        return "\n".join(lines)

    gpu_type, partition, qos = selection
    gpu_info = slurm_mcp.GPU_BY_NAME[gpu_type]
    golden = avail.golden.get(gpu_type)
    cluster = avail.cluster.get(gpu_type)
    gstr = f"golden {golden.free}/{golden.total}" if golden else "no golden"
    cstr = f"cluster {cluster.free}/{cluster.total}" if cluster else "N/A"

    return (
        f"Recommendation: {gpu_type} ({gpu_info.vram_gb}GB VRAM)\n"
        f"  Partition: {partition}, QoS: {qos}\n"
        f"  Availability: {gstr}, {cstr}"
    )


@mcp.tool()
def cluster_summary(
    view: Literal["full", "jobs", "gpu"] = "full",
    qos: str | None = None,
) -> str:
    """Single-call cluster dashboard: your jobs + golden tickets (every
    configured QoS) + cluster-wide GPU availability.

    Args:
        view: "full" (default), "jobs" (table only), or "gpu" (availability only).
        qos: Restrict to one QoS — filters both the job list and the Golden
            sections. None = show all configured QoS.
    """
    parts: list[str] = []

    if view in ("full", "jobs"):
        jobs = slurm_mcp.my_jobs(qos=qos)
        if view == "jobs":
            parts.append(render.render_jobs_table(jobs))
        else:
            parts.append(render.render_jobs_summary(jobs))

    if view in ("full", "gpu"):
        avail = slurm_mcp.check_availability()
        queues = slurm_mcp.golden_queues(avail, qos_filter=qos)
        golden = render.render_golden_all(avail, qos_filter=qos, queues=queues)
        if golden:
            parts.append(golden)
        parts.append(render.render_cluster_wide(avail))

    return "\n\n".join(p for p in parts if p)


@mcp.tool()
def get_job_status(job_id: int) -> str:
    """Get the status of a specific SLURM job.

    Args:
        job_id: The SLURM job ID.
    """
    status = slurm_mcp.get_job_status(job_id)
    return json.dumps(status.to_dict(), indent=2)


@mcp.tool()
def read_job_log(job_id: int, output_dir: str = "logs", tail: int = 100) -> str:
    """Read the SLURM log file for a job.

    Args:
        job_id: The SLURM job ID.
        output_dir: Directory to search for log files (default: 'logs').
        tail: Number of lines from the end to return (default: 100, 0 = all).
    """
    content = slurm_mcp.read_job_log(job_id, output_dir=output_dir, tail=tail)
    if content is None:
        return f"No log file found for job {job_id} in {output_dir}/"
    return content


@mcp.tool()
def diagnose_job(job_id: int, output_dir: str = "logs", log_lines: int = 50) -> str:
    """Diagnose a SLURM job failure: gets status, reads log, classifies the error.

    Returns a structured diagnosis with failure classification and suggested action.

    Args:
        job_id: The SLURM job ID.
        output_dir: Directory to search for log files (default: 'logs').
        log_lines: Number of tail lines to include (default: 50).
    """
    return slurm_mcp.diagnose_job(job_id, output_dir=output_dir, log_lines=log_lines)


@mcp.tool()
def submit_job(
    cmd: str,
    vram_gb: int,
    job_name: str | None = None,
    num_gpus: int = 1,
    workdir: str | None = None,
    output_dir: str = "logs",
    gpu_type: str | None = None,
    golden_only: bool = True,
    dependency: str | None = None,
    dry_run: bool = False,
) -> str:
    """Submit a SLURM job. Auto-selects the smallest GPU with enough VRAM.

    IMPORTANT: Set dry_run=true first to preview the sbatch script before actually submitting.

    Multi-GPU: Set num_gpus=2 for multi-GPU jobs (max 2 per cluster policy).
    Uses --gres=gpu:TYPE:N and --nodes=1 to ensure GPUs are on the same node.
    For multi-GPU training, use torchrun: 'torchrun --nproc_per_node=2 train.py'.

    Golden vs main pool: by default a job is golden-only (qos=yisroel on the
    card's dedicated partition, preemption-immune) — it queues on the golden
    partition until a slot frees and NEVER accepts a preemptible main-pool slot.
    Pass golden_only=false to allow the fallback: golden-first, then the
    preemptible main pool when the golden ticket is full. golden_only is ignored
    for CPU jobs.

    Maintenance: Job time limits are automatically capped to finish before scheduled
    maintenance windows. Submissions are blocked if <5 min remain before a window.

    Args:
        cmd: Command to run (e.g. 'python train.py --lr 1e-4').
        vram_gb: GPU VRAM needed in GB (0 for CPU-only jobs).
        job_name: Job name (default: derived from cmd).
        num_gpus: Number of GPUs, 1 or 2 (default: 1). Max 2 per cluster policy.
        workdir: Working directory on compute node.
        output_dir: Directory for SLURM logs (default: 'logs').
        gpu_type: Force a specific GPU type (e.g. 'rtx_pro_6000').
        golden_only: Force qos=yisroel on the card's dedicated golden partition
            (preemption-immune) and never fall back to the preemptible main pool —
            the job stays queued if golden is full. Ignored for CPU jobs. Default
            true; pass false to allow the golden-first-then-main fallback.
        dependency: Job dependency (e.g. 'afterok:12345').
        dry_run: If true, preview the script without submitting.
    """
    result = slurm_mcp.submit_job(
        cmd=cmd,
        vram_gb=vram_gb,
        job_name=job_name,
        num_gpus=num_gpus,
        workdir=workdir,
        output_dir=output_dir,
        gpu_type=gpu_type,
        golden_only=golden_only,
        dependency=dependency,
        wait_until_running=not dry_run,  # don't block on dry runs
        dry_run=dry_run,
    )

    parts = [
        f"success: {result.success}",
        f"job_id: {result.job_id}",
        f"gpu_type: {result.gpu_type}",
        f"partition: {result.partition}",
        f"qos: {result.qos}",
        f"message: {result.message}",
    ]
    if dry_run and result.sbatch_script:
        parts.append(f"\n--- sbatch script ---\n{result.sbatch_script}")
    return "\n".join(parts)


@mcp.tool()
def cancel_jobs(job_ids: list[int] | None = None, all_jobs: bool = False, pending_only: bool = False) -> str:
    """Cancel SLURM jobs.

    Args:
        job_ids: Specific job IDs to cancel.
        all_jobs: Cancel all your jobs.
        pending_only: Only cancel pending jobs (use with all_jobs=true).
    """
    count = slurm_mcp.cancel_jobs(
        job_ids=job_ids,
        all_jobs=all_jobs,
        pending_only=pending_only,
    )
    return f"Cancelled {count} job(s)."


@mcp.tool()
def wait_for_job(job_id: int, poll_interval: int = 30, timeout: int = 600) -> str:
    """Block until a SLURM job finishes and return its final status.

    Args:
        job_id: The SLURM job ID.
        poll_interval: Seconds between status checks (default: 30).
        timeout: Max seconds to wait (default: 600, 0 = no limit).
    """
    status = slurm_mcp.wait_for_job(job_id, poll_interval=poll_interval, timeout=timeout)
    return json.dumps(status.to_dict(), indent=2)


@mcp.tool()
def job_history(days: int = 3, state: str | None = None, limit: int = 30) -> str:
    """Show recent completed/failed jobs from SLURM accounting.

    Unlike my_jobs() which only shows running/pending, this shows finished jobs too.

    Args:
        days: Number of days of history (default: 3).
        state: Filter by state: COMPLETED, FAILED, TIMEOUT, OOM, CANCELLED, or None for all.
        limit: Max jobs to return (default: 30).
    """
    return slurm_mcp.job_history(days=days, state=state, limit=limit)


if __name__ == "__main__":
    mcp.run(transport="stdio")
