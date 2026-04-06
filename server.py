"""
MCP server exposing slurm_mcp tools to Claude Code.

Run: uv run --with "mcp[cli]" python ~/.claude/mcp-servers/slurm-mcp/server.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import os

# Ensure slurm_mcp, claude_job, config are importable from same directory
sys.path.insert(0, os.path.dirname(__file__))

import slurm_mcp
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("slurm-mcp", instructions="""\
## SLURM Job Submission Rules

When the user says "job" or "a job running", they mean a SLURM job — not a local process.
Check with MCP my_jobs or squeue, not ps aux.

### Key Rules
- Always use dry_run=true first to preview the sbatch script before submitting.
- Always specify vram_gb — never pick GPU types manually unless the user asks.
- Default to 1 GPU. Use num_gpus=2 only when the user explicitly requests multi-GPU
  or the workload requires it (e.g. model too large for a single card).
  Max 2 GPUs per cluster policy — requesting more raises an error.
- For multi-GPU training, use `torchrun --nproc_per_node=2 train.py` as the command.
- Maintenance windows are enforced automatically — job time limits are capped to finish
  before scheduled maintenance. If a window is imminent (<5 min), submissions are blocked.
- Do NOT write raw sbatch scripts or Python code that imports slurm_mcp — always use MCP tools.
- Use cluster_summary as a single-call dashboard to see jobs + GPU availability.
- Use diagnose_job to classify failures (OOM, timeout, missing module, code error).
""")


@mcp.tool()
def check_gpu_availability() -> str:
    """Check GPU availability on the SLURM cluster — both golden tickets and cluster-wide.

    Returns a summary of free/total GPUs by type and who is using golden tickets.
    """
    avail = slurm_mcp.check_availability()

    lines = [f"=== Golden Tickets ({slurm_mcp.GOLDEN_QOS} QoS) ==="]
    for name, g in avail.golden.items():
        lines.append(f"  {name}: {g.free}/{g.total} free")
        if g.users:
            for user, count in g.users.items():
                lines.append(f"    {user}: {count} GPU(s)")

    lines.append("")
    lines.append("=== Cluster-Wide ===")
    for name, c in avail.cluster.items():
        lines.append(f"  {name}: {c.free}/{c.total} free")

    return "\n".join(lines)


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
def cluster_summary() -> str:
    """Single-call cluster dashboard: your jobs + GPU availability.

    Combines job listing and GPU status into one overview.
    """
    avail = slurm_mcp.check_availability()
    jobs = slurm_mcp.my_jobs()

    # Summarize user's jobs
    running = [j for j in jobs if j["state"] == "RUNNING"]
    pending = [j for j in jobs if j["state"] == "PENDING"]

    lines = ["=== Your Jobs ==="]
    if not jobs:
        lines.append("  No jobs.")
    else:
        gpu_count = 0
        for j in running:
            m = re.search(r":(\d+)", j.get("gpu_gres", ""))
            if m:
                gpu_count += int(m.group(1))
        lines.append(f"  {len(running)} running, {len(pending)} pending ({gpu_count} GPUs in use)")
        for j in running:
            lines.append(f"    {j['job_id']} {j['name']} ({j['gpu_gres']}) on {j['node']}")
        for j in pending:
            lines.append(f"    {j['job_id']} {j['name']} (pending)")

    lines.append("")
    lines.append("=== Golden Tickets ===")
    for name, g in avail.golden.items():
        lines.append(f"  {name}: {g.free}/{g.total} free")

    lines.append("")
    lines.append("=== Cluster-Wide ===")
    for name, c in avail.cluster.items():
        if c.total > 0:
            lines.append(f"  {name}: {c.free}/{c.total} free")

    return "\n".join(lines)


@mcp.tool()
def my_jobs(qos: str | None = None) -> str:
    """List my current SLURM jobs.

    Args:
        qos: Optional QoS filter (e.g. 'yisroel' for golden tickets only).
    """
    jobs = slurm_mcp.my_jobs(qos=qos)
    if not jobs:
        return "No jobs found."

    lines = [f"{'JOB_ID':<12} {'NAME':<30} {'STATE':<12} {'QOS':<10} {'GPU':<20} {'RUNTIME':<12} {'NODE'}"]
    lines.append("-" * 110)
    for j in jobs:
        lines.append(
            f"{j['job_id']:<12} {j['name']:<30} {j['state']:<12} {j['qos']:<10} "
            f"{j['gpu_gres']:<20} {j['runtime']:<12} {j['node']}"
        )
    return "\n".join(lines)


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


_OOM_MARKERS = [
    "CUDA out of memory",
    "OutOfMemoryError",
    "torch.cuda.OutOfMemoryError",
    "CUDA error: out of memory",
]
_VRAM_ESCALATION = {48: 96, 24: 48, 11: 24, 8: 11}


@mcp.tool()
def diagnose_job(job_id: int, output_dir: str = "logs", log_lines: int = 50) -> str:
    """Diagnose a SLURM job failure: gets status, reads log, classifies the error.

    Returns a structured diagnosis with failure classification and suggested action.

    Args:
        job_id: The SLURM job ID.
        output_dir: Directory to search for log files (default: 'logs').
        log_lines: Number of tail lines to include (default: 50).
    """
    status = slurm_mcp.get_job_status(job_id)

    if status.state in ("RUNNING", "PENDING"):
        reason = f" (reason: {status.reason})" if status.reason else ""
        return f"Job {job_id} is {status.state}{reason}. No diagnosis needed."
    if status.state == "COMPLETED" and status.exit_code == 0:
        return f"Job {job_id} completed successfully (elapsed: {status.elapsed})."

    # Fetch GPU type from sacct AllocTRES
    gpu_used = ""
    try:
        raw = subprocess.run(
            ["sacct", "-j", str(job_id), "--format=JobID,AllocTRES%60", "-n", "-P", "--noconvert"],
            capture_output=True, text=True, timeout=10,
        )
        for line in raw.stdout.splitlines():
            parts = line.split("|")
            if len(parts) >= 2 and parts[0] == str(job_id):
                m = re.search(r"gres/gpu:([^:,]+):(\d+)", parts[1])
                if m:
                    gpu_used = f"{m.group(1)}:{m.group(2)}"
                break
    except Exception:
        pass

    # Read log
    log = slurm_mcp.read_job_log(job_id, output_dir=output_dir, tail=log_lines)

    # Classify failure
    log_text = log or ""
    classification = "UNKNOWN"
    suggestion = "Check the log output below."

    if status.state == "OUT_OF_MEMORY" or any(m in log_text for m in _OOM_MARKERS):
        classification = "OOM"
        gpu_name = gpu_used.split(":")[0] if gpu_used else ""
        gpu_info = slurm_mcp.GPU_BY_NAME.get(gpu_name)
        if gpu_info and gpu_info.vram_gb in _VRAM_ESCALATION:
            next_vram = _VRAM_ESCALATION[gpu_info.vram_gb]
            suggestion = f"Retry with more VRAM: {gpu_info.vram_gb}GB -> {next_vram}GB."
        else:
            suggestion = "Already on largest GPU. Reduce batch size or enable gradient accumulation."
    elif status.state == "TIMEOUT" or "DUE TO TIME LIMIT" in log_text:
        classification = "TIMEOUT"
        suggestion = "Job hit time limit. Add checkpointing or request more time."
    elif "DependencyNeverSatisfied" in log_text or status.reason == "DependencyNeverSatisfied":
        classification = "DEPENDENCY_FAILED"
        suggestion = "A dependency job failed or was cancelled. Check upstream jobs."
    elif "ModuleNotFoundError" in log_text or "ImportError" in log_text:
        classification = "MISSING_MODULE"
        mod_match = re.search(r"(?:ModuleNotFoundError|ImportError).*?'([^']+)'", log_text)
        mod_name = mod_match.group(1) if mod_match else "unknown"
        suggestion = f"Missing module: {mod_name}. Install it in the job environment."
    elif "Killed" in log_text:
        classification = "KILLED"
        suggestion = "Job was killed (possibly system OOM or admin). Check memory usage."
    elif "Traceback (most recent call last)" in log_text:
        classification = "CODE_ERROR"
        # Extract last error line
        tb_lines = log_text.splitlines()
        for line in reversed(tb_lines):
            stripped = line.strip()
            if stripped and not stripped.startswith("File ") and not stripped.startswith("Traceback"):
                suggestion = f"Python error: {stripped}"
                break

    parts = [
        f"=== Job Diagnosis: {job_id} ===",
        f"State: {status.state} (exit code: {status.exit_code})",
        f"Classification: {classification}",
    ]
    if gpu_used:
        parts.append(f"GPU: {gpu_used}")
    if status.elapsed:
        parts.append(f"Elapsed: {status.elapsed}")
    parts.append("")
    parts.append(f"Suggested action: {suggestion}")

    if log:
        parts.append("")
        parts.append(f"--- Log tail (last {log_lines} lines) ---")
        parts.append(log)
    elif log is None:
        parts.append("")
        parts.append(f"No log file found in {output_dir}/.")

    return "\n".join(parts)


@mcp.tool()
def submit_job(
    cmd: str,
    vram_gb: int,
    job_name: str | None = None,
    num_gpus: int = 1,
    workdir: str | None = None,
    output_dir: str = "logs",
    gpu_type: str | None = None,
    dependency: str | None = None,
    dry_run: bool = False,
) -> str:
    """Submit a SLURM job. Auto-selects the smallest GPU with enough VRAM.

    IMPORTANT: Set dry_run=true first to preview the sbatch script before actually submitting.

    Multi-GPU: Set num_gpus=2 for multi-GPU jobs (max 2 per cluster policy).
    Uses --gres=gpu:TYPE:N and --nodes=1 to ensure GPUs are on the same node.
    For multi-GPU training, use torchrun: 'torchrun --nproc_per_node=2 train.py'.

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


@mcp.tool()
def launch_claude(
    name: str | None = None,
    days: int = 7,
    workdir: str | None = None,
    permission_mode: str = "bypassPermissions",
    prompt: str | None = None,
    model: str | None = None,
    effort: str | None = None,
) -> str:
    """Launch Claude Code as a SLURM CPU job. Two modes:

    **Interactive (remote-control)**: No prompt — starts a persistent server.
    Connect via claude.ai/code to interact. Good for ongoing work.

    **One-shot (send and forget)**: With prompt — runs the task and exits.
    Results appear in the SLURM log. Good for autonomous tasks.

    IMPORTANT: Before calling, ask the user:
    1. Interactive or one-shot? (Does the user want to provide a prompt?)
    2. Permission mode? (default: bypassPermissions — fully autonomous)
    3. For one-shot: model and effort level?

    Args:
        name: Session name (default: auto-generated).
        days: Job duration in days (default: 7).
        workdir: Working directory (default: cwd).
        permission_mode: Permission mode for the session (default: bypassPermissions).
            Options: default, acceptEdits, bypassPermissions, dontAsk, plan.
        prompt: Task prompt for one-shot mode. If None, launches remote-control.
        model: Model to use — opus, sonnet, haiku (one-shot only).
        effort: Effort level — low, medium, high, max (one-shot only).
    """
    from claude_job import submit_claude_job
    from maintenance import MaintenanceWindowError

    try:
        job_id, log_path = submit_claude_job(
            name=name, days=days, workdir=workdir,
            permission_mode=permission_mode,
            prompt=prompt, model=model, effort=effort,
        )
    except (RuntimeError, MaintenanceWindowError) as e:
        return f"Failed: {e}"

    mode = "one-shot task" if prompt else "remote-control"
    lines = [
        f"Claude {mode} job submitted!",
        f"  Job ID: {job_id}",
        f"  Log: {log_path}",
        f"  Permission mode: {permission_mode}",
    ]
    if prompt:
        lines.append(f"  Model: {model or 'default'}")
        lines.append(f"  Effort: {effort or 'default'}")
        lines.append(f"  View results: read_job_log({job_id})")
    else:
        lines.append(f"  Check log for connection URL: read_job_log({job_id})")

    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run(transport="stdio")
