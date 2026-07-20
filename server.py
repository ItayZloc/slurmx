"""
MCP server exposing slurm_mcp tools to Claude Code.

Run: uv run --with "mcp[cli]" python ~/.claude/mcp-servers/slurmx/server.py
"""

from __future__ import annotations

import json
import re
import subprocess
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
  the user asks. (For launch_remote_session use gpu_type instead — see below.)
- Default to 1 GPU. Use num_gpus=2 only when the user explicitly requests multi-GPU
  or the workload requires it (e.g. model too large for a single card).
  Max 2 GPUs per cluster policy — requesting more raises an error.
- For multi-GPU training, use `torchrun --nproc_per_node=2 train.py` as the command.
- Maintenance windows are enforced automatically — job time limits are capped to finish
  before scheduled maintenance. If a window is imminent (<5 min), submissions are blocked.
- Do NOT write raw sbatch scripts or Python code that imports slurm_mcp — always use MCP tools.
- Use cluster_summary as the single dashboard tool: it covers jobs AND GPU availability.
  Use view="jobs" or view="gpu" to narrow the output.
- Use diagnose_job to classify failures (OOM, timeout, missing module, code error).
- Before calling launch_remote_session, ASK the user for hardware (cpu/gpu),
  time limit (1/2/3/7 days), AND permission_mode (default/acceptEdits/plan).
  All three are required — do not guess. If they pick gpu, ALSO ask for
  gpu_type (gtx_1080/rtx_2080/rtx_3090/rtx_4090/rtx_6000/rtx_pro_6000).
  Do not use vram_gb — gpu_type is the exact selection. The tool
  docstring has the option descriptions to present.
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
    golden_only: bool = False,
    dependency: str | None = None,
    dry_run: bool = False,
) -> str:
    """Submit a SLURM job. Auto-selects the smallest GPU with enough VRAM.

    IMPORTANT: Set dry_run=true first to preview the sbatch script before actually submitting.

    Multi-GPU: Set num_gpus=2 for multi-GPU jobs (max 2 per cluster policy).
    Uses --gres=gpu:TYPE:N and --nodes=1 to ensure GPUs are on the same node.
    For multi-GPU training, use torchrun: 'torchrun --nproc_per_node=2 train.py'.

    Golden vs main pool: by default a job is placed golden-first (qos=yisroel on
    the card's dedicated partition, preemption-immune) then falls back to the
    preemptible main pool if the golden ticket is full. Set golden_only=true to
    force the golden ticket and NEVER accept a preemptible slot — the job queues
    on the golden partition until a slot frees. Recommended for training you don't
    want evicted.

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
        golden_only: If true, force qos=yisroel on the card's dedicated golden
            partition (preemption-immune) and never fall back to the preemptible
            main pool — the job stays queued if golden is full. Ignored for CPU
            jobs. Default false (golden-first, then main).
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
def launch_remote_session(
    hardware: Literal["cpu", "gpu"],
    days: Literal[1, 2, 3, 7],
    permission_mode: Literal["default", "acceptEdits", "plan"],
    name: str | None = None,
    gpu_type: Literal[
        "gtx_1080", "rtx_2080", "rtx_3090", "rtx_4090",
        "rtx_6000", "rtx_pro_6000",
    ] | None = None,
    vram_gb: int = 24,
    golden_only: bool = False,
    workdir: str | None = None,
    resume: str | None = None,
    wait_url_seconds: int = 90,
) -> str:
    """Launch a Claude Code remote-control server as a SLURM job.

    Starts `claude remote-control` in server mode on a compute node. By
    default the tool polls the SLURM log for the
    `https://claude.ai/code/session_<id>` URL and returns it inline in
    the response (so the user just clicks). Set wait_url_seconds=0 to
    skip the wait and use read_job_log later.

    Connect via claude.ai/code or the Claude mobile app.

    REQUIRED — the assistant MUST ASK the user before calling. Present
    each option to the user verbatim and wait for their choice:

      1. hardware: "cpu" or "gpu"?
           - cpu: editor-only sessions (Claude reading/writing code, no
             local GPU work). Cheapest, no idle-GPU cancellation risk.
           - gpu: only if you'll run GPU code IN the session. The cluster
             cancels GPU jobs whose GPU sits idle.

      2. days: 1, 2, 3, or 7?
           Job time limit. Pick the shortest that covers your work.

      3. permission_mode: which level of approval prompts?
           - "default":     Claude asks before every edit, bash command, or
                            tool call. Most safety, most clicks.
           - "acceptEdits": Auto-approves file edits and routine filesystem
                            bash (mkdir / touch / cp / mv / rm / sed) inside
                            the working directory. Still asks before
                            arbitrary bash. Recommended balance.
           - "plan":        Read-only. Claude researches and writes a plan
                            but won't touch files. For exploring a codebase.

           NOTE: "auto" and "bypassPermissions" are NOT available in
           remote-control sessions per Anthropic policy.

      4. ONLY IF hardware="gpu", ALSO ASK: gpu_type?
           Pick the EXACT GPU class (passed as gpu_type=...).
             - gtx_1080      8GB  — small inference
             - rtx_2080      11GB — small training
             - rtx_3090      24GB — medium models
             - rtx_4090      24GB — medium models (newer)
             - rtx_6000      48GB — large models (golden ticket)
             - rtx_pro_6000  96GB — very large (golden ticket)
           Do NOT pass vram_gb instead — that triggers a "smallest
           fitting golden" fallback that may upgrade an 8GB request to
           a 48GB card. Always ask the user for their GPU class
           explicitly.

    Args:
        hardware: "cpu" or "gpu".
        days: 1, 2, 3, or 7.
        permission_mode: "default" | "acceptEdits" | "plan".
        name: Session title shown at claude.ai/code (default: auto).
        gpu_type: Exact GPU type when hardware="gpu". One of gtx_1080,
            rtx_2080, rtx_3090, rtx_4090, rtx_6000, rtx_pro_6000.
            Takes precedence over vram_gb.
        vram_gb: VRAM fallback when gpu_type is None (default 24).
            Triggers "smallest fitting golden" auto-selection.
        golden_only: If true, force qos=yisroel on the card's dedicated golden
            partition (preemption-immune) and never fall back to the preemptible
            main pool. Ignored for hardware="cpu". Default false.
        workdir: Working directory (default: cwd).
        resume: Optional session ID to resume from a previous Claude Code
            chat. Find session IDs in the claude.ai/code session list.
            None starts a fresh session.
        wait_url_seconds: Block this many seconds polling the SLURM log
            for the `claude.ai/code/session_<id>` URL, returning it in
            the response when found. Default 90 — covers typical
            sbatch-schedule + claude-startup time. Set to 0 to return
            immediately without waiting (the agent must then call
            read_job_log itself).

    Notes:
        - Remote-control sessions time out after ~10 min of network outage.
          Reconnect by submitting a fresh session.
    """
    from slurm_mcp.remote_session import submit_remote_session_job, extract_session_url
    from maintenance import MaintenanceWindowError

    try:
        job_id, log_path = submit_remote_session_job(
            name=name, hardware=hardware, days=days,
            permission_mode=permission_mode,
            gpu_type=gpu_type,
            vram_gb=vram_gb, golden_only=golden_only,
            workdir=workdir, resume=resume,
        )
    except (RuntimeError, MaintenanceWindowError) as e:
        return f"Failed: {e}"

    lines = [
        "Remote-control session job submitted!",
        f"  Job ID: {job_id}",
        f"  Log: {log_path}",
        f"  Hardware: {hardware}",
        f"  Time limit: {days} day(s)",
        f"  Permission mode: {permission_mode}",
    ]
    if resume:
        lines.append(f"  Resuming session: {resume}")

    if wait_url_seconds > 0:
        url = extract_session_url(log_path, timeout=wait_url_seconds)
        if url:
            lines.append(f"  Session URL: {url}")
        else:
            lines.append(
                f"  Session URL: pending — not in log after {wait_url_seconds}s. "
                f"Retry with read_job_log({job_id})."
            )
    else:
        lines.append(f"  Check log for connection URL: read_job_log({job_id})")
    if hardware == "gpu":
        lines.append("")
        lines.append("  WARNING: GPU jobs are cancelled by the cluster if the GPU")
        lines.append("  sits idle. Keep the session actively running GPU code, or")
        lines.append("  switch to hardware='cpu' if you only need the editor.")

    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run(transport="stdio")
