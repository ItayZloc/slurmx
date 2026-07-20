"""Job-failure diagnosis — status + log + error classification.

Shared by the `diagnose_job` MCP tool (server.py) and the `slurmx diagnose` CLI
so both classify failures the same way.
"""

from __future__ import annotations

import re
import subprocess

from .gpu_catalog import GPU_BY_NAME
from .monitoring import get_job_status, read_job_log


_OOM_MARKERS = [
    "CUDA out of memory",
    "OutOfMemoryError",
    "torch.cuda.OutOfMemoryError",
    "CUDA error: out of memory",
]
_VRAM_ESCALATION = {48: 96, 24: 48, 11: 24, 8: 11}


def diagnose_job(job_id: int, output_dir: str = "logs", log_lines: int = 50) -> str:
    """Diagnose a SLURM job failure: status, log tail, error classification.

    Returns a human-readable diagnosis string with a failure classification and a
    suggested action.
    """
    status = get_job_status(job_id)

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
    log = read_job_log(job_id, output_dir=output_dir, tail=log_lines)

    # Classify failure
    log_text = log or ""
    classification = "UNKNOWN"
    suggestion = "Check the log output below."

    if status.state == "OUT_OF_MEMORY" or any(m in log_text for m in _OOM_MARKERS):
        classification = "OOM"
        gpu_name = gpu_used.split(":")[0] if gpu_used else ""
        gpu_info = GPU_BY_NAME.get(gpu_name)
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
