"""Metadata-aware SLURM job submission and sbatch script generation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shlex
import tempfile
import time

from config import CPU_CPUS, CPU_MEM, EXCLUDE_NODES, MAIL_USER, MAX_MEM_GB, START_TIMEOUT, TIME_LIMIT
from config_defaults import CPU_PARTITION, CPU_QOS, MAIL_TYPE
from maintenance import cap_time_limit

from . import monitoring, shell
from .monitoring import _FINISHED_STATES, _UNRECOVERABLE_REASONS
from .selection import GPUChoice, select_resources
from .types import JobResult


_METADATA_KEYS = {"total_vram_gb", "supports_gpu_sharding", "preemption_safe"}
# Kept as import compatibility for older callers of this private policy helper.
# Submission no longer accepts a pool policy from callers.
ASK_POLICY_MESSAGE = "Submission pool policy is declared in the script metadata header."


def resolve_golden_only(golden_only: bool | None) -> bool | None:
    """Legacy helper retained for import compatibility; not used by submission."""
    return golden_only


@dataclass(frozen=True)
class ScriptMetadata:
    total_vram_gb: int
    supports_gpu_sharding: bool
    preemption_safe: bool


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def parse_script_metadata(script_path: str) -> tuple[ScriptMetadata | None, str | None]:
    """Read and validate the mandatory SLURMx header in an executable script."""
    path = Path(script_path)
    if not path.is_file():
        return None, f"Script is not a regular file: {path}"
    if not os.access(path, os.X_OK):
        return None, f"Script is not executable: {path}"
    try:
        with path.open(encoding="utf-8") as script:
            if not script.readline().startswith("#!"):
                return None, "Script must start with a shebang."
            header = script.readline()
    except OSError as exc:
        return None, f"Could not read script: {exc}"
    prefix = "# slurmx: "
    if not header.startswith(prefix):
        return None, "Script must put its slurmx metadata immediately after the shebang."
    try:
        values = json.loads(header[len(prefix):].rstrip("\n"), object_pairs_hook=_json_object)
    except (json.JSONDecodeError, ValueError):
        return None, "slurmx metadata must be strict JSON."
    if not isinstance(values, dict) or set(values) != _METADATA_KEYS:
        return None, "slurmx metadata must contain exactly total_vram_gb, supports_gpu_sharding, and preemption_safe."
    total_vram_gb = values["total_vram_gb"]
    if type(total_vram_gb) is not int or total_vram_gb < 0:
        return None, "slurmx metadata total_vram_gb must be a non-negative integer."
    for key in ("supports_gpu_sharding", "preemption_safe"):
        if type(values[key]) is not bool:
            return None, f"slurmx metadata {key} must be a boolean."
    if total_vram_gb == 0 and values["supports_gpu_sharding"]:
        return None, "slurmx metadata cannot enable GPU sharding for a CPU job."
    return ScriptMetadata(**values), None


def _build_sbatch_script(
    cmd: str, partition: str, qos: str, gpu_type: str, num_gpus: int,
    job_name: str, output_path: str, workdir: str | None,
    preemption_safe: bool, dependency: str | None = None,
) -> str:
    """Generate a batch script that preserves the header's preemption policy."""
    lines = [
        "#!/bin/bash", "", "### --- Slurm Job Configuration ---", "",
        f"#SBATCH --partition {partition}", f"#SBATCH --qos={qos}",
        f"#SBATCH --time {cap_time_limit(TIME_LIMIT, dependency)}",
        f"#SBATCH --job-name {job_name}", f"#SBATCH --output {output_path}",
        "#SBATCH --requeue" if preemption_safe else "#SBATCH --no-requeue",
    ]
    if preemption_safe:
        lines.append("#SBATCH --signal=B:USR1@120")
    if gpu_type:
        lines += [f"#SBATCH --gres=gpu:{gpu_type}:{num_gpus}", "#SBATCH --nodes=1", f"#SBATCH --mem={MAX_MEM_GB}G"]
    else:
        lines += [f"#SBATCH --cpus-per-task={CPU_CPUS}", f"#SBATCH --mem={CPU_MEM}"]
    if EXCLUDE_NODES:
        lines.append(f"#SBATCH --exclude={','.join(EXCLUDE_NODES)}")
    if dependency:
        lines.append(f"#SBATCH --dependency={dependency}")
    mail_types = [item for item in MAIL_TYPE if item and item.upper() != "NONE"]
    if MAIL_USER and mail_types:
        lines += [f"#SBATCH --mail-user={MAIL_USER}", f"#SBATCH --mail-type={','.join(mail_types)}"]
    lines += [
        "", "# --- Scratch directory (fallback to /tmp if /scratch unavailable) ---",
        "export SCRATCH_DIR=/scratch/$USER/$SLURM_JOB_ID",
        'mkdir -p "$SCRATCH_DIR" 2>/dev/null || { export SCRATCH_DIR=/tmp/$USER/slurm_$SLURM_JOB_ID; mkdir -p "$SCRATCH_DIR"; }',
        "trap 'rm -rf \"$SCRATCH_DIR\"' EXIT", "",
    ]
    log_dir = os.path.dirname(output_path)
    if log_dir and log_dir != ".":
        lines += [f'mkdir -p "{log_dir}"', ""]
    if workdir:
        lines += [f"cd {shlex.quote(workdir)}", ""]
    if preemption_safe:
        lines += [
            f"{cmd} &", "child_pid=$!",
            "trap 'kill -USR1 \"$child_pid\" 2>/dev/null || true' USR1",
            "trap 'kill -TERM \"$child_pid\" 2>/dev/null || true' TERM",
            "while true; do", "  wait \"$child_pid\"", "  status=$?",
            "  kill -0 \"$child_pid\" 2>/dev/null || break", "done", "exit \"$status\"",
        ]
    else:
        lines.append(cmd)
    return "\n".join(lines) + "\n"


def _wait_for_running(job_result: JobResult, timeout: int) -> tuple[JobResult, str]:
    """Poll until a job starts, finishes, is fatal, or stays pending."""
    start = time.time()
    while True:
        status = monitoring.get_job_status(job_result.job_id)
        if status.state == "RUNNING":
            job_result.message = f"Job {job_result.job_id} is RUNNING on {status.node}"
            return job_result, "running"
        if status.state in _FINISHED_STATES:
            job_result.success = False
            job_result.message = f"Job {job_result.job_id} ended before running: {status.state} (exit_code={status.exit_code})"
            return job_result, "finished"
        if status.reason in _UNRECOVERABLE_REASONS:
            shell._run_quiet(["scancel", str(job_result.job_id)])
            job_result.success = False
            job_result.message = f"Job {job_result.job_id} cancelled - fatal error: {status.reason}"
            return job_result, "fatal"
        elapsed = time.time() - start
        if timeout > 0 and elapsed >= timeout:
            job_result.message = f"Job {job_result.job_id} still pending after {int(elapsed)}s (reason: {status.reason or 'unknown'}). Job remains queued."
            return job_result, "still_pending"
        time.sleep(5)


def _do_submit(script: str, choice: GPUChoice | None, partition: str, qos: str) -> JobResult:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", prefix="slurm-submit-", dir="/tmp", delete=False) as handle:
        handle.write(script)
        tmpfile = handle.name
    gpu_type = choice.gpu_type if choice else "cpu"
    try:
        os.chmod(tmpfile, 0o755)
        message = shell._run(["sbatch", tmpfile])
        match = re.search(r"(\d+)", message)
        return JobResult(True, int(match.group(1)) if match else None, gpu_type, partition, qos, message.strip(), script)
    except RuntimeError as exc:
        return JobResult(False, None, gpu_type, partition, qos, str(exc), script)
    finally:
        os.unlink(tmpfile)


def _failure(message: str) -> JobResult:
    return JobResult(False, None, "", "", "", message, "")


def submit_job(
    script_path: str,
    args: list[str] | None = None,
    job_name: str | None = None,
    workdir: str | None = None,
    output_dir: str = "logs",
    dependency: str | None = None,
    wait_until_running: bool = True,
    dry_run: bool = False,
) -> JobResult:
    """Submit only an executable script whose metadata selects its resources."""
    resolved_path = os.path.abspath(script_path if os.path.isabs(script_path) else os.path.join(workdir or os.getcwd(), script_path))
    metadata, error = parse_script_metadata(resolved_path)
    if error:
        return _failure(error)
    assert metadata is not None
    job_name = job_name or Path(resolved_path).stem.replace(".", "-") or "job"
    command = shlex.join([resolved_path, *(args or [])])
    if metadata.total_vram_gb == 0:
        choice = None
        partition, qos = CPU_PARTITION, CPU_QOS
    else:
        choice = select_resources(metadata.total_vram_gb, metadata.supports_gpu_sharding, metadata.preemption_safe)
        if choice is None:
            return _failure(f"No GPU configuration can satisfy {metadata.total_vram_gb}GB under this script's policy.")
        partition, qos = choice.partition, choice.qos
    output_path = os.path.join(output_dir, f"slurm-{job_name}-%J.out")
    script = _build_sbatch_script(command, partition, qos, choice.gpu_type if choice else "", choice.num_gpus if choice else 0, job_name, output_path, workdir, metadata.preemption_safe, dependency)
    gpu_type = choice.gpu_type if choice else "cpu"
    if dry_run:
        return JobResult(True, None, gpu_type, partition, qos, "[DRY RUN] Would submit job", script)
    result = _do_submit(script, choice, partition, qos)
    if result.success and wait_until_running and result.job_id is not None:
        result, _ = _wait_for_running(result, START_TIMEOUT)
    return result
