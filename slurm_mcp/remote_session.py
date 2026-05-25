#!/usr/bin/env python3
"""Submit `claude remote-control` (server mode) as a long-running SLURM job.

Two hardware modes:
  - cpu: runs on the CPU partition (CPU_PARTITION/CPU_QOS in config).
  - gpu: prefers an exact `gpu_type` (e.g. "gtx_1080") if given; otherwise
         falls back to slurm_mcp.select_gpu(vram_gb)'s smallest-fitting
         logic. The cluster cancels GPU jobs that sit idle, so only use
         this mode if the session will actively run GPU code.

Importable:
    from slurm_mcp.remote_session import submit_remote_session_job
    job_id, log_path = submit_remote_session_job(
        name="my-session", hardware="cpu", days=1,
    )
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

from config import CPU_PARTITION, CPU_QOS, CPU_CPUS, CPU_MEM, CLAUDE_LOG_DIR
from maintenance import cap_time_limit

from . import selection
from .gpu_catalog import GPU_BY_NAME, PRIMARY_QOS

VALID_HARDWARE = ("cpu", "gpu")
VALID_DAYS = (1, 2, 3, 7)
# Remote-control sessions only allow these three modes per Anthropic policy;
# auto and bypassPermissions are explicitly unavailable in RC sessions.
VALID_PERMISSION_MODES = ("default", "acceptEdits", "plan")

# claude remote-control prints session URLs in the form
#   https://claude.ai/code/session_<id>?from=cli
# This is the URL the user actually wants to click. There's a separate
# https://claude.ai/code?environment=<env_id> line (the lobby) which we
# DO NOT match — it doesn't open the specific session.
_SESSION_URL_PATTERN = re.compile(
    r"https://claude\.ai/code/session_[A-Za-z0-9_-]+(?:\?[^\s\x07\x1b]*)?"
)


def extract_session_url(log_path: str, timeout: float = 90.0,
                        poll_interval: float = 2.0) -> str | None:
    """Poll the SLURM log file for the `claude.ai/code/session_<id>` URL.

    Returns the first matching URL once it appears, or None if `timeout`
    elapses without finding one. The log file may not exist yet when
    called (sbatch hasn't created it yet) — that's fine, we keep polling.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            with open(log_path) as f:
                content = f.read()
            m = _SESSION_URL_PATTERN.search(content)
            if m:
                return m.group(0)
        except OSError:
            pass
        if time.monotonic() >= deadline:
            return None
        time.sleep(poll_interval)


def _ensure_workspace_trusted(workdir: str, config_path: str | None = None) -> None:
    """Pre-accept Claude Code's workspace-trust dialog for `workdir`.

    Without this, `claude remote-control` on a compute node fails with
    "Workspace not trusted. Please run `claude` in <DIR> first to review
    and accept the workspace trust dialog." SLURM batch jobs can't accept
    interactive dialogs, so we write the acceptance into ~/.claude.json
    (where Claude Code stores the per-project flag) before sbatch.
    """
    if config_path is None:
        config_path = os.path.expanduser("~/.claude.json")

    try:
        with open(config_path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}

    projects = data.setdefault("projects", {})
    project = projects.setdefault(workdir, {})
    if project.get("hasTrustDialogAccepted") is True:
        return
    project["hasTrustDialogAccepted"] = True

    # Atomic write so a concurrent Claude Code read either sees the old
    # file or the new one, never a half-written one.
    tmp = config_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, config_path)


def submit_remote_session_job(
    name=None,
    hardware="cpu",
    days=1,
    permission_mode="acceptEdits",
    vram_gb=24,
    workdir=None,
    resume=None,
    gpu_type=None,
):
    """Submit a `claude remote-control` server as a SLURM job.

    Args:
        name: Session name (default: auto-generated).
        hardware: "cpu" or "gpu".
        days: 1, 2, 3, or 7.
        permission_mode: "default" (ask before each operation), "acceptEdits"
            (auto-approve file edits + routine filesystem bash), or "plan"
            (read-only). auto and bypassPermissions are not available in
            remote-control sessions.
        gpu_type: Exact GPU type to request (e.g. "gtx_1080", "rtx_4090").
            Takes precedence over vram_gb. Use this when the caller knows
            which GPU they want — the auto-select fallback otherwise picks
            the smallest golden GPU with enough VRAM, which may upgrade an
            8GB request to a 48GB card.
        vram_gb: VRAM requirement when hardware="gpu" and gpu_type is None
            (default 24). Falls back to slurm_mcp.select_gpu's
            "smallest-fitting, golden-preferred" logic.
        workdir: Working directory (default: cwd).
        resume: Session ID to resume from a previous Claude Code chat.
            None starts a fresh session. The resume path uses the
            interactive `claude --remote-control NAME --resume ID` form
            since `--resume` isn't a flag on the `remote-control` subcommand.

    Returns:
        (job_id, log_path) tuple.
    """
    if hardware not in VALID_HARDWARE:
        raise ValueError(f"hardware must be one of {VALID_HARDWARE}, got {hardware!r}")
    if days not in VALID_DAYS:
        raise ValueError(f"days must be one of {VALID_DAYS}, got {days!r}")
    if permission_mode not in VALID_PERMISSION_MODES:
        raise ValueError(
            f"permission_mode must be one of {VALID_PERMISSION_MODES}, "
            f"got {permission_mode!r}"
        )

    name = name or f"claude-{int(time.time()) % 10000}"
    workdir = workdir or os.getcwd()
    workdir = os.path.abspath(workdir)
    _ensure_workspace_trusted(workdir)
    time_str = cap_time_limit(f"{days}-0:00:00")
    log_dir = CLAUDE_LOG_DIR or "logs"
    log_path = os.path.join(log_dir, f"{name}-%J.out")

    # Unset inference-only tokens so `claude` falls back to the cached
    # full-scope OAuth credentials in ~/.claude/. Remote Control rejects
    # inference-only tokens (CLAUDE_CODE_OAUTH_TOKEN / ANTHROPIC_API_KEY)
    # with "Remote Control requires a full-scope login token".
    env_prefix = "unset CLAUDE_CODE_OAUTH_TOKEN ANTHROPIC_API_KEY; "

    pmode = f" --permission-mode {permission_mode}"
    if resume:
        # Interactive form supports --resume; server-mode subcommand does not.
        cmd = (
            f"{env_prefix}echo y | claude --remote-control '{name}'"
            f" --resume '{resume}'{pmode}"
        )
    else:
        cmd = (
            f"{env_prefix}echo y | claude remote-control --name '{name}'{pmode}"
        )

    sbatch_args = [
        "sbatch",
        f"--job-name={name}",
        f"--time={time_str}",
        f"--output={log_path}",
    ]

    if hardware == "cpu":
        sbatch_args.extend([
            f"--partition={CPU_PARTITION}",
            f"--qos={CPU_QOS}",
            f"--cpus-per-task={CPU_CPUS}",
            f"--mem={CPU_MEM}",
        ])
    else:
        if gpu_type:
            # Explicit GPU type — honor the caller's choice exactly.
            if gpu_type not in GPU_BY_NAME:
                raise ValueError(
                    f"Unknown gpu_type {gpu_type!r}. "
                    f"Valid: {sorted(GPU_BY_NAME)}"
                )
            gpu_info = GPU_BY_NAME[gpu_type]
            if gpu_info.golden_quota > 0 and gpu_info.golden_partition:
                partition = gpu_info.golden_partition
                qos = PRIMARY_QOS
            else:
                partition = "main"
                qos = "normal"
        else:
            sel = selection.select_gpu(vram_gb)
            if sel is None:
                raise RuntimeError(
                    f"No GPU with >= {vram_gb}GB VRAM is currently available."
                )
            gpu_type, partition, qos = sel
        sbatch_args.extend([
            f"--partition={partition}",
            f"--qos={qos}",
            f"--gres=gpu:{gpu_type}:1",
            "--nodes=1",
            "--cpus-per-task=4",
            "--mem=16G",
        ])

    sbatch_args.append(f"--wrap=cd {workdir} && {cmd}")

    os.makedirs(log_dir, exist_ok=True)
    result = subprocess.run(sbatch_args, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed: {result.stderr.strip()}")

    m = re.search(r"(\d+)", result.stdout)
    job_id = int(m.group(1)) if m else None
    actual_log = log_path.replace("%J", str(job_id)) if job_id else log_path

    return job_id, actual_log


def _prompt_choice(label, options):
    """Numbered menu over `options`, each a (value, description) tuple.
    Reads a 1-based index OR a literal value from stdin. Re-prompts on
    invalid input. Returns the chosen value (same type as options[i][0])."""
    print(f"\n{label}")
    for i, (val, desc) in enumerate(options, 1):
        print(f"  {i}. {str(val):<14} {desc}")
    while True:
        raw = input(f"Select 1-{len(options)}: ").strip()
        if not raw:
            continue
        # Try numeric index
        try:
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1][0]
        except ValueError:
            pass
        # Try literal match (case-insensitive)
        for val, _ in options:
            if raw.lower() == str(val).lower():
                return val
        print(f"  Invalid. Enter 1-{len(options)} or the literal value.")


def _ensure_required_interactive(args):
    """Fill in `args` values left as None by interactively prompting the user.
    Errors out if stdin isn't a TTY and a required value is missing."""
    needs_prompt = (
        args.hardware is None
        or args.days is None
        or args.permission_mode is None
    )
    if needs_prompt and not sys.stdin.isatty():
        sys.exit(
            "error: --hardware, --days, --permission-mode are required when "
            "stdin is not a TTY (no way to prompt interactively).\n"
            "Pass them as flags or run from a terminal."
        )

    if needs_prompt:
        print("=== slurmx remote-session: interactive setup ===")

    if args.hardware is None:
        args.hardware = _prompt_choice(
            "Hardware?",
            [
                ("cpu", "editor-only sessions (no in-session GPU work)"),
                ("gpu", "in-session GPU work (cluster cancels idle GPUs)"),
            ],
        )
    if args.days is None:
        args.days = _prompt_choice(
            "Time limit (days)?",
            [
                (1, "1 day"),
                (2, "2 days"),
                (3, "3 days"),
                (7, "7 days (max)"),
            ],
        )
    if args.permission_mode is None:
        args.permission_mode = _prompt_choice(
            "Permission mode?",
            [
                ("default",     "ask before each operation (safest)"),
                ("acceptEdits", "auto-approve file edits + routine fs bash"),
                ("plan",        "read-only; Claude writes a plan, no edits"),
            ],
        )
    if args.hardware == "gpu" and args.gpu_type is None:
        args.gpu_type = _prompt_choice(
            "GPU?",
            [
                ("gtx_1080",     "8GB  — small inference"),
                ("rtx_2080",     "11GB — small training"),
                ("rtx_3090",     "24GB — medium models"),
                ("rtx_4090",     "24GB — medium models (newer)"),
                ("rtx_6000",     "48GB — large models (golden ticket)"),
                ("rtx_pro_6000", "96GB — very large (golden ticket)"),
            ],
        )
    return args


def add_arguments(parser):
    parser.add_argument("name", nargs="?", help="Session name (default: auto)")
    parser.add_argument("--hardware", choices=VALID_HARDWARE, default=None,
                        help="cpu or gpu (prompted if omitted)")
    parser.add_argument("--days", type=int, choices=VALID_DAYS, default=None,
                        help="Job duration in days (prompted if omitted)")
    parser.add_argument("--gpu-type", default=None,
                        help="Exact GPU type when --hardware=gpu (prompted if gpu)")
    parser.add_argument("--vram-gb", type=int, default=None,
                        help="VRAM fallback when --hardware=gpu and --gpu-type "
                             "is not specified (default 24)")
    parser.add_argument("--permission-mode", choices=VALID_PERMISSION_MODES,
                        default=None,
                        help="default | acceptEdits | plan (prompted if omitted)")
    parser.add_argument("--workdir", help="Working directory (default: cwd)")
    parser.add_argument("--resume", help="Session ID to resume from a previous chat")
    parser.add_argument("--wait-url-seconds", type=int, default=90,
                        help="Poll the log this long for the session URL (default 90; "
                             "0 disables the wait)")


def run(args):
    _ensure_required_interactive(args)

    vram_gb = args.vram_gb if args.vram_gb is not None else 24

    detail = f", gpu_type={args.gpu_type}" if args.hardware == "gpu" else ""
    print(
        f"\nSubmitting: hardware={args.hardware}, days={args.days}, "
        f"permission_mode={args.permission_mode}{detail}"
    )

    job_id, log_path = submit_remote_session_job(
        name=args.name,
        hardware=args.hardware,
        days=args.days,
        permission_mode=args.permission_mode,
        gpu_type=args.gpu_type,
        vram_gb=vram_gb,
        workdir=args.workdir,
        resume=args.resume,
    )
    print(f"Submitted remote-control job {job_id} ({args.hardware}, {args.days}d)")
    print(f"Watch log: tail -f {log_path}")

    print(f"Waiting up to {args.wait_url_seconds}s for the session URL...")
    url = extract_session_url(log_path, timeout=args.wait_url_seconds)
    if url:
        print(f"\nSession URL: {url}")
    else:
        print(f"\nSession URL not in log yet. Check with: tail {log_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Submit `claude remote-control` as a SLURM job. "
            "Required values left off the command line are prompted interactively."
        ),
    )
    add_arguments(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
