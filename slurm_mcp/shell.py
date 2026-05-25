"""Subprocess helpers used by every module that shells out to SLURM commands."""

from __future__ import annotations

import subprocess


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
