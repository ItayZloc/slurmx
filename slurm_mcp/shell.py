"""Subprocess helpers used by every module that shells out to SLURM commands."""

from __future__ import annotations

import os
import subprocess


# sbatch option aliases from https://slurm.schedmd.com/sbatch.html#SECTION_INPUT-ENVIRONMENT-VARIABLES
_SBATCH_ENV_ALIASES = {"SLURM_CLUSTERS", "SLURM_HINT"}


def _run(cmd: list[str]) -> str:
    """Run a command and return stdout. Raises on failure."""
    env = None
    if os.path.basename(cmd[0]) == "sbatch":
        # Environment options override #SBATCH directives; preserve only other variables.
        env = {
            name: value for name, value in os.environ.items()
            if not name.startswith("SBATCH_") and name not in _SBATCH_ENV_ALIASES
        }
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, env=env)
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
