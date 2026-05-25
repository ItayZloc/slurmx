#!/usr/bin/env python3
"""slurmx — umbrella CLI for this project.

Dispatches subcommands the way `git`, `aws-cli`, or `kubectl` do — each
subcommand exposes `add_arguments(parser)` and `run(args)` in its own module,
and slurmx wires them as argparse subparsers. `slurmx <cmd> --help` shows
that subcommand's options properly; tab-completion works the standard way.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cli import status as status_mod
from cli import submit as submit_mod
from slurm_mcp import remote_session

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _add_subcommand(subparsers, name, module, *, aliases=(), help=None):
    p = subparsers.add_parser(name, aliases=list(aliases), help=help)
    module.add_arguments(p)
    p.set_defaults(_run=module.run)
    return p


def _exec_repo_script(script_name):
    """Build a `run(args)` that execs a bash script living at the repo root,
    forwarding any extra positional args."""
    def run(args):
        path = os.path.join(REPO, script_name)
        os.execvp(path, [path] + list(getattr(args, "script_args", []) or []))
    return run


def _add_script_subcommand(subparsers, name, script_name, help):
    p = subparsers.add_parser(name, help=help)
    p.add_argument("script_args", nargs=argparse.REMAINDER,
                   help=f"Additional args forwarded to {script_name}")
    p.set_defaults(_run=_exec_repo_script(script_name))
    return p


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="slurmx",
        description=(
            "Cluster CLI: SLURM job management + Claude Code remote sessions. "
            "Use `slurmx <subcommand> --help` for per-command options."
        ),
    )
    subparsers = parser.add_subparsers(
        dest="subcommand",
        metavar="<subcommand>",
        required=True,
    )

    _add_subcommand(
        subparsers, "status", status_mod,
        aliases=("s",),
        help="One-shot SLURM dashboard.",
    )
    _add_subcommand(
        subparsers, "submit", submit_mod,
        help="Submit a GPU/CPU job (auto-selects GPU by VRAM).",
    )
    _add_subcommand(
        subparsers, "remote-session", remote_session,
        aliases=("rc", "claude"),
        help="Launch `claude remote-control` as a SLURM job.",
    )
    _add_script_subcommand(
        subparsers, "setup", "setup.sh",
        help="Run the project setup script (uv sync + symlink CLIs into ~/.local/bin/).",
    )
    _add_script_subcommand(
        subparsers, "update", "update.sh",
        help="Fast-forward git pull; re-runs uv sync if dependencies changed.",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args._run(args)


if __name__ == "__main__":
    main()
