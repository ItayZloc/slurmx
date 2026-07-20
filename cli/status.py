#!/usr/bin/env python3
"""Backing module for `slurmx status` (also runnable as `python -m cli.status`).

In an interactive terminal, `slurmx status` opens a live, scrollable dashboard
(see cli/watch.py): your jobs + golden tickets + cluster-wide GPU availability,
auto-refreshing on an interval. Piped, redirected, or run under `watch` (any
non-TTY), it prints the classic one-shot text and exits, so scripts are
unaffected. `--once` forces the one-shot text even in a terminal.

    slurmx status                # live dashboard (in a terminal)
    slurmx status --once         # one-shot text snapshot
    slurmx status -n 2           # live dashboard, refresh every 2 seconds
    slurmx status | grep 42      # one-shot text (non-TTY)
"""

import argparse
import curses
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import slurm_mcp
from cli import render


def render_squeue_me() -> str:
    """Raw `squeue --me` text — unmodified, for users familiar with the CLI."""
    try:
        r = subprocess.run(["squeue", "--me"], capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            return f"squeue failed: {r.stderr.strip()}"
        return r.stdout.rstrip()
    except Exception as e:
        return f"squeue error: {e}"


def render_dashboard(qos: str | None = None) -> str:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    avail = slurm_mcp.check_availability()
    queues = slurm_mcp.golden_queues(avail, qos_filter=qos)
    golden = render.render_golden_all(avail, qos_filter=qos, queues=queues)
    cluster = render.render_cluster_wide(avail)
    parts = [
        f"slurmx status — {ts}",
        "",
        "=== squeue --me ===",
        render_squeue_me(),
        "",
        golden,
        "",
        cluster,
    ]
    return "\n".join(parts)


def add_arguments(parser):
    parser.add_argument("--qos", help="Restrict golden-ticket sections to one QoS.")
    parser.add_argument("--once", action="store_true",
                        help="Print the one-shot text snapshot and exit (no live view).")
    parser.add_argument("--interval", "-n", type=float, default=5.0,
                        help="Live view refresh interval in seconds (default 5).")


def run(args):
    wants_tui = sys.stdout.isatty() and not args.once
    if wants_tui:
        from cli import watch
        try:
            watch.run_tui(interval=args.interval, qos=args.qos)
            return
        except curses.error:
            # TERM unset/dumb or no usable terminal — fall back to one-shot text.
            pass
    print(render_dashboard(qos=args.qos))


def main():
    p = argparse.ArgumentParser(
        description="Live SLURM dashboard (one-shot text when piped or with --once).",
    )
    add_arguments(p)
    run(p.parse_args())


if __name__ == "__main__":
    main()
