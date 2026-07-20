#!/usr/bin/env python3
"""Backing module for `slurmx status` (also runnable as `python -m cli.status`).

Prints squeue --me + golden ticket holders (per configured QoS) + cluster-wide
GPU availability, then exits. Pair with the unix `watch` command if you want
periodic refresh:

    slurmx status                # one snapshot
    watch -n 2 slurmx status     # refresh every 2 seconds
"""

import argparse
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


def run(args):
    print(render_dashboard(qos=args.qos))


def main():
    p = argparse.ArgumentParser(
        description="One-shot SLURM dashboard. Use `watch slurmx status` to refresh.",
    )
    add_arguments(p)
    run(p.parse_args())


if __name__ == "__main__":
    main()
