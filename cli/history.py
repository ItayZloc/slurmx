#!/usr/bin/env python3
"""Backing module for `slurmx history` — recent finished jobs from sacct.

    slurmx history                       # last 3 days
    slurmx history --days 7 --state OOM  # last week, out-of-memory only
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import slurm_mcp


def add_arguments(parser):
    parser.add_argument("--days", type=int, default=3,
                        help="Days of history to show (default: 3).")
    parser.add_argument("--state", default=None,
                        help="Filter: COMPLETED, FAILED, TIMEOUT, OOM, CANCELLED.")
    parser.add_argument("--limit", type=int, default=30,
                        help="Max jobs to show, most recent first (default: 30).")


def run(args):
    print(slurm_mcp.job_history(days=args.days, state=args.state, limit=args.limit))


def main():
    p = argparse.ArgumentParser(description="Recent finished jobs from SLURM accounting.")
    add_arguments(p)
    run(p.parse_args())


if __name__ == "__main__":
    main()
