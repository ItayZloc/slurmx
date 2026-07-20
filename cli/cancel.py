#!/usr/bin/env python3
"""Backing module for `slurmx cancel` — cancel jobs by ID or all.

    slurmx cancel 12345 12346        # cancel specific jobs
    slurmx cancel --all              # cancel all your jobs
    slurmx cancel --all --pending-only
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import slurm_mcp
from cli._style import RED, GREEN, NC


def add_arguments(parser):
    parser.add_argument("job_ids", nargs="*", type=int,
                        help="Specific job IDs to cancel.")
    parser.add_argument("--all", action="store_true", dest="all_jobs",
                        help="Cancel all your jobs.")
    parser.add_argument("--pending-only", action="store_true",
                        help="Only cancel pending jobs (use with --all).")


def run(args):
    if not args.job_ids and not args.all_jobs:
        print(f"{RED}Error: specify job IDs or --all.{NC}", file=sys.stderr)
        sys.exit(1)
    count = slurm_mcp.cancel_jobs(
        job_ids=args.job_ids or None,
        all_jobs=args.all_jobs,
        pending_only=args.pending_only,
    )
    print(f"{GREEN}Cancelled {count} job(s).{NC}")


def main():
    p = argparse.ArgumentParser(description="Cancel SLURM jobs.")
    add_arguments(p)
    run(p.parse_args())


if __name__ == "__main__":
    main()
