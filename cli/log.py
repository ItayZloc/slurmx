#!/usr/bin/env python3
"""Backing module for `slurmx log` — read a job's SLURM log file.

    slurmx log 12345               # last 100 lines
    slurmx log 12345 --tail 0      # whole log
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import slurm_mcp
from cli._style import RED, NC


def add_arguments(parser):
    parser.add_argument("job_id", type=int, help="SLURM job ID.")
    parser.add_argument("-o", "--output-dir", default="logs",
                        help="Directory to search for log files (default: logs).")
    parser.add_argument("--tail", type=int, default=100,
                        help="Lines from the end to show (default: 100, 0 = all).")


def run(args):
    content = slurm_mcp.read_job_log(args.job_id, output_dir=args.output_dir, tail=args.tail)
    if content is None:
        print(f"{RED}No log file found for job {args.job_id} in {args.output_dir}/{NC}",
              file=sys.stderr)
        sys.exit(1)
    print(content)


def main():
    p = argparse.ArgumentParser(description="Read a SLURM job's log file.")
    add_arguments(p)
    run(p.parse_args())


if __name__ == "__main__":
    main()
