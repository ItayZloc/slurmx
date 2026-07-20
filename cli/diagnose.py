#!/usr/bin/env python3
"""Backing module for `slurmx diagnose` — classify a job failure.

    slurmx diagnose 12345
    slurmx diagnose 12345 --log-lines 100
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import slurm_mcp


def add_arguments(parser):
    parser.add_argument("job_id", type=int, help="SLURM job ID.")
    parser.add_argument("-o", "--output-dir", default="logs",
                        help="Directory to search for log files (default: logs).")
    parser.add_argument("--log-lines", type=int, default=50,
                        help="Tail lines to include in the diagnosis (default: 50).")


def run(args):
    print(slurm_mcp.diagnose_job(args.job_id, output_dir=args.output_dir,
                                 log_lines=args.log_lines))


def main():
    p = argparse.ArgumentParser(description="Diagnose a SLURM job failure.")
    add_arguments(p)
    run(p.parse_args())


if __name__ == "__main__":
    main()
