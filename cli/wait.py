#!/usr/bin/env python3
"""Backing module for `slurmx wait` — block until a job finishes.

    slurmx wait 12345
    slurmx wait 12345 --poll 15 --timeout 3600
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import slurm_mcp
from cli._style import NC, state_color


def add_arguments(parser):
    parser.add_argument("job_id", type=int, help="SLURM job ID.")
    parser.add_argument("--poll", type=int, default=30,
                        help="Seconds between status checks (default: 30).")
    parser.add_argument("--timeout", type=int, default=0,
                        help="Max seconds to wait (default: 0 = no limit).")
    parser.add_argument("--json", action="store_true", dest="json_output",
                        help="Output the final status as JSON.")


def run(args):
    status = slurm_mcp.wait_for_job(args.job_id, poll_interval=args.poll, timeout=args.timeout)
    if args.json_output:
        print(json.dumps(status.to_dict(), indent=2))
        return
    sc = state_color(status.state)
    tag = "finished" if status.finished else "still pending (timeout reached)"
    print(f"Job {args.job_id} {tag}: {sc}{status.state}{NC}")


def main():
    p = argparse.ArgumentParser(description="Block until a SLURM job finishes.")
    add_arguments(p)
    run(p.parse_args())


if __name__ == "__main__":
    main()
