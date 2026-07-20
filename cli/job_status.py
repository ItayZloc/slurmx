#!/usr/bin/env python3
"""Backing module for `slurmx job-status` (alias `job`) — status of one job.

    slurmx job-status 12345
    slurmx job 12345 --json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import slurm_mcp
from cli._style import BOLD, NC, state_color


def add_arguments(parser):
    parser.add_argument("job_id", type=int, help="SLURM job ID.")
    parser.add_argument("--json", action="store_true", dest="json_output",
                        help="Output the raw status as JSON.")


def run(args):
    status = slurm_mcp.get_job_status(args.job_id)
    if args.json_output:
        print(json.dumps(status.to_dict(), indent=2))
        return

    d = status.to_dict()
    sc = state_color(d["state"])
    print(f"{BOLD}Job {d['job_id']}{NC}")
    print(f"  State:    {sc}{d['state']}{NC}")
    if d.get("reason"):
        print(f"  Reason:   {d['reason']}")
    if d.get("node"):
        print(f"  Node:     {d['node']}")
    if d.get("elapsed"):
        print(f"  Elapsed:  {d['elapsed']}")
    print(f"  Exit:     {d['exit_code']}")


def main():
    p = argparse.ArgumentParser(description="Status of a specific SLURM job.")
    add_arguments(p)
    run(p.parse_args())


if __name__ == "__main__":
    main()
