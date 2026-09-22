#!/usr/bin/env python3
"""Backing module for ``slurmx submit``.

Usage: slurmx submit [options] -- SCRIPT [ARG ...]
"""

import argparse
import json
import sys

from slurm_mcp import submit_job


def add_arguments(parser):
    parser.add_argument("-j", "--job-name", default=None, help="Job name (default: script name)")
    parser.add_argument("-w", "--workdir", default=None, help="Working directory on the compute node")
    parser.add_argument("-o", "--output-dir", default="logs", help="SLURM log directory (default: logs)")
    parser.add_argument("--after", nargs="+", type=int, metavar="JOBID", default=None, help="Require these jobs to succeed first")
    parser.add_argument("-d", "--dependency", default=None, help="singleton or TYPE:JOBID[:JOBID...] (TYPE: after, afterany, afterok, afternotok, aftercorr)")
    parser.add_argument("--no-wait", action="store_true", help="Do not wait for the job to start")
    parser.add_argument("--dry-run", action="store_true", help="Print the generated script without submitting")
    parser.add_argument("--json", action="store_true", dest="json_output", help="Output result as JSON")
    parser.add_argument("script", nargs=argparse.REMAINDER, help="Executable script and its arguments, after --")


def run(args):
    script_args = list(args.script)
    if script_args and script_args[0] == "--":
        script_args.pop(0)
    if not script_args:
        raise SystemExit("Error: No executable script specified. Usage: slurmx submit [options] -- SCRIPT [ARG ...]")
    if args.after and args.dependency:
        raise SystemExit("Error: Use either --after or --dependency, not both.")
    dependency = args.dependency or ("afterok:" + ":".join(map(str, args.after)) if args.after else None)
    result = submit_job(
        script_args[0], args=script_args[1:], job_name=args.job_name,
        workdir=args.workdir, output_dir=args.output_dir, dependency=dependency,
        wait_until_running=not args.no_wait, dry_run=args.dry_run,
    )
    if args.json_output:
        payload = {
            "success": result.success, "job_id": result.job_id,
            "gpu_type": result.gpu_type, "partition": result.partition,
            "qos": result.qos, "message": result.message,
        }
        if args.dry_run:
            payload["sbatch_script"] = result.sbatch_script
        print(json.dumps(payload))
    elif result.success:
        if args.dry_run:
            print(result.sbatch_script)
        else:
            print(result.message)
    else:
        print(f"Error: {result.message}", file=sys.stderr)
    if not result.success:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(usage="%(prog)s [options] -- SCRIPT [ARG ...]")
    add_arguments(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
