#!/usr/bin/env python3
"""Backing module for `slurmx submit` (also runnable as `python -m cli.submit`).

Auto-selects GPU based on VRAM requirement. Which pool a job lands on comes from
config's GOLDEN_POLICY unless --golden-only or --allow-main says otherwise;
under the "ask" policy the choice is asked for at the terminal.

Usage (via slurmx):
    slurmx submit --vram 48 -- python train.py --lr 1e-4
    slurmx submit --vram 48 -j train-bert -- python train.py
    slurmx submit --gpu-type rtx_pro_6000 -- python eval.py
    slurmx submit --vram 48 --after 12345 -- python eval.py   # wait for job 12345
    slurmx submit --vram 48 --allow-main -- python train.py   # allow main fallback
    slurmx submit --vram 48 --golden-only -- python train.py  # never preemptible
    slurmx submit --vram 48 --dry-run -- python train.py
"""

import argparse
import json
import os
import sys

# Allow import from the same directory as this script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from slurm_mcp import submit_job, resolve_golden_only, GPU_TYPES

BOLD = "\033[1m"
GREEN = "\033[0;32m"
RED = "\033[0;31m"
YELLOW = "\033[1;33m"
NC = "\033[0m"

if not sys.stdout.isatty():
    BOLD = GREEN = RED = YELLOW = NC = ""


def add_arguments(parser):
    parser.add_argument("--vram", type=int, default=None,
                        help="VRAM needed in GB (required unless --gpu-type is set)")
    parser.add_argument("--gpu-type", default=None,
                        help="Override auto GPU selection (e.g., rtx_pro_6000)")
    parser.add_argument("-n", "--num-gpus", type=int, default=1,
                        help="Number of GPUs (default: 1)")
    parser.add_argument("-q", "--qos", default=None,
                        help="Override QoS (default: auto)")
    pool = parser.add_mutually_exclusive_group()
    pool.add_argument("--allow-main", action="store_true",
                      help="Allow falling back to the preemptible main pool when "
                           "the golden ticket is full.")
    pool.add_argument("--golden-only", action="store_true",
                      help="Force the golden ticket (qos on the card's dedicated "
                           "partition, preemption-immune): queue on golden rather "
                           "than drop to main. With neither flag the pool comes "
                           "from config's GOLDEN_POLICY, which `slurmx config` "
                           "edits; under the 'ask' policy you are prompted.")
    parser.add_argument("-j", "--job-name", default=None,
                        help="Job name (default: from command)")
    parser.add_argument("-w", "--workdir", default=None,
                        help="Working directory on compute node")
    parser.add_argument("-o", "--output-dir", default="logs",
                        help="Directory for SLURM log files (default: logs)")
    parser.add_argument("--after", nargs="+", type=int, metavar="JOBID", default=None,
                        help="Wait for these job IDs to finish successfully before "
                             "starting (shorthand for --dependency afterok:ID[:ID...]).")
    parser.add_argument("-d", "--dependency", default=None,
                        help="Raw SLURM dependency expression (e.g., afterok:12345, "
                             "afterany:111:222, singleton). Use --after for the "
                             "common 'finish first' case.")
    parser.add_argument("--no-wait", action="store_true",
                        help="Don't wait for job to reach RUNNING state (default: wait)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print generated script without submitting")
    parser.add_argument("--json", action="store_true", dest="json_output",
                        help="Output result as JSON (for programmatic use)")
    # Trailing positional: everything after `--` (or after the last option).
    parser.add_argument("cmd", nargs=argparse.REMAINDER,
                        help="Command to run; precede with `--` to be safe.")


def _choose_pool(args, is_cpu_job):
    """The golden_only to submit with, asking the user if the policy says to."""
    if args.golden_only:
        return True
    if args.allow_main:
        return False
    chosen = resolve_golden_only(None)
    if chosen is not None or is_cpu_job:
        # CPU jobs ignore golden_only, so never hold one up for an answer.
        return bool(chosen)
    if not sys.stdin.isatty():
        print(f"{RED}Error: GOLDEN_POLICY is 'ask' and there is no terminal to "
              f"ask on.{NC} Pass --golden-only or --allow-main.", file=sys.stderr)
        sys.exit(1)
    # The prompt goes to stderr so it can't land in --json output.
    while True:
        print("Golden-only (preemption-immune, queues when the ticket is full) "
              "or the preemptible main pool? [g/m]: ", end="", file=sys.stderr,
              flush=True)
        answer = (sys.stdin.readline() or "").strip().lower()
        if answer in ("g", "golden", "golden-only"):
            return True
        if answer in ("m", "main", "allow-main"):
            return False
        if not answer:
            print(f"{RED}Nothing chosen — aborted.{NC}", file=sys.stderr)
            sys.exit(1)


def run(args):
    # Strip a leading `--` token if the user used the canonical separator.
    cmd_args = list(args.cmd)
    if cmd_args and cmd_args[0] == "--":
        cmd_args = cmd_args[1:]
    if not cmd_args:
        print(f"{RED}Error: No command specified.{NC}", file=sys.stderr)
        print("Usage: ... [options] -- <command> [args...]", file=sys.stderr)
        print("Available GPU types (name: VRAM):", file=sys.stderr)
        for g in GPU_TYPES:
            golden = f"  [golden: {g.golden_quota}]" if g.golden_quota > 0 else ""
            print(f"  {g.name}: {g.vram_gb}GB{golden}", file=sys.stderr)
        sys.exit(1)

    if args.vram is None and args.gpu_type is None:
        print(f"{RED}Error: Must specify either --vram or --gpu-type.{NC}", file=sys.stderr)
        sys.exit(1)

    if args.after and args.dependency:
        print(f"{RED}Error: Use either --after or --dependency, not both.{NC}",
              file=sys.stderr)
        sys.exit(1)
    dependency = args.dependency
    if args.after:
        dependency = "afterok:" + ":".join(str(j) for j in args.after)

    vram_gb = args.vram if args.vram is not None else 0
    cmd = " ".join(cmd_args)
    golden_only = _choose_pool(args, is_cpu_job=vram_gb == 0 and not args.gpu_type)

    result = submit_job(
        cmd=cmd,
        vram_gb=vram_gb,
        job_name=args.job_name,
        num_gpus=args.num_gpus,
        workdir=args.workdir,
        output_dir=args.output_dir,
        gpu_type=args.gpu_type,
        qos=args.qos,
        golden_only=golden_only,
        dependency=dependency,
        wait_until_running=not args.no_wait,
        dry_run=args.dry_run,
    )

    if args.json_output:
        out = {
            "success": result.success,
            "job_id": result.job_id,
            "gpu_type": result.gpu_type,
            "partition": result.partition,
            "qos": result.qos,
            "message": result.message,
        }
        if args.dry_run:
            out["sbatch_script"] = result.sbatch_script
        print(json.dumps(out))
        sys.exit(0 if result.success else 1)

    if not result.success:
        print(f"{RED}Error: {result.message}{NC}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print(f"{BOLD}=== Job Submission Preview ==={NC}")
        print(f"")
        print(f"  GPU:        {GREEN}{result.gpu_type}{NC}")
        print(f"  Partition:  {result.partition}")
        print(f"  QoS:        {result.qos}")
        print(f"  Command:    {cmd}")
        print(f"")
        print(f"{YELLOW}--- Generated sbatch script ---{NC}")
        print(f"")
        print(result.sbatch_script)
    else:
        print(f"{GREEN}{result.message}{NC}")
        print(f"  GPU: {result.gpu_type} | Partition: {result.partition} | QoS: {result.qos}")


def main():
    parser = argparse.ArgumentParser(
        description="Smart SLURM job submission. Auto-selects GPU based on VRAM.",
        usage="%(prog)s [options] -- <command> [args...]",
    )
    add_arguments(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
