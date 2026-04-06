#!/usr/bin/env python3
"""
submit_job.py — Smart SLURM job submission CLI.

Auto-selects GPU based on VRAM requirement. Golden tickets first,
then falls back to normal QoS cluster-wide.

Usage:
    submit_job.py --vram 48 -- python train.py --lr 1e-4
    submit_job.py --vram 48 -j train-bert -- python train.py
    submit_job.py --gpu-type rtx_pro_6000 -- python eval.py
    submit_job.py --vram 48 --dry-run -- python train.py
"""

import argparse
import json
import os
import sys

# Allow import from the same directory as this script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from slurm_mcp import submit_job, GPU_TYPES

BOLD = "\033[1m"
GREEN = "\033[0;32m"
RED = "\033[0;31m"
YELLOW = "\033[1;33m"
NC = "\033[0m"

if not sys.stdout.isatty():
    BOLD = GREEN = RED = YELLOW = NC = ""


def main():
    # Split args at -- separator
    pre_args = []
    cmd_args = []
    found_separator = False
    for arg in sys.argv[1:]:
        if arg == "--" and not found_separator:
            found_separator = True
            continue
        if found_separator:
            cmd_args.append(arg)
        else:
            pre_args.append(arg)

    if not found_separator:
        print(f"{RED}Error: Missing -- separator before command.{NC}", file=sys.stderr)
        print("", file=sys.stderr)
        print("Usage: submit_job.py [options] -- <command> [args...]", file=sys.stderr)
        print("", file=sys.stderr)
        print("Examples:", file=sys.stderr)
        print("  submit_job.py --vram 48 -- python train.py --lr 1e-4", file=sys.stderr)
        print("  submit_job.py --vram 48 -j train-bert -- python train.py", file=sys.stderr)
        print("  submit_job.py --gpu-type rtx_pro_6000 --dry-run -- echo test", file=sys.stderr)
        print("", file=sys.stderr)
        print("Available GPU types (name: VRAM):", file=sys.stderr)
        for g in GPU_TYPES:
            golden = f"  [golden: {g.golden_quota}]" if g.golden_quota > 0 else ""
            print(f"  {g.name}: {g.vram_gb}GB{golden}", file=sys.stderr)
        sys.exit(1)

    if not cmd_args:
        print(f"{RED}Error: No command specified after --.{NC}", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Smart SLURM job submission. Auto-selects GPU based on VRAM.",
        usage="submit_job.py [options] -- <command> [args...]",
    )
    parser.add_argument("--vram", type=int, default=None,
                        help="VRAM needed in GB (required unless --gpu-type is set)")
    parser.add_argument("--gpu-type", default=None,
                        help="Override auto GPU selection (e.g., rtx_pro_6000)")
    parser.add_argument("-n", "--num-gpus", type=int, default=1,
                        help="Number of GPUs (default: 1)")
    parser.add_argument("-q", "--qos", default=None,
                        help="Override QoS (default: auto)")
    parser.add_argument("-j", "--job-name", default=None,
                        help="Job name (default: from command)")
    parser.add_argument("-w", "--workdir", default=None,
                        help="Working directory on compute node")
    parser.add_argument("-o", "--output-dir", default="logs",
                        help="Directory for SLURM log files (default: logs)")
    parser.add_argument("-d", "--dependency", default=None,
                        help="Job dependency (e.g., afterok:12345, afterany:111:222)")
    parser.add_argument("--no-wait", action="store_true",
                        help="Don't wait for job to reach RUNNING state (default: wait)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print generated script without submitting")
    parser.add_argument("--json", action="store_true", dest="json_output",
                        help="Output result as JSON (for programmatic use)")

    args = parser.parse_args(pre_args)

    # Validate: need either --vram or --gpu-type
    if args.vram is None and args.gpu_type is None:
        print(f"{RED}Error: Must specify either --vram or --gpu-type.{NC}", file=sys.stderr)
        sys.exit(1)

    # If --gpu-type but no --vram, set vram to 0 (no VRAM filtering)
    vram_gb = args.vram if args.vram is not None else 0

    # Build command string
    cmd = " ".join(cmd_args)

    # Submit
    result = submit_job(
        cmd=cmd,
        vram_gb=vram_gb,
        job_name=args.job_name,
        num_gpus=args.num_gpus,
        workdir=args.workdir,
        output_dir=args.output_dir,
        gpu_type=args.gpu_type,
        qos=args.qos,
        dependency=args.dependency,
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


if __name__ == "__main__":
    main()
