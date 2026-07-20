#!/usr/bin/env python3
"""Backing module for `slurmx select-gpu` — recommend a GPU for a VRAM need.

    slurmx select-gpu --vram 48              # golden-only recommendation (default)
    slurmx select-gpu --vram 24 --allow-main # also consider the main pool
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import slurm_mcp
from cli._style import BOLD, GREEN, NC


def add_arguments(parser):
    parser.add_argument("--vram", type=int, required=True,
                        help="VRAM needed in GB (0 = CPU-only).")
    parser.add_argument("--allow-main", action="store_true",
                        help="Consider the preemptible main pool, not just golden.")


def _recommend(vram_gb: int, golden_only: bool) -> str:
    if vram_gb == 0:
        return "CPU-only job — no GPU needed. Use `slurmx submit --vram 0 -- <cmd>`."

    selection = slurm_mcp.select_gpu(vram_gb, golden_only=golden_only)
    avail = slurm_mcp.check_availability()

    if selection is None:
        capable = [g for g in slurm_mcp.GPU_TYPES if g.vram_gb >= vram_gb]
        if not capable:
            max_gpu = max(slurm_mcp.GPU_TYPES, key=lambda g: g.vram_gb)
            return f"No GPU has >= {vram_gb}GB VRAM. Max available: {max_gpu.vram_gb}GB ({max_gpu.name})."
        lines = [f"No GPU with >= {vram_gb}GB VRAM is currently free.", ""]
        for g in capable:
            golden = avail.golden.get(g.name)
            cluster = avail.cluster.get(g.name)
            gstr = f"golden {golden.free}/{golden.total}" if golden else "no golden"
            cstr = f"cluster {cluster.free}/{cluster.total}" if cluster else "N/A"
            lines.append(f"  {g.name} ({g.vram_gb}GB): {gstr}, {cstr}")
        return "\n".join(lines)

    gpu_type, partition, qos = selection
    gpu_info = slurm_mcp.GPU_BY_NAME[gpu_type]
    golden = avail.golden.get(gpu_type)
    cluster = avail.cluster.get(gpu_type)
    gstr = f"golden {golden.free}/{golden.total}" if golden else "no golden"
    cstr = f"cluster {cluster.free}/{cluster.total}" if cluster else "N/A"

    return (
        f"{BOLD}Recommendation:{NC} {GREEN}{gpu_type}{NC} ({gpu_info.vram_gb}GB VRAM)\n"
        f"  Partition: {partition}, QoS: {qos}\n"
        f"  Availability: {gstr}, {cstr}"
    )


def run(args):
    print(_recommend(args.vram, golden_only=not args.allow_main))


def main():
    p = argparse.ArgumentParser(description="Recommend a GPU for a VRAM requirement.")
    add_arguments(p)
    run(p.parse_args())


if __name__ == "__main__":
    main()
