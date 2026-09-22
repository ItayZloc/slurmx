"""Read-only scheduler scans deployed to ~/.slurmx/preemption_scan.py."""

from __future__ import annotations

from dataclasses import asdict
import json
import sys

from config_defaults import MAIN_PARTITION
from slurm_mcp.gpu_catalog import GPU_TYPES
from slurm_mcp.preemption import (
    _Budget, _Candidate, _Job, _QueryFailure, _SAFE_ATOM, _expand_nodelist,
    _job_snapshot, _node_gpu_allocations, _node_is_usable, _node_snapshot,
)


def find_candidate(nodes: list[dict[str, str]], jobs: list[_Job], budget: _Budget | None = None) -> _Candidate | None:
    golden = {gpu.name: gpu.golden_partition for gpu in GPU_TYPES if gpu.golden_partition}
    for fields in nodes:
        node = fields.get("NodeName", "")
        if not _SAFE_ATOM.fullmatch(node) or not _node_is_usable(fields.get("State", "")):
            continue
        partitions = set(fields.get("Partitions", "").split(","))
        relevant = {gpu_type: partition for gpu_type, partition in golden.items() if partition in partitions}
        if MAIN_PARTITION not in partitions or not relevant:
            continue
        total, used = _node_gpu_allocations(fields)
        for gpu_type, golden_partition in relevant.items():
            if total.get(gpu_type, 0) - used.get(gpu_type, 0) != 1:
                continue
            if any(node in _expand_nodelist(job.node_list, budget) for job in jobs):
                continue
            return _Candidate(node, gpu_type, golden_partition)
    return None


def post_victim_residency(candidate: _Candidate, victim_id: int, budget: _Budget | None = None) -> bool:
    fields = [node for node in _node_snapshot(budget) if node.get("NodeName") == candidate.node]
    if len(fields) != 1 or not _node_is_usable(fields[0].get("State", "")) or "GresUsed" not in fields[0]:
        return False
    try:
        inventory, usage = _node_gpu_allocations(fields[0])
        total, used = inventory.get(candidate.gpu_type, 0), usage.get(candidate.gpu_type, 0)
        residents = [job for job in _job_snapshot(budget=budget, details=False)
                     if candidate.node in _expand_nodelist(job.node_list, budget)]
        return total > 0 and used == total and len(residents) == 1 and residents[0].job_id == str(victim_id)
    except _QueryFailure:
        return False


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        return 2
    try:
        budget = _Budget(int(argv[2]))
        if argv[1] == "candidate" and len(argv) == 3:
            result = find_candidate(_node_snapshot(budget), _job_snapshot(budget=budget, details=False), budget)
            print(json.dumps(asdict(result) if result is not None else None))
            return 0
        if argv[1] == "post" and len(argv) == 7:
            candidate = _Candidate(argv[3], argv[4], argv[5])
            result = post_victim_residency(candidate, int(argv[6]), budget)
            print(json.dumps({"safe": result}))
            return 0
    except (ValueError, _QueryFailure):
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
