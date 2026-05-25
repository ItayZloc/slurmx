"""GPU availability scanning — squeue per golden-QoS + sinfo for cluster-wide."""

from __future__ import annotations

import re

from config import GOLDEN_QOS

from . import shell
from .gpu_catalog import GPU_TYPES, GPU_TYPES_BY_QOS, PRIMARY_QOS
from .types import Availability, GPUAvailability


def _golden_availability_for_qos(qos: str) -> dict:
    """Build {gpu_type -> GPUAvailability} for one QoS by parsing squeue."""
    qos_gpu_types = GPU_TYPES_BY_QOS.get(qos, [])
    if not qos_gpu_types:
        return {}

    running = {}
    pending = {}
    running_users = {}
    pending_users = {}

    raw = shell._run_quiet([
        "squeue", "--qos", qos, "-h", "-O",
        "UserName:20,tres-per-job:60,tres-per-node:60,State:12"
    ])

    for line in raw.splitlines():
        if not line.strip():
            continue

        user = line[0:20].strip()
        tres_job = line[20:80].strip()
        tres_node = line[80:140].strip()
        state = line[140:152].strip()

        if state not in ("RUNNING", "PENDING"):
            continue

        gres_field = ""
        for f in (tres_job, tres_node):
            if f != "N/A" and "gres/gpu:" in f:
                gres_field = f
                break
        if not gres_field:
            continue

        m = re.search(r"gres/gpu:([^:,]+):(\d+)", gres_field)
        if not m:
            continue

        gpu_type = m.group(1)
        gpu_count = int(m.group(2))

        if state == "RUNNING":
            running[gpu_type] = running.get(gpu_type, 0) + gpu_count
            users_map = running_users.setdefault(gpu_type, {})
        else:
            pending[gpu_type] = pending.get(gpu_type, 0) + gpu_count
            users_map = pending_users.setdefault(gpu_type, {})
        users_map[user] = users_map.get(user, 0) + gpu_count

    result = {}
    for gpu in qos_gpu_types:
        if gpu.golden_quota <= 0:
            continue
        r = running.get(gpu.name, 0)
        p = pending.get(gpu.name, 0)
        result[gpu.name] = GPUAvailability(
            gpu_type=gpu.name,
            total=gpu.golden_quota,
            used=r,
            free=max(0, gpu.golden_quota - r),
            users=running_users.get(gpu.name, {}),
            running=r,
            pending=p,
            running_users=running_users.get(gpu.name, {}),
            pending_users=pending_users.get(gpu.name, {}),
        )
    return result


def check_availability() -> Availability:
    """
    Query GPU availability — every configured golden QoS plus cluster-wide.

    Returns an Availability object with:
      - .golden_by_qos[qos] = {gpu_type -> GPUAvailability} for each QoS
      - .golden               (alias for golden_by_qos[PRIMARY_QOS] for back-compat)
      - .cluster              {gpu_type -> GPUAvailability} cluster-wide
    """
    avail = Availability()

    for qos in GOLDEN_QOS:
        avail.golden_by_qos[qos] = _golden_availability_for_qos(qos)

    avail.golden = avail.golden_by_qos.get(PRIMARY_QOS, {})

    cluster_total = {}
    cluster_alloc = {}

    raw = shell._run_quiet([
        "sinfo", "-N", "-h", "-O",
        "NodeHost:20,Gres:40,GresUsed:40,StateLong:20"
    ])

    seen_nodes = set()
    for line in raw.splitlines():
        if not line.strip():
            continue

        node = line[0:20].strip()
        gres_str = line[20:60].strip()
        gres_used_str = line[60:100].strip()
        state = line[100:].strip()

        if node in seen_nodes:
            continue
        seen_nodes.add(node)

        if "down" in state or "drain" in state:
            continue

        if gres_str and gres_str != "(null)" and "gpu:" in gres_str:
            m = re.search(r"gpu:([^:]+):(\d+)", gres_str)
            if m:
                gtype = m.group(1)
                gcount = int(m.group(2))
                cluster_total[gtype] = cluster_total.get(gtype, 0) + gcount

        if gres_used_str and gres_used_str != "(null)" and "gpu:" in gres_used_str:
            m = re.search(r"gpu:([^:]+):(\d+)", gres_used_str)
            if m:
                gtype = m.group(1)
                gcount = int(m.group(2))
                cluster_alloc[gtype] = cluster_alloc.get(gtype, 0) + gcount

    for gpu in GPU_TYPES:
        total = cluster_total.get(gpu.name, 0)
        alloc = cluster_alloc.get(gpu.name, 0)
        free = max(0, total - alloc)
        avail.cluster[gpu.name] = GPUAvailability(
            gpu_type=gpu.name,
            total=total,
            used=alloc,
            free=free,
        )

    return avail
