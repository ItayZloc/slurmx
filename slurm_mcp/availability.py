"""GPU availability scanning — squeue per golden-QoS + sinfo for cluster-wide."""

from __future__ import annotations

import re

from config import GOLDEN_QOS
from config_defaults import MAIN_PARTITION

from . import shell
from .gpu_catalog import GPU_TYPES, GPU_TYPES_BY_QOS, PRIMARY_QOS
from .types import Availability, GPUAvailability


def _parse_row(line: str, fields: tuple) -> tuple:
    """Split one fixed-width `-O` row into stripped values.

    `fields` is a ((sinfo/squeue field name, width), ...) spec; the last field
    takes the rest of the line so a long tail can't be clipped by its width.
    """
    out, pos = [], 0
    for i, (_, width) in enumerate(fields):
        last = i == len(fields) - 1
        out.append((line[pos:] if last else line[pos:pos + width]).strip())
        pos += width
    return tuple(out)


def _fmt(fields: tuple) -> str:
    """The `-O` format string for a field spec."""
    return ",".join(f"{name}:{width}" for name, width in fields)


_TRES_GPU_RE = re.compile(r"gres/gpu:([^:,]+):(\d+)")


def _gres_from_tres(tres_job: str, tres_node: str) -> tuple:
    """(gpu_type, count) from a job's two TRES fields; ('', 0) if it wants no GPU.

    A job can ask for GPUs at job scope (--gpus / --gres, which lands in
    tres-per-job) or per node (--gpus-per-node, which lands in tres-per-node).
    Both spellings are common, so reading only one of the two silently drops
    every job that used the other.
    """
    for field in (tres_job, tres_node):
        if field and field != "N/A":
            m = _TRES_GPU_RE.search(field)
            if m:
                return m.group(1), int(m.group(2))
    return "", 0


_GOLDEN_FIELDS = (
    ("UserName", 20),
    ("tres-per-job", 60),
    ("tres-per-node", 60),
    ("State", 12),
)


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
        "squeue", "--qos", qos, "-h", "-O", _fmt(_GOLDEN_FIELDS),
    ])

    for line in raw.splitlines():
        if not line.strip():
            continue

        user, tres_job, tres_node, state = _parse_row(line, _GOLDEN_FIELDS)

        if state not in ("RUNNING", "PENDING"):
            continue

        gpu_type, gpu_count = _gres_from_tres(tres_job, tres_node)
        if not gpu_type:
            continue

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


# --------------------------------------------------------------------------- #
# Cluster-wide scan
# --------------------------------------------------------------------------- #

# `sinfo -N` prints one fixed-width row per (node, partition). The widths are
# declared once and drive both the -O format string and the slice offsets, so
# the two can't drift apart. Gres/GresUsed need real room — a dual-card node
# prints "gpu:rtx_3090:0(IDX:N/A),gpu:gtx_1080:1(IDX:1)" (45 chars) and sinfo
# truncates silently at whatever width you declare.
_SINFO_FIELDS = (
    ("NodeHost", 20),
    ("Gres", 72),
    ("GresUsed", 72),
    ("Partition", 24),
    ("StateLong", 24),
)

_GRES_RE = re.compile(r"gpu:([A-Za-z0-9_]+):(\d+)")

# StateLong prints "<base state><flags>" — "mixed", "draining", "completing*".
# Only capacity that can take a new job right now is counted. `sres`
# (/storage/scripts/sres) lands on the same answer from the other direction:
# `sinfo -r` drops non-responding nodes, then it greps out inval|drain|down|reboot.
#
# Slurm renders the same condition as a word or as a flag character depending on
# the base state: an idle maintenance node prints "maint", a busy one prints
# "mixed$" (verified against node_state_string() in libslurm 25.11). So the two
# forms have to be filtered together, or the answer would depend on whether a job
# happened to be running.
#   *  NOT_RESPONDING  — slurmctld can't reach slurmd, so nothing schedules
#   $  MAINT           — inside a maintenance reservation
#   @  REBOOT_REQUESTED / ^ REBOOT_ISSUED
# Flags that still allow scheduling are kept:
#   -  PLANNED         — earmarked by the backfill scheduler
#   ~ # % !            — power save (disabled on this cluster; '~' resumes on
#                        allocation, so it is real capacity where it is enabled)
_UNUSABLE_FLAGS = "*$@^"
_UNUSABLE_STATES = (
    "down",    # DOWN, POWER_DOWN, POWERING_DOWN, POWERED_DOWN
    "drain",   # DRAIN, DRAINING, DRAINED
    "fail",    # FAIL, FAILING
    "inval",   # INVALID_REG — registered with a config slurmctld rejects
    "unk",     # UNKNOWN — hasn't checked in since slurmctld started
    "future",  # FUTURE — configured placeholder with no hardware behind it
    "maint",   # MAINT — inside a maintenance reservation
    "reboot",  # REBOOT_ISSUED, REBOOT_REQUESTED
)


def _node_is_usable(state: str) -> bool:
    """True when a node can accept a new job right now."""
    s = state.lower()
    if any(flag in s for flag in _UNUSABLE_FLAGS):
        return False
    return not any(word in s for word in _UNUSABLE_STATES)


def _submittable_partitions() -> set:
    """Partitions we can actually place a job in: the shared pool plus every
    configured golden partition. A node reachable only through some other queue
    (slurm-bridge, another group's private partition) isn't our capacity, so its
    GPUs shouldn't inflate the count."""
    parts = {MAIN_PARTITION}
    parts.update(g.golden_partition for g in GPU_TYPES if g.golden_partition)
    return parts


def _count_gpus(gres: str) -> dict:
    """{gpu_type -> count} for one Gres/GresUsed string. Dual-card nodes list
    several entries ("gpu:rtx_3090:1(S:0),gpu:gtx_1080:1(S:0)"), so every entry
    counts — matching only the first hides whole card types."""
    counts = {}
    for m in _GRES_RE.finditer(gres):
        counts[m.group(1)] = counts.get(m.group(1), 0) + int(m.group(2))
    return counts


def _cluster_availability() -> dict:
    """{gpu_type -> GPUAvailability} for the cluster.

    `total` counts only GPUs a job could actually land on: the node has to sit in
    a partition we submit to and be in a state that accepts work. GPUs skipped
    for a node-state reason are reported as `offline` instead of vanishing, so a
    total that drops overnight is explainable rather than mysterious.
    """
    allowed = _submittable_partitions()
    raw = shell._run_quiet(["sinfo", "-N", "-h", "-O", _fmt(_SINFO_FIELDS)])

    # One node spans several rows (one per partition); fold them into one entry.
    nodes = {}
    for line in raw.splitlines():
        if not line.strip():
            continue
        node, gres, gres_used, partition, state = _parse_row(line, _SINFO_FIELDS)
        if not node:
            continue
        entry = nodes.setdefault(
            node, {"gres": gres, "gres_used": gres_used, "state": state,
                   "partitions": set()}
        )
        # sinfo marks the cluster default partition with a trailing '*'.
        entry["partitions"].add(partition.rstrip("*"))

    total, alloc, offline = {}, {}, {}
    offline_nodes = {}
    for node, e in sorted(nodes.items()):
        if not e["partitions"] & allowed:
            continue
        counts = _count_gpus(e["gres"])
        if not counts:
            continue
        if _node_is_usable(e["state"]):
            for gtype, n in counts.items():
                total[gtype] = total.get(gtype, 0) + n
            for gtype, n in _count_gpus(e["gres_used"]).items():
                alloc[gtype] = alloc.get(gtype, 0) + n
        else:
            for gtype, n in counts.items():
                offline[gtype] = offline.get(gtype, 0) + n
                offline_nodes.setdefault(gtype, []).append(node)

    cluster = {}
    for gpu in GPU_TYPES:
        t = total.get(gpu.name, 0)
        a = alloc.get(gpu.name, 0)
        cluster[gpu.name] = GPUAvailability(
            gpu_type=gpu.name,
            total=t,
            used=a,
            free=max(0, t - a),
            offline=offline.get(gpu.name, 0),
            offline_nodes=offline_nodes.get(gpu.name, []),
        )
    return cluster


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
    avail.cluster = _cluster_availability()

    return avail


# Deliberately carries BOTH TRES fields, like _GOLDEN_FIELDS. This query feeds the
# per-user "Pending (next first)" list while _golden_availability_for_qos feeds the
# "(N pending)" counter beside it, so the two have to read GPU requests the same
# way. They didn't: this one used `-o %b`, which is tres-per-node only, so a job
# submitted with --gpus instead of --gpus-per-node was counted but never listed —
# the card read "13 pending" above a list adding up to 11.
# PriorityLong is the integer priority (`-O Priority` returns a normalized float).
_QUEUE_FIELDS = (
    ("PriorityLong", 12),
    ("JobArrayID", 26),   # room for a throttled array id: "12345678_[1-10000%5]"
    ("UserName", 20),
    ("tres-per-job", 60),
    ("tres-per-node", 60),
    ("Name", 60),
)


def golden_queue(qos: str) -> list[dict]:
    """Pending jobs on `qos`, in scheduling order (priority desc, then job id).

    Each row: {priority, job_id, user, name, gpu_type, gpu_count}. Lets the
    status view show who/what is ahead in line when a golden ticket is full.
    Order matches SLURM's dispatch order: higher priority first, then lower
    (older) job id — so row 1 is next to run.
    """
    raw = shell._run_quiet([
        "squeue", "--qos", qos, "-t", "PENDING", "-h", "-O", _fmt(_QUEUE_FIELDS),
    ])

    rows = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        prio_s, job_id, user, tres_job, tres_node, name = _parse_row(line, _QUEUE_FIELDS)
        if not job_id:
            continue

        gpu_type, gpu_count = _gres_from_tres(tres_job, tres_node)

        try:
            prio = int(prio_s)
        except ValueError:
            prio = 0

        rows.append({
            "priority": prio,
            "job_id": job_id.strip(),
            "user": user.strip(),
            "name": name.strip(),
            "gpu_type": gpu_type,
            "gpu_count": gpu_count,
        })

    def _order_key(r):
        # Sort by priority desc, then job id asc. Handles array ids ("123_4").
        m = re.match(r"(\d+)(?:_(\d+))?", r["job_id"])
        jid = (int(m.group(1)), int(m.group(2) or 0)) if m else (0, 0)
        return (-r["priority"], jid)

    rows.sort(key=_order_key)
    return rows


def golden_queues(avail: Availability, qos_filter: str | None = None) -> dict:
    """{qos -> ordered pending rows} for every golden QoS with at least one FULL
    card (free==0). Returns {} when nothing is full, so no squeue call is made
    in the common case where golden tickets are available."""
    result = {}
    for qos, gpus in avail.golden_by_qos.items():
        if qos_filter and qos != qos_filter:
            continue
        if any(g.free == 0 for g in gpus.values()):
            result[qos] = golden_queue(qos)
    return result
