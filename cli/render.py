"""Shared render helpers used by the MCP `cluster_summary` tool and the
`slurmx status` CLI. Pure formatting — no SLURM calls."""

from __future__ import annotations

import re

import slurm_mcp

# How many queued jobs to list per full card before collapsing into a "+N more".
QUEUE_DISPLAY_LIMIT = 15


def render_golden_all(avail: slurm_mcp.Availability,
                      qos_filter: str | None = None,
                      queues: dict | None = None,
                      limit: int | None = QUEUE_DISPLAY_LIMIT) -> str:
    """One '=== Golden Tickets ({qos} QoS) ===' section per configured QoS.

    Iterates avail.golden_by_qos. If qos_filter is set, only that QoS is shown.
    Each card lists Running (per user) and a single Pending block: when `queues`
    carries that QoS's ordered waiting queue (from slurm_mcp.golden_queues, fetched
    when the ticket is full), Pending is the queue in dispatch order, one
    'user: N GPU(s)' row per queued job — so the same user can appear at several
    positions. Otherwise it falls back to the per-user pending aggregate.

    `limit` caps how many queued rows are listed per card (default
    QUEUE_DISPLAY_LIMIT); pass None to list all of them (the scrollable TUI does
    this since it can page through the full queue).
    """
    queues = queues or {}
    sections = []
    for qos, gpus in avail.golden_by_qos.items():
        if qos_filter and qos != qos_filter:
            continue
        header = f"=== Golden Tickets ({qos} QoS) ==="
        qos_queue = queues.get(qos, [])
        cards = []
        for name, g in gpus.items():
            card = [
                f"  {name}: {g.free}/{g.total} free "
                f"({g.running} running, {g.pending} pending)"
            ]
            if g.running_users:
                card.append("    Running:")
                for user, count in sorted(g.running_users.items(), key=lambda x: -x[1]):
                    card.append(f"      {user}: {count} GPU(s)")
            # Pending: the ordered waiting queue, one row per queued job in
            # dispatch order (so a user can appear at several positions, e.g.
            # itay: 3 / doron: 2 / itay: 1). Falls back to the per-user aggregate
            # when the ordered queue isn't available (card not full, so
            # golden_queues didn't fetch it). Never both.
            waiting = [r for r in qos_queue if r.get("gpu_type") == name]
            if waiting:
                card.append(f"    Pending ({len(waiting)} waiting, next first):")
                shown = waiting if limit is None else waiting[:limit]
                for r in shown:
                    card.append(f"      {r['user']}: {r['gpu_count']} GPU(s)")
                if limit is not None and len(waiting) > limit:
                    card.append(f"      ... and {len(waiting) - limit} more queued")
            elif g.pending_users:
                card.append("    Pending:")
                for user, count in sorted(g.pending_users.items(), key=lambda x: -x[1]):
                    card.append(f"      {user}: {count} GPU(s)")
            cards.append("\n".join(card))
        # One blank line between adjacent cards (e.g. rtx_pro_6000 vs rtx_6000)
        # so the two ticket types read as distinct blocks under the QoS header.
        # No blank before the first card and no trailing blank; the gap lands
        # deep in the golden block (past the short cluster column), so the TUI's
        # _side_by_side stays aligned.
        body = "\n\n".join(cards)
        sections.append(f"{header}\n{body}" if body else header)
    return "\n\n".join(sections)


def render_cluster_wide(avail: slurm_mcp.Availability) -> str:
    """=== Cluster-Wide === section. Skips GPU types with total=0."""
    lines = ["=== Cluster-Wide ==="]
    for name, c in avail.cluster.items():
        if c.total > 0:
            lines.append(f"  {name}: {c.free}/{c.total} free")
    return "\n".join(lines)


def render_jobs_table(jobs: list[dict]) -> str:
    """Formatted table from slurm_mcp.my_jobs() output."""
    if not jobs:
        return "No jobs found."
    header = (
        f"{'JOB_ID':<12} {'NAME':<30} {'STATE':<12} {'QOS':<10} "
        f"{'GPU':<20} {'RUNTIME':<12} {'NODE'}"
    )
    lines = [header, "-" * 110]
    for j in jobs:
        lines.append(
            f"{j['job_id']:<12} {j['name']:<30} {j['state']:<12} {j['qos']:<10} "
            f"{j['gpu_gres']:<20} {j['runtime']:<12} {j['node']}"
        )
    return "\n".join(lines)


def render_jobs_summary(jobs: list[dict]) -> str:
    """Compact 'Your Jobs' summary: counts + GPU sum + one line per job."""
    lines = ["=== Your Jobs ==="]
    if not jobs:
        lines.append("  No jobs.")
        return "\n".join(lines)

    running = [j for j in jobs if j["state"] == "RUNNING"]
    pending = [j for j in jobs if j["state"] == "PENDING"]
    gpu_count = 0
    for j in running:
        m = re.search(r":(\d+)", j.get("gpu_gres", ""))
        if m:
            gpu_count += int(m.group(1))
    lines.append(
        f"  {len(running)} running, {len(pending)} pending "
        f"({gpu_count} GPUs in use)"
    )
    for j in running:
        lines.append(f"    {j['job_id']} {j['name']} ({j['gpu_gres']}) on {j['node']}")
    for j in pending:
        lines.append(f"    {j['job_id']} {j['name']} (pending)")
    return "\n".join(lines)
