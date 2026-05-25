"""Shared render helpers used by the MCP `cluster_summary` tool and the
`slurmx status` CLI. Pure formatting — no SLURM calls."""

from __future__ import annotations

import re

import slurm_mcp


def render_golden_all(avail: slurm_mcp.Availability,
                      qos_filter: str | None = None) -> str:
    """One '=== Golden Tickets ({qos} QoS) ===' section per configured QoS.

    Iterates avail.golden_by_qos. If qos_filter is set, only that QoS is shown.
    """
    sections = []
    for qos, gpus in avail.golden_by_qos.items():
        if qos_filter and qos != qos_filter:
            continue
        lines = [f"=== Golden Tickets ({qos} QoS) ==="]
        for name, g in gpus.items():
            lines.append(
                f"  {name}: {g.free}/{g.total} free "
                f"({g.running} running, {g.pending} pending)"
            )
            if g.running_users:
                lines.append("    Running:")
                for user, count in sorted(g.running_users.items(), key=lambda x: -x[1]):
                    lines.append(f"      {user}: {count} GPU(s)")
            if g.pending_users:
                lines.append("    Pending:")
                for user, count in sorted(g.pending_users.items(), key=lambda x: -x[1]):
                    lines.append(f"      {user}: {count} GPU(s)")
        sections.append("\n".join(lines))
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
