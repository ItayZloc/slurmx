"""Read-only preemption inspection and a deliberately narrow scheduler probe."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shlex
import time
import uuid

from config import GOLDEN_QOS
from config_defaults import MAIN_PARTITION

from . import shell
from .gpu_catalog import GPU_TYPES, PRIMARY_QOS


PROBE_ROOT = Path.home() / ".slurmx" / "probes"
_JOB_FORMAT = "%i|%u|%q|%T|%N|%b"
_OWNER_QOS_FORMAT = "%i|%u|%q"
_GPU_RE = re.compile(r"gpu:([^:,(]+):(\d+)")
_CONFIG_KEYS = (
    ("SLURM version", "SLURM_VERSION"),
    ("PreemptType", "PreemptType"),
    ("PreemptMode", "PreemptMode"),
    ("PreemptParameters", "PreemptParameters"),
    ("JobRequeue", "JobRequeue"),
    ("KillWait", "KillWait"),
)


class _QueryFailure(RuntimeError):
    pass


class _Refusal(RuntimeError):
    pass


@dataclass(frozen=True)
class _Candidate:
    node: str
    gpu_type: str
    golden_partition: str


@dataclass(frozen=True)
class _Job:
    job_id: str
    user: str
    qos: str
    state: str
    node: str
    gpu_type: str
    gpu_count: int


def _required(cmd: list[str]) -> str:
    """Run a scheduler query, keeping every failure distinguishable from empty."""
    try:
        return shell._run(cmd)
    except Exception as exc:
        raise _QueryFailure(str(exc)) from exc


def _config_values(raw: str) -> dict[str, str]:
    values = {}
    for line in raw.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _qos_rows(raw: str) -> list[tuple[str, str, str, str]]:
    rows = []
    for line in raw.splitlines():
        fields = line.strip().split("|")
        if len(fields) >= 4 and fields[0]:
            rows.append(tuple(field.strip() for field in fields[:4]))
    return rows


def preemption_info() -> str:
    """Report controller and configured QoS preemption settings without mutation."""
    try:
        config = _config_values(_required(["scontrol", "show", "config"]))
        lines = [f"{label}: {config.get(key, 'unavailable (field missing)')}" for label, key in _CONFIG_KEYS]
    except _QueryFailure as exc:
        lines = [f"{label}: unavailable (query failed: {exc})" for label, _ in _CONFIG_KEYS]

    try:
        rows = _qos_rows(_required([
            "sacctmgr", "-nP", "show", "qos",
            "format=Name,Preempt,PreemptMode,GraceTime",
        ]))
        lines.append("QoS relationships:")
        by_name = {name: (preempt, mode, grace) for name, preempt, mode, grace in rows}
        wanted = ["normal", *[q for q in GOLDEN_QOS if q != "normal"]]
        for name in wanted:
            if name not in by_name:
                lines.append(f"{name}: unavailable (QoS not returned)")
                continue
            preempt, mode, grace = by_name[name]
            lines.append(
                f"{name}: preempts {preempt or '<none>'}; "
                f"PreemptMode={mode or '<none>'}; GraceTime={grace or '<none>'}"
            )
    except _QueryFailure as exc:
        lines.append(f"QoS relationships: unavailable (query failed: {exc})")
    return "\n".join(lines)


def _count_gpus(value: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for gpu_type, count in _GPU_RE.findall(value or ""):
        counts[gpu_type] = counts.get(gpu_type, 0) + int(count)
    return counts


def _node_fields(line: str) -> dict[str, str]:
    return dict(re.findall(r"(\w+)=([^\s]+)", line))


def _jobs(raw: str) -> list[_Job]:
    parsed = []
    for line in raw.splitlines():
        fields = line.split("|")
        if len(fields) != 6:
            continue
        gpu_counts = _count_gpus(fields[5])
        gpu_type, gpu_count = next(iter(gpu_counts.items()), ("", 0))
        parsed.append(_Job(
            job_id=fields[0].strip(), user=fields[1].strip(), qos=fields[2].strip(),
            state=fields[3].strip(), node=fields[4].strip(), gpu_type=gpu_type,
            gpu_count=gpu_count,
        ))
    return parsed


def _node_snapshot() -> list[dict[str, str]]:
    return [_node_fields(line) for line in _required(["scontrol", "show", "node", "-o"]).splitlines() if line.strip()]


def _job_snapshot() -> list[_Job]:
    return _jobs(_required(["squeue", "-h", "-o", _JOB_FORMAT]))


def _find_candidate(nodes: list[dict[str, str]], jobs: list[_Job]) -> _Candidate | None:
    golden = {gpu.name: gpu.golden_partition for gpu in GPU_TYPES if gpu.golden_partition}
    for fields in nodes:
        node = fields.get("NodeName", "")
        partitions = set(fields.get("Partitions", "").split(","))
        total = _count_gpus(fields.get("Gres", ""))
        used = _count_gpus(fields.get("GresUsed", ""))
        for gpu_type, golden_partition in golden.items():
            if MAIN_PARTITION not in partitions or golden_partition not in partitions:
                continue
            if total.get(gpu_type, 0) - used.get(gpu_type, 0) != 1:
                continue
            unsafe = any(
                job.state == "RUNNING" and job.qos == "normal" and job.node == node
                and job.gpu_count > 0
                for job in jobs
            )
            if not unsafe:
                return _Candidate(node=node, gpu_type=gpu_type, golden_partition=golden_partition)
    return None


def _probe_scripts(candidate: _Candidate, probe_dir: Path) -> tuple[str, str]:
    event_log = probe_dir / "victim-events.log"
    victim = f"""#!/bin/bash
#SBATCH --job-name=slurmx-preemption-victim
#SBATCH --partition={MAIN_PARTITION}
#SBATCH --qos=normal
#SBATCH --nodelist={candidate.node}
#SBATCH --gres=gpu:{candidate.gpu_type}:1
#SBATCH --time=00:02:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output={probe_dir}/victim-%j.out
set -u
event_log={shlex.quote(str(event_log))}
record_signal() {{ test -s "$event_log" && grep -q '^signal ' "$event_log" || printf 'signal USR1 %s\\n' "$(date +%s)" >> "$event_log"; }}
if test "${{SLURM_RESTART_COUNT:-0}}" -gt 0; then printf 'restart %s\\n' "$(date +%s)" >> "$event_log"; exit 0; fi
trap record_signal USR1
while :; do printf 'heartbeat %s\\n' "$(date +%s)" >> "$event_log"; sleep 1; done
"""
    preemptor = f"""#!/bin/bash
#SBATCH --job-name=slurmx-preemption-preemptor
#SBATCH --partition={candidate.golden_partition}
#SBATCH --qos={PRIMARY_QOS}
#SBATCH --nodelist={candidate.node}
#SBATCH --gres=gpu:{candidate.gpu_type}:1
#SBATCH --time=00:02:00
#SBATCH --output={probe_dir}/preemptor-%j.out
sleep 30
"""
    return victim, preemptor


def _submit(script: str, path: Path) -> int:
    path.write_text(script)
    path.chmod(0o700)
    output = _required(["sbatch", "--parsable", str(path)]).strip()
    match = re.match(r"(\d+)(?:;|$)", output)
    if not match:
        raise _QueryFailure(f"sbatch returned an unrecognized job ID: {output!r}")
    return int(match.group(1))


def _job_is_running(job_id: int, candidate: _Candidate) -> bool:
    raw = _required(["squeue", "-h", "-j", str(job_id), "-o", _JOB_FORMAT])
    rows = _jobs(raw)
    return any(
        row.job_id == str(job_id) and row.state == "RUNNING" and row.node == candidate.node
        and row.qos == "normal" and row.gpu_type == candidate.gpu_type and row.gpu_count == 1
        for row in rows
    )


def _post_victim_safe(candidate: _Candidate, victim_id: int) -> bool:
    nodes = _node_snapshot()
    jobs = _job_snapshot()
    matching = [fields for fields in nodes if fields.get("NodeName") == candidate.node]
    if len(matching) != 1:
        return False
    fields = matching[0]
    total = _count_gpus(fields.get("Gres", "")).get(candidate.gpu_type, 0)
    used = _count_gpus(fields.get("GresUsed", "")).get(candidate.gpu_type, 0)
    normal = [
        job for job in jobs if job.state == "RUNNING" and job.qos == "normal"
        and job.node == candidate.node and job.gpu_count > 0
    ]
    return total > 0 and used == total and len(normal) == 1 and normal[0].job_id == str(victim_id)


def _measurement(event_log: Path) -> tuple[int | None, bool]:
    if not event_log.exists():
        return None, False
    signal_at = None
    heartbeats = []
    restarted = False
    for line in event_log.read_text().splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0] == "heartbeat" and fields[1].isdigit():
            heartbeats.append(int(fields[1]))
        elif len(fields) == 3 and fields[:2] == ["signal", "USR1"] and fields[2].isdigit():
            signal_at = signal_at or int(fields[2])
        elif len(fields) == 2 and fields[0] == "restart":
            restarted = True
    if signal_at is None:
        return None, restarted
    later = [timestamp for timestamp in heartbeats if timestamp >= signal_at]
    return (max(later) - signal_at if later else 0), restarted


def _cleanup(created: list[tuple[int, str]], user: str) -> list[str]:
    results = []
    for job_id, expected_qos in created:
        try:
            rows = _required(["squeue", "-h", "-j", str(job_id), "-o", _OWNER_QOS_FORMAT]).splitlines()
            owned = any(line.strip() == f"{job_id}|{user}|{expected_qos}" for line in rows)
            if not owned:
                results.append(f"left {job_id}: ownership/QoS verification failed")
                continue
            _required(["scancel", str(job_id)])
            results.append(f"cancelled {job_id}")
        except _QueryFailure as exc:
            results.append(f"left {job_id}: cleanup query failed: {exc}")
    return results


def probe_preemption(dry_run: bool = True, max_seconds: int = 600) -> str:
    """Safely preview or run a node-pinned, disposable normal-vs-golden probe."""
    if max_seconds <= 0:
        return "refused: max_seconds must be positive; no jobs submitted."
    try:
        candidate = _find_candidate(_node_snapshot(), _job_snapshot())
    except _QueryFailure as exc:
        return f"refused: scheduler safety query failed: {exc}; no jobs submitted."
    if candidate is None:
        return "refused: no isolated node exists; no jobs submitted."

    preview_dir = PROBE_ROOT / "DRY_RUN"
    victim_script, preemptor_script = _probe_scripts(candidate, preview_dir)
    evidence = "safety evidence: one free " + candidate.gpu_type + " GPU; no running normal-QoS GPU job"
    if dry_run:
        return "\n".join([
            "dry run: no jobs submitted.",
            f"candidate: node={candidate.node} gpu={candidate.gpu_type} golden_partition={candidate.golden_partition}",
            evidence,
            "--- victim script ---", victim_script,
            "--- preemptor script ---", preemptor_script,
        ])

    user = os.environ.get("USER", "")
    try:
        # A preview snapshot can be stale by the time an explicit real probe is
        # requested. Re-run the complete isolation check just before sbatch.
        candidate = _find_candidate(_node_snapshot(), _job_snapshot())
    except _QueryFailure as exc:
        return f"refused: scheduler safety query failed: {exc}; no jobs submitted."
    if candidate is None:
        return "refused: no isolated node exists; no jobs submitted."
    probe_dir = PROBE_ROOT / f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
    probe_dir.mkdir(parents=True, mode=0o700)
    victim_script, preemptor_script = _probe_scripts(candidate, probe_dir)
    created: list[tuple[int, str]] = []
    outcome = ""
    try:
        victim_id = _submit(victim_script, probe_dir / "victim.sh")
        created.append((victim_id, "normal"))
        deadline = time.monotonic() + max_seconds
        while time.monotonic() < deadline:
            if _job_is_running(victim_id, candidate):
                break
            time.sleep(1)
        else:
            raise _Refusal("timeout waiting for disposable victim to run")
        if not _post_victim_safe(candidate, victim_id):
            raise _Refusal("post-victim safety check failed; preemptor was not submitted")
        preemptor_id = _submit(preemptor_script, probe_dir / "preemptor.sh")
        created.append((preemptor_id, PRIMARY_QOS))
        while time.monotonic() < deadline:
            estimate, restarted = _measurement(probe_dir / "victim-events.log")
            if restarted:
                if estimate is None:
                    outcome = "probe completed: victim restart observed; first signal timestamp was unavailable"
                else:
                    outcome = f"probe completed: victim restart observed; warning/grace estimate: {estimate}s (one-second precision)"
                break
            time.sleep(1)
        if not outcome:
            outcome = "timeout: no requeue signal observed before the bounded wait expired"
    except _Refusal as exc:
        outcome = f"refused: {exc}"
    except _QueryFailure as exc:
        outcome = f"refused: scheduler query failed: {exc}"
    except Exception as exc:
        outcome = f"refused: probe exception: {exc}"
    finally:
        cleanup = _cleanup(created, user)
    return "\n".join([outcome, f"probe logs retained: {probe_dir}", "cleanup: " + "; ".join(cleanup or ["no jobs created"])])
