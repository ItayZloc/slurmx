"""Read-only preemption inspection and a tightly constrained scheduler probe."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import pwd
import re
import shlex
import time
import uuid

from config import GOLDEN_QOS
from config_defaults import MAIN_PARTITION

from . import shell
from .gpu_catalog import GPU_TYPES, PRIMARY_QOS


_LIST_JOBS = ("squeue", "-h", "-t", "RUNNING", "-o", "%i|%u|%q|%T|%N")
_CONFIG_KEYS = (
    ("SLURM version", "SLURM_VERSION"), ("PreemptType", "PreemptType"),
    ("PreemptMode", "PreemptMode"), ("PreemptParameters", "PreemptParameters"),
    ("JobRequeue", "JobRequeue"), ("KillWait", "KillWait"),
)
_GPU_TRES_RE = re.compile(r"(?:gres/)?gpu(?::([^,:=]+))?[:=](\d+)")
_SAFE_ATOM = re.compile(r"^[A-Za-z0-9_.-]+$")
_SAFE_USER = re.compile(r"^[a-z_][a-z0-9_-]*$")
_USABLE_STATES = frozenset({"IDLE", "MIXED", "ALLOCATED"})


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
class _Policy:
    victim_qos: str
    preemptible_qos: frozenset[str]


@dataclass(frozen=True)
class _Job:
    job_id: str
    user: str
    qos: str
    state: str
    node_list: str
    gpu_sources: tuple[dict[str, int], ...]

    @property
    def uses_gpu(self) -> bool:
        return any(sum(source.values()) > 0 for source in self.gpu_sources)


def _required(cmd: list[str] | tuple[str, ...]) -> str:
    try:
        return shell._run(list(cmd))
    except Exception as exc:
        raise _QueryFailure(str(exc)) from exc


def _config_values(raw: str) -> dict[str, str]:
    values = {}
    for line in raw.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def _qos_rows(raw: str) -> list[tuple[str, str, str, str]]:
    rows = []
    for line in raw.splitlines():
        fields = line.strip().split("|")
        if len(fields) < 4 or not fields[0].strip():
            raise _QueryFailure("unparseable QoS row")
        rows.append(tuple(field.strip() for field in fields[:4]))
    return rows


def _setting(values: dict[str, str], key: str) -> str:
    value = values.get(key, "").strip()
    return value or "unavailable (field missing or empty)"


def preemption_info() -> str:
    """Report controller/QoS preemption configuration without modifying jobs."""
    try:
        config = _config_values(_required(("scontrol", "show", "config")))
        lines = [f"{label}: {_setting(config, key)}" for label, key in _CONFIG_KEYS]
    except _QueryFailure as exc:
        lines = [f"{label}: unavailable (query failed: {exc})" for label, _ in _CONFIG_KEYS]
    try:
        rows = _qos_rows(_required((
            "sacctmgr", "-nP", "show", "qos", "format=Name,Preempt,PreemptMode,GraceTime",
        )))
        by_name = {name: (preempt, mode, grace) for name, preempt, mode, grace in rows}
        lines.append("QoS relationships:")
        for name in ["normal", *[q for q in GOLDEN_QOS if q != "normal"]]:
            row = by_name.get(name)
            if row is None:
                lines.append(f"{name}: unavailable (QoS not returned)")
                continue
            preempt, mode, grace = row
            lines.append(
                f"{name}: preempts {preempt or '<none configured>'}; "
                f"PreemptMode={mode or 'unavailable (unset)'}; "
                f"GraceTime={grace or 'unavailable (unset)'}"
            )
    except _QueryFailure as exc:
        lines.append(f"QoS relationships: unavailable (query failed: {exc})")
    return "\n".join(lines)


def _gpu_counts(value: str) -> dict[str, int]:
    value = value.strip()
    if not value or value in {"(null)", "N/A", "None"}:
        return {}
    counts: dict[str, int] = {}
    for gpu_type, count in _GPU_TRES_RE.findall(value):
        name = gpu_type or ""
        counts[name] = counts.get(name, 0) + int(count)
    if "gpu" in value.lower() and not counts:
        raise _QueryFailure(f"unparseable GPU allocation: {value}")
    return counts


def _node_fields(line: str) -> dict[str, str]:
    return dict(re.findall(r"(\w+)=([^\s]+)", line))


def _parse_job_rows(raw: str) -> list[_Job]:
    """Parse complete pipe-delimited job rows used by tests and job details."""
    rows = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split("|")]
        if len(fields) not in {7, 8} or any(not field for field in fields[:5]):
            raise _QueryFailure("unparseable or incomplete running-job row")
        sources = tuple(source for source in (_gpu_counts(value) for value in fields[5:]) if source)
        rows.append(_Job(*fields[:5], gpu_sources=sources))
    return rows


def _detail_job(listed: list[str]) -> _Job:
    if len(listed) != 5 or any(not field.strip() for field in listed):
        raise _QueryFailure("unparseable running-job listing")
    detail = _node_fields(_required(("scontrol", "show", "job", "-o", listed[0])))
    job_id = detail.get("JobId", "").split(".", 1)[0]
    user = detail.get("UserId", "").split("(", 1)[0]
    qos = detail.get("QOS", "")
    state = detail.get("JobState", "")
    node_list = detail.get("NodeList", "")
    if not all((job_id, user, qos, state, node_list)):
        raise _QueryFailure(f"incomplete scheduler detail for job {listed[0]}")
    if (job_id, user, qos, state) != tuple(listed[:4]):
        raise _QueryFailure(f"scheduler detail disagrees with job listing for {listed[0]}")
    values = (detail.get("TresPerJob", ""), detail.get("TresPerNode", ""), detail.get("AllocTRES", ""))
    if not any(values):
        raise _QueryFailure(f"missing GPU allocation detail for job {listed[0]}")
    return _parse_job_rows("|".join([job_id, user, qos, state, node_list, *values]))[0]


def _job_snapshot(job_id: int | None = None) -> list[_Job]:
    command = list(_LIST_JOBS)
    if job_id is not None:
        command = ["squeue", "-h", "-j", str(job_id), "-o", "%i|%u|%q|%T|%N"]
    listed = []
    for line in _required(command).splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split("|")]
        if len(fields) != 5 or any(not field for field in fields):
            raise _QueryFailure("unparseable or incomplete running-job row")
        listed.append(fields)
    return [_detail_job(fields) for fields in listed]


def _node_snapshot() -> list[dict[str, str]]:
    return [_node_fields(line) for line in _required(("scontrol", "show", "node", "-d", "-o")).splitlines() if line.strip()]


def _node_is_usable(state: str) -> bool:
    upper = state.upper()
    forbidden = ("DOWN", "DRAIN", "FAIL", "MAINT", "NO_RESPOND", "POWER", "UNKNOWN")
    return upper in _USABLE_STATES and not any(word in upper for word in forbidden)


def _expand_nodelist(value: str) -> set[str]:
    if not value or any(char.isspace() or ord(char) < 32 for char in value):
        raise _QueryFailure("unsafe or empty scheduler nodelist")
    nodes = {line.strip() for line in _required(("scontrol", "show", "hostnames", value)).splitlines() if line.strip()}
    if not nodes or any(not _SAFE_ATOM.fullmatch(node) for node in nodes):
        raise _QueryFailure("unparseable expanded scheduler nodelist")
    return nodes


def _job_on_node(job: _Job, node: str) -> bool:
    return node in _expand_nodelist(job.node_list)


def _probe_policy() -> _Policy:
    config = _config_values(_required(("scontrol", "show", "config")))
    if "preempt/qos" not in _setting(config, "PreemptType").lower():
        raise _Refusal("controller PreemptType does not enable QoS preemption")
    rows = {name: preempt for name, preempt, _mode, _grace in _qos_rows(_required((
        "sacctmgr", "-nP", "show", "qos", "format=Name,Preempt,PreemptMode,GraceTime",
    )))}
    preempt = rows.get(PRIMARY_QOS)
    if preempt is None or not preempt.strip():
        raise _Refusal(f"primary golden QoS {PRIMARY_QOS!r} has no configured preemption relationship")
    tokens = {token for token in re.split(r"[,:\s]+", preempt) if token}
    if "ALL" in {token.upper() for token in tokens}:
        tokens = set(rows) - {PRIMARY_QOS}
    if "normal" not in tokens:
        raise _Refusal("normal QoS is not preemptible by the primary golden QoS")
    return _Policy(victim_qos="normal", preemptible_qos=frozenset(tokens))


def _find_candidate(nodes: list[dict[str, str]], jobs: list[_Job], policy: _Policy) -> _Candidate | None:
    golden = {gpu.name: gpu.golden_partition for gpu in GPU_TYPES if gpu.golden_partition}
    for fields in nodes:
        node = fields.get("NodeName", "")
        if not _SAFE_ATOM.fullmatch(node) or not _node_is_usable(fields.get("State", "")):
            continue
        if "GresUsed" not in fields:
            continue
        partitions = set(fields.get("Partitions", "").split(","))
        total, used = _gpu_counts(fields.get("Gres", "")), _gpu_counts(fields.get("GresUsed", ""))
        for gpu_type, golden_partition in golden.items():
            if MAIN_PARTITION not in partitions or golden_partition not in partitions:
                continue
            if total.get(gpu_type, 0) - used.get(gpu_type, 0) != 1:
                continue
            if any(job.state == "RUNNING" and job.qos in policy.preemptible_qos and job.uses_gpu and _job_on_node(job, node) for job in jobs):
                continue
            return _Candidate(node, gpu_type, golden_partition)
    return None


def _authenticated_probe_root() -> str:
    account = pwd.getpwuid(os.getuid())
    user, home = account.pw_name, account.pw_dir
    expected = f"/home/{user}"
    if not _SAFE_USER.fullmatch(user) or home != expected:
        raise _Refusal("authenticated account does not have a safe /home/<user> directory")
    root = f"{expected}/.slurmx/probes"
    if any(ord(char) < 32 or char.isspace() for char in root) or not root.startswith("/home/"):
        raise _Refusal("unsafe authenticated probe directory")
    return root


def _time_limit(seconds: int) -> str:
    seconds += 300
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _probe_scripts(candidate: _Candidate, probe_dir: Path, *, max_seconds: int, victim_qos: str) -> tuple[str, str]:
    directives = (MAIN_PARTITION, candidate.node, candidate.gpu_type, candidate.golden_partition, PRIMARY_QOS, victim_qos)
    if not all(_SAFE_ATOM.fullmatch(value) for value in directives):
        raise _Refusal("unsafe scheduler candidate")
    output_dir = str(probe_dir)
    if not output_dir.startswith("/home/") or any(char.isspace() or ord(char) < 32 for char in output_dir):
        raise _Refusal("unsafe probe output directory")
    event_log = probe_dir / "victim-events.log"
    victim = f"""#!/bin/bash
#SBATCH --job-name=slurmx-preemption-victim
#SBATCH --partition={MAIN_PARTITION}
#SBATCH --qos={victim_qos}
#SBATCH --nodelist={candidate.node}
#SBATCH --gres=gpu:{candidate.gpu_type}:1
#SBATCH --time={_time_limit(max_seconds)}
#SBATCH --requeue
#SBATCH --output={output_dir}/victim-%j.out
set -u
event_log={shlex.quote(str(event_log))}
record_signal() {{ printf 'signal %s %s\\n' "$1" "$(date +%s)" >> "$event_log"; }}
if test "${{SLURM_RESTART_COUNT:-0}}" -gt 0; then printf 'restart %s\\n' "$(date +%s)" >> "$event_log"; exit 0; fi
trap 'record_signal USR1' USR1
trap 'record_signal TERM; exit 0' TERM
while :; do printf 'heartbeat %s\\n' "$(date +%s)" >> "$event_log"; sleep 1; done
"""
    preemptor = f"""#!/bin/bash
#SBATCH --job-name=slurmx-preemption-preemptor
#SBATCH --partition={candidate.golden_partition}
#SBATCH --qos={PRIMARY_QOS}
#SBATCH --nodelist={candidate.node}
#SBATCH --gres=gpu:{candidate.gpu_type}:1
#SBATCH --time={_time_limit(max_seconds)}
#SBATCH --output={output_dir}/preemptor-%j.out
sleep 30
"""
    return victim, preemptor


def _submit(script: str, path: Path) -> int:
    path.write_text(script)
    path.chmod(0o700)
    output = _required(("sbatch", "--parsable", str(path))).strip()
    match = re.fullmatch(r"(\d+)(?:;[^;\s]+)?", output)
    if not match:
        raise _QueryFailure(f"sbatch returned an unrecognized job ID: {output!r}")
    return int(match.group(1))


def _exact_gpu(job: _Job, candidate: _Candidate) -> bool:
    nonempty = [source for source in job.gpu_sources if source]
    return bool(nonempty) and all(source == {candidate.gpu_type: 1} for source in nonempty)


def _verify_victim(job_id: int, candidate: _Candidate, policy: _Policy, user: str) -> bool:
    rows = _job_snapshot(job_id)
    return len(rows) == 1 and (
        (job := rows[0]).job_id == str(job_id) and job.user == user and job.qos == policy.victim_qos
        and job.state == "RUNNING" and _job_on_node(job, candidate.node) and _exact_gpu(job, candidate)
    )


def _post_victim_safe(candidate: _Candidate, victim_id: int, policy: _Policy) -> bool:
    fields = [node for node in _node_snapshot() if node.get("NodeName") == candidate.node]
    if len(fields) != 1 or not _node_is_usable(fields[0].get("State", "")) or "GresUsed" not in fields[0]:
        return False
    total = _gpu_counts(fields[0].get("Gres", "")).get(candidate.gpu_type, 0)
    used = _gpu_counts(fields[0].get("GresUsed", "")).get(candidate.gpu_type, 0)
    preemptible = [job for job in _job_snapshot() if job.qos in policy.preemptible_qos and job.uses_gpu and _job_on_node(job, candidate.node)]
    return total > 0 and used == total and len(preemptible) == 1 and preemptible[0].job_id == str(victim_id)


def _measurement(event_log: Path) -> tuple[int | None, str | None, bool]:
    if not event_log.exists():
        return None, None, False
    signal = None
    heartbeats, restarted = [], False
    for line in event_log.read_text().splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0] == "heartbeat" and fields[1].isdigit():
            heartbeats.append(int(fields[1]))
        elif len(fields) == 3 and fields[0] == "signal" and fields[1] in {"TERM", "USR1"} and fields[2].isdigit() and signal is None:
            signal = (fields[1], int(fields[2]))
        elif len(fields) == 2 and fields[0] == "restart" and fields[1].isdigit():
            restarted = True
    if signal is None:
        return None, None, restarted
    name, at = signal
    after = [heartbeat for heartbeat in heartbeats if heartbeat >= at]
    return (max(after) - at if after else 0), name, restarted


def _cleanup(created: list[tuple[int, str]], user: str) -> list[str]:
    results = []
    for job_id, qos in created:
        try:
            raw = _required(("squeue", "-h", "-j", str(job_id), "-o", "%i|%u|%q"))
            owned = any(line.strip() == f"{job_id}|{user}|{qos}" for line in raw.splitlines())
            if not owned:
                results.append(f"left {job_id}: ownership/QoS verification failed")
                continue
            _required(("scancel", str(job_id)))
            results.append(f"cancelled {job_id}")
        except _QueryFailure as exc:
            results.append(f"left {job_id}: cleanup query failed: {exc}")
    return results


def probe_preemption(dry_run: bool = True, max_seconds: int = 600) -> str:
    """Preview or run a disposable, internally pinned preemption diagnostic."""
    if not 0 < max_seconds <= 3600:
        return "refused: max_seconds must be between 1 and 3600; no jobs submitted."
    try:
        policy = _probe_policy()
        candidate = _find_candidate(_node_snapshot(), _job_snapshot(), policy)
    except _Refusal as exc:
        return f"refused: {exc}; no jobs submitted."
    except _QueryFailure as exc:
        return f"refused: scheduler safety query failed: {exc}; no jobs submitted."
    if candidate is None:
        return "refused: no isolated node exists; no jobs submitted."
    try:
        root = _authenticated_probe_root()
        preview_dir = Path(root) / "DRY_RUN"
        victim_script, preemptor_script = _probe_scripts(candidate, preview_dir, max_seconds=max_seconds, victim_qos=policy.victim_qos)
    except _Refusal as exc:
        return f"refused: {exc}; no jobs submitted."
    if dry_run:
        return "\n".join(("dry run: no jobs submitted.", f"candidate: node={candidate.node} gpu={candidate.gpu_type} golden_partition={candidate.golden_partition}", "safety evidence: one free GPU; no running QoS that the primary golden QoS can preempt", "--- victim script ---", victim_script, "--- preemptor script ---", preemptor_script))

    user = pwd.getpwuid(os.getuid()).pw_name
    try:
        policy = _probe_policy()
        candidate = _find_candidate(_node_snapshot(), _job_snapshot(), policy)
    except (_QueryFailure, _Refusal) as exc:
        return f"refused: scheduler safety query failed: {exc}; no jobs submitted."
    if candidate is None:
        return "refused: no isolated node exists; no jobs submitted."
    probe_dir = Path(root) / f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
    probe_dir.mkdir(parents=True, mode=0o700)
    victim_script, preemptor_script = _probe_scripts(candidate, probe_dir, max_seconds=max_seconds, victim_qos=policy.victim_qos)
    created: list[tuple[int, str]] = []
    outcome = ""
    try:
        victim_id = _submit(victim_script, probe_dir / "victim.sh")
        created.append((victim_id, policy.victim_qos))
        deadline = time.monotonic() + max_seconds
        while time.monotonic() < deadline:
            if _verify_victim(victim_id, candidate, policy, user):
                break
            time.sleep(1)
        else:
            raise _Refusal("timeout waiting for an exactly verified disposable victim")
        if not _post_victim_safe(candidate, victim_id, policy) or not _verify_victim(victim_id, candidate, policy, user):
            raise _Refusal("post-victim safety check failed; preemptor was not submitted")
        preemptor_id = _submit(preemptor_script, probe_dir / "preemptor.sh")
        created.append((preemptor_id, PRIMARY_QOS))
        while time.monotonic() < deadline:
            estimate, signal, restarted = _measurement(probe_dir / "victim-events.log")
            if restarted:
                outcome = "probe completed: victim restart observed; " + (f"{signal} warning/grace estimate: {estimate}s (one-second precision)" if signal is not None else "timing unavailable: no recorded preemption signal")
                break
            time.sleep(1)
        if not outcome:
            outcome = "timeout: no recorded preemption signal observed before the bounded wait expired"
    except _Refusal as exc:
        outcome = f"refused: {exc}"
    except _QueryFailure as exc:
        outcome = f"refused: scheduler query failed: {exc}"
    except Exception as exc:
        outcome = f"refused: probe exception: {exc}"
    finally:
        cleanup = _cleanup(created, user)
    return "\n".join((outcome, f"probe logs retained: {probe_dir}", "cleanup: " + "; ".join(cleanup or ["no jobs created"])))
