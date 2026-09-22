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


_LIST_JOBS = ("squeue", "--all", "-h", "-t", "RUNNING", "-o", "%A|%u|%q|%T|%N")
_CONFIG_KEYS = (
    ("SLURM version", "SLURM_VERSION"), ("PreemptType", "PreemptType"),
    ("PreemptMode", "PreemptMode"), ("PreemptParameters", "PreemptParameters"),
    ("JobRequeue", "JobRequeue"), ("KillWait", "KillWait"),
)
_SAFE_ATOM = re.compile(r"^[A-Za-z0-9_.-]+$")
_SAFE_USER = re.compile(r"^[a-z_][a-z0-9_-]*$")
_USABLE_STATES = frozenset({"IDLE", "MIXED", "ALLOCATED"})


class _QueryFailure(RuntimeError):
    pass


class _Refusal(RuntimeError):
    pass


@dataclass(frozen=True)
class _Budget:
    max_seconds: int
    deadline: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "deadline", time.monotonic() + self.max_seconds)

    def remaining(self) -> float:
        return self.deadline - time.monotonic()

    def before_mutation(self) -> None:
        if self.remaining() <= 0:
            raise _Refusal("probe deadline expired before scheduler mutation")

    def query_timeout(self) -> float:
        remaining = self.remaining()
        if remaining <= 0:
            raise _Refusal("probe deadline expired before scheduler mutation")
        return min(30.0, remaining)


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
    per_node_raw: str = ""
    per_job_raw: str = ""
    allocated_raw: str = ""
    exclusive: str = ""
    oversubscribe: str = ""


def _required(cmd: list[str] | tuple[str, ...], budget: _Budget | None = None) -> str:
    try:
        if budget is None:
            return shell._run(list(cmd))
        return shell._run(list(cmd), timeout=budget.query_timeout())
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


def _top_level_tokens(value: str) -> list[str]:
    tokens, start, depth = [], 0, 0
    for index, char in enumerate(value):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise _QueryFailure(f"unbalanced GPU allocation annotation: {value}")
        elif char == "," and depth == 0:
            tokens.append(value[start:index].strip())
            start = index + 1
    if depth:
        raise _QueryFailure(f"unbalanced GPU allocation annotation: {value}")
    tokens.append(value[start:].strip())
    if any(not token for token in tokens):
        raise _QueryFailure(f"unparseable empty allocation fragment: {value}")
    return tokens


def _gpu_source(value: str) -> dict[str, int] | None:
    value = value.strip()
    if not value or value == "(null)":
        return None
    aggregate: int | None = None
    typed: dict[str, int] = {}
    for fragment in _top_level_tokens(value):
        if "(" in fragment:
            if not fragment.endswith(")"):
                raise _QueryFailure(f"unparseable GPU allocation annotation: {fragment}")
            clean = fragment.split("(", 1)[0].strip()
        else:
            clean = fragment
        if "gpu" not in clean.lower():
            if not re.fullmatch(
                r"[A-Za-z][A-Za-z0-9_./-]*(?:=|:)\d+(?:\.\d+)?(?:[KMGTPE](?:i?B)?)?",
                clean, re.IGNORECASE,
            ):
                raise _QueryFailure(f"unparseable allocation fragment: {fragment}")
            continue
        match = re.fullmatch(r"(?:gres/)?gpu(?::([A-Za-z0-9_.-]+))?(?:=|:)(\d+)", clean)
        if match is None:
            raise _QueryFailure(f"unparseable GPU allocation fragment: {fragment}")
        gpu_type, count = match.groups()
        if gpu_type is None:
            if aggregate is not None:
                raise _QueryFailure(f"duplicate aggregate GPU allocation: {value}")
            aggregate = int(count)
        else:
            typed[gpu_type] = typed.get(gpu_type, 0) + int(count)
    if aggregate is not None and typed and aggregate != sum(typed.values()):
        raise _QueryFailure(f"inconsistent aggregate and typed GPU allocation: {value}")
    if typed:
        return {gpu_type: count for gpu_type, count in typed.items() if count > 0}
    if aggregate is not None:
        return {"": aggregate} if aggregate else {}
    return {}


def _gpu_counts(value: str) -> dict[str, int]:
    source = _gpu_source(value)
    if source is None:
        raise _QueryFailure("missing GPU allocation evidence")
    return source


def _normalize_allocation_sources(sources: list[dict[str, int]]) -> dict[str, int]:
    """Coalesce a matching aggregate and typed scheduler representation."""
    typed = [source for source in sources if "" not in source]
    aggregate = [source for source in sources if set(source) == {""}]
    if typed:
        canonical = typed[0]
        if any(source != canonical for source in typed) or any(source[""] != sum(canonical.values()) for source in aggregate):
            raise _QueryFailure("inconsistent aggregate and typed GPU allocation sources")
        return canonical
    if len({tuple(sorted(source.items())) for source in sources}) != 1:
        raise _QueryFailure("inconsistent GPU allocation sources for running job")
    return sources[0]


def _node_fields(line: str) -> dict[str, str]:
    return dict(re.findall(r"(\w+)=([^\s]+)", line))


def _detail_job(listed: list[str], budget: _Budget | None = None) -> _Job:
    if len(listed) != 5 or any(not field.strip() for field in listed):
        raise _QueryFailure("unparseable running-job listing")
    detail = _node_fields(_required(("scontrol", "show", "job", "-o", listed[0]), budget))
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
    return _Job(
        job_id, user, qos, state, node_list, per_job_raw=values[0],
        per_node_raw=values[1], allocated_raw=values[2],
        exclusive=detail.get("Exclusive", ""), oversubscribe=detail.get("OverSubscribe", ""),
    )


def _job_snapshot(job_id: int | None = None, budget: _Budget | None = None, *, details: bool = True) -> list[_Job]:
    command = list(_LIST_JOBS)
    if job_id is not None:
        command = ["squeue", "-h", "-j", str(job_id), "-o", "%A|%u|%q|%T|%N"]
    listed = []
    for line in _required(command, budget).splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split("|")]
        if (len(fields) != 5 or any(not field for field in fields)
                or not re.fullmatch(r"[0-9]+", fields[0]) or fields[3] != "RUNNING"):
            raise _QueryFailure("unparseable or incomplete running-job row")
        listed.append(fields)
    if details:
        return [_detail_job(fields, budget) for fields in listed]
    return [_Job(*fields) for fields in listed]


def _node_snapshot(budget: _Budget | None = None) -> list[dict[str, str]]:
    return [_node_fields(line) for line in _required(("scontrol", "show", "node", "-d", "-o"), budget).splitlines() if line.strip()]


def _node_is_usable(state: str) -> bool:
    upper = state.upper()
    forbidden = ("DOWN", "DRAIN", "FAIL", "MAINT", "NO_RESPOND", "POWER", "UNKNOWN")
    return upper in _USABLE_STATES and not any(word in upper for word in forbidden)


def _expand_nodelist(value: str, budget: _Budget | None = None) -> set[str]:
    if not value or any(char.isspace() or ord(char) < 32 for char in value):
        raise _QueryFailure("unsafe or empty scheduler nodelist")
    if _SAFE_ATOM.fullmatch(value):
        return {value}
    nodes = {line.strip() for line in _required(("scontrol", "show", "hostnames", value), budget).splitlines() if line.strip()}
    if not nodes or any(not _SAFE_ATOM.fullmatch(node) for node in nodes):
        raise _QueryFailure("unparseable expanded scheduler nodelist")
    return nodes


def _multiply_allocation(source: dict[str, int], nodes: int) -> dict[str, int]:
    return {gpu_type: count * nodes for gpu_type, count in source.items()}


def _job_gpu_on_nodes(job: _Job, nodes: set[str]) -> dict[str, int]:
    """Return one node's allocation only when scheduler fields prove it."""
    if not nodes:
        raise _QueryFailure("empty scheduler job allocation")
    per_node = _gpu_source(job.per_node_raw)
    totals = [
        source for value in (job.per_job_raw, job.allocated_raw)
        if (source := _gpu_source(value)) is not None
    ]
    if per_node is None and not totals:
        raise _QueryFailure("missing GPU allocation evidence for running job")
    if per_node is not None:
        totals.append(_multiply_allocation(per_node, len(nodes)))
    total = _normalize_allocation_sources(totals)
    if per_node is None and total and len(nodes) != 1:
        raise _QueryFailure("cannot attribute a multi-node total GPU allocation to one node")
    if per_node is not None and "" in per_node and len(nodes) > 1 and len(total) > 1:
        # Mixed job totals do not identify the GPU type placed on each node.
        return per_node
    return {gpu_type: count // len(nodes) for gpu_type, count in total.items()}


def _node_gpu_allocations(fields: dict[str, str]) -> tuple[dict[str, int], dict[str, int]]:
    total = _gpu_counts(fields.get("Gres", ""))
    used = _gpu_counts(fields.get("GresUsed", ""))
    if "" in total:
        raise _QueryFailure("node GPU inventory does not identify GPU types")
    if "" in used:
        if len(total) != 1:
            raise _QueryFailure("cannot attribute untyped GPU usage to a node GPU type")
        used = {next(iter(total)): used[""]}
    if any(count > total.get(gpu_type, 0) for gpu_type, count in used.items()):
        raise _QueryFailure("node GPU usage exceeds its typed inventory")
    return total, used


def _job_gpu_on_candidate(job: _Job, node: str, budget: _Budget | None = None) -> dict[str, int] | None:
    nodes = _expand_nodelist(job.node_list, budget)
    if node not in nodes:
        return None
    return _job_gpu_on_nodes(job, nodes)


def _probe_policy(budget: _Budget | None = None) -> _Policy:
    config = _config_values(_required(("scontrol", "show", "config"), budget))
    if "preempt/qos" not in _setting(config, "PreemptType").lower():
        raise _Refusal("controller PreemptType does not enable QoS preemption")
    rows = {name: preempt for name, preempt, _mode, _grace in _qos_rows(_required((
        "sacctmgr", "-nP", "show", "qos", "format=Name,Preempt,PreemptMode,GraceTime",
    ), budget))}
    preempt = rows.get(PRIMARY_QOS)
    if preempt is None or not preempt.strip():
        raise _Refusal(f"primary golden QoS {PRIMARY_QOS!r} has no configured preemption relationship")
    tokens = {token for token in re.split(r"[,:\s]+", preempt) if token}
    if "ALL" in {token.upper() for token in tokens}:
        tokens = set(rows) - {PRIMARY_QOS}
    if "normal" not in tokens:
        raise _Refusal("normal QoS is not preemptible by the primary golden QoS")
    return _Policy(victim_qos="normal", preemptible_qos=frozenset(tokens))


def _find_candidate(nodes: list[dict[str, str]], jobs: list[_Job], budget: _Budget | None = None) -> _Candidate | None:
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
            occupied = False
            for job in jobs:
                if node in _expand_nodelist(job.node_list, budget):
                    occupied = True
                    break
            if occupied:
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
#SBATCH --exclusive
#SBATCH --output={output_dir}/victim-%j.out
set -u
event_log={shlex.quote(str(event_log))}
record_signal() {{ printf 'signal %s %s\\n' "$1" "$(date +%s)" >> "$event_log"; }}
if test "${{SLURM_RESTART_COUNT:-0}}" -gt 0; then printf 'restart %s\\n' "$(date +%s)" >> "$event_log"; exit 0; fi
trap 'record_signal USR1' USR1
trap 'record_signal TERM' TERM
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


def _submit(script: str, path: Path, budget: _Budget) -> int:
    budget.before_mutation()
    path.write_text(script)
    path.chmod(0o700)
    output = _required(("sbatch", "--parsable", str(path)), budget).strip()
    match = re.fullmatch(r"(\d+)(?:;[^;\s]+)?", output)
    if not match:
        raise _QueryFailure(f"sbatch returned an unrecognized job ID: {output!r}")
    return int(match.group(1))


def _exact_gpu(job: _Job, candidate: _Candidate, nodes: set[str]) -> bool:
    allocation = _job_gpu_on_nodes(job, nodes)
    # The probe submitted this exact ID with a typed, pinned one-GPU request.
    return allocation in ({candidate.gpu_type: 1}, {"": 1})


def _victim_matches(job: _Job, job_id: int, candidate: _Candidate, policy: _Policy, user: str, nodes: set[str]) -> bool:
    if not (
        job.job_id == str(job_id) and job.user == user and job.qos == policy.victim_qos
        and job.state == "RUNNING" and nodes == {candidate.node} and _exact_gpu(job, candidate, nodes)
    ):
        return False
    if job.exclusive != "NODE" or job.oversubscribe != "NO":
        raise _Refusal("scheduler did not establish a node-exclusive victim; preemptor was not submitted")
    return True


def _verify_victim(job_id: int, candidate: _Candidate, policy: _Policy, user: str, budget: _Budget | None = None) -> bool:
    rows = _job_snapshot(job_id, budget)
    if len(rows) != 1:
        return False
    job = rows[0]
    try:
        nodes = _expand_nodelist(job.node_list, budget)
        return _victim_matches(job, job_id, candidate, policy, user, nodes)
    except _QueryFailure:
        return False


def _post_victim_safe(candidate: _Candidate, victim_id: int, policy: _Policy, user: str, budget: _Budget | None = None) -> bool:
    fields = [node for node in _node_snapshot(budget) if node.get("NodeName") == candidate.node]
    if len(fields) != 1 or not _node_is_usable(fields[0].get("State", "")) or "GresUsed" not in fields[0]:
        return False
    try:
        inventory, usage = _node_gpu_allocations(fields[0])
        total, used = inventory.get(candidate.gpu_type, 0), usage.get(candidate.gpu_type, 0)
        residents = []
        for job in _job_snapshot(budget=budget, details=False):
            if candidate.node in _expand_nodelist(job.node_list, budget):
                residents.append(job)
        if total <= 0 or used != total or len(residents) != 1:
            return False
        return residents[0].job_id == str(victim_id) and _verify_victim(victim_id, candidate, policy, user, budget)
    except _QueryFailure:
        return False


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


def _cleanup(created: list[tuple[int, str]], user: str, budget: _Budget) -> list[str]:
    results = []
    for job_id, qos in created:
        try:
            raw = _required(("squeue", "-h", "-j", str(job_id), "-o", "%i|%u|%q"), budget)
            owned = any(line.strip() == f"{job_id}|{user}|{qos}" for line in raw.splitlines())
            if not owned:
                results.append(f"left {job_id}: ownership/QoS verification failed")
                continue
            budget.before_mutation()
            _required(("scancel", str(job_id)), budget)
            results.append(f"cancelled {job_id}")
        except (_QueryFailure, _Refusal) as exc:
            results.append(f"left {job_id}: cleanup query failed: {exc}")
    return results


def probe_preemption(dry_run: bool = True, max_seconds: int = 600) -> str:
    """Preview or run a disposable, internally pinned preemption diagnostic."""
    if isinstance(max_seconds, bool) or not isinstance(max_seconds, int) or not 0 < max_seconds <= 3600:
        return "refused: max_seconds must be between 1 and 3600; no jobs submitted."
    budget = _Budget(max_seconds)
    try:
        policy = _probe_policy(budget)
        candidate = _find_candidate(_node_snapshot(budget), _job_snapshot(budget=budget, details=False), budget)
    except _Refusal as exc:
        return f"refused: {exc}; no jobs submitted."
    except _QueryFailure as exc:
        return f"refused: scheduler safety query failed: {exc}; no jobs submitted."
    if candidate is None:
        return "refused: no isolated node exists; no jobs submitted."
    try:
        budget.before_mutation()
        root = _authenticated_probe_root()
        preview_dir = Path(root) / "DRY_RUN"
        victim_script, preemptor_script = _probe_scripts(candidate, preview_dir, max_seconds=max_seconds, victim_qos=policy.victim_qos)
    except _Refusal as exc:
        return f"refused: {exc}; no jobs submitted."
    if dry_run:
        return "\n".join(("dry run: no jobs submitted.", f"candidate: node={candidate.node} gpu={candidate.gpu_type} golden_partition={candidate.golden_partition}", "safety evidence: one free GPU; no running jobs of any QoS on the candidate", "real-mode gate: victim requests --exclusive; scheduler must report Exclusive=NODE and OverSubscribe=NO, with the exact owned victim as the sole running job", "--- victim script ---", victim_script, "--- preemptor script ---", preemptor_script))

    probe_dir: Path | None = None
    created: list[tuple[int, str]] = []
    outcome = ""
    try:
        user = pwd.getpwuid(os.getuid()).pw_name
        budget.before_mutation()
        policy = _probe_policy(budget)
        candidate = _find_candidate(_node_snapshot(budget), _job_snapshot(budget=budget, details=False), budget)
        if candidate is None:
            raise _Refusal("no isolated node exists")
        budget.before_mutation()
        created_dir = Path(root) / f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
        created_dir.mkdir(parents=True, mode=0o700)
        probe_dir = created_dir
        victim_script, preemptor_script = _probe_scripts(candidate, probe_dir, max_seconds=max_seconds, victim_qos=policy.victim_qos)
        victim_id = _submit(victim_script, probe_dir / "victim.sh", budget)
        created.append((victim_id, policy.victim_qos))
        while budget.remaining() > 0:
            if _verify_victim(victim_id, candidate, policy, user, budget):
                break
            time.sleep(1)
        else:
            raise _Refusal("timeout waiting for an exactly verified disposable victim")
        if not _post_victim_safe(candidate, victim_id, policy, user, budget) or not _verify_victim(victim_id, candidate, policy, user, budget):
            raise _Refusal("post-victim safety check failed; preemptor was not submitted")
        budget.before_mutation()
        preemptor_id = _submit(preemptor_script, probe_dir / "preemptor.sh", budget)
        created.append((preemptor_id, PRIMARY_QOS))
        while budget.remaining() > 0:
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
        # Cleanup always has a distinct fixed window. It never broadens the IDs
        # it may cancel, and does not make the diagnostic exceed its deadline.
        cleanup_budget = _Budget(5)
        cleanup = _cleanup(created, user, cleanup_budget) if "user" in locals() else []
    logs = f"probe logs retained: {probe_dir}" if probe_dir is not None and probe_dir.exists() else "probe logs unavailable: probe directory was not created"
    return "\n".join((outcome, logs, "cleanup: " + "; ".join(cleanup or ["no jobs created"])))
