"""Mocked contracts for SLURM preemption inspection and the disposable probe."""

from __future__ import annotations

from pathlib import Path

import pytest


CONFIG = "SLURM_VERSION = 25.11.1\nPreemptType = preempt/qos\nPreemptMode = REQUEUE\nPreemptParameters = send_user_signal\nJobRequeue = 1\nKillWait = 30\n"
QOS = "normal||REQUEUE|0\nyisroel|normal|REQUEUE|120\n"
NODES_ONE_FREE = "NodeName=node-a State=MIXED Partitions=main,rtx6000 Gres=gpu:rtx_6000:2 GresUsed=gpu:rtx_6000:1\n"
NODES_FULL = "NodeName=node-a State=ALLOCATED Partitions=main,rtx6000 Gres=gpu:rtx_6000:2 GresUsed=gpu:rtx_6000:2\n"
NO_JOBS = ""
VICTIM_RUNNING = "101|probe-user|normal|RUNNING|node-a|gpu:rtx_6000:1\n"
JOB_LIST = ("squeue", "-h", "-t", "RUNNING", "-o", "%i|%u|%q|%T|%N")
NODE_DETAIL = ("scontrol", "show", "node", "-d", "-o")


def _job_detail(job_id, user, qos, state, nodes, tres_job="gres/gpu:rtx_6000=1", tres_node=""):
    return f"JobId={job_id} UserId={user}(1) QOS={qos} JobState={state} NodeList={nodes} TresPerJob={tres_job} TresPerNode={tres_node} AllocTRES={tres_job}\n"


def _reply(monkeypatch, responses):
    """Replace external scheduler commands with complete, command-keyed output."""
    from slurm_mcp import preemption

    calls = []

    def run(cmd, timeout=30):
        calls.append(cmd)
        response = responses.get(tuple(cmd), responses.get((cmd[0],)))
        if response is None:
            raise AssertionError(f"unexpected command: {cmd}")
        if isinstance(response, Exception):
            raise response
        if isinstance(response, list):
            return response.pop(0)
        return response

    monkeypatch.setattr(preemption.shell, "_run", run)
    return calls


def _scan_responses(nodes=NODES_ONE_FREE, jobs=NO_JOBS):
    responses = {
        ("scontrol", "show", "config"): CONFIG,
        ("sacctmgr", "-nP", "show", "qos", "format=Name,Preempt,PreemptMode,GraceTime"): QOS,
        NODE_DETAIL: nodes,
        JOB_LIST: NO_JOBS,
    }
    if jobs:
        job_id, user, qos, state, node, _gpu = jobs.strip().split("|")
        responses[JOB_LIST] = "|".join((job_id, user, qos, state, node)) + "\n"
        responses[("scontrol", "show", "job", "-o", job_id)] = _job_detail(job_id, user, qos, state, node)
        responses[("scontrol", "show", "hostnames", node)] = node + "\n"
    return responses


def _allocation_responses(*, per_job="(null)", per_node="(null)", allocated="(null)", node_list="node-a", expanded="node-a\n"):
    row = f"101|probe-user|normal|RUNNING|{node_list}\n"
    responses = _scan_responses()
    responses.update({
        JOB_LIST: row,
        ("squeue", "-h", "-j", "101", "-o", "%i|%u|%q|%T|%N"): row,
        ("scontrol", "show", "job", "-o", "101"): (
            "JobId=101 UserId=probe-user(1) QOS=normal JobState=RUNNING "
            f"NodeList={node_list} TresPerJob={per_job} TresPerNode={per_node} AllocTRES={allocated}\n"
        ),
        ("scontrol", "show", "hostnames", node_list): expanded,
    })
    return responses


@pytest.mark.parametrize("partitions", ["cpu", "main", "rtx6000", "main,other"])
@pytest.mark.parametrize("gres", ["(null)", "unavailable"])
def test_probe_skips_irrelevant_nodes_before_requiring_gpu_evidence(monkeypatch, partitions, gres):
    from slurm_mcp.preemption import probe_preemption

    cpu = f"NodeName=cpu-a State=IDLE Partitions={partitions} Gres={gres} GresUsed={gres}\n"
    _reply(monkeypatch, _scan_responses(nodes=cpu + NODES_ONE_FREE))

    assert "candidate: node=node-a gpu=rtx_6000" in probe_preemption()


@pytest.mark.parametrize("per_node,per_job,allocated,want,exact", [
    ("gres/gpu:rtx_6000=1", "(null)", "gres/gpu=1", {"rtx_6000": 1}, True),
    ("gres/gpu=1", "(null)", "gres/gpu=1,gres/gpu:rtx_6000=1", {"rtx_6000": 1}, True),
    ("gres/gpu=1", "gres/gpu:rtx_6000=1", "gres/gpu=1", {"rtx_6000": 1}, True),
    ("gres/gpu=1", "(null)", "gres/gpu=1", {"": 1}, True),
    ("(null)", "(null)", "gres/gpu=1", {"": 1}, True),
    ("gres/gpu:rtx_6000=1", "(null)", "(null)", {"rtx_6000": 1}, True),
    ("(null)", "gres/gpu=1", "gres/gpu:rtx_6000=1", {"rtx_6000": 1}, True),
    ("(null)", "gres/gpu=1,gres/gpu:rtx_6000=1", "gres/gpu=1,gres/gpu:rtx_6000=1", {"rtx_6000": 1}, True),
    ("gres/gpu:other=1", "(null)", "gres/gpu=1", {"other": 1}, False),
    ("gres/gpu=2", "(null)", "gres/gpu:rtx_6000=2", {"rtx_6000": 2}, False),
    ("(null)", "(null)", "cpu=2,mem=4G,node=1,billing=2", {}, False),
    ("gres/gpu=0", "(null)", "cpu=2", {}, False),
    ("gres/gpu:rtx_6000=0", "gres/gpu=0", "cpu=2", {}, False),
    ("gres/gpu:rtx_6000=1", "(null)", "gres/gpu:other=1", None, False),
    ("gres/gpu:rtx_6000=1", "(null)", "gres/gpu=2", None, False),
    ("gres/gpu=1", "gres/gpu:rtx_6000=1", "gres/gpu:other=1", None, False),
    ("(null)", "(null)", "gres/gpu=2,gres/gpu:rtx_6000=1", None, False),
    ("(null)", "(null)", "(null)", None, False),
    ("", "", "", None, False),
    ("(null)", "(null)", "cpu=unavailable", None, False),
    ("cpu=unknown", "(null)", "gres/gpu=0", None, False),
    ("(null)", "mem=N/A", "cpu=2", None, False),
    ("(null)", "cpu=2", "gres/gpu=1,cpu=unknown", None, False),
    ("gres/gpu:rtx_6000=1", "cpu=unavailable", "gres/gpu=1", None, False),
    ("(null)", "(null)", "gres/gpu:rtx_6000=0,gres/gpu:broken", None, False),
    ("(null)", "(null)", "gpu:rtx_6000:3(IDX:0,2-3)", {"rtx_6000": 3}, False),
])
def test_production_job_allocation_matrix(monkeypatch, per_node, per_job, allocated, want, exact):
    from slurm_mcp.preemption import (
        _Candidate, _Policy, _QueryFailure, _find_candidate, _job_gpu_on_candidate,
        _job_snapshot, _node_snapshot, _verify_victim,
    )

    _reply(monkeypatch, _allocation_responses(per_node=per_node, per_job=per_job, allocated=allocated))
    jobs = _job_snapshot()
    policy = _Policy("normal", frozenset({"normal"}))
    if want is None:
        with pytest.raises(_QueryFailure):
            _job_gpu_on_candidate(jobs[0], "node-a")
        with pytest.raises(_QueryFailure):
            _find_candidate(_node_snapshot(), jobs, policy)
    else:
        assert _job_gpu_on_candidate(jobs[0], "node-a") == want
        candidate = _find_candidate(_node_snapshot(), jobs, policy)
        assert (candidate is not None) == (want == {})
    assert _verify_victim(101, _Candidate("node-a", "rtx_6000", "rtx6000"), policy, "probe-user") is exact


@pytest.mark.parametrize("per_node,per_job,allocated,want", [
    ("gres/gpu:rtx_6000=1", "(null)", "gres/gpu=2", {"rtx_6000": 1}),
    ("gres/gpu=1", "(null)", "gres/gpu=2,gres/gpu:rtx_6000=2", {"rtx_6000": 1}),
    ("gres/gpu=1", "gres/gpu:rtx_6000=2", "gres/gpu=2", {"rtx_6000": 1}),
    ("gres/gpu=1", "(null)", "gres/gpu=2", {"": 1}),
    ("gres/gpu:rtx_6000=1", "(null)", "gres/gpu:rtx_6000=2", {"rtx_6000": 1}),
    ("gres/gpu:rtx_6000=1,gres/gpu:other=1", "(null)", "gres/gpu=4", {"rtx_6000": 1, "other": 1}),
    ("gres/gpu=2", "(null)", "gres/gpu:rtx_6000=2,gres/gpu:other=2", {"": 2}),
    ("gres/gpu=1", "(null)", "gres/gpu:rtx_6000=1,gres/gpu:other=1", {"": 1}),
    ("gres/gpu:rtx_6000=1", "(null)", "gres/gpu=1", None),
    ("gres/gpu:rtx_6000=1", "gres/gpu=2", "gres/gpu:other=2", None),
    ("(null)", "(null)", "gres/gpu=2", None),
    ("(null)", "(null)", "gres/gpu=0", {}),
    ("gres/gpu:rtx_6000=0", "(null)", "cpu=4", {}),
])
def test_production_multinode_allocation_matrix(monkeypatch, per_node, per_job, allocated, want):
    from slurm_mcp.preemption import _QueryFailure, _job_gpu_on_candidate, _job_snapshot

    _reply(monkeypatch, _allocation_responses(
        per_node=per_node, per_job=per_job, allocated=allocated,
        node_list="node-[a-b]", expanded="node-a\nnode-b\n",
    ))
    job = _job_snapshot()[0]
    if want is None:
        with pytest.raises(_QueryFailure):
            _job_gpu_on_candidate(job, "node-a")
    else:
        assert _job_gpu_on_candidate(job, "node-a") == want


@pytest.mark.parametrize("inventory,used,expected", [
    ("gpu:rtx_6000:1", "gres/gpu=1", "full"),
    ("gpu:rtx_6000:2", "gres/gpu=1", "candidate"),
    ("gpu:rtx_6000:1", "gres/gpu=0", "candidate"),
    ("gpu:rtx_6000:1", "gres/gpu=1,gres/gpu:rtx_6000=1", "full"),
    ("gpu:rtx_6000:1,gpu:other:1", "gres/gpu=1", "unknown"),
    ("gpu:rtx_6000:1,gpu:other:1", "gres/gpu=0", "candidate"),
    ("gpu:rtx_6000:1,gpu:other:1", "gpu:rtx_6000:0,gpu:other:1", "candidate"),
    ("gpu:rtx_6000:1", "cpu=unavailable", "unknown"),
    ("gpu:rtx_6000:1", "(null)", "unknown"),
    ("gpu:rtx_6000:1", "unavailable", "unknown"),
    ("(null)", "(null)", "unknown"),
    ("gpu:rtx_6000:1", "gpu:other:1", "unknown"),
    ("gpu:rtx_6000:1", "gpu:rtx_6000:2", "unknown"),
    ("gpu:rtx_6000:4", "gpu:rtx_6000:3(IDX:0,2-3)", "candidate"),
])
def test_production_node_allocation_matrix(monkeypatch, inventory, used, expected):
    from slurm_mcp.preemption import probe_preemption

    node = f"NodeName=node-a State=MIXED Partitions=main,rtx6000 Gres={inventory} GresUsed={used}\n"
    _reply(monkeypatch, _scan_responses(nodes=node))
    result = probe_preemption()

    if expected == "candidate":
        assert "candidate: node=node-a gpu=rtx_6000" in result
    elif expected == "full":
        assert result.startswith("refused: no isolated node exists")
    else:
        assert result.startswith("refused: scheduler safety query failed")


@pytest.mark.parametrize("evidence", ["unavailable", "(null)", ""])
def test_production_unrelated_job_allocation_is_not_required(monkeypatch, evidence):
    from slurm_mcp.preemption import probe_preemption

    _reply(monkeypatch, _allocation_responses(
        per_node=evidence, per_job=evidence, allocated=evidence,
        node_list="node-[b-c]", expanded="node-b\nnode-c\n",
    ))

    assert "candidate: node=node-a gpu=rtx_6000" in probe_preemption()


@pytest.mark.parametrize("used,safe", [
    ("gres/gpu=2", True),
    ("gres/gpu=1", False),
    ("gres/gpu=2,gres/gpu:rtx_6000=2", True),
    ("cpu=unavailable", False),
    ("(null)", False),
    ("gpu:other:2", False),
])
def test_post_victim_reconciles_node_usage_at_scheduler_boundary(monkeypatch, used, safe):
    from slurm_mcp.preemption import _Candidate, _Policy, _post_victim_safe

    responses = _allocation_responses(per_node="gres/gpu:rtx_6000=1", allocated="gres/gpu=1")
    responses[NODE_DETAIL] = f"NodeName=node-a State=MIXED Partitions=main,rtx6000 Gres=gpu:rtx_6000:2 GresUsed={used}\n"
    _reply(monkeypatch, responses)

    assert _post_victim_safe(
        _Candidate("node-a", "rtx_6000", "rtx6000"), 101,
        _Policy("normal", frozenset({"normal"})),
    ) is safe


def test_preemption_info_parses_controller_and_qos_settings(monkeypatch):
    """Dropping a controller/QoS field must change the inspection result."""
    from slurm_mcp.preemption import preemption_info

    _reply(monkeypatch, {
        ("scontrol", "show", "config"): CONFIG,
        ("sacctmgr", "-nP", "show", "qos", "format=Name,Preempt,PreemptMode,GraceTime"): QOS,
    })

    result = preemption_info()

    assert "SLURM version: 25.11.1" in result
    assert "PreemptType: preempt/qos" in result
    assert "PreemptParameters: send_user_signal" in result
    assert "JobRequeue: 1" in result
    assert "KillWait: 30" in result
    assert "normal: preempts <none configured>; PreemptMode=REQUEUE; GraceTime=0" in result
    assert "yisroel: preempts normal; PreemptMode=REQUEUE; GraceTime=120" in result


def test_preemption_info_makes_failed_queries_explicit(monkeypatch):
    """A failed QoS query must never look like an empty preemption policy."""
    from slurm_mcp.preemption import preemption_info

    _reply(monkeypatch, {
        ("scontrol", "show", "config"): RuntimeError("controller unavailable"),
        ("sacctmgr", "-nP", "show", "qos", "format=Name,Preempt,PreemptMode,GraceTime"): RuntimeError("accounting unavailable"),
    })

    result = preemption_info()

    assert "SLURM version: unavailable (query failed: controller unavailable)" in result
    assert "QoS relationships: unavailable (query failed: accounting unavailable)" in result


def test_probe_refuses_when_no_isolated_node_exists(monkeypatch):
    """Removing the one-free-GPU condition must submit no disposable jobs."""
    from slurm_mcp.preemption import probe_preemption

    calls = _reply(monkeypatch, _scan_responses(nodes=NODES_FULL))
    result = probe_preemption()

    assert result.startswith("refused: no isolated node exists; no jobs submitted.")
    assert not any(command[0] in {"sbatch", "scancel"} for command in calls)


def test_probe_refuses_when_normal_gpu_job_is_already_running(monkeypatch):
    """A normal GPU job on an otherwise suitable node makes the probe unsafe."""
    from slurm_mcp.preemption import probe_preemption

    calls = _reply(monkeypatch, _scan_responses(jobs="88|someone|normal|RUNNING|node-a|gpu:rtx_6000:1\n"))
    result = probe_preemption()

    assert result.startswith("refused: no isolated node exists; no jobs submitted.")
    assert not any(command[0] == "sbatch" for command in calls)


def test_probe_dry_run_reports_internal_pinning_scripts_and_evidence(monkeypatch):
    """The default path must expose its plan without calling sbatch."""
    from slurm_mcp.preemption import probe_preemption

    calls = _reply(monkeypatch, _scan_responses())
    result = probe_preemption()

    assert "dry run: no jobs submitted." in result
    assert "candidate: node=node-a gpu=rtx_6000 golden_partition=rtx6000" in result
    assert "#SBATCH --nodelist=node-a" in result
    assert "#SBATCH --qos=normal" in result
    assert "#SBATCH --qos=yisroel" in result
    assert "safety evidence: one free GPU; no running QoS that the primary golden QoS can preempt" in result
    assert not any(command[0] == "sbatch" for command in calls)


def test_victim_verification_requires_exact_identity_owner_qos_node_state_and_gpu(monkeypatch):
    """Relaxing any post-victim identity field would allow an unrelated job to trigger preemption."""
    from slurm_mcp.preemption import _Candidate, _Job, _Policy, _verify_victim

    victim = _Job("101", "probe-user", "normal", "RUNNING", "node-01", per_node_raw="gres/gpu:rtx_6000=1")
    monkeypatch.setattr("slurm_mcp.preemption._job_snapshot", lambda *args, **kwargs: [victim])
    monkeypatch.setattr("slurm_mcp.preemption._expand_nodelist", lambda *args, **kwargs: {"node-01"})
    candidate = _Candidate("node-01", "rtx_6000", "rtx6000")
    policy = _Policy("normal", frozenset({"normal", "alternate"}))

    assert _verify_victim(101, candidate, policy, "probe-user")
    assert not _verify_victim(101, candidate, policy, "other-user")


def test_post_victim_check_refuses_an_alternate_preemptible_race(monkeypatch):
    """A second QoS in the controller-derived preemptible set blocks the golden submit."""
    from slurm_mcp.preemption import _Candidate, _Job, _Policy, _post_victim_safe

    jobs = [
        _Job("101", "probe-user", "normal", "RUNNING", "node-a", per_node_raw="gres/gpu:rtx_6000=1"),
        _Job("102", "other", "alternate", "RUNNING", "node-a", per_node_raw="gres/gpu:rtx_6000=1"),
    ]
    monkeypatch.setattr("slurm_mcp.preemption._node_snapshot", lambda *args, **kwargs: [{"NodeName": "node-a", "State": "ALLOCATED", "Gres": "gpu:rtx_6000:2", "GresUsed": "gpu:rtx_6000:2"}])
    monkeypatch.setattr("slurm_mcp.preemption._job_snapshot", lambda *args, **kwargs: jobs)
    monkeypatch.setattr("slurm_mcp.preemption._expand_nodelist", lambda value, *args, **kwargs: {value})

    assert not _post_victim_safe(_Candidate("node-a", "rtx_6000", "rtx6000"), 101, _Policy("normal", frozenset({"normal", "alternate"})))


def test_measurement_requires_a_received_preemption_signal(tmp_path):
    """A restart without TERM/USR1 cannot fabricate a warning/grace interval."""
    from slurm_mcp.preemption import _measurement

    events = tmp_path / "victim-events.log"
    events.write_text("heartbeat 100\nheartbeat 112\nrestart 113\n")

    assert _measurement(events) == (None, None, True)


def test_cli_exposes_inspection_and_dry_run_probe(monkeypatch, capsys):
    """The CLI must keep the real probe opt-in and pass its bounded wait through."""
    from cli import preemption as cli_preemption
    from cli import slurmx

    parser = slurmx.build_parser()
    inspect_args = parser.parse_args(["preemption-info"])
    probe_args = parser.parse_args(["probe-preemption", "--real", "--max-seconds", "42"])
    monkeypatch.setattr(cli_preemption.slurm_mcp, "preemption_info", lambda: "INFO")
    monkeypatch.setattr(cli_preemption.slurm_mcp, "probe_preemption", lambda **kwargs: repr(kwargs))

    inspect_args._run(inspect_args)
    assert capsys.readouterr().out == "INFO\n"
    probe_args._run(probe_args)
    assert capsys.readouterr().out == "{'dry_run': False, 'max_seconds': 42}\n"


def test_mcp_exposes_preemption_operations(monkeypatch):
    """The server adapter must preserve the probe's dry-run default."""
    import sys
    import types

    mcp_module = types.ModuleType("mcp")
    mcp_server = types.ModuleType("mcp.server")
    fastmcp = types.ModuleType("mcp.server.fastmcp")

    class FakeMCP:
        def __init__(self, *args, **kwargs):
            pass

        def tool(self):
            return lambda function: function

    fastmcp.FastMCP = FakeMCP
    monkeypatch.setitem(sys.modules, "mcp", mcp_module)
    monkeypatch.setitem(sys.modules, "mcp.server", mcp_server)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fastmcp)
    previous = sys.modules.pop("server", None)
    try:
        import server
        monkeypatch.setattr(server.slurm_mcp, "preemption_info", lambda: "INFO")
        monkeypatch.setattr(server.slurm_mcp, "probe_preemption", lambda **kwargs: repr(kwargs))

        assert server.preemption_info() == "INFO"
        assert server.probe_preemption() == "{'dry_run': True, 'max_seconds': 600}"
    finally:
        if previous is None:
            sys.modules.pop("server", None)
        else:
            sys.modules["server"] = previous


def test_preemption_info_marks_empty_controller_and_qos_values_unavailable(monkeypatch):
    """Blank controller/QoS fields must not be rendered as known settings."""
    from slurm_mcp.preemption import preemption_info

    _reply(monkeypatch, {
        ("scontrol", "show", "config"): "SLURM_VERSION = \nPreemptType = \nPreemptMode = \nPreemptParameters = \nJobRequeue = \nKillWait = \n",
        ("sacctmgr", "-nP", "show", "qos", "format=Name,Preempt,PreemptMode,GraceTime"): "normal|||\nyisroel|||\n",
    })

    result = preemption_info()

    assert "PreemptType: unavailable (field missing or empty)" in result
    assert "normal: preempts <none configured>; PreemptMode=unavailable (unset); GraceTime=unavailable (unset)" in result


def test_tres_job_parser_treats_untyped_gpu_as_real_usage_and_rejects_malformed_rows(monkeypatch):
    """Changing a running untyped GPU row into an ignored row would make isolation unsafe."""
    from slurm_mcp.preemption import _QueryFailure, _job_gpu_on_candidate, _job_snapshot

    responses = _allocation_responses(per_job="gres/gpu=1")
    _reply(monkeypatch, responses)

    assert _job_gpu_on_candidate(_job_snapshot()[0], "node-a") == {"": 1}
    responses[JOB_LIST] = "not a complete scheduler row\n"
    with pytest.raises(_QueryFailure):
        _job_snapshot()


def test_tres_job_parser_uses_allocated_tres_when_job_and_node_requests_are_empty(monkeypatch):
    """Allocation detail is still GPU use when request-source fields are absent."""
    from slurm_mcp.preemption import _job_gpu_on_candidate, _job_snapshot

    _reply(monkeypatch, _allocation_responses(allocated="gres/gpu=1"))

    assert _job_gpu_on_candidate(_job_snapshot()[0], "node-a") == {"": 1}


def test_candidate_rejects_compressed_nodelist_and_alternative_preemptible_qos(monkeypatch):
    """A compressed allocation on the candidate node must block every preemptible QoS."""
    from slurm_mcp.preemption import _Policy, _QueryFailure, _find_candidate, _job_snapshot

    nodes = [{
        "NodeName": "node-01", "State": "MIXED", "Partitions": "main,rtx6000",
        "Gres": "gpu:rtx_6000:2", "GresUsed": "gpu:rtx_6000:1",
    }]
    responses = _allocation_responses(per_job="gres/gpu=1", node_list="node[01-02]", expanded="node-01\nnode-02\n")
    responses[JOB_LIST] = responses[JOB_LIST].replace("normal", "alternate")
    detail = ("scontrol", "show", "job", "-o", "101")
    responses[detail] = responses[detail].replace("normal", "alternate")
    _reply(monkeypatch, responses)
    jobs = _job_snapshot()
    policy = _Policy(victim_qos="normal", preemptible_qos=frozenset({"normal", "alternate"}))

    with pytest.raises(_QueryFailure, match="cannot attribute"):
        _find_candidate(nodes, jobs, policy)


@pytest.mark.parametrize("state", ["DOWN", "DRAINING", "FAIL", "MAINT", "NO_RESPOND", "POWER_DOWN", "UNKNOWN"])
def test_candidate_rejects_non_usable_node_states(state):
    """A node state outside the explicitly usable set cannot host a probe."""
    from slurm_mcp.preemption import _node_is_usable

    assert not _node_is_usable(state)


def test_candidate_requires_detailed_gres_used_field():
    """Missing GresUsed must not be interpreted as an unused GPU."""
    from slurm_mcp.preemption import _Policy, _QueryFailure, _find_candidate

    nodes = [{"NodeName": "node-a", "State": "MIXED", "Partitions": "main,rtx6000", "Gres": "gpu:rtx_6000:2"}]
    with pytest.raises(_QueryFailure, match="missing GPU allocation"):
        _find_candidate(nodes, [], _Policy("normal", frozenset({"normal"})))


def test_authenticated_probe_root_ignores_malicious_home_and_rejects_bad_account_path(monkeypatch):
    """Inherited HOME must not control a generated SBATCH output directive."""
    import types
    from slurm_mcp.preemption import _authenticated_probe_root

    monkeypatch.setattr("slurm_mcp.preemption.os.environ", {"HOME": "/tmp/evil\n#SBATCH --qos=evil"})
    monkeypatch.setattr("slurm_mcp.preemption.os.getuid", lambda: 123)
    monkeypatch.setattr("slurm_mcp.preemption.pwd.getpwuid", lambda _: types.SimpleNamespace(pw_name="probe-user", pw_dir="/home/probe-user"))
    assert _authenticated_probe_root() == "/home/probe-user/.slurmx/probes"
    monkeypatch.setattr("slurm_mcp.preemption.pwd.getpwuid", lambda _: types.SimpleNamespace(pw_name="probe-user", pw_dir="/home/probe-user\n#SBATCH --qos=evil"))
    with pytest.raises(Exception):
        _authenticated_probe_root()


def test_victim_script_has_no_time_limit_usr1_and_records_distinct_preemption_signals(tmp_path):
    """A scheduled time-limit warning would contaminate the preemption measurement."""
    from slurm_mcp.preemption import _Candidate, _probe_scripts

    victim, _ = _probe_scripts(_Candidate("node-a", "rtx_6000", "rtx6000"), Path("/home/probe-user/.slurmx/probes/example"), max_seconds=600, victim_qos="normal")

    assert "#SBATCH --signal=" not in victim
    assert "trap 'record_signal TERM' TERM" in victim
    assert "trap 'record_signal USR1' USR1" in victim
    assert "#SBATCH --time=00:15:00" in victim


def test_probe_scripts_reject_directive_injection_from_scheduler_fields():
    """A malicious scheduler/config value must not become a second SBATCH directive."""
    from slurm_mcp.preemption import _Candidate, _Refusal, _probe_scripts

    with pytest.raises(_Refusal):
        _probe_scripts(_Candidate("node-a", "rtx_6000", "rtx6000\n#SBATCH --qos=evil"), Path("/home/probe-user/.slurmx/probes/run"), max_seconds=600, victim_qos="normal")


def test_allocation_parser_normalizes_aggregate_and_typed_tres_without_double_counting(monkeypatch):
    """An aggregate plus matching typed GPU entry describes one allocation, not two."""
    from slurm_mcp.preemption import _job_gpu_on_candidate, _job_snapshot

    _reply(monkeypatch, _allocation_responses(
        per_job="gres/gpu=1,gres/gpu:rtx_6000=1", allocated="gres/gpu=1,gres/gpu:rtx_6000=1",
    ))

    assert _job_gpu_on_candidate(_job_snapshot()[0], "node-a") == {"rtx_6000": 1}


def test_allocation_parser_normalizes_matching_aggregate_and_typed_separate_sources(monkeypatch):
    """Per-job aggregate and allocated typed evidence agree on one typed GPU."""
    from slurm_mcp.preemption import _job_gpu_on_candidate, _job_snapshot

    _reply(monkeypatch, _allocation_responses(per_job="gres/gpu=1", allocated="gres/gpu:rtx_6000=1"))

    assert _job_gpu_on_candidate(_job_snapshot()[0], "node-a") == {"rtx_6000": 1}


@pytest.mark.parametrize("evidence", [
    "||",
    "gres/gpu:rtx_6000:0,gres/gpu:broken||",
    "gres/gpu:rtx_6000=1|gres/gpu:rtx_6000=2|",
])
def test_allocation_parser_refuses_missing_malformed_or_inconsistent_gpu_evidence(monkeypatch, evidence):
    """Any ambiguous GPU-bearing source makes a running job unsafe to ignore."""
    from slurm_mcp.preemption import _QueryFailure, _job_gpu_on_candidate, _job_snapshot

    per_job, per_node, allocated = evidence.split("|")
    _reply(monkeypatch, _allocation_responses(per_job=per_job, per_node=per_node, allocated=allocated))
    with pytest.raises(_QueryFailure):
        _job_gpu_on_candidate(_job_snapshot()[0], "node-a")


def test_exact_victim_requires_single_expanded_candidate_node(monkeypatch):
    """A multi-node job that merely contains the candidate is not a pinned victim."""
    from slurm_mcp.preemption import _Candidate, _Job, _Policy, _verify_victim

    victim = _Job("101", "probe-user", "normal", "RUNNING", "node[01-02]", per_node_raw="gres/gpu:rtx_6000=1")
    monkeypatch.setattr("slurm_mcp.preemption._job_snapshot", lambda *args, **kwargs: [victim])
    monkeypatch.setattr("slurm_mcp.preemption._expand_nodelist", lambda *args, **kwargs: {"node-01", "node-02"})

    assert not _verify_victim(101, _Candidate("node-01", "rtx_6000", "rtx6000"), _Policy("normal", frozenset({"normal"})), "probe-user")


def test_victim_signal_handlers_keep_heartbeating_until_scheduler_requeues():
    """A handler that exits itself cannot measure scheduler grace time."""
    from slurm_mcp.preemption import _Candidate, _probe_scripts

    victim, _ = _probe_scripts(_Candidate("node-a", "rtx_6000", "rtx6000"), Path("/home/probe-user/.slurmx/probes/run"), max_seconds=600, victim_qos="normal")

    assert "trap 'record_signal TERM' TERM" in victim
    assert "trap 'record_signal TERM; exit 0' TERM" not in victim


def test_budget_refuses_before_a_mutation_after_a_slow_safety_query(monkeypatch):
    """Once the bounded wall-clock budget is gone, the preemptor must not submit."""
    from slurm_mcp.preemption import _Budget, _Refusal

    ticks = iter((10.0, 11.1))
    monkeypatch.setattr("slurm_mcp.preemption.time.monotonic", lambda: next(ticks))
    budget = _Budget(1)
    with pytest.raises(_Refusal, match="deadline"):
        budget.before_mutation()


def _mock_real_probe(monkeypatch, tmp_path, *, verify=True, post=True, measurement=(4, "TERM", True)):
    """Install an end-to-end fake scheduler; no command reaches the cluster."""
    import types
    from slurm_mcp.preemption import _Candidate, _Policy

    candidate = _Candidate("node-a", "rtx_6000", "rtx6000")
    policy = _Policy("normal", frozenset({"normal"}))
    submitted, cleaned = [], []
    monkeypatch.setattr("slurm_mcp.preemption._probe_policy", lambda *args, **kwargs: policy)
    monkeypatch.setattr("slurm_mcp.preemption._node_snapshot", lambda *args, **kwargs: [])
    monkeypatch.setattr("slurm_mcp.preemption._job_snapshot", lambda *args, **kwargs: [])
    monkeypatch.setattr("slurm_mcp.preemption._find_candidate", lambda *args, **kwargs: candidate)
    monkeypatch.setattr("slurm_mcp.preemption._authenticated_probe_root", lambda: str(tmp_path))
    monkeypatch.setattr("slurm_mcp.preemption._probe_scripts", lambda *args, **kwargs: ("victim", "preemptor"))
    monkeypatch.setattr("slurm_mcp.preemption.pwd.getpwuid", lambda _: types.SimpleNamespace(pw_name="probe-user"))
    monkeypatch.setattr("slurm_mcp.preemption._verify_victim", lambda *args, **kwargs: verify)
    monkeypatch.setattr("slurm_mcp.preemption._post_victim_safe", lambda *args, **kwargs: post)
    monkeypatch.setattr("slurm_mcp.preemption._measurement", lambda *args, **kwargs: measurement)
    monkeypatch.setattr("slurm_mcp.preemption._submit", lambda script, path, budget: submitted.append(script) or len(submitted) + 100)
    monkeypatch.setattr("slurm_mcp.preemption._cleanup", lambda created, user, budget: cleaned.append((created, user, budget)) or ["cleaned"])
    return submitted, cleaned


def test_real_probe_success_is_fully_mocked_and_reports_signal_timing(monkeypatch, tmp_path):
    """A complete real-mode success path must be testable without SLURM access."""
    from slurm_mcp.preemption import probe_preemption

    submitted, cleaned = _mock_real_probe(monkeypatch, tmp_path)
    result = probe_preemption(dry_run=False, max_seconds=5)

    assert "TERM warning/grace estimate: 4s (one-second precision)" in result
    assert submitted == ["victim", "preemptor"]
    assert cleaned[0][:2] == ([(101, "normal"), (102, "yisroel")], "probe-user")


def test_real_probe_post_victim_refusal_never_submits_preemptor(monkeypatch, tmp_path):
    """A failed post-victim recheck leaves only the disposable victim to clean up."""
    from slurm_mcp.preemption import probe_preemption

    submitted, cleaned = _mock_real_probe(monkeypatch, tmp_path, post=False)
    result = probe_preemption(dry_run=False, max_seconds=5)

    assert result.startswith("refused: post-victim safety check failed")
    assert submitted == ["victim"]
    assert cleaned[0][:2] == ([(101, "normal")], "probe-user")


def test_real_probe_query_failure_cleans_created_victim(monkeypatch, tmp_path):
    """A scheduler query error after victim submission still enters disposable cleanup."""
    from slurm_mcp.preemption import _QueryFailure, probe_preemption

    submitted, cleaned = _mock_real_probe(monkeypatch, tmp_path)
    monkeypatch.setattr("slurm_mcp.preemption._post_victim_safe", lambda *args, **kwargs: (_ for _ in ()).throw(_QueryFailure("controller lost")))
    result = probe_preemption(dry_run=False, max_seconds=5)

    assert "scheduler query failed: controller lost" in result
    assert submitted == ["victim"]
    assert cleaned[0][:2] == ([(101, "normal")], "probe-user")


def test_real_probe_timeout_still_uses_bounded_cleanup(monkeypatch, tmp_path):
    """Deadline expiry must skip the preemptor but retain a cleanup attempt for the victim."""
    from slurm_mcp.preemption import probe_preemption

    submitted, cleaned = _mock_real_probe(monkeypatch, tmp_path, verify=False)
    clock = [0.0]
    monkeypatch.setattr("slurm_mcp.preemption.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("slurm_mcp.preemption.time.sleep", lambda _: clock.__setitem__(0, 2.0))
    result = probe_preemption(dry_run=False, max_seconds=1)

    assert result.startswith("refused: timeout waiting for an exactly verified disposable victim")
    assert submitted == ["victim"]
    assert cleaned[0][:2] == ([(101, "normal")], "probe-user")


def test_cleanup_refuses_foreign_job_even_after_timeout(monkeypatch):
    """A foreign job ID must never receive scancel during disposable cleanup."""
    from slurm_mcp.preemption import _Budget, _cleanup

    calls = _reply(monkeypatch, {("squeue", "-h", "-j", "101", "-o", "%i|%u|%q"): "101|other-user|normal\n"})
    result = _cleanup([(101, "normal")], "probe-user", _Budget(5))

    assert result == ["left 101: ownership/QoS verification failed"]
    assert not any(command[0] == "scancel" for command in calls)


@pytest.mark.parametrize("value", ["unavailable", "N/A", "mystery-value"])
def test_gpu_allocation_unknown_markers_are_not_cpu_zero(value):
    """Unknown nonempty allocation text must refuse rather than imply no GPU."""
    from slurm_mcp.preemption import _QueryFailure, _gpu_source

    with pytest.raises(_QueryFailure):
        _gpu_source(value)


def test_gpu_allocation_parser_keeps_parenthesized_index_commas_in_one_token():
    """GRES index annotations contain commas but still describe one typed allocation."""
    from slurm_mcp.preemption import _gpu_source

    assert _gpu_source("gpu:rtx_6000:3(IDX:0,2-3)") == {"rtx_6000": 3}


def test_gpu_allocation_parser_rejects_text_after_an_annotation():
    """Only a complete parenthesized scheduler annotation may follow a GRES token."""
    from slurm_mcp.preemption import _QueryFailure, _gpu_source

    with pytest.raises(_QueryFailure):
        _gpu_source("gpu:rtx_6000:3(IDX:0) unexpected")


def test_candidate_refuses_unknown_node_gpu_usage():
    """An unavailable GresUsed field cannot create a false isolated GPU."""
    from slurm_mcp.preemption import _Policy, _QueryFailure, _find_candidate

    nodes = [{
        "NodeName": "node-a", "State": "MIXED", "Partitions": "main,rtx6000",
        "Gres": "gpu:rtx_6000:2", "GresUsed": "unavailable",
    }]
    with pytest.raises(_QueryFailure):
        _find_candidate(nodes, [], _Policy("normal", frozenset({"normal"})))


def test_candidate_ignores_unrelated_multinode_allocation_before_gpu_parsing(monkeypatch):
    """Malformed allocation data outside the candidate's expanded node set is irrelevant."""
    from slurm_mcp.preemption import _Job, _Policy, _find_candidate

    nodes = [{
        "NodeName": "node-a", "State": "MIXED", "Partitions": "main,rtx6000",
        "Gres": "gpu:rtx_6000:2", "GresUsed": "gpu:rtx_6000:1",
    }]
    job = _Job("99", "other", "normal", "RUNNING", "node[02-03]", per_node_raw="unavailable", allocated_raw="unavailable")
    monkeypatch.setattr("slurm_mcp.preemption._expand_nodelist", lambda value, *args, **kwargs: {"node-02", "node-03"})

    assert _find_candidate(nodes, [job], _Policy("normal", frozenset({"normal"}))).node == "node-a"


def test_multinode_per_node_and_total_gpu_evidence_are_scope_aware(monkeypatch):
    """A per-node request and a two-node total must agree after multiplying by nodes."""
    from slurm_mcp.preemption import _job_gpu_on_candidate, _job_snapshot

    _reply(monkeypatch, _allocation_responses(
        per_node="gres/gpu:rtx_6000=1", allocated="gres/gpu:rtx_6000=2",
        node_list="node[01-02]", expanded="node-01\nnode-02\n",
    ))

    assert _job_gpu_on_candidate(_job_snapshot()[0], "node-01") == {"rtx_6000": 1}


def test_probe_cleanup_has_its_own_budget_after_diagnostic_deadline(monkeypatch, tmp_path):
    """A spent diagnostic budget must not suppress exact-ID ownership cleanup."""
    from slurm_mcp.preemption import _Budget, probe_preemption

    submitted, cleaned = _mock_real_probe(monkeypatch, tmp_path, verify=False)
    clock = [0.0]
    monkeypatch.setattr("slurm_mcp.preemption.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("slurm_mcp.preemption.time.sleep", lambda _: clock.__setitem__(0, 2.0))
    result = probe_preemption(dry_run=False, max_seconds=1)

    assert result.startswith("refused: timeout waiting")
    assert submitted == ["victim"]
    assert cleaned[0][0] == [(101, "normal")]
    assert isinstance(cleaned[0][2], _Budget)
    assert cleaned[0][2].max_seconds == 5


def _boundary_scheduler(
    monkeypatch, tmp_path, *, post_race=False, query_failure=False,
    foreign_cleanup=False, never_runs=False, per_node="gres/gpu:rtx_6000=1",
    allocated="gres/gpu=1,gres/gpu:rtx_6000=1", untyped_node_usage=False,
):
    """Mock the scheduler process boundary and reject every unmodelled command."""
    import types
    from slurm_mcp import preemption

    state = {"victim": False, "preemptor": False, "cancelled": [], "config_calls": 0}
    victim = "101|probe-user|normal|RUNNING|node-a"
    alternate = "202|other|alternate|RUNNING|node-a"

    def detail(row):
        job_id, user, qos, job_state, node = row.split("|")
        return (
            f"JobId={job_id} UserId={user}(1) QOS={qos} JobState={job_state} "
            f"NodeList={node} TresPerNode={per_node} AllocTRES={allocated}\n"
        )

    def run(command, timeout=30):
        cmd = tuple(command)
        if cmd == ("scontrol", "show", "config"):
            state["config_calls"] += 1
            return CONFIG
        if cmd == ("sacctmgr", "-nP", "show", "qos", "format=Name,Preempt,PreemptMode,GraceTime"):
            return "normal||REQUEUE|0\nalternate||REQUEUE|0\nyisroel|normal,alternate|REQUEUE|120\n"
        if cmd == NODE_DETAIL:
            if query_failure and state["victim"]:
                raise RuntimeError("controller lost")
            output = NODES_FULL if state["victim"] else NODES_ONE_FREE
            if untyped_node_usage:
                output = output.replace("GresUsed=gpu:rtx_6000:", "GresUsed=gres/gpu=")
            return "NodeName=cpu-a State=IDLE Partitions=cpu Gres=(null) GresUsed=(null)\n" + output
        if cmd == JOB_LIST:
            if not state["victim"]:
                return ""
            return victim + "\n" + (alternate + "\n" if post_race else "")
        if cmd[:4] == ("squeue", "-h", "-j", "101"):
            if cmd[-1] == "%i|%u|%q":
                owner = "other-user" if foreign_cleanup else "probe-user"
                return f"101|{owner}|normal\n"
            return victim + "\n" if state["victim"] and not never_runs else ""
        if cmd[:4] == ("squeue", "-h", "-j", "102") and cmd[-1] == "%i|%u|%q":
            return "102|probe-user|yisroel\n"
        if cmd[:4] == ("scontrol", "show", "job", "-o"):
            job_id = cmd[4]
            if job_id == "101":
                return detail(victim)
            if job_id == "202":
                return detail(alternate)
        if cmd == ("scontrol", "show", "hostnames", "node-a"):
            return "node-a\n"
        if cmd[:2] == ("sbatch", "--parsable"):
            if not state["victim"]:
                state["victim"] = True
                return "101\n"
            state["preemptor"] = True
            (Path(cmd[2]).parent / "victim-events.log").write_text("heartbeat 10\nsignal TERM 11\nheartbeat 12\nrestart 13\n")
            return "102\n"
        if cmd[:1] == ("scancel",):
            state["cancelled"].append(cmd[1])
            return ""
        raise AssertionError(f"unexpected scheduler command: {command}")

    monkeypatch.setattr(preemption.shell, "_run", run)
    monkeypatch.setattr(preemption, "_authenticated_probe_root", lambda: str(tmp_path))
    monkeypatch.setattr(preemption, "_probe_scripts", lambda *args, **kwargs: ("victim", "preemptor"))
    monkeypatch.setattr(preemption.pwd, "getpwuid", lambda _: types.SimpleNamespace(pw_name="probe-user"))
    return state


@pytest.mark.parametrize("per_node,allocated,untyped_node_usage", [
    ("gres/gpu:rtx_6000=1", "gres/gpu=1,gres/gpu:rtx_6000=1", False),
    ("gres/gpu:rtx_6000=1", "gres/gpu=1", True),
    ("gres/gpu=1", "gres/gpu=1,gres/gpu:rtx_6000=1", True),
    ("gres/gpu=1", "gres/gpu=1", True),
    ("(null)", "gres/gpu=1", True),
])
def test_real_probe_boundary_success_uses_only_modelled_scheduler_commands(monkeypatch, tmp_path, per_node, allocated, untyped_node_usage):
    """The complete real lifecycle is isolated at the subprocess scheduler boundary."""
    from slurm_mcp.preemption import probe_preemption

    state = _boundary_scheduler(monkeypatch, tmp_path, per_node=per_node, allocated=allocated, untyped_node_usage=untyped_node_usage)
    result = probe_preemption(dry_run=False, max_seconds=5)

    assert "probe completed: victim restart observed; TERM warning/grace estimate: 1s" in result
    assert state["preemptor"]
    assert state["cancelled"] == ["101", "102"]


def test_real_probe_boundary_post_victim_race_refuses_before_preemptor(monkeypatch, tmp_path):
    """A second controller-preemptible allocation blocks the golden submission."""
    from slurm_mcp.preemption import probe_preemption

    state = _boundary_scheduler(monkeypatch, tmp_path, post_race=True)
    result = probe_preemption(dry_run=False, max_seconds=5)

    assert result.startswith("refused: post-victim safety check failed")
    assert not state["preemptor"]
    assert state["cancelled"] == ["101"]


def test_real_probe_boundary_query_failure_cleans_exact_owned_victim(monkeypatch, tmp_path):
    """A failed recheck still uses the independent cleanup window for the created victim."""
    from slurm_mcp.preemption import probe_preemption

    state = _boundary_scheduler(monkeypatch, tmp_path, query_failure=True)
    result = probe_preemption(dry_run=False, max_seconds=5)

    assert "scheduler query failed: controller lost" in result
    assert not state["preemptor"]
    assert state["cancelled"] == ["101"]


def test_real_probe_boundary_timeout_and_foreign_cleanup_never_cancel(monkeypatch, tmp_path):
    """A timing expiry has cleanup time, but a foreign owner prevents scancel."""
    from slurm_mcp.preemption import probe_preemption

    state = _boundary_scheduler(monkeypatch, tmp_path, foreign_cleanup=True, never_runs=True)
    clock = [0.0]
    monkeypatch.setattr("slurm_mcp.preemption.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("slurm_mcp.preemption.time.sleep", lambda _: clock.__setitem__(0, 2.0))
    result = probe_preemption(dry_run=False, max_seconds=1)

    assert result.startswith("refused: timeout waiting")
    assert not state["preemptor"]
    assert state["cancelled"] == []
