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

    def run(cmd):
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

    victim = _Job("101", "probe-user", "normal", "RUNNING", "node[01-02]", ({"rtx_6000": 1},))
    monkeypatch.setattr("slurm_mcp.preemption._job_snapshot", lambda job_id: [victim])
    monkeypatch.setattr("slurm_mcp.preemption._expand_nodelist", lambda _: {"node-01", "node-02"})
    candidate = _Candidate("node-01", "rtx_6000", "rtx6000")
    policy = _Policy("normal", frozenset({"normal", "alternate"}))

    assert _verify_victim(101, candidate, policy, "probe-user")
    assert not _verify_victim(101, candidate, policy, "other-user")


def test_post_victim_check_refuses_an_alternate_preemptible_race(monkeypatch):
    """A second QoS in the controller-derived preemptible set blocks the golden submit."""
    from slurm_mcp.preemption import _Candidate, _Job, _Policy, _post_victim_safe

    jobs = [
        _Job("101", "probe-user", "normal", "RUNNING", "node-a", ({"rtx_6000": 1},)),
        _Job("102", "other", "alternate", "RUNNING", "node-a", ({"rtx_6000": 1},)),
    ]
    monkeypatch.setattr("slurm_mcp.preemption._node_snapshot", lambda: [{"NodeName": "node-a", "State": "ALLOCATED", "Gres": "gpu:rtx_6000:2", "GresUsed": "gpu:rtx_6000:2"}])
    monkeypatch.setattr("slurm_mcp.preemption._job_snapshot", lambda: jobs)
    monkeypatch.setattr("slurm_mcp.preemption._expand_nodelist", lambda value: {value})

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


def test_tres_job_parser_treats_untyped_gpu_as_real_usage_and_rejects_malformed_rows():
    """Changing a running untyped GPU row into an ignored row would make isolation unsafe."""
    from slurm_mcp.preemption import _parse_job_rows

    rows = _parse_job_rows("88|other|alternate|RUNNING|node[01-02]|gres/gpu=1|\n")

    assert rows[0].uses_gpu
    assert rows[0].gpu_sources == ({"": 1},)
    with pytest.raises(Exception):
        _parse_job_rows("not a complete scheduler row\n")


def test_tres_job_parser_uses_allocated_tres_when_job_and_node_requests_are_empty():
    """Allocation detail is still GPU use when request-source fields are absent."""
    from slurm_mcp.preemption import _parse_job_rows

    rows = _parse_job_rows("88|other|alternate|RUNNING|node-a|||gres/gpu=1\n")

    assert rows[0].uses_gpu


def test_candidate_rejects_compressed_nodelist_and_alternative_preemptible_qos(monkeypatch):
    """A compressed allocation on the candidate node must block every preemptible QoS."""
    from slurm_mcp.preemption import _Candidate, _Policy, _find_candidate, _parse_job_rows

    nodes = [{
        "NodeName": "node-01", "State": "MIXED", "Partitions": "main,rtx6000",
        "Gres": "gpu:rtx_6000:2", "GresUsed": "gpu:rtx_6000:1",
    }]
    jobs = _parse_job_rows("88|other|alternate|RUNNING|node[01-02]|gres/gpu=1|\n")
    policy = _Policy(victim_qos="normal", preemptible_qos=frozenset({"normal", "alternate"}))
    monkeypatch.setattr("slurm_mcp.preemption._expand_nodelist", lambda value: {"node-01", "node-02"})

    assert _find_candidate(nodes, jobs, policy) is None


@pytest.mark.parametrize("state", ["DOWN", "DRAINING", "FAIL", "MAINT", "NO_RESPOND", "POWER_DOWN", "UNKNOWN"])
def test_candidate_rejects_non_usable_node_states(state):
    """A node state outside the explicitly usable set cannot host a probe."""
    from slurm_mcp.preemption import _node_is_usable

    assert not _node_is_usable(state)


def test_candidate_requires_detailed_gres_used_field():
    """Missing GresUsed must not be interpreted as an unused GPU."""
    from slurm_mcp.preemption import _Policy, _find_candidate

    nodes = [{"NodeName": "node-a", "State": "MIXED", "Partitions": "main,rtx6000", "Gres": "gpu:rtx_6000:2"}]
    assert _find_candidate(nodes, [], _Policy("normal", frozenset({"normal"}))) is None


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
    assert "trap 'record_signal TERM; exit 0' TERM" in victim
    assert "trap 'record_signal USR1' USR1" in victim
    assert "#SBATCH --time=00:15:00" in victim


def test_probe_scripts_reject_directive_injection_from_scheduler_fields():
    """A malicious scheduler/config value must not become a second SBATCH directive."""
    from slurm_mcp.preemption import _Candidate, _Refusal, _probe_scripts

    with pytest.raises(_Refusal):
        _probe_scripts(_Candidate("node-a", "rtx_6000", "rtx6000\n#SBATCH --qos=evil"), Path("/home/probe-user/.slurmx/probes/run"), max_seconds=600, victim_qos="normal")
