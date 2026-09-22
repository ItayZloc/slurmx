"""Mocked contracts for SLURM preemption inspection and the disposable probe."""

from __future__ import annotations

import pytest


CONFIG = "SLURM_VERSION = 25.11.1\nPreemptType = preempt/qos\nPreemptMode = REQUEUE\nPreemptParameters = send_user_signal\nJobRequeue = 1\nKillWait = 30\n"
QOS = "normal||REQUEUE|0\nyisroel|normal|REQUEUE|120\n"
NODES_ONE_FREE = "NodeName=node-a State=MIXED Partitions=main,rtx6000 Gres=gpu:rtx_6000:2 GresUsed=gpu:rtx_6000:1\n"
NODES_FULL = "NodeName=node-a State=ALLOCATED Partitions=main,rtx6000 Gres=gpu:rtx_6000:2 GresUsed=gpu:rtx_6000:2\n"
NO_JOBS = ""
VICTIM_RUNNING = "101|probe-user|normal|RUNNING|node-a|gpu:rtx_6000:1\n"


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
    return {
        ("scontrol", "show", "node", "-o"): nodes,
        ("squeue", "-h", "-o", "%i|%u|%q|%T|%N|%b"): jobs,
    }


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
    assert "normal: preempts <none>; PreemptMode=REQUEUE; GraceTime=0" in result
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
    assert "safety evidence: one free rtx_6000 GPU; no running normal-QoS GPU job" in result
    assert not any(command[0] == "sbatch" for command in calls)


def test_probe_rechecks_after_victim_and_refuses_a_race(monkeypatch, tmp_path):
    """A second normal job after victim start must prevent the preemptor submit."""
    from slurm_mcp.preemption import probe_preemption

    responses = _scan_responses()
    responses[("sbatch",)] = "101;cluster\n"
    responses[("squeue", "-h", "-j", "101", "-o", "%i|%u|%q|%T|%N|%b")] = VICTIM_RUNNING
    responses[("scontrol", "show", "node", "-o")] = [NODES_ONE_FREE, NODES_ONE_FREE, NODES_FULL]
    responses[("squeue", "-h", "-o", "%i|%u|%q|%T|%N|%b")] = [NO_JOBS, NO_JOBS, VICTIM_RUNNING + "102|other|normal|RUNNING|node-a|gpu:rtx_6000:1\n"]
    responses[("squeue", "-h", "-j", "101", "-o", "%i|%u|%q")] = "101|probe-user|normal\n"
    responses[("scancel", "101")] = ""
    calls = _reply(monkeypatch, responses)
    monkeypatch.setattr("slurm_mcp.preemption.PROBE_ROOT", tmp_path)
    monkeypatch.setattr("slurm_mcp.preemption.os.environ", {"USER": "probe-user"})

    result = probe_preemption(dry_run=False, max_seconds=2)

    assert result.startswith("refused: post-victim safety check failed")
    assert [command[0] for command in calls].count("sbatch") == 1
    assert [command[0] for command in calls].count("scontrol") == 3
    assert ["scancel", "101"] in calls


def test_probe_records_requeue_measurement_and_cleans_verified_jobs(monkeypatch, tmp_path):
    """A signal and later heartbeat must produce a one-second-precision estimate."""
    from slurm_mcp.preemption import probe_preemption

    responses = _scan_responses()
    responses[("sbatch",)] = ["101;cluster\n", "102;cluster\n"]
    responses[("squeue", "-h", "-j", "101", "-o", "%i|%u|%q|%T|%N|%b")] = VICTIM_RUNNING
    responses[("scontrol", "show", "node", "-o")] = [NODES_ONE_FREE, NODES_ONE_FREE, NODES_FULL]
    responses[("squeue", "-h", "-o", "%i|%u|%q|%T|%N|%b")] = [NO_JOBS, NO_JOBS, VICTIM_RUNNING]
    responses[("squeue", "-h", "-j", "101", "-o", "%i|%u|%q")] = "101|probe-user|normal\n"
    responses[("squeue", "-h", "-j", "102", "-o", "%i|%u|%q")] = "102|probe-user|yisroel\n"
    responses[("scancel", "101")] = ""
    responses[("scancel", "102")] = ""
    calls = _reply(monkeypatch, responses)
    monkeypatch.setattr("slurm_mcp.preemption.PROBE_ROOT", tmp_path)
    monkeypatch.setattr("slurm_mcp.preemption.os.environ", {"USER": "probe-user"})

    written = {"done": False}

    def sleep(_):
        if not written["done"]:
            probe_dir = next(tmp_path.iterdir())
            (probe_dir / "victim-events.log").write_text("heartbeat 100\nsignal USR1 105\nheartbeat 112\nrestart 113\n")
            written["done"] = True

    monkeypatch.setattr("slurm_mcp.preemption.time.sleep", sleep)
    result = probe_preemption(dry_run=False, max_seconds=5)

    assert "warning/grace estimate: 7s (one-second precision)" in result
    assert "victim restart observed" in result
    assert ["scancel", "101"] in calls
    assert ["scancel", "102"] in calls


def test_probe_timeout_cleans_only_verified_disposable_jobs(monkeypatch, tmp_path):
    """A timeout must not cancel a job whose live owner/QoS check disagrees."""
    from slurm_mcp.preemption import probe_preemption

    responses = _scan_responses()
    responses[("sbatch",)] = ["101;cluster\n", "102;cluster\n"]
    responses[("squeue", "-h", "-j", "101", "-o", "%i|%u|%q|%T|%N|%b")] = VICTIM_RUNNING
    responses[("scontrol", "show", "node", "-o")] = [NODES_ONE_FREE, NODES_ONE_FREE, NODES_FULL]
    responses[("squeue", "-h", "-o", "%i|%u|%q|%T|%N|%b")] = [NO_JOBS, NO_JOBS, VICTIM_RUNNING]
    responses[("squeue", "-h", "-j", "101", "-o", "%i|%u|%q")] = "101|probe-user|normal\n"
    responses[("squeue", "-h", "-j", "102", "-o", "%i|%u|%q")] = "102|other-user|yisroel\n"
    responses[("scancel", "101")] = ""
    calls = _reply(monkeypatch, responses)
    monkeypatch.setattr("slurm_mcp.preemption.PROBE_ROOT", tmp_path)
    monkeypatch.setattr("slurm_mcp.preemption.os.environ", {"USER": "probe-user"})

    result = probe_preemption(dry_run=False, max_seconds=1)

    assert result.startswith("timeout: no requeue signal observed")
    assert ["scancel", "101"] in calls
    assert ["scancel", "102"] not in calls


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
