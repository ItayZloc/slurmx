"""Contract tests for metadata-aware SLURMx submission."""

from __future__ import annotations

import inspect
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_config = types.ModuleType("config")
_config.MAIL_USER = "test@example.com"
_config.MAX_MEM_GB = 80
_config.TIME_LIMIT = "1-0:00:00"
_config.START_TIMEOUT = 1
_config.CPU_MEM = "16G"
_config.CPU_CPUS = 4
_config.GOLDEN_QOS = ["yisroel"]
_config.EXCLUDE_NODES = []
_config.GPU_DEFINITIONS = [
    ("rtx_6000", "RTX 6000", 48, 12, "rtx6000"),
    ("rtx_3090", "RTX 3090", 24, 0, "rtx3090"),
]
_config.GPU_DEFINITIONS_BY_QOS = {"yisroel": _config.GPU_DEFINITIONS}
sys.modules.setdefault("config", _config)

from slurm_mcp.submission import parse_script_metadata, submit_job
from slurm_mcp.types import Availability, GPUAvailability


def _script(tmp_path, header: str, body: str = "echo hello"):
    path = tmp_path / "job.sh"
    path.write_text(f"#!/bin/bash\n{header}\n{body}\n")
    path.chmod(0o755)
    return path


def _availability(*, golden=None, cluster=None):
    return Availability(golden=golden or {}, cluster=cluster or {})


def test_metadata_requires_exact_schema_and_types(tmp_path):
    """A malformed header must not be able to bypass resource policy."""
    script = _script(
        tmp_path,
        '# slurmx: {"total_vram_gb": true, "supports_gpu_sharding": false, "preemption_safe": true}',
    )

    metadata, error = parse_script_metadata(str(script))

    assert metadata is None
    assert error == "slurmx metadata total_vram_gb must be a non-negative integer."


def test_metadata_must_follow_shebang_and_reject_cpu_sharding(tmp_path):
    """The only accepted header position prevents accidental metadata parsing."""
    script = _script(
        tmp_path,
        '# slurmx: {"total_vram_gb": 0, "supports_gpu_sharding": true, "preemption_safe": true}',
    )

    metadata, error = parse_script_metadata(str(script))

    assert metadata is None
    assert error == "slurmx metadata cannot enable GPU sharding for a CPU job."


def test_submit_resolves_relative_executable_and_quotes_arguments(tmp_path, monkeypatch):
    """The wrapper must invoke exactly the executable script and arguments supplied."""
    script = _script(
        tmp_path,
        '# slurmx: {"total_vram_gb": 0, "supports_gpu_sharding": false, "preemption_safe": false}',
    )
    monkeypatch.chdir(tmp_path)

    result = submit_job("job.sh", args=["two words", "--flag"], dry_run=True)

    assert result.success
    assert f"{script} 'two words' --flag" in result.sbatch_script
    assert "#SBATCH --no-requeue" in result.sbatch_script


def test_safe_job_prefers_live_golden_then_uses_main(monkeypatch, tmp_path):
    """Safe jobs may use main, but only after a live golden option is exhausted."""
    script = _script(
        tmp_path,
        '# slurmx: {"total_vram_gb": 40, "supports_gpu_sharding": true, "preemption_safe": true}',
    )
    monkeypatch.setattr(
        "slurm_mcp.selection.availability.check_availability",
        lambda: _availability(
            golden={"rtx_6000": GPUAvailability("rtx_6000", 12, 0, 1)},
            cluster={
                "rtx_6000": GPUAvailability("rtx_6000", 8, 0, 8),
                "rtx_4090": GPUAvailability("rtx_4090", 8, 0, 8),
            },
        ),
    )

    result = submit_job(str(script), dry_run=True)

    assert result.success
    assert (result.gpu_type, result.partition, result.qos) == (
        "rtx_6000", "rtx6000", "yisroel",
    )
    assert "#SBATCH --requeue" in result.sbatch_script
    assert "#SBATCH --signal=B:USR1@120" in result.sbatch_script
    assert 'trap \'kill -USR1 "$child_pid"' in result.sbatch_script
    assert 'trap \'kill -TERM "$child_pid"' in result.sbatch_script


def test_safe_job_uses_two_same_type_main_cards_when_needed(monkeypatch, tmp_path):
    """A sharding-capable job can receive two cards, never mixed GPU types."""
    script = _script(
        tmp_path,
        '# slurmx: {"total_vram_gb": 40, "supports_gpu_sharding": true, "preemption_safe": true}',
    )
    monkeypatch.setattr(
        "slurm_mcp.selection.availability.check_availability",
        lambda: _availability(
            golden={},
            cluster={"rtx_3090": GPUAvailability("rtx_3090", 9, 0, 2)},
        ),
    )

    result = submit_job(str(script), dry_run=True)

    assert result.success
    assert (result.gpu_type, result.partition, result.qos) == (
        "rtx_3090", "main", "normal",
    )
    assert "#SBATCH --gres=gpu:rtx_3090:2" in result.sbatch_script


def test_unsafe_job_queues_on_best_golden_candidate_without_availability(monkeypatch, tmp_path):
    """Unsafe work must never bypass golden even when no card is currently free."""
    script = _script(
        tmp_path,
        '# slurmx: {"total_vram_gb": 40, "supports_gpu_sharding": false, "preemption_safe": false}',
    )
    monkeypatch.setattr(
        "slurm_mcp.selection.availability.check_availability",
        lambda: (_ for _ in ()).throw(AssertionError("unsafe jobs do not inspect availability")),
    )

    result = submit_job(str(script), dry_run=True)

    assert result.success
    assert (result.gpu_type, result.partition, result.qos) == (
        "rtx_6000", "rtx6000", "yisroel",
    )
    assert "#SBATCH --no-requeue" in result.sbatch_script


def test_submission_has_only_the_script_metadata_interface():
    """Callers cannot override metadata policy through public resource parameters."""
    assert list(inspect.signature(submit_job).parameters) == [
        "script_path", "args", "job_name", "workdir", "output_dir", "dependency",
        "wait_until_running", "dry_run",
    ]


def test_mcp_and_cli_expose_script_arguments_without_resource_flags():
    """The public adapters cannot reintroduce a raw-command submission bypass."""
    mcp_module = types.ModuleType("mcp")
    mcp_server = types.ModuleType("mcp.server")
    fastmcp = types.ModuleType("mcp.server.fastmcp")

    class FakeMCP:
        def __init__(self, *args, **kwargs):
            pass

        def tool(self):
            return lambda function: function

    fastmcp.FastMCP = FakeMCP
    sys.modules.update({
        "mcp": mcp_module, "mcp.server": mcp_server,
        "mcp.server.fastmcp": fastmcp,
    })
    import server
    from cli.submit import add_arguments
    import argparse

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    parsed = parser.parse_args(["--dry-run", "--", "./job.sh", "--epochs", "3"])

    assert list(inspect.signature(server.submit_job).parameters) == [
        "script_path", "args", "job_name", "workdir", "output_dir", "dependency",
        "wait_until_running", "dry_run",
    ]
    assert parsed.script == ["--", "./job.sh", "--epochs", "3"]
    with __import__("pytest").raises(SystemExit):
        parser.parse_args(["--vram", "48", "--", "./job.sh"])
