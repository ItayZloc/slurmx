"""Tests for slurm_mcp.remote_session.submit_remote_session_job — the backing
function for the launch_remote_session MCP tool. Monkeypatches subprocess.run
to capture the sbatch invocation without actually submitting.
"""

import json
import os
import sys
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slurm_mcp import remote_session
from config import CPU_PARTITION, CPU_QOS, CPU_CPUS, CPU_MEM


def _ok_sbatch_run(*_args, **_kwargs):
    m = MagicMock()
    m.returncode = 0
    m.stdout = "Submitted batch job 12345"
    m.stderr = ""
    return m


@patch("slurm_mcp.remote_session.os.makedirs")
@patch("slurm_mcp.remote_session.subprocess.run", side_effect=_ok_sbatch_run)
def test_cpu_mode_sbatch_args(mock_run, _mock_mkdir):
    job_id, log_path = remote_session.submit_remote_session_job(
        name="myproj", hardware="cpu", days=1,
    )

    assert job_id == 12345
    args = mock_run.call_args[0][0]

    assert args[0] == "sbatch"
    assert "--job-name=myproj" in args
    assert f"--partition={CPU_PARTITION}" in args
    assert f"--qos={CPU_QOS}" in args
    assert f"--cpus-per-task={CPU_CPUS}" in args
    assert f"--mem={CPU_MEM}" in args
    # No --gres for CPU mode
    assert not any(a.startswith("--gres=") for a in args)


@patch("slurm_mcp.remote_session.selection.select_gpu",
       return_value=("rtx_4090", "main", "normal"))
@patch("slurm_mcp.remote_session.os.makedirs")
@patch("slurm_mcp.remote_session.subprocess.run", side_effect=_ok_sbatch_run)
def test_gpu_mode_sbatch_args(mock_run, _mock_mkdir, _mock_select):
    job_id, log_path = remote_session.submit_remote_session_job(
        name="gpuproj", hardware="gpu", days=2, vram_gb=24,
    )

    assert job_id == 12345
    args = mock_run.call_args[0][0]

    assert "--partition=main" in args
    assert "--qos=normal" in args
    assert "--gres=gpu:rtx_4090:1" in args
    assert "--nodes=1" in args
    # CPU-mode partition must NOT leak through
    assert f"--partition={CPU_PARTITION}" not in args


@patch("slurm_mcp.remote_session.selection.select_gpu", return_value=None)
def test_gpu_mode_no_capacity_raises(_mock_select):
    with pytest.raises(RuntimeError, match="No GPU"):
        remote_session.submit_remote_session_job(
            name="x", hardware="gpu", days=1, vram_gb=200,
        )


@patch("slurm_mcp.remote_session.selection.select_gpu")
@patch("slurm_mcp.remote_session.os.makedirs")
@patch("slurm_mcp.remote_session.subprocess.run", side_effect=_ok_sbatch_run)
def test_gpu_type_param_bypasses_select_gpu(mock_run, _mock_mkdir, mock_select):
    """When gpu_type is set, select_gpu must NOT be called (user wants the
    exact card, not the smallest-fitting fallback)."""
    remote_session.submit_remote_session_job(
        name="exact-pick", hardware="gpu", days=1,
        gpu_type="gtx_1080",
    )
    mock_select.assert_not_called()
    args = mock_run.call_args[0][0]
    assert "--gres=gpu:gtx_1080:1" in args
    # gtx_1080 has golden_quota=0 → main/normal partition
    assert "--partition=main" in args
    assert "--qos=normal" in args


@patch("slurm_mcp.remote_session.os.makedirs")
@patch("slurm_mcp.remote_session.subprocess.run", side_effect=_ok_sbatch_run)
def test_gpu_type_param_routes_golden_to_yisroel(mock_run, _mock_mkdir):
    """A golden GPU type (e.g. rtx_6000) should route to its golden_partition
    + the primary QoS."""
    remote_session.submit_remote_session_job(
        name="golden", hardware="gpu", days=1,
        gpu_type="rtx_6000",
    )
    args = mock_run.call_args[0][0]
    assert "--gres=gpu:rtx_6000:1" in args
    assert "--partition=rtx6000" in args
    assert "--qos=yisroel" in args


def test_unknown_gpu_type_rejected():
    with pytest.raises(ValueError, match="gpu_type"):
        remote_session.submit_remote_session_job(
            name="x", hardware="gpu", days=1, gpu_type="rtx_imaginary",
        )


@patch("slurm_mcp.remote_session.os.makedirs")
@patch("slurm_mcp.remote_session.subprocess.run", side_effect=_ok_sbatch_run)
def test_wrap_command_uses_remote_control_server_mode(mock_run, _mock_mkdir):
    remote_session.submit_remote_session_job(
        name="myproj", hardware="cpu", days=1,
    )
    args = mock_run.call_args[0][0]
    wrap = next((a for a in args if a.startswith("--wrap=")), "")
    assert wrap, "expected --wrap=... in sbatch args"
    assert "claude remote-control --name 'myproj'" in wrap
    # Legacy one-shot flags must not appear
    for forbidden in (" -p ", "--model", "--effort"):
        assert forbidden not in wrap, f"unexpected legacy flag {forbidden!r} in --wrap"


@patch("slurm_mcp.remote_session.os.makedirs")
@patch("slurm_mcp.remote_session.subprocess.run", side_effect=_ok_sbatch_run)
def test_permission_mode_lands_in_wrap_server_form(mock_run, _mock_mkdir):
    remote_session.submit_remote_session_job(
        name="myproj", hardware="cpu", days=1,
        permission_mode="acceptEdits",
    )
    args = mock_run.call_args[0][0]
    wrap = next((a for a in args if a.startswith("--wrap=")), "")
    assert "--permission-mode acceptEdits" in wrap


@patch("slurm_mcp.remote_session.os.makedirs")
@patch("slurm_mcp.remote_session.subprocess.run", side_effect=_ok_sbatch_run)
def test_permission_mode_lands_in_wrap_resume_form(mock_run, _mock_mkdir):
    remote_session.submit_remote_session_job(
        name="myproj", hardware="cpu", days=1,
        permission_mode="plan",
        resume="sess_abc",
    )
    args = mock_run.call_args[0][0]
    wrap = next((a for a in args if a.startswith("--wrap=")), "")
    assert "--permission-mode plan" in wrap
    assert "--resume 'sess_abc'" in wrap


def test_invalid_permission_mode_rejected():
    with pytest.raises(ValueError, match="permission_mode"):
        remote_session.submit_remote_session_job(
            name="x", hardware="cpu", days=1, permission_mode="bypassPermissions",
        )


@patch("slurm_mcp.remote_session.os.makedirs")
@patch("slurm_mcp.remote_session.subprocess.run", side_effect=_ok_sbatch_run)
def test_wrap_unsets_inference_only_tokens(mock_run, _mock_mkdir):
    """The wrap command must `unset CLAUDE_CODE_OAUTH_TOKEN ANTHROPIC_API_KEY`
    before calling claude, so the compute node falls back to the cached
    full-scope OAuth credentials instead of inheriting an inference-only token
    from the user's shell environment."""
    # Pin a workdir without "claude" in its path so the substring search
    # below isn't fooled by paths like ~/.claude/mcp-servers/slurmx.
    remote_session.submit_remote_session_job(
        name="auth-test", hardware="cpu", days=1, workdir="/tmp",
    )
    args = mock_run.call_args[0][0]
    wrap = next((a for a in args if a.startswith("--wrap=")), "")
    assert "unset CLAUDE_CODE_OAUTH_TOKEN ANTHROPIC_API_KEY" in wrap
    # The unset must come before the claude binary invocation
    assert wrap.index("unset") < wrap.index("claude remote-control")


@patch("slurm_mcp.remote_session.os.makedirs")
@patch("slurm_mcp.remote_session.subprocess.run", side_effect=_ok_sbatch_run)
def test_resume_switches_to_interactive_form(mock_run, _mock_mkdir):
    remote_session.submit_remote_session_job(
        name="myproj", hardware="cpu", days=1,
        resume="sess_abc123",
    )
    args = mock_run.call_args[0][0]
    wrap = next((a for a in args if a.startswith("--wrap=")), "")
    assert wrap, "expected --wrap=... in sbatch args"
    # Interactive form with --resume; NOT the server-mode subcommand
    assert "claude --remote-control 'myproj'" in wrap
    assert "--resume 'sess_abc123'" in wrap
    assert "claude remote-control --name" not in wrap


def test_ensure_workspace_trusted_writes_new_entry(tmp_path):
    cfg = tmp_path / ".claude.json"
    cfg.write_text(json.dumps({
        "projects": {
            "/preexisting": {"hasTrustDialogAccepted": False, "other": "keep"},
        },
        "other_top_field": "untouched",
    }))
    remote_session._ensure_workspace_trusted("/new/dir", config_path=str(cfg))

    data = json.loads(cfg.read_text())
    assert data["projects"]["/new/dir"]["hasTrustDialogAccepted"] is True
    # Other entries untouched
    assert data["projects"]["/preexisting"]["hasTrustDialogAccepted"] is False
    assert data["projects"]["/preexisting"]["other"] == "keep"
    assert data["other_top_field"] == "untouched"


def test_ensure_workspace_trusted_idempotent(tmp_path):
    cfg = tmp_path / ".claude.json"
    cfg.write_text(json.dumps({
        "projects": {"/dir": {"hasTrustDialogAccepted": True}},
    }))
    mtime_before = cfg.stat().st_mtime_ns
    remote_session._ensure_workspace_trusted("/dir", config_path=str(cfg))
    # No write should happen on a no-op
    assert cfg.stat().st_mtime_ns == mtime_before


def test_ensure_workspace_trusted_creates_file_when_missing(tmp_path):
    cfg = tmp_path / ".claude.json"
    # File does not exist yet
    assert not cfg.exists()
    remote_session._ensure_workspace_trusted("/dir", config_path=str(cfg))
    data = json.loads(cfg.read_text())
    assert data["projects"]["/dir"]["hasTrustDialogAccepted"] is True


@patch("slurm_mcp.remote_session._ensure_workspace_trusted")
@patch("slurm_mcp.remote_session.os.makedirs")
@patch("slurm_mcp.remote_session.subprocess.run", side_effect=_ok_sbatch_run)
def test_submit_calls_ensure_workspace_trusted(_mock_run, _mock_mkdir, mock_trust):
    remote_session.submit_remote_session_job(
        name="t", hardware="cpu", days=1, workdir="/some/path",
    )
    mock_trust.assert_called_once()
    called_workdir = mock_trust.call_args[0][0]
    assert called_workdir == "/some/path"


def _make_cli_args(**overrides):
    """Build an argparse.Namespace mirroring the CLI defaults (all None)."""
    import argparse
    ns = argparse.Namespace(
        name=None, hardware=None, days=None, vram_gb=None, gpu_type=None,
        permission_mode=None, workdir=None, resume=None,
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def test_prompt_choice_numeric_index():
    with patch("builtins.input", side_effect=["2"]):
        chosen = remote_session._prompt_choice(
            "x?", [("a", "first"), ("b", "second")],
        )
    assert chosen == "b"


def test_prompt_choice_literal_match():
    with patch("builtins.input", side_effect=["acceptEdits"]):
        chosen = remote_session._prompt_choice(
            "mode?",
            [("default", "x"), ("acceptEdits", "y"), ("plan", "z")],
        )
    assert chosen == "acceptEdits"


def test_prompt_choice_reprompts_on_invalid():
    # First two inputs invalid, third is valid.
    with patch("builtins.input", side_effect=["nope", "99", "1"]):
        chosen = remote_session._prompt_choice("x?", [(1, "one"), (2, "two")])
    assert chosen == 1


@patch("slurm_mcp.remote_session.sys.stdin.isatty", return_value=True)
@patch("builtins.input", side_effect=["cpu", "1", "plan"])
def test_ensure_required_interactive_fills_in_cpu(_mock_in, _mock_tty):
    args = _make_cli_args()
    remote_session._ensure_required_interactive(args)
    assert args.hardware == "cpu"
    assert args.days == 1
    assert args.permission_mode == "plan"
    assert args.vram_gb is None  # cpu path: no vram prompt


@patch("slurm_mcp.remote_session.sys.stdin.isatty", return_value=True)
@patch("builtins.input", side_effect=["gpu", "7", "acceptEdits", "gtx_1080"])
def test_ensure_required_interactive_asks_gpu_type_on_gpu(_mock_in, _mock_tty):
    args = _make_cli_args()
    remote_session._ensure_required_interactive(args)
    assert args.hardware == "gpu"
    assert args.days == 7
    assert args.permission_mode == "acceptEdits"
    assert args.gpu_type == "gtx_1080"
    assert args.vram_gb is None  # only gpu_type path; vram_gb stays unset


@patch("slurm_mcp.remote_session.sys.stdin.isatty", return_value=True)
@patch("builtins.input", side_effect=["plan"])
def test_ensure_required_interactive_only_prompts_missing(_mock_in, _mock_tty):
    # hardware and days already set on the cli → only permission_mode is asked.
    args = _make_cli_args(hardware="cpu", days=2)
    remote_session._ensure_required_interactive(args)
    assert args.hardware == "cpu"
    assert args.days == 2
    assert args.permission_mode == "plan"


@patch("slurm_mcp.remote_session.sys.stdin.isatty", return_value=False)
def test_ensure_required_interactive_exits_without_tty(_mock_tty):
    args = _make_cli_args()  # everything None and no TTY
    with pytest.raises(SystemExit):
        remote_session._ensure_required_interactive(args)


def test_extract_session_url_finds_url(tmp_path):
    log = tmp_path / "session.out"
    log.write_text(
        "Continue coding in the Claude mobile app or "
        "https://claude.ai/code?environment=env_01abc\n"
        "·✔︎· Connected · slurmx · main\n"
        "    \x1b]8;;https://claude.ai/code/session_01E9bcTHpWgEj8ERAVHCnCmx?from=cli\x07myproj\x1b]8;;\x07\n"
    )
    url = remote_session.extract_session_url(str(log), timeout=0, poll_interval=0)
    assert url == "https://claude.ai/code/session_01E9bcTHpWgEj8ERAVHCnCmx?from=cli"


def test_extract_session_url_ignores_environment_url(tmp_path):
    log = tmp_path / "session.out"
    # Only the lobby URL is present, NOT the per-session URL.
    log.write_text(
        "Continue coding in the Claude mobile app or "
        "https://claude.ai/code?environment=env_01abc\n"
    )
    url = remote_session.extract_session_url(str(log), timeout=0, poll_interval=0)
    assert url is None


def test_extract_session_url_missing_file_returns_none(tmp_path):
    url = remote_session.extract_session_url(
        str(tmp_path / "nope.out"), timeout=0, poll_interval=0,
    )
    assert url is None


@patch("slurm_mcp.remote_session.time.sleep")
def test_extract_session_url_polls_until_appears(mock_sleep, tmp_path):
    """File starts empty, then a URL appears on the 3rd read."""
    log = tmp_path / "session.out"
    log.write_text("")  # empty
    reads = {"n": 0}
    real_open = open

    def fake_open(path, *a, **kw):
        if str(path) == str(log):
            reads["n"] += 1
            if reads["n"] >= 3:
                log.write_text(
                    "session URL: https://claude.ai/code/session_01XYZ?from=cli\n"
                )
        return real_open(path, *a, **kw)

    with patch("builtins.open", fake_open):
        url = remote_session.extract_session_url(
            str(log), timeout=10.0, poll_interval=0.01,
        )
    assert url == "https://claude.ai/code/session_01XYZ?from=cli"


def test_invalid_hardware_rejected():
    with pytest.raises(ValueError, match="hardware"):
        remote_session.submit_remote_session_job(name="x", hardware="tpu", days=1)


def test_invalid_days_rejected():
    with pytest.raises(ValueError, match="days"):
        remote_session.submit_remote_session_job(name="x", hardware="cpu", days=5)
