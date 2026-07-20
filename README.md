# slurmx

MCP server and unified CLI that lets you (and Claude Code) submit, monitor, and manage SLURM GPU jobs. Auto-selects the smallest GPU that fits your VRAM needs, tries golden tickets first, falls back to cluster-wide. Also spawns `claude remote-control` as a SLURM job so you can keep coding from your phone or another machine.

After install (`./setup.sh` or `slurmx setup`), see [WELCOME.md](WELCOME.md) for a one-page summary of what's available and how to drive it from a Claude Code chat. The setup script prints the same content at the end of `uv sync`.

## MCP tools

| Tool | Description |
|------|-------------|
| `cluster_summary` | Single-call dashboard: your jobs + golden tickets (per QoS) + cluster-wide GPU availability. `view="jobs"` or `"gpu"` narrows the output. |
| `submit_job` | Submit GPU/CPU jobs (auto-selects GPU by VRAM). `golden_only=true` forces a preemption-immune golden slot. |
| `select_gpu` | Recommend best GPU for a VRAM requirement |
| `job_history` | Show recent completed/failed jobs (via sacct) |
| `get_job_status` | Get detailed status of a specific job |
| `wait_for_job` | Block until a job finishes |
| `read_job_log` | Read SLURM log file for a job |
| `diagnose_job` | Classify job failures (OOM, timeout, missing module, code error) |
| `cancel_jobs` | Cancel jobs by ID or all |
| `launch_remote_session` | Spawn `claude remote-control` as a SLURM job. Required: `hardware` (cpu/gpu), `days` (1/2/3/7), `permission_mode` (default/acceptEdits/plan). For gpu, also asks the user for `gpu_type` (gtx_1080…rtx_pro_6000) and takes `golden_only`. Polls the SLURM log and returns the `claude.ai/code/session_<id>` URL inline (default 90s wait; `wait_url_seconds=0` to skip). |

## Golden tickets (preemption) vs the main pool

On this cluster a job's **QoS**, not its card type, decides whether it can be
evicted. `qos=normal` (partition `main`/`gpu`) is the shared pool — everyone can
use it, but a job there is **preemptible**: any group's golden QoS can requeue it.
Your golden QoS (e.g. `yisroel`) runs on the per-card dedicated partitions
(`rtx_pro_6000`, `rtx6000`, `rtx4090`, …) and is **preemption-immune** — it bumps
`normal` jobs and nothing bumps it. A golden QoS is invalid on `main`/`gpu`, so
"golden" always means a dedicated partition.

`submit_job` / `launch_remote_session` / `slurmx submit` take a **`golden_only`**
flag (`--golden-only` on the CLI):

- **default (`golden_only=false`)** — golden-first on the cards you own
  (`golden_quota > 0`), then fall back to the preemptible main pool if golden is
  full. Unchanged behavior.
- **`golden_only=true`** — force `qos=yisroel` on the card's dedicated partition
  and **never** accept a preemptible slot. If the golden ticket is full the job
  waits in the golden queue and starts automatically when a slot frees (it is not
  downgraded). Works on every card, including the smaller ones the group doesn't
  own (`golden_quota=0`) — those then preempt other groups' `normal` jobs there.
  Recommended for training you don't want evicted.

When a golden ticket is **full**, `slurmx status` and `cluster_summary` list the
waiting queue for that card in dispatch order (position, job id, user, job name)
so you can see who is ahead of you.

## Installation

```bash
# 1. Clone
git clone https://github.com/ItayZloc/slurmx.git ~/.claude/mcp-servers/slurmx

# 2. Configure
cd ~/.claude/mcp-servers/slurmx
cp config-examples/default.py config.py
# Or pre-filled: cp config-examples/yisroel.py config.py
# Edit config.py — fill in MAIL_USER (CLAUDE_LOG_DIR is auto-detected)

# 3. Bootstrap: create venv, install deps, symlink `slurmx` into ~/.local/bin/
./setup.sh

# 4. Register the MCP server with Claude Code
claude mcp add slurmx \
  ~/.claude/mcp-servers/slurmx/.venv/bin/python \
  ~/.claude/mcp-servers/slurmx/server.py
```

Verify it works:
```bash
claude mcp list
```

To pull updates later: `slurmx update` (or `./update.sh`) — fast-forward `git pull`, re-runs `uv sync` if dependencies changed.

## Before using `launch_remote_session`

This step is **only required if you plan to use `launch_remote_session`** (the tool that spawns `claude remote-control` as a SLURM job). The other tools work without it.

Remote Control requires a **full-scope OAuth session token**, not a long-lived API token. Run this once on a node that mounts your home directory (your login node is fine — `~/.claude/` is NFS-shared):

```bash
claude auth login        # interactive browser OAuth flow
```

If you previously authenticated via `claude setup-token` (which writes an inference-only token), swap it for an OAuth token:

```bash
claude auth logout
claude auth login
```

**Gotcha:** `claude auth status` reports both token types as `authMethod: "claude.ai"`, so you can't tell them apart from the status output. The symptom that you're on the wrong token is `Remote Control requires a full-scope login token` in the SLURM log of a failed `launch_remote_session` job. If you see that, run `claude auth logout && claude auth login`.

## Configuration

Edit `config.py` (copied from one of the templates in `config-examples/`):

| Field | What to fill in |
|-------|----------------|
| `MAIL_USER` | Your cluster email (for SLURM mail notifications) |
| `GOLDEN_QOS` | List of your QoS, e.g. `["yisroel"]` or `["yisroel", "shared"]`. First entry is primary for job submission. |
| `GPU_DEFINITIONS_BY_QOS` | Dict keyed by QoS name; each value is a list of `(name, display_name, vram_gb, golden_quota, golden_partition)` tuples for that QoS. |

`CLAUDE_LOG_DIR` and other paths are auto-populated from `$USER`. You can also set `SLURM_GOLDEN_QOS="a,b"` in your shell to override the list at runtime.

## Maintenance windows

When cluster maintenance is announced, update the `WINDOWS` list in `maintenance.py`:

```python
WINDOWS = [
    (datetime(2026, 5, 1, 8, 0), datetime(2026, 5, 1, 20, 0)),
]
```

Set `WINDOWS = []` when no maintenance is scheduled. Job time limits are automatically capped to finish before the next window; submissions are blocked when less than 5 minutes remain.

## Usage in Claude Code

The server embeds usage rules that Claude reads automatically. Ask naturally:

- "Check GPU availability"
- "Submit a training job needing 48GB VRAM"
- "What happened to job 12345?"
- "Show me a cluster summary"
- "Launch a remote-control session on CPU for 1 day"

## CLI (`slurmx`)

`setup.sh` symlinks `bin/slurmx.sh` into `~/.local/bin/slurmx`, so the command is on PATH globally. Real `argparse` subparsers, same shape as `git` or `aws-cli`:

```bash
slurmx --help                              # list subcommands
slurmx status                              # live scrollable dashboard (in a terminal)
slurmx status --once                       # one-shot text snapshot (+ golden queue when full)
slurmx status -n 2                         # live dashboard, refresh every 2s
slurmx submit --vram 48 -- python train.py # submit a job (golden-first, main fallback)
slurmx submit --vram 48 --golden-only -- python train.py  # preemption-immune, never main
slurmx remote-session                      # interactive launch_remote_session
slurmx rc                                  # short alias for remote-session
slurmx setup                               # = ./setup.sh
slurmx update                              # = ./update.sh
slurmx <subcommand> --help                 # per-subcommand options
```

### Live dashboard

Run in a terminal, `slurmx status` opens a **live, scrollable** dashboard: your jobs
(compact, one line each — no truncation, so 30 queued jobs stay readable), the golden
tickets with the full waiting queue, and cluster-wide GPU availability. It auto-refreshes
(default 5s, `-n/--interval N` to change) without losing your scroll position.

Keys: `↑/↓` or `j/k` scroll, `PgUp/PgDn` page, `g/G` top/bottom, `←/→` or `h/l` pan, `q` quit.

Piped, redirected, or run under `watch` (any non-TTY), it prints the classic one-shot text
and exits, so scripts are unaffected. `--once` forces the one-shot text even in a terminal.

```bash
slurmx status                # live dashboard
slurmx status --once | grep  # one-shot text (also happens automatically when piped)
```

It's stdlib `curses` (no extra dependency) and works over SSH.

## Running tests

```bash
uv sync --extra dev
uv run python -m pytest tests/ -v -k "not live"
```
