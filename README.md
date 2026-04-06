# SLURM MCP Server

MCP server that lets Claude Code submit, monitor, and manage SLURM GPU jobs directly. Auto-selects the smallest GPU that fits your VRAM needs, tries golden tickets first, falls back to cluster-wide.

## Tools

| Tool | Description |
|------|-------------|
| `cluster_summary` | Single-call dashboard: your jobs + GPU availability |
| `submit_job` | Submit GPU/CPU jobs (auto-selects GPU by VRAM) |
| `check_gpu_availability` | See free GPUs (golden tickets + cluster-wide) |
| `select_gpu` | Recommend best GPU for a VRAM requirement |
| `my_jobs` | List your current running/pending jobs |
| `job_history` | Show recent completed/failed jobs (via sacct) |
| `get_job_status` | Get detailed status of a specific job |
| `wait_for_job` | Block until a job finishes |
| `read_job_log` | Read SLURM log file for a job |
| `diagnose_job` | Classify job failures (OOM, timeout, missing module, code error) |
| `cancel_jobs` | Cancel jobs by ID or all |
| `launch_claude` | Spawn Claude Code as a SLURM CPU job |

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/ItayZloc/slurm-mcp.git ~/.claude/mcp-servers/slurm-utils

# 2. Configure
cd ~/.claude/mcp-servers/slurm-utils
cp config.example.py config.py
# Edit config.py — fill in MAIL_USER and CLAUDE_LOG_DIR

# 3. Create venv and install dependencies
uv sync

# 4. Register with Claude Code
claude mcp add slurm-utils \
  ~/.claude/mcp-servers/slurm-utils/.venv/bin/python \
  ~/.claude/mcp-servers/slurm-utils/server.py
```

Verify it works:
```bash
claude mcp list
```

## Configuration

Edit `config.py` (copied from `config.example.py`):

| Field | What to fill in |
|-------|----------------|
| `MAIL_USER` | Your cluster email (for SLURM mail notifications) |
| `CLAUDE_LOG_DIR` | Path to store Claude job logs, e.g. `/home/USERNAME/.claude/logs` |

The rest (GPU definitions, QoS, partitions) is pre-filled for the shared cluster.

## Maintenance Windows

When cluster maintenance is announced, update the `WINDOWS` list in `maintenance.py`:

```python
WINDOWS = [
    (datetime(2026, 5, 1, 8, 0), datetime(2026, 5, 1, 20, 0)),
]
```

Set `WINDOWS = []` when no maintenance is scheduled. Job time limits are automatically capped to finish before the next window.

## Usage in Claude Code

The server embeds usage rules that Claude reads automatically. Just ask naturally:

- "Check GPU availability"
- "Submit a training job needing 48GB VRAM"
- "What happened to job 12345?"
- "Show me a cluster summary"

## CLI Usage

You can also use the CLI wrapper directly:

```bash
python submit_job.py --vram 48 -- python train.py --lr 1e-4
python submit_job.py --vram 48 --dry-run -- python train.py
```

## Running Tests

```bash
uv sync --extra dev
uv run python -m pytest tests/ -v -k "not live"
```
