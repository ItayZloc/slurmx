# slurmx

MCP server and unified CLI that lets you (and Claude Code) submit, monitor, and manage SLURM GPU jobs. Auto-selects the smallest GPU that fits your VRAM needs, tries golden tickets first, falls back to cluster-wide.

After install (`./setup.sh` or `slurmx setup`), see [WELCOME.md](WELCOME.md) for a one-page summary of what's available and how to drive it from a Claude Code chat. The setup script prints the same content at the end of `uv sync`.

## MCP tools

| Tool | Description |
|------|-------------|
| `cluster_summary` | Single-call dashboard: your jobs + golden tickets (per QoS) + cluster-wide GPU availability. `view="jobs"` or `"gpu"` narrows the output. |
| `submit_job` | Submit GPU/CPU jobs (auto-selects GPU by VRAM). An omitted `golden_only` resolves from your `GOLDEN_POLICY`; pass `true`/`false` to decide per job. Supports `dependency` (e.g. `afterok:12345`). Blocks until the job is RUNNING. |
| `select_gpu` | Recommend a GPU for a VRAM requirement, with current availability. Advisory — it always reports the non-golden selection, so it can disagree with what a default `submit_job` picks. |
| `job_history` | Recent jobs from sacct, finished ones included. Yours only, newest first. |
| `get_job_status` | One job's status as JSON (squeue, falling back to sacct). Carries the pending reason; branch on `state`, not `exit_code`. |
| `wait_for_job` | Block until a job reaches a terminal state. Returns the last polled status on timeout rather than raising. |
| `read_job_log` | Read a job's SLURM log. `output_dir` must be the exact directory the job's `--output` points at — no recursion. |
| `diagnose_job` | Classify a *finished* job's failure (OOM, timeout, missing module, dependency, killed, code error) and show the log tail. Running/pending jobs short-circuit. |
| `cancel_jobs` | Cancel by ID, or every job you own. The count returned is cancels requested, not confirmed. |

Every tool reports failure in its return value instead of raising, so a call that
returned isn't necessarily a call that worked. Each docstring spells out its own
failure strings; the common ones are `success: false` from `submit_job`, `No log
file found ...` from `read_job_log`, and state `UNKNOWN` from `get_job_status`.

## Golden tickets (preemption) vs the main pool

On this cluster a job's **QoS**, not its card type, decides whether it can be
evicted. `qos=normal` (partition `main`/`gpu`) is the shared pool — everyone can
use it, but a job there is **preemptible**: any group's golden QoS can requeue it.
Your golden QoS (e.g. `yisroel`) runs on the per-card dedicated partitions
(`rtx_pro_6000`, `rtx6000`, `rtx4090`, …) and is **preemption-immune** — it bumps
`normal` jobs and nothing bumps it. A golden QoS is invalid on `main`/`gpu`, so
"golden" always means a dedicated partition.

Which pool a job takes is set per job with **`--golden-only`** / **`--allow-main`**
(`golden_only=true`/`false` on the MCP tool):

- **golden-only** — force `qos=yisroel` on the card's dedicated partition
  and **never** accept a preemptible slot. If the golden ticket is full the job
  waits in the golden queue and starts automatically when a slot frees (it is not
  downgraded). Works on every card, including the smaller ones the group doesn't
  own (`golden_tickets=0`) — those then preempt other groups' `normal` jobs there.
  Recommended for training you don't want evicted. Ignored for CPU jobs.
- **`--allow-main` / `golden_only=false`** — golden-first on the cards you own
  (`golden_tickets > 0`), then fall back to the preemptible main pool if golden is
  full.

With neither flag, `GOLDEN_POLICY` in `config.py` decides — `slurmx config` edits
it, and it defaults to `golden_only`:

| `GOLDEN_POLICY` | what an unspecified `golden_only` becomes |
|---|---|
| `golden_only` | golden-only, the default |
| `allow_main` | the main-pool fallback |
| `ask` | no default at all: `submit_job` returns `success: false` until Claude asks you and passes `golden_only` explicitly (dry runs included), and `slurmx submit` prompts `[g/m]` at the terminal |

An explicit flag or argument always wins, so a one-off "burst this onto main"
never means editing the config. `SLURM_GOLDEN_POLICY=allow_main` overrides the
policy for one shell — useful for a script that must not stop to prompt.

Claude reads the policy from the MCP server's instructions, which are built when
the session connects. After changing it, reconnect with `/mcp` or restart the
session, or Claude keeps following the old rule.

When a golden ticket is **full**, `slurmx status` and `cluster_summary` list the
card's pending GPUs by user in dispatch order — like the Running block but
ordered: consecutive jobs from the same user merge into one `user: N GPU(s)` row
(GPUs summed), and a user split by another user shows at each position — so you
can see who is ahead of you.

`pending` counts only jobs waiting for a **free GPU**. A job pending on a
dependency (`--after`/`--dependency`), a `scontrol hold`, or a future `--begin`
time keeps its place in the priority order, but the scheduler skips it, so it
won't take the next card that frees. Those are reported separately as
`N blocked` and left out of the ordered list:

```
  rtx_pro_6000: 2/16 free (14 running, 0 pending, 1 blocked)
```

Quota waits (`QOSMaxGRESPerUser`, `MaxGRESPerAccount`, …) still count as pending:
they clear when someone's running job ends, which is exactly waiting for a slot.
Same for a `%N`-throttled job array.

## Installation

Copy this line into any shell on the cluster — a login node, a `sinteractive`
session, wherever you are. It doesn't care about your working directory, and
re-running it updates an existing install instead of failing.

```bash
D=$HOME/.claude/mcp-servers/slurmx; git clone https://github.com/ItayZloc/slurmx.git "$D" 2>/dev/null || git -C "$D" pull --ff-only; "$D"/setup.sh
```

Then `slurmx config` to fill in who you are. That's the whole install.

It works from anywhere because `$HOME` is the same NFS mount on every node, so
one install covers the cluster. `setup.sh` creates the venv, symlinks `slurmx`
into `~/.local/bin/`, and registers the MCP server with Claude Code if `claude`
is on your PATH. Every step is idempotent.

You need `uv` and `git`; `setup.sh` stops with a link if `uv` is missing.

<details>
<summary>The same thing, step by step</summary>

```bash
# 1. Clone
git clone https://github.com/ItayZloc/slurmx.git ~/.claude/mcp-servers/slurmx

# 2. Bootstrap: venv, deps, `slurmx` on PATH, MCP registration
cd ~/.claude/mcp-servers/slurmx
./setup.sh

# 3. Configure: pick a template, then edit it in the form
slurmx config

# 4. Only if setup.sh said it couldn't register the server
claude mcp add slurmx \
  ~/.claude/mcp-servers/slurmx/.venv/bin/python \
  ~/.claude/mcp-servers/slurmx/server.py
```
</details>

Verify it works:
```bash
slurmx status --once     # talks to SLURM
claude mcp list          # slurmx should say "✔ Connected"
```

To pull updates later: `slurmx update` (or `./update.sh`) — fast-forward `git pull`, re-runs `uv sync` if dependencies changed. The install line above does the same thing.

## Configuration

Run `slurmx config` — a terminal form over `config.py`. It creates the file from
a template on first run, validates every field before it writes, and keeps your
comments and `os.environ.get` fallbacks intact (it replaces one literal, not the
file). `slurmx config --show` prints the resolved values instead, and that is
what you get automatically when the output is piped.

`config.py` holds personal things only: who you are, which QoS you belong to,
how many golden tickets your group owns.

| Field | What to fill in |
|-------|----------------|
| `MAIL_USER` | Your cluster email for SLURM notifications. Defaults to `$USER@post.bgu.ac.il`. |
| `MAIL_TYPE` | Which events mail you, passed to `sbatch --mail-type`. A checklist in the form (`⏎` opens it, space ticks an event). Defaults to `["END", "FAIL"]`; unticking everything, or ticking `NONE`, turns mail off entirely. `SLURM_MAIL_TYPE="BEGIN,END"` overrides it for one shell. |
| `GOLDEN_QOS` | List of your QoS, e.g. `["yisroel"]` or `["yisroel", "shared"]`. First entry is primary for job submission. |
| `GOLDEN_POLICY` | What an unspecified `golden_only` becomes: `golden_only`, `allow_main`, or `ask`. A radio list in the form (`⏎` opens it). Defaults to `golden_only`. See [Golden tickets](#golden-tickets-preemption-vs-the-main-pool). |
| `GPU_DEFINITIONS_BY_QOS` | Dict keyed by QoS name; each value is a list of `(name, display_name, vram_gb, golden_tickets, golden_partition)` tuples for that QoS. |

Edit through the form where you can: it validates. `sbatch` keeps a `--mail-type`
list as long as **one** token is recognised and drops the rest without a word, so
a hand-typed `["END", "FIAL"]` silently costs you the failure mail. `slurmx config`
warns when it sees one.

A save writes `config.py.bak` first, so the previous version is always one `mv`
away. The form refuses to write a config whose primary QoS has no GPU cards:
`GPU_DEFINITIONS = GPU_DEFINITIONS_BY_QOS[GOLDEN_QOS[0]]` would raise at import
and take down every subcommand.

Paths are auto-populated from `$USER`. You can also set `SLURM_GOLDEN_QOS="a,b"` in your shell to override the list at runtime.

`config.py` is gitignored, so `slurmx update` never touches it — your copy keeps
whatever template it was created from, and updating never requires editing it. Any
key added to the templates later is therefore missing from every `config.py` already
on disk, so new keys live in `config_defaults.py` with a fallback rather than being
imported straight from `config`.

### Fixed cluster facts

`CPU_PARTITION`, `CPU_QOS` and `MAIN_PARTITION` are **not** config keys. A
partition name is a property of the cluster, the same for every user, so it lives
in `config_defaults.py` (which is tracked) and is read from nowhere else. They
don't appear in `slurmx config` at all. To change one for a single shell:

```bash
SLURM_CPU_PARTITION=bigcpu slurmx submit --vram 0 -- ./job.sh
```

They used to be config keys, so a `config.py` created before 2026-07-30 still
assigns them. That copy is ignored. If it assigns a value that differs from the
fixed one, `slurmx config` says so and names the env var that restores the old
behaviour, rather than letting your jobs quietly move partition.

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

## CLI (`slurmx`)

`setup.sh` symlinks `bin/slurmx.sh` into `~/.local/bin/slurmx`, so the command is on PATH globally. Real `argparse` subparsers, same shape as `git` or `aws-cli`:

```bash
slurmx --help                              # list subcommands
slurmx status                              # live scrollable dashboard (in a terminal)
slurmx status --once                       # one-shot text snapshot (+ golden queue when full)
slurmx status -n 2                         # live dashboard, refresh every 2s
slurmx submit --vram 48 -- python train.py # submit a job (pool from GOLDEN_POLICY)
slurmx submit --vram 48 --after 12345 -- python eval.py   # wait for job 12345 first
slurmx submit --vram 48 --allow-main -- python train.py   # allow the main-pool fallback
slurmx submit --vram 48 --golden-only -- python train.py  # never preemptible
slurmx select-gpu --vram 48                # recommend a GPU for a VRAM need
slurmx job-status 12345                    # status of one job (alias: slurmx job)
slurmx wait 12345                          # block until a job finishes
slurmx log 12345                           # read a job's SLURM log (--tail N)
slurmx diagnose 12345                      # classify a failed job (OOM/timeout/...)
slurmx history --days 7                    # recent finished jobs (sacct)
slurmx cancel 12345                        # cancel jobs by ID (or --all)
slurmx config                              # edit config.py in a terminal form
slurmx config --show                       # print the resolved config as text
slurmx setup                               # = ./setup.sh
slurmx update                              # = ./update.sh
slurmx <subcommand> --help                 # per-subcommand options
```

### Live dashboard

Run in a terminal, `slurmx status` opens a **live, scrollable** dashboard: your jobs in
`squeue --me` format (full list, no truncation, so 30 queued jobs stay readable), then the
golden tickets (pending listed by user in dispatch order) and cluster-wide GPU availability
shown side by side. It auto-refreshes (default 5s, `-n/--interval N` to change) without
losing your scroll position.

Keys: `↑/↓` or `j/k` scroll, `PgUp/PgDn` page, `g/G` top/bottom, `←/→` or `h/l` pan, `q` quit.

It's colorized with a cyan/teal accent (green = free/running, yellow = pending, red = full),
dim secondary labels, and ●/○ status glyphs when the terminal supports color; it degrades to
plain text otherwise.

Piped, redirected, or run under `watch` (any non-TTY), it prints the classic one-shot text
and exits, so scripts are unaffected. `--once` forces the one-shot text even in a terminal.

```bash
slurmx status                # live dashboard
slurmx status --once | grep  # one-shot text (also happens automatically when piped)
```

It's stdlib `curses` (no extra dependency) and works over SSH.

### What "cluster-wide free" counts

The Cluster-Wide totals count GPUs a job could actually land on, so they line up with
`sres`. A node is skipped when:

- **it can't take work** — `down`, `drain*`, `fail*`, `inval`, `unknown`, `future`,
  `maint`, `reboot*`, or one of the flags `*` (slurmctld can't reach slurmd), `$`
  (maintenance reservation), `@`/`^` (reboot pending/issued). Slurm renders the same
  condition as a word or a flag depending on the base state — an idle maintenance node
  prints `maint`, a busy one prints `mixed$` — so both forms are filtered together.
  Those GPUs are reported as `(N offline: <nodes>)` next to the card, so a total that
  drops overnight has a visible reason. Flags that still allow scheduling stay counted:
  `-` (earmarked by the backfill scheduler) and the power-save states.
- **we can't submit to it** — the node is only reachable through a partition that isn't
  `MAIN_PARTITION` or one of the configured golden partitions. That capacity belongs to
  someone else, so it isn't listed at all (not even as offline).

Nodes carrying more than one card type (`gpu:rtx_3090:1,gpu:gtx_1080:1`) count under
both.

## Running tests

```bash
uv sync --extra dev
uv run python -m pytest tests/ -v -k "not live"
```
