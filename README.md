# slurmx

MCP server and CLI for submitting, monitoring, and managing SLURM jobs. Executable scripts declare their VRAM needs, sharding support, and preemption safety. SLURMx chooses the GPU allocation and pool from that metadata.

## Install

Paste this into any shell on the cluster — login node, `sinteractive` session,
wherever you are, whatever directory you're in:

```bash
D=$HOME/.claude/mcp-servers/slurmx; git -C "$D" pull --ff-only 2>/dev/null || git clone https://github.com/ItayZloc/slurmx.git "$D"; "$D"/setup.sh
```

Then:

```bash
source ~/.bashrc     # only the first time, so `slurmx` is on PATH
slurmx config        # pick a template, fill in who you are
```

That's the whole install. All you need beforehand is `git` and either `curl` or
`wget` — `setup.sh` installs `uv` itself if you don't have it, creates the venv,
puts `slurmx` on your PATH, and registers the MCP server with Claude Code. It
tells you at the end if anything is left for you to do.

It works from anywhere because `$HOME` is the same NFS mount on every node, so
one install covers the whole cluster. Re-running the same line updates an
existing install instead of failing, so it doubles as `slurmx update`.

Verify:

```bash
slurmx status --once     # talks to SLURM
claude mcp list          # slurmx should say "✔ Connected"
```

See [WELCOME.md](WELCOME.md) for a one-page summary of what's available and how
to drive it from a Claude Code chat — `setup.sh` prints the same thing when it
finishes.

## MCP tools

| Tool | Description |
|------|-------------|
| `cluster_summary` | Single-call dashboard: your jobs + golden tickets (per QoS) + cluster-wide GPU availability. `view="jobs"` or `"gpu"` narrows the output. |
| `submit_job` | Submit an executable script with a `# slurmx:` metadata header. The header, not the caller, selects GPU resources and preemption policy. Supports `dependency` (e.g. `afterok:12345`). Blocks until the job is RUNNING. |
| `select_gpu` | Recommend one GPU from current availability, trying golden first and main second. Advisory; use `submit_job` with a script path and `dry_run=true` to preview that script's allocation. |
| `job_history` | Recent jobs from sacct, finished ones included. Yours only, newest first. |
| `get_job_status` | One job's status as JSON (squeue, falling back to sacct). Carries the pending reason; branch on `state`, not `exit_code`. |
| `wait_for_job` | Block until a job reaches a terminal state. Returns the last polled status on timeout rather than raising. |
| `read_job_log` | Read a job's SLURM log. `output_dir` must be the exact directory the job's `--output` points at — no recursion. |
| `diagnose_job` | Classify a *finished* job's failure (OOM, timeout, missing module, dependency, killed, code error) and show the log tail. Running/pending jobs short-circuit. |
| `cancel_jobs` | Cancel by ID, or every job you own. The count returned is cancels requested, not confirmed. |
| `preemption_info` | Read the controller and configured QoS preemption settings. Failed scheduler queries are shown as unavailable. |
| `probe_preemption` | Default dry run for a guarded, node-pinned normal-vs-golden scheduler diagnostic. Real mode is explicit and retains logs under `~/.slurmx/probes/`. |

Every tool reports failure in its return value instead of raising, so a call that
returned isn't necessarily a call that worked. Each docstring spells out its own
failure strings; the common ones are `success: false` from `submit_job`, `No log
file found ...` from `read_job_log`, and state `UNKNOWN` from `get_job_status`.

## Metadata-aware submission

On this cluster a job's **QoS**, not its card type, decides whether it can be
evicted. `qos=normal` (partition `main`/`gpu`) is the shared pool — everyone can
use it, but a job there is **preemptible**: any group's golden QoS can requeue it.
Your golden QoS (e.g. `yisroel`) runs on the per-card dedicated partitions
(`rtx_pro_6000`, `rtx6000`, `rtx4090`, …) and is **preemption-immune** — it bumps
`normal` jobs and nothing bumps it. A golden QoS is invalid on `main`/`gpu`, so
"golden" always means a dedicated partition.

`submit_job` and `slurmx submit` accept an executable script path and optional
arguments, never a raw command or resource override. Put this as the second line
of the script, immediately after its shebang:

```bash
#!/bin/bash
# slurmx: {"total_vram_gb": 48, "supports_gpu_sharding": false, "preemption_safe": false}
exec python train.py "$@"
```

The header must contain exactly those three JSON keys. `total_vram_gb` must be
a non-negative integer; the other fields must be JSON booleans. A script that
supports sharding may receive one or two cards of the same type on one node.
Their combined VRAM must meet the total requirement. Within each pool,
selection minimizes allocated VRAM, then GPU count, per-card VRAM, and card name.

A safe job checks live golden capacity first, then main, and requests requeue.
The batch requests a USR1 warning 120 seconds before its time limit and forwards
USR1 and TERM to the child while it waits for the child to exit. Set
`preemption_safe` to true only when the script can save durable state and resume
after interruption. An unsafe job uses its golden partition, queues if needed,
and disables requeue. CPU jobs set `total_vram_gb` to zero and must set
`supports_gpu_sharding` to false; they use the configured CPU partition and QoS.

Save the example as `train.sh`, then preview and submit it:

```bash
chmod +x train.sh
slurmx submit --dry-run -- ./train.sh --epochs 3
slurmx submit -- ./train.sh --epochs 3
```

The MCP equivalent is `submit_job(script_path="/path/to/train.sh",
args=["--epochs", "3"], dry_run=true)`. Relative script paths resolve against
`workdir` when supplied, otherwise the caller's working directory. Reusable
scripts whose hardware needs vary by argument should have concrete wrapper
scripts with headers for the resources each invocation needs.

Submission normally waits for the job to start; `--no-wait` (MCP
`wait_until_running=false`) returns after `sbatch`. A start timeout leaves the
job queued. Safe jobs cancel on quota errors: a golden per-account quota race
retries once on main, while a per-user quota fails without retry. Unsafe jobs
remain queued on quota errors. Scratch storage is removed on exit, so save
checkpoints outside `$SCRATCH_DIR`.

SLURMx removes inherited `SBATCH_*` variables and the `SLURM_CLUSTERS` and
`SLURM_HINT` option aliases from the `sbatch` environment so they cannot override
the generated directives. Matching is case-sensitive; other variables, including
`PATH`, `SLURM_CONF`, and runtime job identity, are preserved. Dry runs still
return the generated script without invoking `sbatch`.

`--dependency` accepts `singleton` or `TYPE:JOBID[:JOBID...]`, where `TYPE` is
`after`, `afterany`, `afterok`, `afternotok`, or `aftercorr` and each ID is numeric.
`--after JOBID [JOBID ...]` is shorthand for `afterok`.

Wrap shell pipelines and compound commands in a metadata-bearing script. This
keeps the submit interface auditable and prevents callers from bypassing the
preemption policy.

## Checking preemption behavior

`preemption_info` (or `slurmx preemption-info`) is read-only. It reports the
controller's preemption settings and the normal plus configured golden QoS
relationships. A query that fails is labeled unavailable, so an empty-looking
result never implies a default policy.

`probe_preemption` is a scheduler diagnostic, not a normal submission path.
It defaults to dry run and reports a candidate node, safety evidence, and its
two generated scripts without calling `sbatch`. The CLI equivalent is:

```bash
slurmx probe-preemption
slurmx probe-preemption --real --max-seconds 600
```

Real mode is only appropriate after an explicit decision to test the scheduler.
It rechecks isolation before each submission: the node must be in `main` and
the chosen golden partition, have exactly one free GPU of that type, and have
no running normal-QoS GPU job. It pins only its disposable victim and
preemptor internally; `submit_job` still accepts no caller resource or node
overrides. The probe cancels only created IDs that live scheduler output
confirms belong to the current user with the expected QoS. Its scripts and
event logs stay in `~/.slurmx/probes/` for diagnosis.

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

## Installing by hand

The [one-liner at the top](#install) is the same as this, and re-running it
updates. Do it step by step if you'd rather see each part:

```bash
# 1. Clone
git clone https://github.com/ItayZloc/slurmx.git ~/.claude/mcp-servers/slurmx

# 2. Bootstrap: uv, venv, deps, `slurmx` on PATH, MCP registration
cd ~/.claude/mcp-servers/slurmx
./setup.sh

# 3. Configure: pick a template, then edit it in the form
slurmx config

# 4. Only if setup.sh said it couldn't register the server
claude mcp add slurmx \
  ~/.claude/mcp-servers/slurmx/.venv/bin/python \
  ~/.claude/mcp-servers/slurmx/server.py
```

`setup.sh` appends `~/.local/bin` to your PATH in `~/.bashrc` (or `~/.zshrc`)
when it isn't there already, because otherwise it links `slurmx` somewhere your
shell never looks and setup "succeeds" into a `command not found`. Set
`SLURMX_NO_PATH_EDIT=1` to be told what to add instead of having it added.

To pull updates later: `slurmx update` (or `./update.sh`) — fast-forward `git pull`, re-runs `uv sync` if dependencies changed.

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
| `GOLDEN_POLICY` | Retained legacy setting, default `allow_main`. Current submission and recommendation tools do not read it; submitted jobs take their policy from script metadata. |
| `GPU_DEFINITIONS_BY_QOS` | Dict keyed by QoS name; each value is a list of `(name, display_name, vram_gb, golden_tickets, golden_partition)` tuples for that QoS. |

Edit through the form where you can: it validates. `sbatch` keeps a `--mail-type`
list as long as **one** token is recognised and drops the rest without a word — so
if you hand-edit the file and misspell `FAIL`, the job still submits and you simply
stop getting failure mail. `slurmx config` warns about any event sbatch won't
recognise.

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
SLURM_CPU_PARTITION=bigcpu slurmx submit -- ./job.sh
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
- "Preview train.sh, then submit it with --epochs 3"
- "What happened to job 12345?"
- "Show me a cluster summary"

## CLI (`slurmx`)

`setup.sh` symlinks `bin/slurmx.sh` into `~/.local/bin/slurmx`, so the command is on PATH globally. Real `argparse` subparsers, same shape as `git` or `aws-cli`:

```bash
slurmx --help                              # list subcommands
slurmx status                              # live scrollable dashboard (in a terminal)
slurmx status --once                       # one-shot text snapshot (+ golden queue when full)
slurmx status -n 2                         # live dashboard, refresh every 2s
slurmx submit -- ./train.sh                 # resources come from train.sh metadata
slurmx submit --after 12345 -- ./eval.sh    # wait for job 12345 first
slurmx submit -- ./safe-main.sh --epochs 3  # script arguments follow the path
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
