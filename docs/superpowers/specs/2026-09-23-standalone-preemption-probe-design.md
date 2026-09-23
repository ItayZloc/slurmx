# Standalone preemption probe

Status: superseded by the user's simpler native RTX 6000 test on 2026-09-23.
The current procedure and code are in `scripts/preemption_probe.py` and
`scripts/PREEMPTION_PROBE.md`; the design below is historical.

## Purpose

Measure the warning or grace interval that a normal-QoS GPU job actually sees
when a higher-priority golden job preempts it. The controller currently reports
`PreemptMode=REQUEUE` and `PreemptParameters=grace_time=60`, but that is a
configuration value, not a measured interval. The previous live probe did not
reach preemption: it selected an RTX 3090, for which this account has zero
golden tickets, and submitted only a victim. That victim was later confirmed
`CANCELLED`. No grace-period result came from that run.

The probe is a one-off diagnostic. It must not remain an MCP tool or a `slurmx`
CLI command. The normal `submit_job` interface and read-only `preemption_info`
tool remain unchanged.

## Location and dependencies

Keep a tested source copy at `tools/preemption_probe/` in the SLURMx repo and
install the same files into `/home/itayzloc/preemption-probe/`. The home folder
contains the runner, its tests, a README, and a `runs/` directory for scripts,
logs, job IDs, and results. The runner uses Python's standard library and the
cluster's SLURM commands. It does not import `slurm_mcp`, `config.py`, or code
outside its own folder. This is a user-authorized exception to the usual
MCP-only rule for SLURM operations, limited to this diagnostic.

The eligible golden cards are fixed in this one-off runner:

| GPU type | Golden partition | Golden QoS |
| --- | --- | --- |
| `rtx_6000` | `rtx6000` | `yisroel` |
| `rtx_pro_6000` | `rtx_pro_6000` | `yisroel` |

No other GPU type is a candidate even if it has a configured golden partition
or free capacity. A candidate must also be in `main`, be usable, have exactly
one free GPU of its selected type, and have no running co-resident job of any
QoS. Unclear scheduler output is a refusal, not an empty-cluster assumption.

## Commands and run flow

`python probe.py preview` is the default and performs read-only policy and
candidate checks. It prints the selected node, safety evidence, and the exact
victim and preemptor scripts without submitting a job.

`python probe.py run --max-seconds 600` explicitly runs the diagnostic. It
rechecks policy and node isolation immediately before mutation, then:

1. Creates a private run directory and submits a pinned, requeue-enabled
   normal-QoS victim with an exclusive node request and a finite SLURM time
   limit. The victim logs one-second heartbeats, received signals, and restart
   count to the run directory.
2. Waits for the exact victim ID to be running. Before the preemptor is
   submitted, it verifies the owner, QoS, node, GPU allocation, scheduler
   exclusivity fields, and that no other job occupies the node. A queued,
   malformed, or contradictory victim never passes this gate.
3. Submits a pinned `yisroel` preemptor on that same card. It measures the
   timeline from preemptor submission through the victim's first TERM or USR1
   signal, final heartbeat, and recorded restart. It reports both the
   submission-to-signal and signal-to-final-heartbeat intervals at one-second
   precision, without assuming which one corresponds to the controller's
   configured grace period. A run without signal and restart evidence reports
   "not measured" rather than a guessed grace period.

The process writes each returned job ID to a manifest as soon as `sbatch`
returns. It records its outcome in the run directory. A run-wide deadline
bounds diagnostic waiting; finite SLURM time limits backstop orphaned jobs.

## Cleanup and interruption

The runner's normal exit and SIGINT/SIGTERM paths request cleanup for only
IDs in its manifest. Before each `scancel`, it checks the live ID, owner, and
expected QoS. It never uses `scancel -u` or cancels a job merely because it
shares a node or name. Cleanup results are written to disk, and the caller
must verify terminal states rather than infer them from a successful
`scancel` request. SIGKILL cannot run a Python handler, so the finite batch
time limits remain the last backstop.

`python probe.py cleanup RUN_ID` offers the same exact-ID cleanup for a
disconnected or killed runner. It refuses unknown run IDs, missing manifests,
and identity mismatches. Existing `~/.slurmx/probes/` logs remain untouched.

## MCP removal and testing

Remove the `probe_preemption` MCP registration, `slurmx probe-preemption`
command, package export, old scanner installer, and their documentation.
Keep `preemption_info` read-only and keep normal submission behavior intact.
The old `~/.slurmx/preemption_scan.py` becomes unused; do not delete past run
logs. Remove that generated scanner only after confirming it still matches
the versioned file that installed it.

Port the probe's safety tests to the standalone source. Tests must cover the
RTX 3090 exclusion, dry-run no-submission behavior, exact victim and resident
checks, malformed scheduler responses, interrupted cleanup, and the distinction
between configured and measured grace. Run the standalone tests and the SLURMx
suite before copying the runner to home. Compare the installed files with the
tested source, run a preview, and start real mode only if that preview selects
an RTX 6000 or RTX 6000 Pro candidate. If no such candidate exists, stop
without submitting anything and report that the interval remains unmeasured.
