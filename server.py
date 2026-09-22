"""
MCP server exposing slurm_mcp tools to Claude Code.

Run: uv run --with "mcp[cli]" python ~/.claude/mcp-servers/slurmx/server.py
"""

from __future__ import annotations

import json
import sys
import os
from typing import Literal

# Ensure slurm_mcp, cli, config, maintenance are importable from same directory
sys.path.insert(0, os.path.dirname(__file__))

import slurm_mcp
from cli import render
from config_defaults import GOLDEN_POLICY
from mcp.server.fastmcp import FastMCP

def build_instructions(policy: str = GOLDEN_POLICY) -> str:
    """Return static submission guidance; policy now comes from script metadata."""
    return _INSTRUCTIONS


_INSTRUCTIONS = """\
## SLURM Job Submission Rules

When the user says "job" or "a job running", they mean a SLURM job — not a local process.
Check with MCP cluster_summary or squeue, not ps aux.

### Key Rules
- Always use dry_run=true first to preview the sbatch script before submitting.
- submit_job accepts only an executable script path and its arguments. The script
  must put a strict `# slurmx: {...}` JSON header immediately after its shebang.
  Its total_vram_gb, supports_gpu_sharding, and preemption_safe values select
  resources and preemption behavior. Do not request a GPU, pool, or QoS directly.
- Preemption-safe GPU jobs search live golden capacity first and then main. Unsafe
  jobs are golden-only and queue if necessary. CPU jobs use the configured CPU pool.
- Use a metadata-bearing wrapper script for shell pipelines or compound commands.
- Maintenance windows are enforced automatically — job time limits are capped to finish
  before scheduled maintenance. If a window is imminent (<5 min), submissions are blocked.
- Do NOT write raw sbatch scripts or Python code that imports slurm_mcp — always use MCP tools.
- Use cluster_summary as the single dashboard tool: it covers jobs AND GPU availability.
  Use view="jobs" or view="gpu" to narrow the output.
- Use diagnose_job to classify failures (OOM, timeout, missing module, code error).
- Use preemption_info for read-only controller/QoS preemption settings. The
  probe_preemption tool is dry-run-first; do not use real mode unless a
  scheduler diagnostic was explicitly authorized.
- These tools report failure in their return value rather than raising, so read what
  comes back: "success: false" from submit_job, "No log file found ..." from
  read_job_log, state UNKNOWN from get_job_status. A call that returned is not a
  call that worked.
- submit_job blocks until the job is RUNNING, so expect it to take a while; it does
  not wait for the job to finish.
"""

mcp = FastMCP("slurmx", instructions=build_instructions())


@mcp.tool()
def select_gpu(vram_gb: int) -> str:
    """Recommend a GPU for a VRAM requirement and show what's free right now.

    Advisory only — it submits nothing, and submit_job runs its own selection
    anyway. Use it to decide or to report, not as a required pre-step.

    Selection here always takes the availability-driven path: the smallest
    fitting card with a free golden slot, else the smallest fitting card free
    anywhere, reported on the main pool with qos "normal". Submission uses
    the executable script's metadata and also checks node capacity; it may
    select two cards when the script supports sharding. Unsafe scripts queue
    in a golden partition even when no card is free. To preview the resources
    for a specific script, call submit_job with its path and dry_run=true.

    Only cards given a non-zero golden quota in config.py count as golden, so a
    small ask is upgraded to a bigger owned card whenever that card has a free
    golden slot — 8 GB can come back as a 48 GB card.

    Returns three lines of text, not JSON: the card and its VRAM, "Partition:
    P, QoS: Q", then "Availability: golden f/t, cluster f/t". Golden f/t is the
    configured quota minus running jobs, not physical cards, and "no golden"
    means the card has no golden quota in config rather than that golden is
    full. Cluster f/t counts only GPUs on usable nodes in partitions you can
    submit to.

    Three outcomes arrive as strings rather than errors: vram_gb=0 returns a
    fixed CPU-only note without touching the cluster, an ask bigger than every
    configured card returns "No GPU has >= NGB VRAM", and nothing free right
    now returns "No GPU with >= NGB VRAM is currently free" plus a per-card
    list. That last one is a snapshot, not a verdict: an unsafe submission
    still queues and starts when a slot frees.

    cluster_summary view="gpu" is the fuller picture: every card, every
    configured QoS, and who is ahead of you in the golden queue.

    Args:
        vram_gb: VRAM needed in GB. 0 and anything above the largest configured
            card short-circuit to the fixed strings above.
    """
    if vram_gb == 0:
        return "CPU-only work requires an executable script with total_vram_gb=0 and supports_gpu_sharding=false in its slurmx header."

    selection = slurm_mcp.select_gpu(vram_gb)
    avail = slurm_mcp.check_availability()

    if selection is None:
        capable = [g for g in slurm_mcp.GPU_TYPES if g.vram_gb >= vram_gb]
        if not capable:
            max_gpu = max(slurm_mcp.GPU_TYPES, key=lambda g: g.vram_gb)
            return f"No GPU has >= {vram_gb}GB VRAM. Max available: {max_gpu.vram_gb}GB ({max_gpu.name})."
        lines = [f"No GPU with >= {vram_gb}GB VRAM is currently free.", ""]
        for g in capable:
            golden = avail.golden.get(g.name)
            cluster = avail.cluster.get(g.name)
            gstr = f"golden {golden.free}/{golden.total}" if golden else "no golden"
            cstr = f"cluster {cluster.free}/{cluster.total}" if cluster else "N/A"
            lines.append(f"  {g.name} ({g.vram_gb}GB): {gstr}, {cstr}")
        return "\n".join(lines)

    gpu_type, partition, qos = selection
    gpu_info = slurm_mcp.GPU_BY_NAME[gpu_type]
    golden = avail.golden.get(gpu_type)
    cluster = avail.cluster.get(gpu_type)
    gstr = f"golden {golden.free}/{golden.total}" if golden else "no golden"
    cstr = f"cluster {cluster.free}/{cluster.total}" if cluster else "N/A"

    return (
        f"Recommendation: {gpu_type} ({gpu_info.vram_gb}GB VRAM)\n"
        f"  Partition: {partition}, QoS: {qos}\n"
        f"  Availability: {gstr}, {cstr}"
    )


@mcp.tool()
def cluster_summary(
    view: Literal["full", "jobs", "gpu"] = "full",
    qos: str | None = None,
) -> str:
    """Show your jobs, golden tickets per QoS, and cluster-wide GPU counts.

    Returns preformatted text, not JSON: "=== Your Jobs ===", one "=== Golden
    Tickets ({qos} QoS) ===" per configured QoS, then "=== Cluster-Wide ===".
    It's a snapshot, so call it again to poll.

    Jobs come from `squeue -u $USER` (yours only); other people appear only as
    per-user GPU counts in the golden sections. view="full" prints an "N
    running, M pending (K GPUs in use)" line plus one line per RUNNING and per
    PENDING job, so a job in any other state (COMPLETING, SUSPENDED,
    CONFIGURING) lands in neither the counts nor the list. view="jobs" tables
    every row instead. Neither view prints the pending reason or the partition,
    so this can't say why a job is stuck, and finished jobs are out of squeue
    entirely. The GPU column is squeue's tres-per-node, so a CPU job and a job
    that asked for GPUs at job scope (--gpus rather than --gres) both show
    "N/A" and add 0 to the GPU total; submit_job writes --gres, so its jobs
    are counted.

    In the golden sections free = quota - running, where quota is the
    golden_quota from config.py. Pending is printed beside it but never
    subtracted, so "3 free" with a queue behind it can still mean someone ahead
    of you takes the card. Only cards with a non-zero golden quota are listed,
    so a QoS whose cards are all quota 0 renders as a bare header. The per-user
    "Pending (next first)" list in dispatch order is fetched only when some card
    of that QoS is at 0 free, and it's capped at 15 rows per card. "N blocked"
    is a GPU count like running and pending beside it, covering jobs gated on a
    dependency, a hold, or a future --begin; the scheduler skips those, so they
    stay out of both the pending count and the ordered list.

    Cluster-Wide counts only GPUs a job could actually land on: the node has to
    sit in the main pool or a configured golden partition and be in a state
    that accepts work. Capacity lost to a node that's down, drained,
    unreachable, or inside a maintenance reservation shows as "(N offline:
    <nodes>)" rather than quietly leaving the total.

    Every squeue/sinfo call here swallows a non-zero exit, a 30s timeout, or a
    missing binary and returns empty, so a broken SLURM never raises. It
    renders as "No jobs.", golden cards sitting at free == full quota, and an
    empty Cluster-Wide block — read that combination as a failed scan, not an
    idle cluster.

    For one job's pending reason use get_job_status; for finished jobs,
    job_history; select_gpu answers "which card fits N GB" off this same scan.

    Args:
        view: "full" (default) = jobs summary + golden + cluster-wide; "jobs" =
            the job table alone; "gpu" = golden + cluster-wide, no jobs.
        qos: Restrict to one QoS. Filters the job list and which golden sections
            render; Cluster-Wide ignores it. A name that isn't a configured
            golden QoS silently drops the golden block instead of erroring.
            Default None shows all of them.
    """
    parts: list[str] = []

    if view in ("full", "jobs"):
        jobs = slurm_mcp.my_jobs(qos=qos)
        if view == "jobs":
            parts.append(render.render_jobs_table(jobs))
        else:
            parts.append(render.render_jobs_summary(jobs))

    if view in ("full", "gpu"):
        avail = slurm_mcp.check_availability()
        queues = slurm_mcp.golden_queues(avail, qos_filter=qos)
        golden = render.render_golden_all(avail, qos_filter=qos, queues=queues)
        if golden:
            parts.append(golden)
        parts.append(render.render_cluster_wide(avail))

    return "\n\n".join(p for p in parts if p)


@mcp.tool()
def preemption_info() -> str:
    """Show SLURM controller and QoS preemption settings without changing jobs.

    Reports SLURM version, controller PreemptType/PreemptMode/Parameters,
    JobRequeue, KillWait, and normal plus primary-golden QoS preemption
    relationships. A failed scontrol or sacctmgr query is marked unavailable,
    never reported as an empty/default setting.
    """
    return slurm_mcp.preemption_info()


@mcp.tool()
def probe_preemption(dry_run: bool = True, max_seconds: int = 600) -> str:
    """Preview or explicitly run a tightly scoped normal-vs-golden preemption probe.

    Dry run is the default and submits nothing. It reports the isolated node,
    safety evidence, and the two generated scripts. Real mode repeats the
    safety scan, resolves node-list membership, uses internal node pinning,
    submits only disposable jobs, and cancels only IDs it can re-verify as
    owned by the authenticated user with the expected QoS. Probe logs are
    retained under /home/<user>/.slurmx/probes/ rather than inherited $HOME.

    Args:
        dry_run: Keep true unless a real scheduler diagnostic is authorized.
        max_seconds: Real-probe diagnostic wall-clock bound, from 1 to 3600 seconds.
            Exact-ID ownership cleanup has a separate fixed five-second bound.
    """
    return slurm_mcp.probe_preemption(dry_run=dry_run, max_seconds=max_seconds)


@mcp.tool()
def get_job_status(job_id: int) -> str:
    """Get one SLURM job's status as JSON, from squeue with an sacct fallback.

    squeue is tried first (PENDING/RUNNING); if the job isn't there, sacct is
    queried, which covers finished jobs. Neither call is filtered to your user,
    unlike cluster_summary and job_history.

    Returns pretty-printed JSON with job_id, state, exit_code, node, elapsed,
    reason, finished. What those keys don't tell you on their own:

    - Branch on `state`, not `finished`. On the squeue path `finished` is
      hardcoded false whatever the state; only the sacct path sets it, and only
      for COMPLETED, FAILED, CANCELLED, CANCELLED+, TIMEOUT, NODE_FAIL,
      PREEMPTED, OUT_OF_MEMORY.
    - `exit_code` defaults to 0 and is filled only from sacct, so it's 0 for
      every pending or running job, and it's the exit half of sacct's
      "code:signal" pair — a signal-killed job (0:9) reports 0. Judge success
      from the state.
    - `reason` is squeue's pending reason ("Resources", "Priority",
      "QOSMaxGRESPerUser", ...), "" once the job runs and always "" on the
      sacct path. A quota reason on a golden-only job means it's queued for a
      golden slot, not broken; cluster_summary shows who's ahead.
    - `node` is squeue's NodeList, empty while the job is pending. `elapsed` is
      TimeUsed while queued or running, sacct Elapsed afterwards.

    state "UNKNOWN" with finished false is the nothing-found path: not in
    squeue and no matching sacct row. That covers a wrong ID, a job aged out of
    accounting, and a squeue/sacct call that failed or timed out (the shell
    helper swallows the error). The tool never raises, so UNKNOWN looks like a
    successful call — and polling it for `finished` never terminates. Finished
    array tasks land here too: the sacct parser takes only a JobID that is
    exactly the integer you passed, so 12345_1 is skipped (the same rule that
    correctly skips the 12345.batch and 12345.extern step rows).

    For a job that already failed, diagnose_job adds the log tail and a
    classification. cluster_summary answers "what are all my jobs doing" and
    carries name, partition, QoS, and GPU gres, none of which are here.

    Args:
        job_id: SLURM job ID. For an array job this reports whatever squeue
            prints first for the array while it's queued, and UNKNOWN after.
    """
    status = slurm_mcp.get_job_status(job_id)
    return json.dumps(status.to_dict(), indent=2)


@mcp.tool()
def read_job_log(job_id: int, output_dir: str = "logs", tail: int = 100) -> str:
    """Read a job's SLURM log file.

    Globs output_dir for `slurm-*-<job_id>.out`, then `slurm-<job_id>.out`,
    then `*-<job_id>.out`, and returns the first match as raw text — no header,
    no path, no job state. One directory level, no recursion. The trailing
    `-<job_id>.out` anchor stops job 12345 picking up 123456789, and the last
    pattern finds logs from hand-written sbatch scripts as long as the filename
    ends that way. submit_job's script sets --output and no --error, so its
    stdout and stderr share one file; a job submitted elsewhere with a separate
    --error= path keeps its tracebacks in a file this won't find.

    output_dir decides whether the call works at all. It has to be the exact
    directory the job's --output points at. Nothing is searched recursively, so
    a job logging to `logs/training/` needs that full path, not `logs`. A
    relative path resolves against the MCP server process's working directory —
    wherever the session started, not wherever you've been working since — so
    pass an absolute path when the job wasn't submitted from here.

    No match returns the literal string "No log file found for job <id> in
    <output_dir>/" rather than raising; check for it before treating the result
    as log content. An empty result means something else: the file exists and
    the job hasn't written to it yet (still pending, or stdout buffered).

    Logs outlive the job, so finished jobs read fine. For one that already
    failed, diagnose_job pairs this same tail with the exit state and a
    classification. Each call is a snapshot, not a follow — poll it again to
    watch a running job.

    Args:
        job_id: The SLURM job ID.
        output_dir: Directory holding the log (default 'logs', which is
            submit_job's default and is only right for jobs submitted with that
            default from this same directory).
        tail: Lines from the end (default 100). 0 or negative returns the whole
            file, which for a training log can be tens of thousands of lines.
    """
    content = slurm_mcp.read_job_log(job_id, output_dir=output_dir, tail=tail)
    if content is None:
        return f"No log file found for job {job_id} in {output_dir}/"
    return content


@mcp.tool()
def diagnose_job(job_id: int, output_dir: str = "logs", log_lines: int = 50) -> str:
    """Classify why a finished SLURM job failed: state, GPU, log tail, fix.

    Returns plain text, not JSON: a `=== Job Diagnosis: <id> ===` header, then
    State (with exit code), Classification, GPU (only while sacct still reports
    the job's AllocTRES), Elapsed, a `Suggested action:` line, and the log tail
    under `--- Log tail (last N lines) ---`. With no log file matched, the last
    line is `No log file found in <output_dir>/.` instead; a log that exists
    but is empty prints neither. Nothing raises — an unknown or purged job id
    comes back as State: UNKNOWN.

    Only ended jobs get diagnosed. RUNNING and PENDING short-circuit to one
    line ("Job N is PENDING (reason: Resources). No diagnosis needed.") and a
    clean COMPLETED to "completed successfully (elapsed: ...)"; the log isn't
    read in those cases.

    Classification is the first match over the job state plus the log tail, in
    order: OOM, TIMEOUT, DEPENDENCY_FAILED, MISSING_MODULE, KILLED, CODE_ERROR,
    else UNKNOWN. Only the tail is scanned, so a traceback or CUDA OOM that
    scrolled off more than log_lines ago reads as UNKNOWN — raise log_lines
    before concluding the log is clean. Two suggestion lines are heuristics
    worth distrusting: the OOM one walks a hardcoded ladder (8 -> 11 -> 24 ->
    48 -> 96 GB) up from whatever card sacct reports, and prints "Already on
    largest GPU" both when the card really is top-tier and when no GPU could be
    determined at all; the CODE_ERROR one prints the last tail line that isn't
    a `File ...` frame, which is the real exception only if nothing logged
    after the traceback.

    For a live job use read_job_log or get_job_status; job_history lists which
    recent jobs failed.

    Args:
        job_id: SLURM job ID, resolved via squeue then sacct.
        output_dir: Where to glob for the log — same patterns and the same
            relative-to-server-cwd rule as read_job_log (default 'logs'). A
            wrong directory isn't an error: you get a log-less diagnosis that
            usually classifies UNKNOWN.
        log_lines: Tail lines both shown and scanned for classification
            (default 50). 0 means the whole file, though the header still says
            "last 0 lines".
    """
    return slurm_mcp.diagnose_job(job_id, output_dir=output_dir, log_lines=log_lines)


@mcp.tool()
def submit_job(
    script_path: str,
    args: list[str] | None = None,
    job_name: str | None = None,
    workdir: str | None = None,
    output_dir: str = "logs",
    dependency: str | None = None,
    wait_until_running: bool = True,
    dry_run: bool = False,
) -> str:
    """Submit an executable metadata-bearing script and wait for it to start.

    Preview with dry_run=true before submitting. The script must be a regular
    executable file, start with a shebang, and put one strict JSON metadata
    line immediately after it:
    # slurmx: {"total_vram_gb": 48, "supports_gpu_sharding": false, "preemption_safe": false}

    Those are the only header keys. total_vram_gb is a non-negative integer
    (not a boolean); the other values are booleans. Zero requests a CPU job
    and requires supports_gpu_sharding=false. CPU jobs use the configured
    CPU partition and QoS. Put shell pipelines in an executable wrapper with
    its own header; neither raw commands nor caller resource overrides work.

    GPU jobs receive one card, or up to two of the same type on one node when
    supports_gpu_sharding=true. Combined VRAM must meet total_vram_gb. Within
    each pool, selection minimizes allocated VRAM, then GPU count, per-card
    VRAM, and card name. A script must handle the selected GPU count itself.

    With preemption_safe=true, selection checks live free capacity in the
    primary golden pool first, then main/normal. The batch requests requeue
    and a USR1 warning 120 seconds before its time limit, forwards USR1 and
    TERM to the child, and waits until it exits. The script must checkpoint
    and resume its own state; these signals alone do not make it safe.

    With preemption_safe=false, selection uses the best configured golden
    partition without checking availability, so the job can queue there.
    The batch disables requeue and executes the script directly.

    A real submission polls every 5 seconds until RUNNING, a terminal/fatal
    state, or START_TIMEOUT. Safe jobs cancel on quota errors: a golden
    per-account quota race retries once on main, while a per-user limit
    fails without retry. Unsafe jobs remain queued on quota errors. A start
    timeout leaves the job queued and returns success with that fact in
    message. This call does not wait for completion; use wait_for_job.

    Time, memory, CPU count, mail events, and excluded nodes come from the
    configuration. Maintenance caps the time limit and can block submission.
    The batch creates $SCRATCH_DIR under /scratch, falling back to /tmp, and
    removes it on exit. Save checkpoints and other durable output elsewhere.

    Returns plain text fields: success, job_id, gpu_type, partition, qos,
    message, and the generated script on dry runs. Validation, unavailable
    resources, sbatch rejection, and polling failures return success: false.
    Maintenance rejection raises. Always read success and message.

    Args:
        script_path: Executable script to submit. A relative path resolves
            against workdir when supplied, otherwise the server process cwd.
        args: Literal script arguments, shell-quoted without interpreting
            pipelines, substitutions, or redirects.
        job_name: Defaults to the script filename stem with dots replaced
            by hyphens. Use letters, digits, dots, underscores, or hyphens,
            starting with a letter or digit.
        workdir: Directory entered before the script runs. None preserves
            the server process's submission directory.
        output_dir: Log directory, default "logs". Uses
            {output_dir}/slurm-{job_name}-%J.out. A relative path is relative
            to the server cwd, not workdir. Create it before submitting:
            SLURM opens the log before the batch's mkdir runs. Use the same
            directory for read_job_log and diagnose_job. Paths accept
            letters, digits, dots, underscores, hyphens, and slashes.
        dependency: "singleton" or a supported after* dependency followed
            by numeric IDs, e.g. "afterok:12345:12346". Also used when
            estimating start time for maintenance.
        wait_until_running: Default true. False returns after sbatch.
        dry_run: Return the generated batch without submitting or waiting.
            Safe GPU dry runs still inspect live availability and can fail
            when no suitable configuration is currently free.
    """
    result = slurm_mcp.submit_job(
        script_path=script_path,
        args=args,
        job_name=job_name,
        workdir=workdir,
        output_dir=output_dir,
        dependency=dependency,
        wait_until_running=wait_until_running and not dry_run,
        dry_run=dry_run,
    )

    parts = [
        f"success: {result.success}",
        f"job_id: {result.job_id}",
        f"gpu_type: {result.gpu_type}",
        f"partition: {result.partition}",
        f"qos: {result.qos}",
        f"message: {result.message}",
    ]
    if dry_run and result.sbatch_script:
        parts.append(f"\n--- sbatch script ---\n{result.sbatch_script}")
    return "\n".join(parts)


@mcp.tool()
def cancel_jobs(job_ids: list[int] | None = None, all_jobs: bool = False, pending_only: bool = False) -> str:
    """Cancel SLURM jobs, by explicit ID or every job $USER owns.

    job_ids wins over all_jobs. If both are set only the listed IDs are
    cancelled and pending_only is ignored, so a RUNNING job in the list dies
    too. With job_ids it's one `scancel <id>` per ID; with all_jobs it's a
    single `scancel -u $USER` with no partition, name, or state filter beyond
    pending_only, so it also kills interactive and holder jobs, not just
    training.

    Passing neither job_ids nor all_jobs is a silent no-op that still returns a
    success-looking string. Same for pending_only on its own, and an empty list
    behaves like None.

    Returns one line, "Cancelled N job(s).". N is how many cancels were
    *requested* — len(job_ids), or for all_jobs the `squeue -u $USER` line
    count taken just before the scancel — not how many were confirmed. scancel's
    exit status is discarded, so an unknown ID, an already-finished job, or a
    permission error all count as cancelled. Nothing is re-queried afterwards:
    confirm with get_job_status, or cluster_summary view="jobs", which is also
    how to look IDs up beforehand.

    Args:
        job_ids: IDs to cancel. None or [] falls through to the all_jobs branch.
        all_jobs: Cancel everything for $USER (default false). Ignored when
            job_ids is non-empty.
        pending_only: Restrict the all_jobs sweep to PENDING jobs, on both the
            count and the scancel (default false, which takes running jobs
            too). No effect unless all_jobs is true.
    """
    count = slurm_mcp.cancel_jobs(
        job_ids=job_ids,
        all_jobs=all_jobs,
        pending_only=pending_only,
    )
    return f"Cancelled {count} job(s)."


@mcp.tool()
def wait_for_job(job_id: int, poll_interval: int = 30, timeout: int = 600) -> str:
    """Block until a SLURM job reaches a terminal state, then return JSON.

    Polls get_job_status on a loop, checking immediately first, so a job that
    has already ended returns without sleeping. Terminal means any of
    COMPLETED, FAILED, CANCELLED, CANCELLED+, TIMEOUT, NODE_FAIL, PREEMPTED,
    OUT_OF_MEMORY — "finished" is not "succeeded".

    Returns the same JSON object get_job_status does, with the same caveats.
    Check `finished` first: on timeout this doesn't raise and never says the
    word timeout, it just returns the last polled status with finished=false
    and state still PENDING or RUNNING.

    A job id in neither squeue nor sacct (typo, or accounting purged it) comes
    back as UNKNOWN, which is not terminal, so the call burns the whole timeout
    first. With timeout=0 that's an infinite block.

    Nothing is emitted until it returns, which makes it a poor way to babysit a
    long training run — poll get_job_status and read_job_log on your own
    cadence instead, or chain the follow-up work at submit time with
    submit_job(dependency="afterok:<id>"). You don't need it to see a job start
    either, since submit_job already waits for RUNNING. Use it for short jobs,
    or to wait out a queue with a bounded timeout.

    Args:
        job_id: SLURM job ID.
        poll_interval: Seconds slept between checks (default 30). The elapsed
            check runs before the sleep, so the call can overshoot timeout by
            up to this much.
        timeout: Max seconds to wait (default 600 — this tool gives up after 10
            minutes even though the underlying library defaults to no limit).
            0 blocks until the job actually ends.
    """
    status = slurm_mcp.wait_for_job(job_id, poll_interval=poll_interval, timeout=timeout)
    return json.dumps(status.to_dict(), indent=2)


@mcp.tool()
def job_history(days: int = 3, state: str | None = None, limit: int = 30) -> str:
    """List your recent SLURM jobs from accounting (sacct), finished included.

    Returns preformatted text, not JSON: a "Recent jobs (last N day(s)):"
    header, one line per job with JOB_ID / NAME / STATE / EXIT / ELAPSED / GPU
    / NODE, and a trailing "N job(s) shown." count. There's no timestamp
    column — sacct's Start field is queried but never printed — so this can't
    say when a job ran.

    Scope is always $USER (yours only); there's no parameter for someone else's
    jobs. Rows arrive in sacct's own order and are reversed before `limit`
    truncates the tail, so the table is newest-first by whatever sacct sorted
    on and an older job that's still running can be cut. The window is
    `--starttime now-{days}days`, which selects jobs active in it: a job that
    started a week ago and is still running is listed under days=3, and RUNNING
    jobs show alongside finished ones despite the tool name. A job cancelled
    before it ever started shows ELAPSED 00:00:00 and NODE "None assigned".
    Array tasks (JobID `12345_0`) drop out along with the `.batch`/`.extern`
    step rows, since the parser keeps only plain-integer JobIDs.

    Failures come back as strings, never exceptions: a bad `state` returns
    "sacct query failed: ...", and no matching rows returns "No jobs found in
    the last N day(s)."

    For what's queued or running right now, with QoS and pending reason, use
    cluster_summary; for one job, get_job_status. STATE and EXIT here won't say
    why something failed — diagnose_job classifies that and adds the log tail.

    Args:
        days: History window, default 3.
        state: sacct state filter, default None = every state in the window.
            Uppercased before use, with "OOM" translated to OUT_OF_MEMORY;
            everything else passes straight through, so any name sacct accepts
            works (PREEMPTED, NODE_FAIL, RUNNING, ...) and any name it doesn't
            comes back as the error string above.
        limit: Max rows, default 30, applied after the newest-first reversal.
    """
    return slurm_mcp.job_history(days=days, state=state, limit=limit)


if __name__ == "__main__":
    mcp.run(transport="stdio")
