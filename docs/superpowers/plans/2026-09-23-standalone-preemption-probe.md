# Standalone preemption probe implementation plan

Status: superseded by the user's simpler native RTX 6000 test on 2026-09-23.
The current procedure and code are in `scripts/preemption_probe.py` and
`scripts/PREEMPTION_PROBE.md`; do not execute the plan below.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the MCP preemption probe with a self-contained one-off diagnostic in `/home/itayzloc/preemption-probe/` and measure only on the user's RTX 6000 or RTX 6000 Pro golden allocation.

**Architecture:** Keep tested source under `tools/preemption_probe/` in SLURMx and copy it to the home folder. `scheduler.py` owns SLURM reads, parsing, candidate selection, and identity checks; `probe.py` owns the CLI, disposable submissions, manifest, timing, and exact-ID cleanup. Neither imports SLURMx code. Remove the old MCP and CLI probe interfaces while retaining `preemption_info`.

**Tech stack:** Python 3.9 standard library at `/usr/bin/python3`, SLURM commands, `unittest` for the standalone tests, and the existing `uv` environment for the SLURMx suite.

**Spec:** `docs/superpowers/specs/2026-09-23-standalone-preemption-probe-design.md`

## Global constraints

- The installed folder is `/home/itayzloc/preemption-probe/`; it contains the runnable source, tests, README, and `runs/` output.
- Runtime code uses only the Python 3.9 standard library and SLURM commands. It imports neither `slurm_mcp` nor `config.py`.
- Direct SLURM commands are a user-authorized exception to the usual MCP-only rule for this one-off diagnostic.
- Candidate GPU types are exactly `rtx_6000` on `rtx6000` and `rtx_pro_6000` on `rtx_pro_6000`, with golden QoS `yisroel`. RTX 3090 must never be selected.
- `preview` submits nothing and is the default. `run --max-seconds 600` is explicit. A real run rescans before each submission.
- Only an exactly verified, exclusive, one-GPU victim with no co-resident job permits preemptor submission. Ambiguous scheduler output refuses mutation.
- Cleanup targets only IDs the runner created and recorded, after matching ID, owner, and QoS. Never use an account-wide cancellation.
- A measured result requires a signal, final heartbeat, and victim restart. Record raw timestamps and both submission-to-signal and signal-to-final-heartbeat intervals; do not equate the configured 60 seconds to an observed interval.
- Keep `preemption_info` and normal `submit_job` unchanged. Preserve historical `~/.slurmx/probes/` logs.

## Review focus

These cases can break an otherwise plausible migration; the task tests below pin each one down.

1. A zero-ticket RTX 3090 with a valid-looking golden partition is skipped, even when it appears before an owned card (Task 1).
2. A pending victim is waited for, while a malformed, terminal, or running-but-invalid victim refuses promptly instead of burning the whole deadline (Tasks 1 and 4).
3. A compressed node list or inconsistent typed versus aggregate GPU allocation cannot make an occupied or overallocated node look free (Task 1).
4. Interruption after `sbatch` cannot broaden cleanup beyond persisted, identity-checked IDs; SIGKILL falls back to finite batch limits (Tasks 3 and 4).
5. A signal without restart, or a restart without signal, yields `not measured` rather than a numeric grace estimate (Task 4).

---

### Task 1: Standalone scheduler reads and candidate selection

**Files:**
- Create: `tools/preemption_probe/scheduler.py`
- Create: `tools/preemption_probe/test_probe.py`
- Reference: `slurm_mcp/preemption.py:37-369,486-515`, `slurm_mcp/preemption_scan_runtime.py:17-49`

**Interfaces:**
- Produces: `Budget`, `Candidate`, `Policy`, `Job`, `QueryFailure`, `Refusal`, `query(argv, budget)`, `job_snapshot(job_id=None, budget=None, details=True)`, `find_candidate(nodes, jobs, budget=None)`, `inspect(budget) -> Tuple[Policy, Optional[Candidate]]`, `verify_victim(job_id, candidate, policy, user, budget) -> bool`, `post_victim_safe(candidate, victim_id, policy, user, budget) -> bool`, and `owned_job(job_id, user, qos, budget) -> bool`. Use Python 3.9-compatible annotations (`Optional`, `Tuple`) in the implementation.
- Consumes: no other project module.

- [ ] **Step 1: Write the failing candidate and parser tests.** Use `unittest` and `unittest.mock.patch` in the new test file. Put these methods on `class SchedulerTests(unittest.TestCase)` with `import scheduler` and `from unittest.mock import patch`:

```python
def test_zero_ticket_card_is_not_a_candidate(self):
    nodes = [
        {"NodeName": "node-3090", "State": "IDLE", "Partitions": "main,rtx3090", "Gres": "gpu:rtx_3090:1", "GresUsed": "gpu:rtx_3090:0"},
        {"NodeName": "node-6000", "State": "IDLE", "Partitions": "main,rtx6000", "Gres": "gpu:rtx_6000:1", "GresUsed": "gpu:rtx_6000:0"},
    ]
    self.assertEqual(scheduler.find_candidate(nodes, []).node, "node-6000")

def test_targeted_pending_is_not_a_running_victim(self):
    with patch.object(scheduler, "query", return_value="101|itayzloc|normal|PENDING|(Resources)\n"):
        self.assertEqual(scheduler.job_snapshot(101, details=False), [])

def test_malformed_targeted_state_refuses(self):
    with patch.object(scheduler, "query", return_value="101|itayzloc|normal|RUNNIGN|node-a\n"):
        with self.assertRaises(scheduler.QueryFailure):
            scheduler.job_snapshot(101, details=False)
```

Add compressed-nodelist, inconsistent-allocation, and controller-without-REQUEUE cases from `tests/test_preemption.py` to this same test file. A new module import failure is the expected red result.

- [ ] **Step 2: Verify red.** Run `/usr/bin/python3 -m unittest discover -s tools/preemption_probe -p 'test_*.py' -v`. Expect import or assertion failures tied to the missing standalone scheduler.
- [ ] **Step 3: Build the standalone scheduler.** Port the existing parsing and safety functions named in the reference range, replacing `shell._run` with local read-only `query` calls backed by `subprocess.run(..., check=True, capture_output=True, text=True, timeout=budget.query_timeout())`. Inline the scanner's `find_candidate` and `post_victim_residency`; there is no subprocess bridge. Set the only eligible mapping as:

```python
ELIGIBLE_GOLDEN = {
    "rtx_6000": "rtx6000",
    "rtx_pro_6000": "rtx_pro_6000",
}
MAIN_PARTITION = "main"
GOLDEN_QOS = "yisroel"
VICTIM_QOS = "normal"
```

`find_candidate` iterates only `ELIGIBLE_GOLDEN`, requires exactly one free GPU and zero running residents, and raises `QueryFailure` on unclear counts. `inspect` verifies `preempt/qos`, requeue mode, and that `yisroel` preempts `normal` before returning a candidate. `verify_victim` returns false only while the target is pending; a running target with wrong identity, allocation, or exclusivity raises `Refusal` immediately. `owned_job` accepts only a complete exact `squeue -j <id>` row with matching ID, user, and QoS.
- [ ] **Step 4: Verify green.** Run the standalone `unittest` command again. Expect all Task 1 tests to pass under `/usr/bin/python3` 3.9.
- [ ] **Step 5: Commit.** Stage only `tools/preemption_probe/scheduler.py` and `tools/preemption_probe/test_probe.py`; commit `Add standalone preemption scheduler checks` to `main`.

### Task 2: Default preview and exact batch scripts

**Files:**
- Create: `tools/preemption_probe/probe.py`
- Modify: `tools/preemption_probe/test_probe.py`
- Reference: `slurm_mcp/preemption.py:429-474`

**Interfaces:**
- Consumes: `scheduler.inspect`, `scheduler.Candidate`, `scheduler.Budget`.
- Produces: `RUNS_DIR = Path(__file__).resolve().parent / "runs"`, `build_scripts(candidate, run_dir, max_seconds, victim_qos) -> Tuple[str, str]`, `preview(max_seconds=600) -> str`, and `main(argv=None) -> int` with no-argument and `preview` routing. Later tasks add `run` and `cleanup` routing.

- [ ] **Step 1: Write failing preview tests.** Add `import probe` to the standalone test file. Put this method on a `unittest.TestCase`, patch `scheduler.inspect` to return an owned RTX 6000 candidate, and reject any additional scheduler call:

```python
def test_preview_never_submits(self):
    chosen = scheduler.Candidate("node-6000", "rtx_6000", "rtx6000")
    with patch.object(probe.scheduler, "inspect", return_value=(scheduler.Policy("normal", frozenset({"normal"})), chosen)):
        with patch.object(probe.scheduler, "query", side_effect=AssertionError("unexpected scheduler call")):
            output = probe.preview()
    self.assertIn("node-6000", output)
    self.assertIn("#SBATCH --exclusive", output)
```

Also assert that no candidate prints a refusal and that both generated scripts pin the same node and matching golden partition.
- [ ] **Step 2: Verify red.** Run the standalone `unittest` command; expect the missing `probe.preview` or `probe.build_scripts` assertion.
- [ ] **Step 3: Implement preview.** Port the batch bodies from `_probe_scripts` exactly, with a finite `--time` and event-log paths under `Path(__file__).resolve().parent / "runs"`. Use `argparse` so no args means `preview`; reject invalid `max_seconds` outside 1..3600. The routing at this stage is:

```python
def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", choices=["preview"], default="preview")
    args = parser.parse_args(argv)
    print(preview())
    return 0
```

Do not add `run` or `cleanup` parser commands until their functions exist.
- [ ] **Step 4: Verify green.** Run the standalone `unittest` command and `/usr/bin/python3 tools/preemption_probe/probe.py --help`; expect passing tests and `preview` as the only active command.
- [ ] **Step 5: Commit.** Commit `Add dry-run standalone preemption preview` to `main`.

### Task 3: Manifest and exact-ID cleanup before live submission

**Files:**
- Modify: `tools/preemption_probe/probe.py`
- Modify: `tools/preemption_probe/test_probe.py`

**Interfaces:**
- Consumes: `scheduler.owned_job`, `scheduler.query`, and authenticated user from `pwd.getpwuid(os.getuid())`.
- Produces: `mutation(argv, budget) -> str`, `record_job(run_dir, job_id, qos, user) -> None`, `cleanup_ids(created, user, budget) -> list[str]`, `cleanup_run(run_id) -> list[str]`, `scheduler.job_state(job_id, budget) -> str`, and `cleanup RUN_ID` CLI routing. `run_id` is a safe basename under the fixed `runs/` directory, never a caller path.

- [ ] **Step 1: Write failing manifest and cleanup tests.** Cover an owned victim, an ID with a different user or QoS, a missing manifest, `../` traversal, and a symlinked run directory. In the test class, use `tempfile.TemporaryDirectory()` in `setUp`, create a `runs/run-01` directory, and patch `probe.RUNS_DIR` to that temporary `runs` path:

```python
def test_cleanup_refuses_an_unowned_id(self):
    probe.record_job(self.run_dir, 101, "normal", "itayzloc")
    with patch.object(probe.scheduler, "owned_job", return_value=False):
        with patch.object(probe, "mutation", side_effect=AssertionError("scancel called")):
            result = probe.cleanup_run(self.run_dir.name)
    self.assertIn("identity", " ".join(result).lower())
```

Use a temporary `RUNS_DIR` in tests; ensure an owned ID produces exactly `scancel 101`, not an account-wide command. Add `job_state` tests for an exact `sacct` terminal row and for a mismatched job ID that must not count as terminal.
- [ ] **Step 2: Verify red.** Run the standalone `unittest` command; expect missing manifest/cleanup functions.
- [ ] **Step 3: Implement state and cleanup.** Store `{"user": "...", "jobs": [{"id": 101, "qos": "normal"}]}` in `manifest.json` mode 0600 using a same-directory temporary file and `os.replace`:

```python
payload = {"user": user, "jobs": jobs}
temp = run_dir / "manifest.json.tmp"
temp.write_text(json.dumps(payload))
temp.chmod(0o600)
os.replace(temp, run_dir / "manifest.json")
```

Validate `run_id` with `^[A-Za-z0-9_-]+$`, reject symlinked run directories and manifest files, and reject IDs or QoS values outside the probe's recorded schema. `cleanup_run` reads this manifest and calls `cleanup_ids`. `cleanup_ids` queries each ID's live owner and QoS before `scancel <id>`; it also accepts the in-memory IDs from a run whose manifest write failed. Implement `mutation` in this file with a bounded `subprocess.run` and `SBATCH_`/`SLURM_CLUSTERS`/`SLURM_HINT` filtering when the command is `sbatch`. Add `scheduler.job_state` with an exact-ID `squeue` query and `sacct` fallback; an unknown row is not proof of completion. Return/report scheduler failures rather than broadening scope. Add the `cleanup` CLI command.
- [ ] **Step 4: Verify green.** Run the standalone `unittest` command; expect no cancellation in every mismatch case.
- [ ] **Step 5: Commit.** Commit `Add exact-ID cleanup for standalone probe` to `main`.

### Task 4: Bounded live run and evidence-based timing

**Files:**
- Modify: `tools/preemption_probe/probe.py`
- Modify: `tools/preemption_probe/test_probe.py`
- Create: `tools/preemption_probe/README.md`
- Reference: `slurm_mcp/preemption.py:475-628`

**Interfaces:**
- Consumes: Tasks 1-3 scheduler, batch-script, and cleanup functions.
- Produces: `run(max_seconds=600) -> dict`, `measure(events_path, preemptor_submitted_at) -> dict`, and `run --max-seconds N` CLI routing. The result contains raw epoch timestamps and `measured: bool`.

- [ ] **Step 1: Write failing live-run tests.** Test that a pending victim waits, invalid exclusivity or wrong allocation stops promptly without submitting the preemptor, a verified victim triggers exactly one preemptor submission, interruption after `sbatch` invokes exact-ID cleanup, and signal/restart pairs are required:

```python
def test_restart_without_signal_is_not_measured(self):
    events = self.run_dir / "victim-events.log"
    events.write_text("heartbeat 100\nrestart 130\n")
    self.assertFalse(probe.measure(events, 90)["measured"])

def test_signal_without_restart_is_not_measured(self):
    events = self.run_dir / "victim-events.log"
    events.write_text("heartbeat 100\nsignal TERM 110\nheartbeat 169\n")
    self.assertFalse(probe.measure(events, 90)["measured"])

def test_signal_and_restart_preserve_both_intervals(self):
    events = self.run_dir / "victim-events.log"
    events.write_text("heartbeat 100\nsignal TERM 110\nheartbeat 169\nrestart 170\n")
    result = probe.measure(events, 90)
    self.assertEqual(result["submission_to_signal_seconds"], 20)
    self.assertEqual(result["signal_to_last_heartbeat_seconds"], 59)
```

Use a fake `mutation` runner for submission tests; no test calls a real SLURM binary. Simulate `record_job` raising after `sbatch` returns and assert that the in-memory ID still reaches `cleanup_ids`.
- [ ] **Step 2: Verify red.** Run the standalone `unittest` command; expect missing `run` or `measure` behavior.
- [ ] **Step 3: Implement the live flow.** Port `_submit`, the post-victim gate, and the bounded polling loop from the existing probe, using Task 1's `scheduler.verify_victim` and Task 3's `mutation`. Parse only a numeric `sbatch --parsable` ID; add it to an in-memory `created` list immediately and persist it with `record_job` before the next query. The mutation-to-manifest sequence is:

```python
output = mutation(["sbatch", "--parsable", str(script_path)], budget).strip()
match = re.fullmatch(r"([0-9]+)(?:;[^;\s]+)?", output)
if match is None:
    raise scheduler.QueryFailure("sbatch did not return one numeric job ID")
job_id = int(match.group(1))
created.append((job_id, qos))
record_job(run_dir, job_id, qos, user)
```

Wrap every mutation and wait in the run-wide budget. Save `result.json` in the run directory. In a `finally` block, call `cleanup_ids(created, user, cleanup_budget)` so a failed manifest write does not strand a known ID; install SIGINT/SIGTERM handlers that stop the loop and allow that `finally` block to run, then restore the previous handlers. If a process receives SIGKILL, the finite batch limits remain the backstop. Add explicit `run` routing only now.
- [ ] **Step 4: Verify green.** Run the standalone `unittest` command and `/usr/bin/python3 -m py_compile tools/preemption_probe/probe.py tools/preemption_probe/scheduler.py`. Expect all tests to pass and no syntax error under Python 3.9.
- [ ] **Step 5: Document use and commit.** In the home-folder README, show `preview`, `run --max-seconds 600`, `cleanup RUN_ID`, log/result locations, the two allowed GPU types, and how to read `not measured`. Commit `Add bounded standalone preemption run` to `main`.

### Task 5: Remove the permanent probe interfaces

**Files:**
- Modify: `server.py`, `cli/slurmx.py`, `cli/preemption.py`, `slurm_mcp/__init__.py`, `slurm_mcp/preemption.py`, `setup.sh`, `README.md`, `WELCOME.md`, `tests/test_preemption.py`
- Delete: `slurm_mcp/preemption_scan_runtime.py`
- Reference: `server.py:48-50,226-247`, `cli/slurmx.py:129-139`, `setup.sh:70-83`

**Interfaces:**
- Removes: MCP `probe_preemption`, CLI `slurmx probe-preemption`, and package `slurm_mcp.probe_preemption`.
- Retains: MCP/CLI `preemption_info`, `slurm_mcp.preemption_info`, and script-only `submit_job` unchanged.

- [ ] **Step 1: Write failing absence tests.** Update the server/CLI tests first:

```python
def test_probe_is_not_registered_on_server():
    import server
    assert not hasattr(server, "probe_preemption")

def test_probe_is_not_a_slurmx_command():
    from cli.slurmx import build_parser
    with pytest.raises(SystemExit):
        build_parser().parse_args(["probe-preemption"])
```

Keep a positive `preemption_info` test. The old server and parser make these tests fail.
- [ ] **Step 2: Verify red.** Run `UV_CACHE_DIR=/tmp/slurmx-uv-cache uv run pytest -q tests/test_preemption.py -k 'probe_is_not or preemption_info'`; expect only the new absence tests to fail.
- [ ] **Step 3: Remove registrations and old runtime.** Delete the server decorator and its instructions, the CLI subparser and probe helper, the package export, scanner installer stanza, and scanner runtime file. The remaining package export is exactly:

```python
from .preemption import preemption_info
```

Keep `slurm_mcp/preemption.py` only for the read-only inspection functions it needs; move or delete probe-specific tests after equivalent standalone coverage exists. Update README and WELCOME to point to `~/preemption-probe/` without calling it an MCP tool. Do not change `submit_job`.
- [ ] **Step 4: Verify green and regression.** Run `UV_CACHE_DIR=/tmp/slurmx-uv-cache uv run pytest -q tests/test_preemption.py tests/test_submission_metadata.py` and `bash -n setup.sh`. Then run the full non-live suite with a bounded 15-minute command: `UV_CACHE_DIR=/tmp/slurmx-uv-cache timeout 900s uv run pytest -q -m 'not live'`. If a broad test stalls or fails, record the exact test and investigate before deployment.
- [ ] **Step 5: Commit.** Commit `Remove permanent preemption probe interfaces` to `main`.

### Task 6: Deploy the tested folder and run the one-off diagnostic

**Files:**
- Install: `tools/preemption_probe/{probe.py,scheduler.py,test_probe.py,README.md}` to `/home/itayzloc/preemption-probe/`
- Create at runtime: `/home/itayzloc/preemption-probe/runs/`
- Inspect before optional removal: `/home/itayzloc/.slurmx/preemption_scan.py`

**Interfaces:**
- Consumes: the committed standalone source and its CLI.
- Produces: a home-folder copy, preview output, and either a measured timeline or an explicit no-measurement reason; all created probe jobs end in terminal states.

- [ ] **Step 1: Verify the source.** Run `/usr/bin/python3 -m unittest discover -s tools/preemption_probe -p 'test_*.py' -v`, `git diff --check`, and `git status --short`; require green tests and only intended files.
- [ ] **Step 2: Publish.** Push the completed commits to `main`, then fast-forward `/home/itayzloc/.claude/mcp-servers/slurmx`. The current MCP process will still advertise the removed tool until one later reload; the standalone runner does not depend on that reload.
- [ ] **Step 3: Install with exact targets.** Check whether `/home/itayzloc/preemption-probe/` already exists; do not overwrite unrelated files. Create it and its `runs/` subdirectory mode 0700, then install the four source files. Compare each installed file byte-for-byte with the tested source. Do not copy the repository venv or any SLURMx package code.
- [ ] **Step 4: Retire only the generated scanner.** Compare `/home/itayzloc/.slurmx/preemption_scan.py` against the versioned scanner from commit `ac39385`. Delete that exact home file only if it still matches; otherwise preserve it and report the difference. Keep `~/.slurmx/probes/` and all previous run logs.
- [ ] **Step 5: Preview before any job.** Run `/usr/bin/python3 /home/itayzloc/preemption-probe/probe.py preview`. If it refuses or finds no RTX 6000/6000 Pro candidate, report no measurement and submit nothing. If it selects one, inspect the exact scripts and start `/usr/bin/python3 /home/itayzloc/preemption-probe/probe.py run --max-seconds 600` in a tracked local session.
- [ ] **Step 6: Verify outcome and cleanup.** Read `result.json` and the event log, confirm the victim and preemptor IDs from the manifest are terminal using the runner's exact-ID status check, and use `cleanup RUN_ID` only if they remain active. Report the raw timestamps, the two measured intervals, and uncertainty at one-second precision. If either signal or restart is absent, state that the grace interval remains unmeasured. Do not call the controller's `grace_time=60` an observed value.
