# RTX 6000 preemption test: conclusion

On 2026-09-23, a normal-QoS RTX 6000 job was preempted by a `yisroel` golden
job on the same node. SLURM requeued the normal job and started its batch
script again with `SLURM_RESTART_COUNT=1`. Requeue worked in this test; it did
not restore the process's memory or any unsaved work.

The normal job (`21606653`) was submitted to `main` without a node request and
landed alone on `ise-cpu256-23`. The golden job (`21606665`) was then pinned to
that node in the `rtx6000` partition. The normal script logged `SIGTERM` at
Unix time `1790151844.983`, its last heartbeat at `1790152142.869`, and its
restart at `1790152267.300`. Its last observed heartbeat was **297.9 seconds
after TERM**; the restart occurred **422.3 seconds after TERM**. Both jobs
later completed.

The controller reported `PreemptMode=REQUEUE`, `PreemptParameters=grace_time=60`,
and `KillWait=300` at the time of the test. The measured 297.9 seconds is the
signal-to-last-heartbeat interval, not a direct measurement of the configured
60-second grace parameter. It is close to `KillWait=300`. One run does not
establish a guaranteed checkpoint window for other jobs.

For real jobs, save checkpoints to durable storage during normal operation and
on a termination signal. On restart, load the latest checkpoint and continue.
Do not depend on a five-minute warning or expect SLURM to save program state.
The [one-off probe](../scripts/PREEMPTION_PROBE.md) and its retained event log
at `~/preemption-probe/runs/20260923-112354/events.log` contain the procedure
and raw timestamps.
