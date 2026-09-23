# One-off RTX 6000 preemption probe

`probe.py` is installed in `~/preemption-probe/`. Run it with no flags for a
no-submit preview. The live run used:

```bash
/usr/bin/python3 ~/preemption-probe/probe.py --run --seconds 600
```

It submits a normal-QoS, exclusive RTX 6000 victim to `main` without choosing
a node. After it starts, the script checks that the victim is alone on its
assigned node and submits a `yisroel` RTX 6000 job pinned there. Signals,
heartbeats, and restarts go to `~/preemption-probe/runs/<timestamp>/events.log`.
Both test jobs have short SLURM time limits, and the script attempts to cancel
only its two recorded IDs when it finishes. If the controlling process dies,
check those IDs with `squeue` before cancelling them manually.

This is a disposable diagnostic, not a submission interface. Do not run it
again without deciding to spend a golden ticket and preempt a job.

## 2026-09-23 measurement

Normal victim `21606653` landed on `ise-cpu256-23`; golden job `21606665`
was pinned there. The victim logged `SIGTERM` at Unix time
`1790151844.983`, its last heartbeat at `1790152142.869`, and a restarted
batch invocation at `1790152267.300`. The observed signal-to-last-heartbeat
interval was **297.9 seconds**. The golden job began after the victim was
requeued and ran for 45 seconds. Both jobs later reached `COMPLETED`.

This measures the time the victim continued running after TERM, not the
controller's configured `grace_time=60` parameter in isolation. The cluster
also reports `KillWait=300`; the observed interval is close to that value.
