#!/usr/bin/env python3
"""One-off RTX 6000 normal-versus-golden preemption measurement."""

import argparse
import os
from pathlib import Path
import pwd
import re
import shlex
import signal
import subprocess
import time


USER = pwd.getpwuid(os.getuid()).pw_name
ROOT = Path(pwd.getpwuid(os.getuid()).pw_dir) / "preemption-probe"
JOB_ID = re.compile(r"^(\d+)(?:;[^;\s]+)?$")
NODE = re.compile(r"^[a-zA-Z0-9-]+$")


def command(*args):
    result = subprocess.run(args, text=True, capture_output=True, timeout=30)
    if result.returncode:
        raise RuntimeError(f"{' '.join(args)}: {result.stderr.strip()}")
    return result.stdout.strip()


def rows(*args):
    return [line.split("|") for line in command(*args).splitlines() if line]


def owned(job_id, qos):
    for row in rows("squeue", "-h", "-j", str(job_id), "-o", "%i|%u|%q"):
        if row == [str(job_id), USER, qos]:
            return True
    return False


def cancel(job_id, qos):
    if owned(job_id, qos):
        command("scancel", str(job_id))
        print(f"cancelled {job_id}", flush=True)
    else:
        print(f"{job_id} not active with expected owner/QoS; no cancellation", flush=True)


def submit(path):
    output = command("sbatch", "--parsable", str(path))
    match = JOB_ID.fullmatch(output)
    if not match:
        raise RuntimeError(f"unrecognized sbatch response: {output!r}")
    return int(match.group(1))


def write(path, content):
    path.write_text(content)
    path.chmod(0o700)


def victim_script(run_dir):
    events = shlex.quote(str(run_dir / "events.log"))
    return f"""#!/bin/bash
# slurmx: {{"total_vram_gb": 48, "supports_gpu_sharding": false, "preemption_safe": true}}
#SBATCH --job-name=preemption-victim
#SBATCH --partition=main
#SBATCH --qos=normal
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --time=00:15:00
#SBATCH --output={run_dir}/victim-%j.out
events={events}
printf 'start %s %s\\n' "${{SLURM_RESTART_COUNT:-0}}" "$(date +%s.%N)" >> "$events"
if (( ${{SLURM_RESTART_COUNT:-0}} > 0 )); then exit 0; fi
trap 'printf "signal USR1 %s\\n" "$(date +%s.%N)" >> "$events"' USR1
trap 'printf "signal TERM %s\\n" "$(date +%s.%N)" >> "$events"' TERM
while true; do
  printf 'heartbeat %s\\n' "$(date +%s.%N)" >> "$events"
  sleep 1
done
"""


def golden_script(run_dir, node):
    return f"""#!/bin/bash
# slurmx: {{"total_vram_gb": 48, "supports_gpu_sharding": false, "preemption_safe": false}}
#SBATCH --job-name=preemption-golden
#SBATCH --partition=rtx6000
#SBATCH --qos=yisroel
#SBATCH --nodelist={node}
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --time=00:05:00
#SBATCH --output={run_dir}/golden-%j.out
sleep 45
"""


def victim_node(job_id, deadline):
    while time.monotonic() < deadline:
        snapshot = rows("squeue", "-h", "-j", str(job_id), "-o", "%i|%u|%q|%T|%N")
        if not snapshot:
            raise RuntimeError("victim left the queue before running")
        if len(snapshot) != 1 or snapshot[0][:3] != [str(job_id), USER, "normal"]:
            raise RuntimeError(f"unexpected victim: {snapshot}")
        _, _, _, state, node = snapshot[0]
        if state == "RUNNING":
            if not NODE.fullmatch(node):
                raise RuntimeError(f"victim has an unexpected node: {node}")
            return node
        if state not in {"PENDING", "CONFIGURING"}:
            raise RuntimeError(f"victim entered {state}")
        time.sleep(2)
    raise TimeoutError("victim did not start before the deadline")


def check_node(job_id, node):
    partition = command("scontrol", "show", "node", node, "-o")
    if "Partitions=" not in partition or "rtx6000" not in partition.split("Partitions=", 1)[1].split()[0].split(","):
        raise RuntimeError(f"{node} is not in the RTX 6000 golden partition")
    jobs = rows("squeue", "--all", "-h", "-w", node, "-t", "RUNNING", "-o", "%i|%u|%q|%T|%N")
    if jobs != [[str(job_id), USER, "normal", "RUNNING", node]]:
        raise RuntimeError(f"other jobs on {node}, refusing golden submission: {jobs}")


def measure(path, submitted):
    events = path.read_text().splitlines() if path.exists() else []
    signals, heartbeats, restarts = [], [], []
    for event in events:
        fields = event.split()
        try:
            if fields[0] == "signal" and fields[1] in {"USR1", "TERM"}:
                signals.append((fields[1], float(fields[2])))
            elif fields[0] == "heartbeat":
                heartbeats.append(float(fields[1]))
            elif fields[0] == "start" and int(fields[1]) > 0:
                restarts.append(float(fields[2]))
        except (IndexError, ValueError):
            pass
    if not restarts:
        return None
    if not signals:
        return "victim restarted, but no signal was recorded; grace period not measurable"
    name, at = signals[0]
    last = max((t for t in heartbeats if t >= at), default=at)
    return (f"{name} at {at:.3f}; last heartbeat {last:.3f}; restart {restarts[0]:.3f}; "
            f"signal-to-last-heartbeat {last-at:.1f}s; golden-submit-to-signal {at-submitted:.1f}s")


def run(seconds):
    run_dir = ROOT / "runs" / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, mode=0o700)
    victim_path = run_dir / "victim.sh"
    write(victim_path, victim_script(run_dir))
    created = []
    deadline = time.monotonic() + seconds
    try:
        victim = submit(victim_path)
        created.append((victim, "normal"))
        print(f"victim {victim}; logs {run_dir}", flush=True)
        node = victim_node(victim, deadline)
        check_node(victim, node)
        print(f"victim running alone on {node}", flush=True)
        golden_path = run_dir / "golden.sh"
        write(golden_path, golden_script(run_dir, node))
        submitted = time.time()
        golden = submit(golden_path)
        created.append((golden, "yisroel"))
        print(f"golden {golden} pinned to {node}", flush=True)
        while time.monotonic() < deadline:
            result = measure(run_dir / "events.log", submitted)
            if result:
                print(result, flush=True)
                return
            time.sleep(1)
        print("no observed restart before deadline; grace period not measured", flush=True)
    finally:
        for job_id, qos in reversed(created):
            try:
                cancel(job_id, qos)
            except Exception as exc:
                print(f"cleanup of {job_id} failed: {exc}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="submit the two disposable jobs")
    parser.add_argument("--seconds", type=int, default=600, help="maximum observation time")
    args = parser.parse_args()
    if not 60 <= args.seconds <= 900:
        parser.error("--seconds must be 60 to 900")
    if not args.run:
        print("Preview only: normal exclusive RTX 6000 victim on main; then a node-pinned yisroel RTX 6000 job on rtx6000. No jobs submitted.")
        return
    run(args.seconds)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    main()
