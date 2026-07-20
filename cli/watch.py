#!/usr/bin/env python3
"""Live, scrollable, auto-refreshing dashboard for `slurmx status`.

`slurmx status` in an interactive terminal opens this TUI: your jobs (compact,
one line each, no truncation of the list), the golden-ticket sections with the
full waiting queue, and cluster-wide GPU availability — all in a scrollable pad
that refreshes on an interval without losing your scroll position.

Design notes:
  - stdlib `curses` only (no new dependency; works over SSH). Non-interactive
    callers never reach here — `cli/status.py` routes pipes / `watch` / `--once`
    to the one-shot text path, so an agent's Bash can't hang on a blocking UI.
  - A daemon worker thread fetches snapshots off the main thread so scrolling
    stays smooth while `squeue`/`sinfo` run; only the main thread touches curses.
  - `build_dashboard_lines` and `clamp_scroll` are pure and unit-tested; the
    curses loop is thin glue.
"""

from __future__ import annotations

import curses
import re
import threading
import time
from dataclasses import dataclass

import slurm_mcp
from cli import render

REFRESH_INTERVAL = 5.0          # seconds between snapshots (default)
_INPUT_POLL_MS = 100            # how often getch wakes to redraw the clock
_H_STEP = 8                     # columns per horizontal pan keypress
_SPINNER = "|/-\\"
_STATE_TAG = {"RUNNING": "RUN", "PENDING": "PEND"}


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested)
# --------------------------------------------------------------------------- #

def _trunc(s: str, n: int) -> str:
    """Truncate to n chars with an ellipsis so table columns stay aligned."""
    return s if len(s) <= n else s[: n - 1] + "…"


def _jobs_lines(jobs: list[dict]) -> list[str]:
    """Compact 'Your Jobs' block: a summary header + one line per job.

    Running jobs show gres/runtime/node; non-running jobs show the scheduler
    reason (why they're still pending). The list is never truncated — that's
    what the scrollable pad is for.
    """
    running = [j for j in jobs if j.get("state") == "RUNNING"]
    pending = [j for j in jobs if j.get("state") == "PENDING"]
    other = [j for j in jobs if j.get("state") not in ("RUNNING", "PENDING")]

    gpu_count = 0
    for j in running:
        m = re.search(r":(\d+)", j.get("gpu_gres", ""))
        if m:
            gpu_count += int(m.group(1))

    lines = [
        f"=== Your Jobs ({len(running)} running · {len(pending)} pending · "
        f"{gpu_count} GPUs) ==="
    ]
    if not jobs:
        lines.append("  No jobs.")
        return lines

    for j in running:
        lines.append(
            f"  RUN  {j['job_id']:<9} {_trunc(j['name'], 32):<32} "
            f"{j.get('gpu_gres', ''):<20} {j.get('runtime', ''):<10} "
            f"{j.get('node', '')}"
        )
    for j in pending + other:
        tag = _STATE_TAG.get(j.get("state", ""), (j.get("state", "") or "?")[:4])
        reason = j.get("reason") or ""
        reason = "" if reason in ("None", "N/A") else reason
        lines.append(
            f"  {tag:<4} {j['job_id']:<9} {_trunc(j['name'], 32):<32} {reason}"
        )
    return lines


def build_dashboard_lines(avail, jobs, queues, qos: str | None = None) -> list[str]:
    """Full compact line buffer for the TUI: jobs + golden (uncapped queue) +
    cluster-wide. Pure — reuses the existing `render_*` formatters."""
    lines: list[str] = []
    lines.extend(_jobs_lines(jobs or []))
    lines.append("")
    golden = render.render_golden_all(avail, qos_filter=qos, queues=queues, limit=None)
    if golden:
        lines.extend(golden.splitlines())
        lines.append("")
    lines.extend(render.render_cluster_wide(avail).splitlines())
    return lines


def clamp_scroll(offset: int, total: int, viewport: int) -> int:
    """Clamp a scroll offset to [0, max(0, total - viewport)]."""
    max_off = max(0, total - viewport)
    if offset < 0:
        return 0
    if offset > max_off:
        return max_off
    return offset


# --------------------------------------------------------------------------- #
# Background fetch worker
# --------------------------------------------------------------------------- #

@dataclass
class _Snapshot:
    avail: object | None
    jobs: list | None
    queues: dict | None
    stamp: str
    error: str | None = None


class _TuiState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.snapshot: _Snapshot | None = None
        self.version = 0


def _now() -> str:
    return time.strftime("%H:%M:%S")


def _worker(state: _TuiState, qos: str | None, interval: float) -> None:
    """Fetch a snapshot every `interval`s into state under the lock. Never dies
    silently — a failed fetch stores an error snapshot and the loop retries."""
    while not state.stop.is_set():
        try:
            avail = slurm_mcp.check_availability()
            queues = slurm_mcp.golden_queues(avail, qos_filter=qos)
            jobs = slurm_mcp.my_jobs(qos=qos)
            snap = _Snapshot(avail, jobs, queues, _now())
        except Exception as e:  # noqa: BLE001 — surface, don't crash the thread
            snap = _Snapshot(None, None, None, _now(), error=f"refresh failed: {e}")
        with state.lock:
            state.snapshot = snap
            state.version += 1
        state.stop.wait(interval)


# --------------------------------------------------------------------------- #
# curses loop
# --------------------------------------------------------------------------- #

def _addbar(stdscr, y: int, text: str, maxx: int) -> None:
    """Draw a full-width reverse-video bar, safe against edge writes."""
    bar = text[: maxx - 1].ljust(maxx - 1)
    try:
        stdscr.addnstr(y, 0, bar, maxx - 1, curses.A_REVERSE)
    except curses.error:
        pass


def _loop(stdscr, state: _TuiState, interval: float, qos: str | None) -> None:
    for setup in (lambda: curses.curs_set(0),
                  curses.use_default_colors,
                  lambda: curses.mousemask(curses.ALL_MOUSE_EVENTS)):
        try:
            setup()
        except (curses.error, AttributeError):
            pass
    stdscr.timeout(_INPUT_POLL_MS)

    lines = ["loading…"]
    stamp = "--:--:--"
    rendered_version = -1
    pad = None
    pad_w = 0
    last_size = (0, 0)
    row = col = 0
    frame = 0

    while True:
        frame += 1
        maxy, maxx = stdscr.getmaxyx()
        if maxy < 3 or maxx < 10:
            stdscr.erase()
            try:
                stdscr.addnstr(0, 0, "terminal too small", maxx - 1)
            except curses.error:
                pass
            stdscr.refresh()
            if stdscr.getch() in (ord("q"), ord("Q")):
                break
            continue

        body_h = maxy - 2  # top bar + bottom bar

        with state.lock:
            snap = state.snapshot
            version = state.version

        dirty = pad is None
        if snap is not None and version != rendered_version:
            rendered_version = version
            stamp = snap.stamp
            if snap.error:
                lines = [snap.error, "", "(retrying…)"]
            else:
                lines = build_dashboard_lines(snap.avail, snap.jobs, snap.queues, qos)
            dirty = True
        if (maxy, maxx) != last_size:
            last_size = (maxy, maxx)
            dirty = True

        if dirty:
            pad_w = max(maxx, max((len(s) for s in lines), default=0) + 1)
            pad = curses.newpad(max(len(lines) + 1, body_h), pad_w)
            pad.erase()
            for i, s in enumerate(lines):
                try:
                    pad.addstr(i, 0, s)
                except curses.error:
                    pass

        row = clamp_scroll(row, len(lines), body_h)
        col = clamp_scroll(col, pad_w, maxx)

        first = row + 1
        last = min(row + body_h, len(lines))
        spin = _SPINNER[frame % len(_SPINNER)]
        qos_note = f" · qos={qos}" if qos else ""
        _addbar(stdscr, 0,
                f" slurmx status {spin} · data {stamp} · every {interval:g}s{qos_note}",
                maxx)
        _addbar(stdscr, maxy - 1,
                f" ↑↓/jk PgUp/Dn g/G scroll · ←→/hl pan · q quit"
                f"   {first}-{last}/{len(lines)} ",
                maxx)
        stdscr.noutrefresh()
        try:
            pad.refresh(row, col, 1, 0, maxy - 2, maxx - 1)
        except curses.error:
            pass
        curses.doupdate()

        ch = stdscr.getch()
        if ch in (ord("q"), ord("Q")):
            break
        elif ch in (curses.KEY_DOWN, ord("j")):
            row += 1
        elif ch in (curses.KEY_UP, ord("k")):
            row -= 1
        elif ch in (curses.KEY_NPAGE, ord(" ")):
            row += body_h
        elif ch == curses.KEY_PPAGE:
            row -= body_h
        elif ch == ord("g"):
            row = 0
        elif ch == ord("G"):
            row = len(lines)  # clamped next iteration
        elif ch in (curses.KEY_RIGHT, ord("l")):
            col += _H_STEP
        elif ch in (curses.KEY_LEFT, ord("h")):
            col -= _H_STEP
        elif ch in (curses.KEY_HOME, ord("0")):
            col = 0
        elif ch == curses.KEY_MOUSE:
            try:
                _, _, _, _, bstate = curses.getmouse()
                if bstate & curses.BUTTON4_PRESSED:
                    row -= 3
                elif bstate & getattr(curses, "BUTTON5_PRESSED", 0):
                    row += 3
            except curses.error:
                pass
        # ch == -1 (timeout) or KEY_RESIZE: fall through, redraw next iteration.


def run_tui(interval: float = REFRESH_INTERVAL, qos: str | None = None) -> None:
    """Launch the live dashboard. Blocks until the user quits (q / Ctrl-C).

    Raises curses.error if the terminal can't host curses (e.g. TERM=dumb); the
    caller is expected to fall back to the one-shot text dashboard.
    """
    state = _TuiState()
    worker = threading.Thread(target=_worker, args=(state, qos, interval), daemon=True)
    worker.start()
    try:
        curses.wrapper(_loop, state, interval, qos)
    finally:
        state.stop.set()


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Live slurmx status dashboard.")
    p.add_argument("--interval", "-n", type=float, default=REFRESH_INTERVAL)
    p.add_argument("--qos", default=None)
    args = p.parse_args()
    run_tui(interval=args.interval, qos=args.qos)


if __name__ == "__main__":
    main()
