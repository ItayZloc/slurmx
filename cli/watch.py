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
import threading
import time
from dataclasses import dataclass

import slurm_mcp
from cli import render
from cli import theme as theme_mod

REFRESH_INTERVAL = 5.0          # seconds between snapshots (default)
_INPUT_POLL_MS = 100            # how often getch wakes to redraw the clock
_H_STEP = 8                     # columns per horizontal pan keypress
_COL_GAP = 4                    # spaces between the golden and cluster columns
_SPINNER = "|/-\\"


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested)
# --------------------------------------------------------------------------- #

def _side_by_side(left: list[str], right: list[str], gap: int = _COL_GAP) -> list[str]:
    """Merge two blocks into two top-aligned columns.

    The left column is padded to the widest of the rows that sit beside the
    (shorter) right block, so headers and card summaries line up. Rows past the
    right block's height print left-only, so the deep, uncapped waiting queue
    never collides with the right column. If either block is empty, the other is
    returned unchanged.
    """
    if not left:
        return list(right)
    if not right:
        return list(left)
    overlap = min(len(left), len(right))
    left_w = max(len(left[i]) for i in range(overlap))
    out = []
    for i in range(max(len(left), len(right))):
        l = left[i] if i < len(left) else ""
        r = right[i] if i < len(right) else ""
        out.append(l.ljust(left_w) + " " * gap + r if r else l)
    return out


def build_dashboard_lines(avail, squeue_text: str, queues, qos: str | None = None) -> list[str]:
    """Full line buffer for the TUI: raw `squeue --me` (full width) then Golden
    Tickets and Cluster-Wide side by side. Pure — reuses the `render_*`
    formatters and passes limit=None so the whole waiting queue is listed."""
    lines: list[str] = ["=== squeue --me ==="]
    lines.extend((squeue_text or "(no jobs)").splitlines())
    lines.append("")
    golden = render.render_golden_all(avail, qos_filter=qos, queues=queues, limit=None)
    cluster = render.render_cluster_wide(avail)
    lines.extend(_side_by_side(golden.splitlines(), cluster.splitlines()))
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
    squeue_text: str | None
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
            squeue_text = slurm_mcp.squeue_me()
            snap = _Snapshot(avail, squeue_text, queues, _now())
        except Exception as e:  # noqa: BLE001 — surface, don't crash the thread
            snap = _Snapshot(None, None, None, _now(), error=f"refresh failed: {e}")
        with state.lock:
            state.snapshot = snap
            state.version += 1
        state.stop.wait(interval)


# --------------------------------------------------------------------------- #
# curses loop
# --------------------------------------------------------------------------- #

def _addbar(stdscr, y: int, text: str, maxx: int, attr=None) -> None:
    """Draw a full-width status bar, safe against edge writes. `attr` defaults to
    reverse-video (used when the terminal has no color)."""
    if attr is None:
        attr = curses.A_REVERSE
    bar = text[: maxx - 1].ljust(maxx - 1)
    try:
        stdscr.addnstr(y, 0, bar, maxx - 1, attr)
    except curses.error:
        pass


def _loop(stdscr, state: _TuiState, interval: float, qos: str | None) -> None:
    for setup in (lambda: curses.curs_set(0),
                  curses.use_default_colors,
                  # Shrink the escape-disambiguation window from the ~1000ms
                  # default to 25ms. With keypad(True), a stray ESC / "\x1bO" /
                  # "\x1b[" prefix (e.g. a split arrow/function-key sequence)
                  # would otherwise glue a following 'q' into a key sequence for
                  # up to a second, so the keypress got swallowed instead of
                  # quitting. Mouse is intentionally NOT enabled: any mouse mask
                  # leaves ncurses parsing "\x1b[<…" mouse reports, and a partial
                  # one eats the next 'q'. Keyboard scroll covers navigation.
                  lambda: curses.set_escdelay(25)):
        try:
            setup()
        except (curses.error, AttributeError):
            pass
    stdscr.timeout(_INPUT_POLL_MS)

    # Cyan/teal color map (empty on a color-less terminal → plain fallback).
    theme = theme_mod.init_theme()

    lines = ["loading…"]
    roles: list = []
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
        if maxy < 4 or maxx < 10:
            stdscr.erase()
            try:
                stdscr.addnstr(0, 0, "terminal too small", maxx - 1)
            except curses.error:
                pass
            stdscr.refresh()
            if stdscr.getch() in (ord("q"), ord("Q")):
                break
            continue

        body_h = maxy - 3  # top bar + rule + bottom bar

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
                lines = build_dashboard_lines(snap.avail, snap.squeue_text, snap.queues, qos)
            dirty = True
        if (maxy, maxx) != last_size:
            last_size = (maxy, maxx)
            dirty = True

        if dirty:
            roles = theme_mod.classify_dashboard_lines(lines) if theme else []
            pad_w = max(maxx, max((len(s) for s in lines), default=0) + 1)
            pad = curses.newpad(max(len(lines) + 1, body_h), pad_w)
            pad.erase()
            for i, s in enumerate(lines):
                role = roles[i] if i < len(roles) else None
                attr = theme.get(role, 0)
                # Width-preserving status glyph: swap a card line's 2-space indent
                # for ●/○ (2 cells) so side-by-side columns stay aligned.
                if role in (theme_mod.Role.CARD_FREE, theme_mod.Role.CARD_FULL) and s.startswith("  "):
                    glyph = theme_mod.GLYPH_FULL if role == theme_mod.Role.CARD_FULL else theme_mod.GLYPH_FREE
                    s = glyph + s[2:]
                try:
                    pad.addstr(i, 0, s, attr)
                except curses.error:
                    pass

        row = clamp_scroll(row, len(lines), body_h)
        col = clamp_scroll(col, pad_w, maxx)

        first = row + 1
        last = min(row + body_h, len(lines))
        spin = _SPINNER[frame % len(_SPINNER)]
        qos_note = f" · qos={qos}" if qos else ""
        bar_attr = theme.get(theme_mod.Role.BAR, curses.A_REVERSE)
        rule_attr = theme.get(theme_mod.Role.RULE, 0)
        _addbar(stdscr, 0,
                f" slurmx status {spin} · data {stamp} · every {interval:g}s{qos_note}",
                maxx, bar_attr)
        try:
            stdscr.addnstr(1, 0, "─" * (maxx - 1), maxx - 1, rule_attr)
        except curses.error:
            pass
        _addbar(stdscr, maxy - 1,
                f" ↑↓/jk PgUp/Dn g/G scroll · ←→/hl pan · q quit"
                f"   {first}-{last}/{len(lines)} ",
                maxx, bar_attr)
        stdscr.noutrefresh()
        try:
            pad.refresh(row, col, 2, 0, maxy - 2, maxx - 1)
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
    except KeyboardInterrupt:
        pass  # Ctrl-C quits cleanly, same as 'q'
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
