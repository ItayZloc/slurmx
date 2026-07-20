"""Color layer for the live `slurmx status` TUI (cyan/teal accent).

Pure + curses helpers, kept out of cli/render.py so the one-shot text path and
the MCP `cluster_summary` tool stay byte-stable plain strings. `classify_dashboard_lines`
is a pure, unit-tested classifier over the already-built line list; `init_theme`
builds the curses attribute map (guarded so a color-less terminal degrades to plain).

Palette: cyan accent on headers + status bars, green = free/running,
yellow = pending, red = full, dim gray = secondary labels.
"""

from __future__ import annotations

import re
from enum import Enum, auto


class Role(Enum):
    PLAIN = auto()
    HEADER = auto()
    CARD_FREE = auto()      # "  name: X/Y free" with free > 0
    CARD_FULL = auto()      # "  name: 0/Y free"
    LABEL = auto()          # "    Running:" / "    Pending ..."
    ROW_RUNNING = auto()    # "      user: N GPU(s)" under Running
    ROW_PENDING = auto()    # "      user: N GPU(s)" under Pending
    SQUEUE_RUNNING = auto()
    SQUEUE_PENDING = auto()
    BAR = auto()            # top/bottom status bars (not emitted by classify)
    RULE = auto()           # thin rule under the top bar (not emitted by classify)


_CARD_RE = re.compile(r"  \S+:\s+(\d+)/\d+\s+free")
_GPUROW_RE = re.compile(r"      \S+:\s+\d+\s+GPU\(s\)")


def classify_dashboard_lines(lines: list[str]) -> list[Role]:
    """Map each dashboard line to a Role via a stateful prefix walk.

    Tracks whether we're inside the raw `squeue --me` block (state coloring by the
    ST column) vs the golden/cluster block (card free/full + Running/Pending
    section). Side-by-side rows that merge a golden-left and cluster-right segment
    are colored by the left/dominant role — good enough; the pure cluster rows sit
    below the golden overlap and classify on their own.
    """
    roles: list[Role] = []
    in_squeue = False
    section: str | None = None  # "running" | "pending" | None

    for line in lines:
        if "=== squeue" in line:
            in_squeue, section = True, None
            roles.append(Role.HEADER)
            continue
        if "===" in line:
            in_squeue, section = False, None
            roles.append(Role.HEADER)
            continue
        if in_squeue:
            toks = line.split()
            if "R" in toks:
                roles.append(Role.SQUEUE_RUNNING)
            elif "PD" in toks:
                roles.append(Role.SQUEUE_PENDING)
            else:
                roles.append(Role.PLAIN)
            continue

        m = _CARD_RE.match(line)
        if m:
            roles.append(Role.CARD_FULL if int(m.group(1)) == 0 else Role.CARD_FREE)
            continue
        if line.startswith("    Running"):
            section = "running"
            roles.append(Role.LABEL)
            continue
        if line.startswith("    Pending"):
            section = "pending"
            roles.append(Role.LABEL)
            continue
        if _GPUROW_RE.match(line):
            roles.append(Role.ROW_PENDING if section == "pending" else Role.ROW_RUNNING)
            continue
        roles.append(Role.PLAIN)

    return roles


# Width-preserving status glyphs: swapped in for a card line's 2-space indent at
# draw time (TUI only), so `_side_by_side` column widths stay aligned.
GLYPH_FREE = "○ "
GLYPH_FULL = "● "


def init_theme() -> dict:
    """Build the {Role: curses attr} map. Returns {} if the terminal has no color
    (drawing then falls back to plain / A_REVERSE bars). Import curses lazily so
    this module stays importable for the pure classifier tests off a TTY."""
    import curses

    try:
        if not curses.has_colors():
            return {}
    except curses.error:
        return {}

    hi = getattr(curses, "COLORS", 8) >= 256
    cyan = 44 if hi else curses.COLOR_CYAN
    green = 42 if hi else curses.COLOR_GREEN
    yellow = 220 if hi else curses.COLOR_YELLOW
    red = 203 if hi else curses.COLOR_RED
    gray = 244 if hi else curses.COLOR_WHITE

    for idx, fg in ((1, cyan), (2, green), (3, yellow), (4, red), (5, gray)):
        try:
            curses.init_pair(idx, fg, -1)
        except curses.error:
            return {}

    cp = curses.color_pair
    label_attr = cp(5) if hi else (cp(5) | curses.A_DIM)
    return {
        Role.HEADER: cp(1) | curses.A_BOLD,
        Role.CARD_FREE: cp(2),
        Role.CARD_FULL: cp(4) | curses.A_BOLD,
        Role.LABEL: label_attr,
        Role.ROW_RUNNING: 0,
        Role.ROW_PENDING: cp(3),
        Role.SQUEUE_RUNNING: cp(2),
        Role.SQUEUE_PENDING: cp(3),
        Role.PLAIN: 0,
        Role.BAR: cp(1) | curses.A_REVERSE | curses.A_BOLD,
        Role.RULE: cp(1),
    }
