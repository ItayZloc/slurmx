"""Color layer for the live `slurmx status` TUI (muted cyan/teal accent).

Pure classifiers + curses helpers, kept out of cli/render.py so the one-shot text
path and the MCP `cluster_summary` tool stay byte-stable plain strings.

Classification is per single-column block (`classify_block` for a golden/cluster
column, `squeue_role` per squeue line) so a side-by-side row's golden-left and
cluster-right segments are colored independently — the roles are carried through
the merge as spans (see cli/watch.py `dashboard_row_spans`), not applied to the
whole flat line.

Palette: soft cyan accent on headers + status bars, muted green = free/running,
muted gold = pending, muted red = full, dim gray = secondary labels. Kept
low-saturation on purpose so it reads as a calm dashboard, not a warning panel.
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


def squeue_role(line: str) -> Role:
    """Role for one raw `squeue --me` line, by its ST column token."""
    toks = line.split()
    if "R" in toks:
        return Role.SQUEUE_RUNNING
    if "PD" in toks:
        return Role.SQUEUE_PENDING
    return Role.PLAIN


def classify_block(lines: list[str]) -> list[Role]:
    """Classify one single-column golden/cluster block (no side-by-side merge).

    Headers (`=== … ===`), card summaries (free/full from `X/Y free`),
    Running/Pending sub-labels, and per-user GPU rows (Running vs Pending by the
    current section). Each column is homogeneous, so there is no left/right
    ambiguity — the merge happens later, per segment.
    """
    roles: list[Role] = []
    section: str | None = None  # "running" | "pending" | None
    for line in lines:
        if "===" in line:
            section = None
            roles.append(Role.HEADER)
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


# Width-preserving status glyphs: swapped in for a card segment's 2-space indent
# at draw time (TUI only), so column widths stay aligned.
GLYPH_FREE = "○ "
GLYPH_FULL = "● "


def init_theme() -> dict:
    """Build the {Role: curses attr} map. Returns {} if the terminal has no color
    (drawing then falls back to plain / A_REVERSE bars). Import curses lazily so
    this module stays importable for the pure classifier tests off a TTY.

    256-color terminals get a muted, low-saturation palette; 8-color terminals
    fall back to the basic colors softened with A_DIM where it helps.
    """
    import curses

    try:
        if not curses.has_colors():
            return {}
    except curses.error:
        return {}

    hi = getattr(curses, "COLORS", 8) >= 256
    if hi:
        # Muted xterm-256 shades: soft teal / sage / tan / brick / gray.
        cyan, green, yellow, red, gray = 73, 108, 179, 167, 245
    else:
        cyan, green, yellow, red, gray = (
            curses.COLOR_CYAN, curses.COLOR_GREEN, curses.COLOR_YELLOW,
            curses.COLOR_RED, curses.COLOR_WHITE,
        )

    for idx, fg in ((1, cyan), (2, green), (3, yellow), (4, red), (5, gray)):
        try:
            curses.init_pair(idx, fg, -1)
        except curses.error:
            return {}

    cp = curses.color_pair
    soften = 0 if hi else curses.A_DIM  # 8-color needs A_DIM to calm down
    return {
        Role.HEADER: cp(1) | soften,
        Role.CARD_FREE: cp(2) | soften,
        Role.CARD_FULL: cp(4) | soften,
        Role.LABEL: cp(5) | curses.A_DIM,
        Role.ROW_RUNNING: 0,
        Role.ROW_PENDING: cp(3) | soften,
        Role.SQUEUE_RUNNING: cp(2) | soften,
        Role.SQUEUE_PENDING: cp(3) | soften,
        Role.PLAIN: 0,
        Role.BAR: cp(1) | curses.A_REVERSE,
        Role.RULE: cp(5),
    }
