"""Shared ANSI style constants for the plain-text CLI subcommands.

Blanked when stdout is not a TTY, so piped/redirected output stays plain and
byte-stable. This is separate from cli/theme.py (curses attributes for the live
`status` TUI) — this module is for one-shot ANSI printing.
"""

import sys

_TTY = sys.stdout.isatty()


def _c(code: str) -> str:
    return code if _TTY else ""


BOLD = _c("\033[1m")
DIM = _c("\033[2m")
GREEN = _c("\033[0;32m")
RED = _c("\033[0;31m")
YELLOW = _c("\033[1;33m")
CYAN = _c("\033[0;36m")
NC = _c("\033[0m")


# State -> color, for job-status / wait / history output.
def state_color(state: str) -> str:
    s = (state or "").upper()
    if s in ("RUNNING", "COMPLETED"):
        return GREEN
    if s in ("PENDING", "CONFIGURING", "COMPLETING"):
        return YELLOW
    if s in ("FAILED", "TIMEOUT", "OUT_OF_MEMORY", "CANCELLED", "NODE_FAIL", "PREEMPTED"):
        return RED
    return ""
