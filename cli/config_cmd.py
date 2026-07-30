#!/usr/bin/env python3
"""Backing module for `slurmx config` — edit config.py from the terminal.

    slurmx config           # curses form (interactive terminal)
    slurmx config --show    # resolved values as text, no form
    slurmx config | cat     # same as --show (non-TTY routes to text)

Imports nothing from slurm_mcp and nothing from `config`: this subcommand has
to work on a checkout where config.py does not exist yet.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cli import config_model as cm
from cli._style import BOLD, DIM, NC

NAME_W = 24


def _note(doc, name: str) -> str:
    slot = doc.slots[name]
    bits = [slot.provenance] if slot.provenance != "file" else []
    if slot.env_var:
        live = os.environ.get(slot.env_var)
        if live is not None:
            bits.append(f"{slot.env_var}={live} overrides")
    return ", ".join(bits)


def show_text(doc) -> str:
    """One line per field, then the GPU table. Plain, byte-stable."""
    lines = [f"{BOLD}config.py{NC}  {doc.path}", ""]
    for f in cm.FIELDS:
        if f.kind == "table":
            continue
        note = _note(doc, f.name)
        line = f"  {f.name.ljust(NAME_W)}{doc.display_value(f.name)}"
        lines.append(f"{line}   {DIM}{note}{NC}" if note else line)
    lines.append("")
    for qos, cards in doc.groups():
        lines.append(f"  {BOLD}GPU cards · {qos}{NC} ({len(cards)})")
        if not cards:
            lines.append(f"    {DIM}(none){NC}")
        for name, disp, vram, quota, part in cards:
            lines.append(f"    {name.ljust(16)}{str(vram).rjust(4)}GB "
                         f"quota {str(quota).rjust(3)}  {part}   {DIM}{disp}{NC}")
    errs = doc.cross_field_errors()
    warns = doc.warnings()
    if errs or warns:
        lines.append("")
        lines.extend(f"  error: {e}" for e in errs)
        lines.extend(f"  warn:  {w}" for w in warns)
    return "\n".join(lines)


def _bootstrap(path: str) -> str | None:
    """Create config.py from a template. Returns an error string, or None.

    Filled in by Task 9; until then a missing config.py is a plain error.
    """
    return f"{path} does not exist. Copy one of config-examples/ to config.py."


def add_arguments(parser):
    parser.add_argument("--show", action="store_true",
                        help="Print resolved values as text instead of opening the form.")
    parser.add_argument("--path", default=cm.CONFIG_PATH,
                        help=argparse.SUPPRESS)   # tests point this at a tmp file


def run(args):
    path = getattr(args, "path", cm.CONFIG_PATH)
    if not os.path.exists(path):
        err = _bootstrap(path)
        if err:
            print(err, file=sys.stderr)
            raise SystemExit(1)
    doc = cm.load(path)
    if args.show or not sys.stdout.isatty():
        print(show_text(doc))
        return
    from cli import config_form
    import curses
    try:
        config_form.run_form(path)
    except curses.error:
        # TERM unset/dumb or no usable terminal — same fallback as slurmx status.
        print(show_text(cm.load(path)))


def main():
    p = argparse.ArgumentParser(description="Edit slurmx's config.py.")
    add_arguments(p)
    run(p.parse_args())


if __name__ == "__main__":
    main()
