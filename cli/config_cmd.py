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
import shutil
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
        for name, disp, vram, tickets, part in cards:
            lines.append(f"    {name.ljust(16)}{str(vram).rjust(4)}GB "
                         f"tickets {str(tickets).rjust(3)}  {part}   {DIM}{disp}{NC}")
    lines.append("")
    lines.append(f"  {BOLD}fixed{NC}  {DIM}{cm.FIXED_SOURCE}, "
                 f"not part of config.py{NC}")
    for name, value, help_text in cm.FIXED_FACTS:
        lines.append(f"    {name.ljust(NAME_W - 2)}{value}   {DIM}{help_text}{NC}")
    errs = doc.cross_field_errors()
    warns = doc.warnings()
    if errs or warns:
        lines.append("")
        lines.extend(f"  error: {e}" for e in errs)
        lines.extend(f"  warn:  {w}" for w in warns)
    return "\n".join(lines)


def templates() -> list[tuple[str, str]]:
    """(label, path) for each config-examples/*.py, default first."""
    names = sorted(n for n in os.listdir(cm.TEMPLATE_DIR)
                   if n.endswith(".py") and not n.startswith("_"))
    names.sort(key=lambda n: (n != "default.py", n))
    return [(n[:-3], os.path.join(cm.TEMPLATE_DIR, n)) for n in names]


_TEMPLATE_BLURB = {
    "default": "blank template, fill in your own QoS and cards",
    "yisroel": "Yisroel's lab, pre-filled QoS and golden tickets",
}


def _bootstrap(path: str, choose=input) -> str | None:
    """Create `path` from a template chosen interactively. Error, or None."""
    opts = templates()
    print(f"{BOLD}{path} does not exist.{NC} Pick a starting template:\n")
    for i, (label, _) in enumerate(opts, 1):
        blurb = _TEMPLATE_BLURB.get(label, "")
        print(f"  {i}. {label:<10} {DIM}{blurb}{NC}")
    print()
    raw = (choose(f"Template [1-{len(opts)}, Enter to abort]: ") or "").strip()
    if not raw:
        return "aborted — nothing written."
    if not raw.isdigit() or not 1 <= int(raw) <= len(opts):
        return f"'{raw}' is not one of 1-{len(opts)}. Nothing written."
    label, src = opts[int(raw) - 1]
    shutil.copyfile(src, path)
    print(f"created {path} from config-examples/{label}.py")
    return None


def add_arguments(parser):
    parser.add_argument("--show", action="store_true",
                        help="Print resolved values as text instead of opening the form.")
    parser.add_argument("--path", default=cm.CONFIG_PATH,
                        help=argparse.SUPPRESS)   # tests point this at a tmp file


def run(args):
    path = getattr(args, "path", cm.CONFIG_PATH)
    fresh = not os.path.exists(path)
    if fresh:
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
        config_form.run_form(path, start_field="MAIL_USER" if fresh else None)
    except curses.error:
        # TERM unset/dumb or no usable terminal — same fallback as slurmx status.
        print(show_text(cm.load(path)))


def main():
    p = argparse.ArgumentParser(description="Edit slurmx's config.py.")
    add_arguments(p)
    run(p.parse_args())


if __name__ == "__main__":
    main()
