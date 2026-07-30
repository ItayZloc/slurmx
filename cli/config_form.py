#!/usr/bin/env python3
"""Curses form for `slurmx config`.

Split the way cli/watch.py is: build_rows / move / dispatch are pure and unit
tested, the curses loop at the bottom is thin glue over them. All file access
goes through cli/config_model.ConfigDoc.
"""

from __future__ import annotations

import curses
from dataclasses import dataclass, field as dataclass_field

from cli import config_model as cm
from cli.theme import Role

NAME_W = 18
VALUE_W = 30
# name, display, vram, tickets, partition. Each is wider than its header word
# in cm.CARD_CELLS, or the header runs into the next column.
CARD_W = (16, 16, 6, 9, 16)
CARET = "▏"


@dataclass
class Row:
    kind: str                     # blank | field | group | thead | card | add
    spans: list = dataclass_field(default_factory=list)
    field: str | None = None
    qos: str | None = None
    index: int | None = None
    selectable: bool = False


@dataclass
class FormState:
    doc: cm.ConfigDoc
    path: str
    cursor: int = 0
    folds: set = dataclass_field(default_factory=set)
    cell: int = 0
    editing: str | None = None
    edit_pos: int = 0
    status: str = ""
    confirm: str | None = None    # "delete" | "quit" | None
    done: bool = False


def _tag(doc, name: str) -> str:
    slot = doc.slots[name]
    if name in doc.staged_names():
        return "edited"
    return {"derived": "derived", "absent": "default",
            "env-default": "env", "unsupported": "read-only"}.get(slot.provenance, "")


def _field_row(state, name: str, editing: bool) -> Row:
    doc = state.doc
    if editing:
        buf = state.editing or ""
        value = buf[:state.edit_pos] + CARET + buf[state.edit_pos:]
    else:
        value = doc.display_value(name)
    tag = _tag(doc, name)
    value_role = Role.CFG_EDITED if tag == "edited" or editing else Role.CFG_VALUE
    return Row(
        "field",
        spans=[("  " + name.ljust(NAME_W), Role.CFG_NAME),
               (value.ljust(VALUE_W), value_role),
               (tag, Role.CFG_TAG)],
        field=name,
        selectable=doc.is_editable(name),
    )


def _card_row(state, qos: str, index: int, card: tuple, selected: bool) -> Row:
    # A None golden partition renders blank, and editing it prefills blank, so
    # clearing the cell round-trips back to None.
    cells = ["" if c is None else str(c) for c in card]
    if selected and state.editing is not None:
        buf = state.editing
        cells[state.cell] = buf[:state.edit_pos] + CARET + buf[state.edit_pos:]
    spans = [("      ", Role.CFG_VALUE)]
    for i, (text, width) in enumerate(zip(cells, CARD_W)):
        role = Role.CFG_EDITED if selected and i == state.cell else Role.CFG_VALUE
        spans.append((text.ljust(width), role))
    return Row("card", spans=spans, qos=qos, index=index, selectable=True)


def build_rows(state: FormState) -> list[Row]:
    """The whole visible buffer as rows of (text, Role) spans."""
    doc = state.doc
    rows: list[Row] = [Row("blank", spans=[("", Role.PLAIN)])]
    for f in cm.FIELDS:
        if f.kind == "table":
            continue
        editing = state.editing is not None and state.cursor == len(rows)
        rows.append(_field_row(state, f.name, editing))
    rows.append(Row("blank", spans=[("", Role.PLAIN)]))

    groups = doc.groups()
    multi = len(groups) > 1
    for qos, cards in groups:
        folded = qos in state.folds
        label = f"GPU cards · {qos}" if multi else "GPU cards"
        glyph = "▸" if folded else "▾"
        rows.append(Row("group",
                        spans=[(f" {glyph} {label} ({len(cards)})", Role.CFG_THEAD)],
                        qos=qos, selectable=True))
        if folded:
            continue
        head = "      " + "".join(c.header.ljust(w)
                                  for c, w in zip(cm.CARD_CELLS, CARD_W))
        rows.append(Row("thead", spans=[(head, Role.CFG_TAG)], qos=qos))
        for i, card in enumerate(cards):
            rows.append(_card_row(state, qos, i, card, selected=state.cursor == len(rows)))
        rows.append(Row("add", spans=[("      + add card", Role.CFG_TAG)],
                        qos=qos, selectable=True))
        rows.append(Row("blank", spans=[("", Role.PLAIN)]))

    # Cluster facts, not config.py keys — shown so "where do CPU jobs go?" is
    # answerable here, unselectable so nobody mistakes them for settings.
    rows.append(Row("fixed_head",
                    spans=[(f" ─ fixed · {cm.FIXED_SOURCE} "
                            f"(not part of config.py)", Role.CFG_TAG)]))
    for name, value, _help in cm.FIXED_FACTS:
        rows.append(Row("fixed",
                        spans=[("  " + name.ljust(NAME_W), Role.CFG_TAG),
                               (str(value).ljust(VALUE_W), Role.CFG_TAG)],
                        field=name))
    return rows


# --------------------------------------------------------------------------- #
# Key handling (pure reducer)
# --------------------------------------------------------------------------- #

SAVE_KEYS = (ord("s"), ord("S"))
QUIT_KEYS = (ord("q"), ord("Q"))
_UP = (curses.KEY_UP, ord("k"))
_DOWN = (curses.KEY_DOWN, ord("j"))
_ENTER = (ord("\n"), ord("\r"), curses.KEY_ENTER)
_BACKSPACE = (curses.KEY_BACKSPACE, 127, 8)
_ESC = 27

SAVED_HINT = ("saved · config.py.bak written · a running Claude Code session "
              "keeps the old config until you restart it or reconnect with /mcp")


def _selectable(rows) -> list[int]:
    return [i for i, r in enumerate(rows) if r.selectable]


def move(state: FormState, key: int) -> FormState:
    """Cursor movement over selectable rows only."""
    rows = build_rows(state)
    idx = _selectable(rows)
    if not idx:
        return state
    here = min(range(len(idx)), key=lambda i: abs(idx[i] - state.cursor))
    if key in _DOWN:
        here = min(here + 1, len(idx) - 1)
    elif key in _UP:
        here = max(here - 1, 0)
    elif key == curses.KEY_NPAGE:
        here = min(here + 10, len(idx) - 1)
    elif key == curses.KEY_PPAGE:
        here = max(here - 10, 0)
    elif key == ord("G"):
        here = len(idx) - 1
    elif key == ord("g"):
        here = 0
    state.cursor = idx[here]
    state.cell = 0
    return state


def _begin_edit(state: FormState, rows) -> None:
    row = rows[state.cursor]
    if row.kind == "field":
        state.editing = state.doc.text_value(row.field)
    else:
        card = dict(state.doc.groups())[row.qos][row.index]
        cell = card[state.cell]
        state.editing = "" if cell is None else str(cell)
    state.edit_pos = len(state.editing)


def _commit_edit(state: FormState, rows) -> None:
    row = rows[state.cursor]
    buf = state.editing or ""
    if row.kind == "field":
        err = state.doc.set(row.field, buf)
    else:
        err = state.doc.set_card(row.qos, row.index, state.cell, buf)
    if err:
        state.status = err
        return
    state.editing = None
    state.status = ""


def _edit_key(state: FormState, key: int, rows) -> FormState:
    if key in _ENTER:
        _commit_edit(state, rows)
    elif key == _ESC:
        state.editing = None
        state.status = ""
    elif key in _BACKSPACE:
        if state.edit_pos:
            state.editing = (state.editing[:state.edit_pos - 1]
                             + state.editing[state.edit_pos:])
            state.edit_pos -= 1
    elif key == curses.KEY_DC:
        state.editing = (state.editing[:state.edit_pos]
                         + state.editing[state.edit_pos + 1:])
    elif key == curses.KEY_LEFT:
        state.edit_pos = max(0, state.edit_pos - 1)
    elif key == curses.KEY_RIGHT:
        state.edit_pos = min(len(state.editing), state.edit_pos + 1)
    elif key == curses.KEY_HOME:
        state.edit_pos = 0
    elif key == curses.KEY_END:
        state.edit_pos = len(state.editing)
    elif 32 <= key < 127:
        state.editing = (state.editing[:state.edit_pos] + chr(key)
                         + state.editing[state.edit_pos:])
        state.edit_pos += 1
    return state


def dispatch(state: FormState, key: int) -> FormState:
    """Apply one keypress. Mutates and returns `state`.

    Save is performed here rather than in the loop so the whole key contract is
    testable in one place; the loop only draws and reads keys.
    """
    rows = build_rows(state)
    if state.editing is not None:
        return _edit_key(state, key, rows)

    if key != ord("d") and state.confirm == "delete":
        state.confirm = None
        state.status = ""
    if key not in QUIT_KEYS and state.confirm == "quit":
        state.confirm = None
        state.status = ""

    row = rows[state.cursor] if state.cursor < len(rows) else None

    if key in QUIT_KEYS:
        if state.doc.dirty and state.confirm != "quit":
            state.confirm = "quit"
            state.status = "unsaved changes — q again to discard, s to save"
        else:
            state.done = True
    elif key in SAVE_KEYS:
        err = state.doc.save()
        state.status = f"cannot save: {err}" if err else SAVED_HINT
    elif key in _ENTER and row is not None:
        if row.kind == "group":
            state.folds.symmetric_difference_update({row.qos})
        elif row.kind == "add":
            state.doc.add_card(row.qos)
        elif row.kind in ("field", "card"):
            _begin_edit(state, rows)
    elif key == ord("a") and row is not None and row.qos:
        state.doc.add_card(row.qos)
    elif key == ord("d") and row is not None and row.kind == "card":
        if state.confirm == "delete":
            state.doc.delete_card(row.qos, row.index)
            state.confirm = None
            state.status = ""
        else:
            state.confirm = "delete"
            state.status = f"delete {row.qos} card {row.index + 1}? d again to confirm"
    elif key == ord("r") and row is not None and row.kind == "field":
        state.doc.revert(row.field)
        state.status = f"{row.field} reverted"
    elif key == curses.KEY_RIGHT and row is not None and row.kind == "card":
        state.cell = min(state.cell + 1, len(cm.CARD_CELLS) - 1)
    elif key == curses.KEY_LEFT and row is not None and row.kind == "card":
        state.cell = max(state.cell - 1, 0)
    else:
        move(state, key)
    return state


# --------------------------------------------------------------------------- #
# curses glue (no test — everything it needs is in build_rows/dispatch)
# --------------------------------------------------------------------------- #

def _addbar(stdscr, y: int, text: str, maxx: int, attr) -> None:
    try:
        stdscr.addnstr(y, 0, text[:maxx - 1].ljust(maxx - 1), maxx - 1, attr)
    except curses.error:
        pass


def _status_line(state: FormState) -> tuple[str, bool]:
    """(text, is_error) for the bottom bar."""
    if state.status:
        return state.status, state.status.startswith("cannot save")
    errs = state.doc.cross_field_errors()
    if errs:
        return errs[0], True
    warns = state.doc.warnings()
    if warns:
        return warns[0], False
    n = len(state.doc.staged_names())
    return (f"{n} unsaved change{'s' if n != 1 else ''}" if n else "no changes"), False


def _loop(stdscr, state: FormState) -> None:
    from cli import theme as theme_mod

    for setup in (lambda: curses.curs_set(0),
                  curses.use_default_colors,
                  lambda: curses.set_escdelay(25)):
        try:
            setup()
        except (curses.error, AttributeError):
            pass
    theme = theme_mod.init_theme()
    scroll = 0

    while not state.done:
        rows = build_rows(state)
        maxy, maxx = stdscr.getmaxyx()
        if maxy < 6 or maxx < 40:
            stdscr.erase()
            try:
                stdscr.addnstr(0, 0, "terminal too small", maxx - 1)
            except curses.error:
                pass
            stdscr.refresh()
            if stdscr.getch() in QUIT_KEYS:
                break
            continue

        body_h = maxy - 3
        if state.cursor < scroll:
            scroll = state.cursor
        elif state.cursor >= scroll + body_h:
            scroll = state.cursor - body_h + 1

        stdscr.erase()
        bar = theme.get(theme_mod.Role.BAR, curses.A_REVERSE)
        _addbar(stdscr, 0, f" slurmx config · {state.path}", maxx, bar)
        for y, row in enumerate(rows[scroll:scroll + body_h]):
            x = 0
            cursor_here = (scroll + y) == state.cursor
            if cursor_here:
                try:
                    stdscr.addnstr(y + 1, 0, "▸", 1,
                                   theme.get(theme_mod.Role.CFG_THEAD, 0))
                except curses.error:
                    pass
            for text, role in row.spans:
                attr = theme.get(role, 0)
                if cursor_here and not theme:
                    attr |= curses.A_BOLD
                try:
                    stdscr.addnstr(y + 1, x + 1, text, max(0, maxx - x - 2), attr)
                except curses.error:
                    pass
                x += len(text)
        status, is_err = _status_line(state)
        _addbar(stdscr, maxy - 2, " " + status, maxx,
                theme.get(theme_mod.Role.CFG_ERROR, 0) if is_err else 0)
        keys = ("↑↓ move · ⏎ edit · ←→ cell · a add · d delete · r revert · "
                "s save · q quit")
        _addbar(stdscr, maxy - 1, " " + keys, maxx, bar)
        try:
            curses.curs_set(1 if state.editing is not None else 0)
        except curses.error:
            pass
        stdscr.refresh()

        ch = stdscr.getch()
        if ch == curses.KEY_RESIZE:
            continue
        dispatch(state, ch)


def run_form(path: str, start_field: str | None = None) -> None:
    """Open the form on `path`. Blocks until the user quits.

    Raises curses.error when the terminal can't host curses (TERM=dumb, no tty);
    cli/config_cmd.run falls back to the text dump, same as slurmx status.
    """
    state = FormState(doc=cm.load(path), path=path)
    if start_field:
        for i, row in enumerate(build_rows(state)):
            if row.field == start_field and row.selectable:
                state.cursor = i
                break
    try:
        curses.wrapper(_loop, state)
    except KeyboardInterrupt:
        pass
