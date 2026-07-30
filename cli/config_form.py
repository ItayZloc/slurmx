#!/usr/bin/env python3
"""Curses form for `slurmx config`.

Split the way cli/watch.py is: build_rows / move / dispatch are pure and unit
tested, the curses loop at the bottom is thin glue over them. All file access
goes through cli/config_model.ConfigDoc.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field

from cli import config_model as cm
from cli.theme import Role

NAME_W = 18
VALUE_W = 30
CARD_W = (16, 16, 6, 7, 16)      # name, display, vram, quota, partition
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
    slot = doc.slots[name]
    if editing:
        buf = state.editing or ""
        value = buf[:state.edit_pos] + CARET + buf[state.edit_pos:]
    else:
        value = doc.text_value(name)
        if slot.field.kind == "list" and not value:
            value = "(none)"
        if not value:
            value = "(unset)"
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
    cells = [str(c) for c in card]
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
        head = "      " + "".join(n.ljust(w) for (n, _), w in zip(cm.CARD_CELLS, CARD_W))
        rows.append(Row("thead", spans=[(head, Role.CFG_TAG)], qos=qos))
        for i, card in enumerate(cards):
            rows.append(_card_row(state, qos, i, card, selected=state.cursor == len(rows)))
        rows.append(Row("add", spans=[("      + add card", Role.CFG_TAG)],
                        qos=qos, selectable=True))
        rows.append(Row("blank", spans=[("", Role.PLAIN)]))
    return rows
