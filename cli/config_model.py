"""Read, edit, and write config.py without destroying it.

config.py is Python source, not data: it carries comments, os.environ.get
fallbacks, and one derived line. So an edit here replaces the exact source span
of the literal being changed and leaves every other byte alone. With nothing
staged, render() returns the input verbatim.

Imports nothing from slurm_mcp and nothing from `config` itself — `slurmx
config` has to work on a checkout where config.py does not exist yet.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import NamedTuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(REPO, "config.py")
TEMPLATE_DIR = os.path.join(REPO, "config-examples")

if REPO not in sys.path:
    sys.path.insert(0, REPO)

from maintenance import _parse_slurm_time  # stdlib-only module; safe to import
# config_defaults tolerates a missing config.py, so this stays importable on a
# fresh clone.
from config_defaults import (
    CPU_PARTITION, CPU_QOS, GOLDEN_POLICIES, GOLDEN_POLICY_DEFAULT,
    MAIL_TYPE_DEFAULT, MAIN_PARTITION,
)

# What a key that is absent from config.py resolves to at submit time. Shown
# instead of a blank, so a config.py predating the key doesn't read as "off".
ABSENT_DEFAULTS = {
    "MAIL_TYPE": list(MAIL_TYPE_DEFAULT),
    "GOLDEN_POLICY": GOLDEN_POLICY_DEFAULT,
}

DEFAULT_MAIL_DOMAIN = "post.bgu.ac.il"

# Cluster facts that used to be config.py keys and moved to config_defaults.py
# on 2026-07-30. They are not part of the schema any more, so they are neither
# shown nor editable. This map exists for one reason: a config.py written
# before the move may still assign one, and if the value it assigns differs
# from the fixed one, that user's behaviour changes silently on upgrade. So we
# read them back and warn.
RETIRED_KEYS = {
    "CPU_PARTITION": CPU_PARTITION,
    "CPU_QOS": CPU_QOS,
    "MAIN_PARTITION": MAIN_PARTITION,
}
RETIRED_SOURCE = "config_defaults.py"

# How a slot's provenance reads in the UI. "file" is the ordinary case and
# shows nothing at all.
PROVENANCE_LABEL = {
    "derived": "derived",
    "absent": "default",
    "env-default": "env",
    "unsupported": "read-only",
}


def default_mail_user() -> str:
    """Prefill for an empty or absent MAIL_USER."""
    return f"{os.environ.get('USER', '')}@{DEFAULT_MAIL_DOMAIN}"


# --------------------------------------------------------------------------- #
# Validators — each takes the raw typed string, returns an error or None
# --------------------------------------------------------------------------- #

def _v_email(raw: str) -> str | None:
    raw = raw.strip()
    if not raw:
        return "must not be empty"
    if "@" not in raw:
        return "must look like an email (user@host)"
    return None


def _v_word(raw: str) -> str | None:
    raw = raw.strip()
    if not raw:
        return "must not be empty"
    if re.search(r"\s", raw):
        return "must not contain whitespace"
    return None


def _v_text(raw: str) -> str | None:
    return None if raw.strip() else "must not be empty"


def _words(raw: str) -> list[str]:
    return [t.strip() for t in raw.split(",") if t.strip()]


def _v_word_list(raw: str) -> str | None:
    toks = _words(raw)
    if not toks:
        return "needs at least one entry"
    for t in toks:
        if re.search(r"\s", t):
            return f"'{t}' must not contain whitespace"
    return None


def _v_word_list_or_empty(raw: str) -> str | None:
    for t in _words(raw):
        if re.search(r"\s", t):
            return f"'{t}' must not contain whitespace"
    return None


def _v_int(raw: str, *, minimum: int) -> str | None:
    try:
        n = int(raw.strip())
    except ValueError:
        return "must be an integer"
    return None if n >= minimum else f"must be >= {minimum}"


def _v_posint(raw: str) -> str | None:
    return _v_int(raw, minimum=1)


def _v_nonneg(raw: str) -> str | None:
    return _v_int(raw, minimum=0)


def _v_mem(raw: str) -> str | None:
    if re.fullmatch(r"\d+[KMGT]?", raw.strip()):
        return None
    return "must be a SLURM size like 16G or 4096"


def _v_time(raw: str) -> str | None:
    try:
        _parse_slurm_time(raw.strip())
    except (ValueError, IndexError):
        return "must be D-HH:MM:SS, e.g. 7-0:00:00"
    return None


# `sbatch --mail-type` vocabulary, in the order the form lists them. Empty or
# NONE means no mail at all.
MAIL_EVENTS = (
    "NONE", "BEGIN", "END", "FAIL", "REQUEUE", "ALL", "INVALID_DEPEND",
    "STAGE_OUT", "TIME_LIMIT", "TIME_LIMIT_90", "TIME_LIMIT_80",
    "TIME_LIMIT_50", "ARRAY_TASKS",
)
MAIL_EVENT_HELP = {
    "NONE": "no mail at all",
    "BEGIN": "the job starts",
    "END": "the job finishes",
    "FAIL": "the job fails or is killed",
    "REQUEUE": "the job is requeued (preemption)",
    "ALL": "BEGIN, END, FAIL, REQUEUE and the rest",
    "INVALID_DEPEND": "a dependency can never be satisfied",
    "STAGE_OUT": "burst-buffer stage-out finished",
    "TIME_LIMIT": "the job hit its wall clock",
    "TIME_LIMIT_90": "reached 90% of the wall clock",
    "TIME_LIMIT_80": "reached 80% of the wall clock",
    "TIME_LIMIT_50": "reached 50% of the wall clock",
    "ARRAY_TASKS": "mail per array task, not per job",
}
# Checking one of these clears everything else, and checking anything else
# clears them: "no mail" and "every event" don't combine with a specific event.
MAIL_EXCLUSIVE = ("NONE", "ALL")

# What each golden policy does, in the words the form shows next to the radio.
GOLDEN_POLICY_HELP = {
    "golden_only": "always preemption-immune; queue rather than downgrade",
    "allow_main": "golden first, then the preemptible main pool",
    "ask": "no default — Claude has to ask you, the CLI prompts",
}

# Per-field option help, so the form doesn't special-case field names.
OPTION_HELP = {"MAIL_TYPE": MAIL_EVENT_HELP, "GOLDEN_POLICY": GOLDEN_POLICY_HELP}


def _v_word_or_empty(raw: str) -> str | None:
    """A single token, or blank. Blank stores None — the templates use None for
    a card with no golden partition."""
    return None if not raw.strip() else _v_word(raw)


def _v_mail_types(raw: str) -> str | None:
    for t in _words(raw):
        if t.upper() not in MAIL_EVENTS:
            return f"'{t}' is not a SLURM mail event ({', '.join(MAIL_EVENTS[:6])}, ...)"
    return None


def _v_policy(raw: str) -> str | None:
    if raw.strip() in GOLDEN_POLICIES:
        return None
    return f"must be one of {', '.join(GOLDEN_POLICIES)}"


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Field:
    name: str
    kind: str        # "str" | "int" | "list" | "choice" | "table" | "derived"
    help: str
    validator: object = None            # Callable[[str], str | None]
    options: tuple[str, ...] = ()       # closed vocabulary -> fold-out picker


FIELDS: tuple[Field, ...] = (
    Field("USERNAME", "derived", "auto-detected from $USER"),
    Field("MAIL_USER", "str", "address for SLURM mail notifications", _v_email),
    Field("MAIL_TYPE", "list", "SLURM mail events; empty or NONE = no mail",
          _v_mail_types, options=MAIL_EVENTS),
    Field("GOLDEN_QOS", "list", "golden QoS list; the first is primary", _v_word_list),
    Field("GOLDEN_POLICY", "choice", "what an unspecified golden_only becomes",
          _v_policy, options=GOLDEN_POLICIES),
    Field("EXCLUDE_NODES", "list", "nodes to keep jobs off", _v_word_list_or_empty),
    Field("MAX_MEM_GB", "int", "memory ceiling for a GPU job", _v_posint),
    Field("CPU_CPUS", "int", "cores for a CPU-only job", _v_posint),
    Field("CPU_MEM", "str", "memory for a CPU-only job, e.g. 16G", _v_mem),
    Field("TIME_LIMIT", "str", "default wall clock, D-HH:MM:SS", _v_time),
    Field("START_TIMEOUT", "int", "seconds submit_job waits for RUNNING", _v_posint),
    Field("GPU_DEFINITIONS_BY_QOS", "table", "GPU cards per QoS"),
    Field("GPU_DEFINITIONS", "derived", "the primary QoS's cards"),
)
FIELD_BY_NAME = {f.name: f for f in FIELDS}


class Cell(NamedTuple):
    header: str      # column header in the form, kept short
    label: str       # what an error message calls it
    validator: object


CARD_CELLS = (
    Cell("name", "name", _v_word),
    Cell("display", "display name", _v_text),
    Cell("vram", "VRAM", _v_posint),
    Cell("tickets", "golden tickets", _v_nonneg),
    Cell("partition", "golden partition", _v_word_or_empty),
)
NEW_CARD = ("new_card", "New card", 24, 0, "new_partition")
_INT_CELLS = (2, 3)
_PARTITION_CELL = 4


def _py_literal(v) -> str:
    """One card field as Python source. NOT json.dumps: the default template
    ships None for a card with no golden partition, and json renders that as
    `null`, which is a NameError when config.py is imported."""
    if v is None:
        return "None"
    if isinstance(v, bool) or isinstance(v, int):
        return str(v)
    return json.dumps(v)


def _table_literal(table: dict[str, list[list]]) -> str:
    """Re-render the whole GPU_DEFINITIONS_BY_QOS dict, columns aligned.

    Only called when a card edit is staged; an untouched table keeps its
    original bytes like every other field.
    """
    out = ["{"]
    for qos, cards in table.items():
        out.append(f"    {json.dumps(qos)}: [")
        widths = [max((len(_py_literal(c[i])) for c in cards), default=0)
                  for i in range(4)]
        for c in cards:
            name = (_py_literal(c[0]) + ",").ljust(widths[0] + 2)
            disp = (_py_literal(c[1]) + ",").ljust(widths[1] + 2)
            vram = (_py_literal(c[2]) + ",").rjust(widths[2] + 1)
            tickets = (_py_literal(c[3]) + ",").rjust(widths[3] + 1)
            out.append(f"        ({name}{disp}{vram} {tickets} {_py_literal(c[4])}),")
        out.append("    ],")
    out.append("}")
    return "\n".join(out)


def _parse_raw(f: Field, raw: str):
    if f.kind == "int":
        return int(raw.strip())
    if f.kind == "list":
        toks = _words(raw)
        # sbatch only accepts the events upper-case, so store them that way
        # rather than making the reader remember.
        return [t.upper() for t in toks] if f.name == "MAIL_TYPE" else toks
    return raw.strip()


def _literal(f: Field, value, *, as_csv: bool) -> str:
    if f.kind == "int":
        return str(value)
    if f.kind == "list":
        if as_csv:
            return json.dumps(",".join(value))
        return "[" + ", ".join(json.dumps(v) for v in value) + "]"
    return json.dumps(value)


@dataclass
class Slot:
    field: Field
    span: tuple[int, int] | None
    value: object
    provenance: str               # file | env-default | absent | derived | unsupported
    env_var: str | None = None
    as_csv: bool = False          # list stored as one comma-separated string


# --------------------------------------------------------------------------- #
# Source offsets
# --------------------------------------------------------------------------- #

def _line_starts(text: str) -> list[int]:
    starts, pos = [0], 0
    for line in text.splitlines(keepends=True):
        pos += len(line)
        starts.append(pos)
    return starts


def _offset(text: str, starts: list[int], lineno: int, col: int) -> int:
    """Absolute character offset for an ast (lineno, col_offset).

    ast reports col_offset as a UTF-8 *byte* offset within its line, so slice
    the encoded line and measure the decoded prefix.
    """
    line_start = starts[lineno - 1]
    line = text[line_start:starts[lineno]]
    return line_start + len(line.encode()[:col].decode("utf-8", "replace"))


def _env_get_call(node: ast.AST):
    """('SLURM_X', <default node>) when node is os.environ.get("SLURM_X", ...)."""
    if not isinstance(node, ast.Call) or len(node.args) != 2:
        return None
    fn = node.func
    if not (isinstance(fn, ast.Attribute) and fn.attr == "get"):
        return None
    if not (isinstance(fn.value, ast.Attribute) and fn.value.attr == "environ"):
        return None
    if not isinstance(node.args[0], ast.Constant):
        return None
    return node.args[0].value, node.args[1]


_UNREADABLE = object()


def _literal_eval(node: ast.AST):
    """The node's value, or _UNREADABLE when it isn't a literal."""
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        return _UNREADABLE


def _find_env_get(node: ast.AST):
    """The first os.environ.get(...) anywhere inside an expression.

    Not just at the top: the templates wrap GOLDEN_QOS and EXCLUDE_NODES in a
    comprehension that splits the env default on commas, so the editable literal
    sits several levels down.
    """
    for sub in ast.walk(node):
        found = _env_get_call(sub)
        if found is not None:
            return found
    return None


# --------------------------------------------------------------------------- #
# Document
# --------------------------------------------------------------------------- #

class ConfigDoc:
    """A parsed config.py plus any staged edits."""

    def __init__(self, path: str, text: str) -> None:
        self.path = path
        self.text = text
        self._starts = _line_starts(text)
        assigns: dict[str, ast.Assign] = {}
        for node in ast.parse(text).body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)):
                assigns[node.targets[0].id] = node
        self.slots = {f.name: self._slot(f, assigns) for f in FIELDS}
        self._retired = {}
        for name in RETIRED_KEYS:
            node = assigns.get(name)
            if node is None:
                continue
            value = _literal_eval(node.value)
            if value is _UNREADABLE:
                env = _find_env_get(node.value)
                value = _literal_eval(env[1]) if env else _UNREADABLE
            if value is not _UNREADABLE:
                self._retired[name] = value
        self._staged: dict[str, str] = {}      # field -> replacement literal source
        self._values: dict[str, object] = {}   # field -> staged parsed value
        self._appends: list[str] = []          # new assignments for absent keys
        self._table: dict[str, list[list]] | None = None

    # -- parsing ---------------------------------------------------------- #

    def _span(self, node: ast.AST) -> tuple[int, int]:
        return (_offset(self.text, self._starts, node.lineno, node.col_offset),
                _offset(self.text, self._starts, node.end_lineno, node.end_col_offset))

    def _slot(self, f: Field, assigns: dict[str, ast.Assign]) -> Slot:
        if f.kind == "derived":
            return Slot(f, None, None, "derived")

        node = assigns.get(f.name)
        if node is None:
            return Slot(f, None, None, "absent")

        # A plain literal is the easy case: the span is the literal itself.
        value = _literal_eval(node.value)
        if value is not _UNREADABLE:
            return Slot(f, self._span(node.value), value, "file")

        # Otherwise the value is an expression. If an os.environ.get(NAME,
        # DEFAULT) appears anywhere in it, DEFAULT is the editable part and the
        # env var keeps working. Covers the plain wrapper (MAIN_PARTITION), an
        # f-string default (MAIL_USER), the comprehension over an inline default
        # (GOLDEN_QOS), and the comprehension over a named one (EXCLUDE_NODES).
        env = _find_env_get(node.value)
        if env is None:
            return Slot(f, None, None, "unsupported")
        env_var, default_node = env
        if isinstance(default_node, ast.Name):
            helper = assigns.get(default_node.id)
            if helper is None:
                return Slot(f, None, None, "unsupported")
            default_node = helper.value

        raw = _literal_eval(default_node)
        # A list field whose env default is a string is a comma-separated one,
        # because the comprehension around it splits on commas.
        as_csv = f.kind == "list" and not isinstance(default_node, (ast.List, ast.Tuple))
        if as_csv:
            value = [t.strip() for t in (raw or "").split(",") if t.strip()]
        else:
            # None when it can't be read statically (an f-string). text_value
            # then falls back to the same default the f-string would compute.
            value = None if raw is _UNREADABLE else raw
        return Slot(f, self._span(default_node), value, "env-default",
                    env_var=env_var, as_csv=as_csv)

    # -- rendering -------------------------------------------------------- #

    def render(self) -> str:
        """The file with staged spans swapped in. Byte-identical when clean."""
        text = self.text
        edits = [(self.slots[n].span, lit) for n, lit in self._staged.items()]
        for (start, end), literal in sorted(edits, key=lambda e: -e[0][0]):
            text = text[:start] + literal + text[end:]
        if self._appends:
            if not text.endswith("\n"):
                text += "\n"
            text += "\n# --- Added by `slurmx config` ---\n" + "".join(self._appends)
        return text

    # -- staged edits ----------------------------------------------------- #

    @property
    def dirty(self) -> bool:
        return bool(self._staged or self._appends)

    def staged_names(self) -> set[str]:
        appended = {a.split(" =", 1)[0] for a in self._appends}
        return set(self._staged) | appended

    def is_editable(self, name: str) -> bool:
        slot = self.slots[name]
        return slot.field.kind not in ("derived", "table") \
            and slot.provenance != "unsupported"

    def value(self, name: str):
        if name in self._values:
            return self._values[name]
        slot = self.slots[name]
        if slot.provenance == "absent" and name in ABSENT_DEFAULTS:
            return ABSENT_DEFAULTS[name]
        return slot.value

    def text_value(self, name: str) -> str:
        v = self.value(name)
        if name == "MAIL_USER" and not v:
            return default_mail_user()
        if v is None:
            return ""
        if self.slots[name].field.kind == "list":
            return ", ".join(v)
        return str(v)

    def display_value(self, name: str) -> str:
        """What the form and `--show` print for one field.

        The two derived fields have no literal to read, so they are resolved the
        way config.py resolves them: USERNAME from $USER, GPU_DEFINITIONS from
        the primary QoS's card list.
        """
        if name == "USERNAME":
            return os.environ.get("USER", "")
        if name == "GPU_DEFINITIONS":
            primary = (self.value("GOLDEN_QOS") or [""])[0]
            n = len(dict(self.groups()).get(primary, []))
            return f"{n} cards ({primary})" if primary else "(no QoS)"
        text = self.text_value(name)
        if text:
            return text
        return "(none)" if self.slots[name].field.kind == "list" else "(unset)"

    def set(self, name: str, raw: str) -> str | None:
        """Validate and stage one edit. Error message, or None on success."""
        slot = self.slots[name]
        if not self.is_editable(name):
            return f"{name} is not editable here ({slot.provenance})"
        err = slot.field.validator(raw)
        if err:
            return f"{name} {err}"
        value = _parse_raw(slot.field, raw)
        literal = _literal(slot.field, value, as_csv=slot.as_csv)
        self._values[name] = value
        if slot.span is None:
            self._appends = [a for a in self._appends
                             if not a.startswith(f"{name} =")]
            self._appends.append(f"{name} = {literal}\n")
        else:
            self._staged[name] = literal
        return None

    def selected_options(self, name: str) -> list[str]:
        """Which of a field's options are currently picked.

        A list field can have several, a choice exactly one — the form draws
        checkboxes or radios off that difference rather than off the field name.
        """
        if self.slots[name].field.kind == "choice":
            v = self.value(name)
            return [v] if v else []
        return self.mail_events()

    def toggle_option(self, name: str, option: str) -> str | None:
        """Pick or unpick one option. Error message, or None on success."""
        if self.slots[name].field.kind == "choice":
            return self.set(name, option)
        return self.toggle_mail_event(option)

    def mail_events(self) -> list[str]:
        """The currently checked events, upper-cased."""
        return [str(e).upper() for e in (self.value("MAIL_TYPE") or [])]

    def toggle_mail_event(self, event: str) -> str | None:
        """Check or uncheck one event. Error message, or None on success."""
        event = event.upper()
        checked = set(self.mail_events())
        if event in checked:
            checked.discard(event)
        elif event in MAIL_EXCLUSIVE:
            checked = {event}
        else:
            checked -= set(MAIL_EXCLUSIVE)
            checked.add(event)
        # Re-emit in MAIL_EVENTS order so the file doesn't churn on click order.
        return self.set("MAIL_TYPE", ", ".join(e for e in MAIL_EVENTS if e in checked))

    def revert(self, name: str) -> None:
        self._staged.pop(name, None)
        self._values.pop(name, None)
        self._appends = [a for a in self._appends
                         if not a.startswith(f"{name} =")]

    # -- GPU card table --------------------------------------------------- #

    def groups(self) -> list[tuple[str, list[tuple]]]:
        """(qos, cards) in GOLDEN_QOS order, then any extra keys in the table.

        A QoS named in GOLDEN_QOS with no entry in the table yields an empty
        list, so the form can show it with an `+ add card` row.
        """
        table = self._table if self._table is not None else \
            (self.slots["GPU_DEFINITIONS_BY_QOS"].value or {})
        order: list[str] = []
        for q in (self.value("GOLDEN_QOS") or []):
            if q not in order:
                order.append(q)
        for q in table:
            if q not in order:
                order.append(q)
        return [(q, [tuple(c) for c in table.get(q, [])]) for q in order]

    def _mutable_table(self) -> dict[str, list[list]]:
        if self._table is None:
            base = self.slots["GPU_DEFINITIONS_BY_QOS"].value or {}
            self._table = {q: [list(c) for c in cards] for q, cards in base.items()}
        return self._table

    def _stage_table(self) -> None:
        literal = _table_literal(self._table)
        slot = self.slots["GPU_DEFINITIONS_BY_QOS"]
        if slot.span is None:
            self._appends = [a for a in self._appends
                             if not a.startswith("GPU_DEFINITIONS_BY_QOS =")]
            self._appends.append(f"GPU_DEFINITIONS_BY_QOS = {literal}\n")
        else:
            self._staged["GPU_DEFINITIONS_BY_QOS"] = literal

    def set_card(self, qos: str, index: int, cell: int, raw: str) -> str | None:
        spec = CARD_CELLS[cell]
        err = spec.validator(raw)
        if err:
            return f"{spec.label} {err}"
        if cell in _INT_CELLS:
            value = int(raw.strip())
        elif cell == _PARTITION_CELL and not raw.strip():
            value = None
        else:
            value = raw.strip()
        table = self._mutable_table()
        table.setdefault(qos, [])
        table[qos][index][cell] = value
        self._stage_table()
        return None

    def add_card(self, qos: str) -> None:
        self._mutable_table().setdefault(qos, []).append(list(NEW_CARD))
        self._stage_table()

    def delete_card(self, qos: str, index: int) -> None:
        del self._mutable_table()[qos][index]
        self._stage_table()


    # -- gate + write ----------------------------------------------------- #

    def cross_field_errors(self) -> list[str]:
        """Conditions that must be fixed before a save is allowed."""
        errs: list[str] = []
        qos = self.value("GOLDEN_QOS") or []
        groups = dict(self.groups())
        if not qos:
            errs.append("GOLDEN_QOS needs at least one QoS")
        else:
            primary = qos[0]
            if not groups.get(primary):
                errs.append(
                    f"GOLDEN_QOS[0] '{primary}' has no GPU cards — "
                    "every slurmx command would fail"
                )
        for q, cards in groups.items():
            names = [c[0] for c in cards]
            dupes = sorted({n for n in names if names.count(n) > 1})
            if dupes:
                errs.append(f"{q}: duplicate card name(s) {', '.join(dupes)}")
        return errs

    def warnings(self) -> list[str]:
        """Non-blocking: the file still imports."""
        groups = dict(self.groups())
        qos = self.value("GOLDEN_QOS") or []
        warns = [f"QoS '{q}' has no GPU cards" for q in qos[1:] if not groups.get(q)]
        # sbatch keeps a --mail-type list as long as ONE token is recognised and
        # drops the others without a word, so a typo costs you that mail and
        # says nothing. Only the form validates; a hand-edited file wouldn't.
        bogus = [e for e in self.mail_events() if e not in MAIL_EVENTS]
        if bogus:
            warns.append(
                f"MAIL_TYPE has {', '.join(repr(e) for e in bogus)}, which sbatch "
                "does not know — it drops them silently, so you never get that "
                "mail. Open the MAIL_TYPE checklist to fix it."
            )
        policy = self.value("GOLDEN_POLICY")
        if policy is not None and policy not in GOLDEN_POLICIES:
            # config_defaults normalises this away rather than crashing, so
            # without the warning the file says one thing and jobs do another.
            warns.append(
                f"GOLDEN_POLICY = {policy!r} is not one of "
                f"{', '.join(GOLDEN_POLICIES)} — it is ignored and "
                f"{GOLDEN_POLICY_DEFAULT!r} applies."
            )
        for name, stale in sorted(self._retired.items()):
            fixed = RETIRED_KEYS[name]
            if stale != fixed:
                warns.append(
                    f"{name} = {stale!r} in config.py is ignored — it is fixed at "
                    f"{fixed!r} in {RETIRED_SOURCE}. Set SLURM_{name}={stale} to "
                    "keep the old behaviour."
                )
        return warns

    def save(self) -> str | None:
        """Validate, back up, replace. Error string on failure, None on success.

        Order matters: the backup is only written once the candidate has passed
        validation, so a rejected save leaves the directory exactly as it was.
        """
        errs = self.cross_field_errors()
        if errs:
            return errs[0]
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            f.write(self.render())
        err = validate_file(tmp)
        if err:
            os.unlink(tmp)
            return err
        if os.path.exists(self.path):
            shutil.copy2(self.path, self.path + BACKUP_SUFFIX)
        os.replace(tmp, self.path)
        self._reload()
        return None

    def _reload(self) -> None:
        """Re-parse from disk after a save, dropping the stage."""
        with open(self.path) as f:
            text = f.read()
        self.__init__(self.path, text)


BACKUP_SUFFIX = ".bak"

_REQUIRED = (
    "MAIL_USER", "GOLDEN_QOS", "CPU_MEM", "CPU_CPUS",
    "MAX_MEM_GB", "TIME_LIMIT", "START_TIMEOUT",
    "GPU_DEFINITIONS_BY_QOS", "GPU_DEFINITIONS",
)

_VALIDATE_SNIPPET = r"""
import runpy, sys
required = %r
try:
    ns = runpy.run_path(sys.argv[1])
except Exception as e:
    raise SystemExit("%%s: %%s" %% (type(e).__name__, e))
missing = [k for k in required if k not in ns]
if missing:
    raise SystemExit("missing keys: " + ", ".join(missing))
if not ns["GPU_DEFINITIONS"]:
    raise SystemExit("GPU_DEFINITIONS is empty: the primary QoS has no cards")
for qos, cards in ns["GPU_DEFINITIONS_BY_QOS"].items():
    for c in cards:
        if len(c) != 5:
            raise SystemExit("%%s: card %%r needs 5 fields" %% (qos, tuple(c)))
""" % (_REQUIRED,)


def validate_file(path: str) -> str | None:
    """Exec a candidate config in a subprocess. Error string, or None if fine.

    A subprocess because a syntax error or a KeyError from the derived
    GPU_DEFINITIONS line must not touch the interpreter running the form, and
    because the check should see a clean import.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _VALIDATE_SNIPPET, path],
        capture_output=True, text=True, timeout=30,
    )
    if proc.returncode == 0:
        return None
    lines = [l for l in proc.stderr.strip().splitlines() if l.strip()]
    return lines[-1] if lines else f"validation failed (exit {proc.returncode})"


def load(path: str = CONFIG_PATH) -> ConfigDoc:
    with open(path) as f:
        return ConfigDoc(path, f.read())
