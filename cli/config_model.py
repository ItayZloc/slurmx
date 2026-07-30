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

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(REPO, "config.py")
TEMPLATE_DIR = os.path.join(REPO, "config-examples")

if REPO not in sys.path:
    sys.path.insert(0, REPO)

from maintenance import _parse_slurm_time  # stdlib-only module; safe to import

DEFAULT_MAIL_DOMAIN = "post.bgu.ac.il"


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


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Field:
    name: str
    kind: str        # "str" | "int" | "list" | "table" | "derived"
    help: str
    validator: object = None   # Callable[[str], str | None]


FIELDS: tuple[Field, ...] = (
    Field("USERNAME", "derived", "auto-detected from $USER"),
    Field("MAIL_USER", "str", "address for SLURM mail notifications", _v_email),
    Field("GOLDEN_QOS", "list", "golden QoS list; the first is primary", _v_word_list),
    Field("CPU_PARTITION", "str", "partition for CPU-only jobs", _v_word),
    Field("CPU_QOS", "str", "QoS for CPU-only jobs", _v_word),
    Field("MAIN_PARTITION", "str", "shared preemptible GPU pool", _v_word),
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


CARD_CELLS = (
    ("name", _v_word),
    ("display", _v_text),
    ("vram", _v_posint),
    ("quota", _v_nonneg),
    ("partition", _v_word),
)
NEW_CARD = ("new_card", "New card", 24, 0, "new_partition")
_INT_CELLS = (2, 3)


def _table_literal(table: dict[str, list[list]]) -> str:
    """Re-render the whole GPU_DEFINITIONS_BY_QOS dict, columns aligned.

    Only called when a card edit is staged; an untouched table keeps its
    original bytes like every other field.
    """
    out = ["{"]
    for qos, cards in table.items():
        out.append(f"    {json.dumps(qos)}: [")
        w0 = max((len(json.dumps(c[0])) for c in cards), default=0)
        w1 = max((len(json.dumps(c[1])) for c in cards), default=0)
        w2 = max((len(str(c[2])) for c in cards), default=0)
        w3 = max((len(str(c[3])) for c in cards), default=0)
        for c in cards:
            name = (json.dumps(c[0]) + ",").ljust(w0 + 2)
            disp = (json.dumps(c[1]) + ",").ljust(w1 + 2)
            vram = (str(c[2]) + ",").rjust(w2 + 1)
            quota = (str(c[3]) + ",").rjust(w3 + 1)
            out.append(f"        ({name}{disp}{vram} {quota} {json.dumps(c[4])}),")
        out.append("    ],")
    out.append("}")
    return "\n".join(out)


def _parse_raw(f: Field, raw: str):
    if f.kind == "int":
        return int(raw.strip())
    if f.kind == "list":
        return _words(raw)
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

        # EXCLUDE_NODES is a comprehension over an env default: the editable
        # literal is the _EXCLUDE_NODES_DEFAULT comma string above it.
        if f.name == "EXCLUDE_NODES":
            helper = assigns.get("_EXCLUDE_NODES_DEFAULT")
            if helper is not None and isinstance(helper.value, ast.Constant):
                csv = helper.value.value or ""
                return Slot(f, self._span(helper.value),
                            [t.strip() for t in csv.split(",") if t.strip()],
                            "env-default", env_var="SLURM_EXCLUDE_NODES", as_csv=True)

        node = assigns.get(f.name)
        if node is None:
            return Slot(f, None, None, "absent")

        env = _env_get_call(node.value)
        if env is not None:
            env_var, default_node = env
            if isinstance(default_node, ast.Constant):
                return Slot(f, self._span(default_node), default_node.value,
                            "env-default", env_var=env_var)
            return Slot(f, None, None, "unsupported")

        try:
            value = ast.literal_eval(node.value)
        except (ValueError, SyntaxError, TypeError):
            return Slot(f, None, None, "unsupported")
        return Slot(f, self._span(node.value), value, "file")

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
        return self.slots[name].value

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
        cell_name, validator = CARD_CELLS[cell]
        err = validator(raw)
        if err:
            return f"{cell_name} {err}"
        table = self._mutable_table()
        table.setdefault(qos, [])
        table[qos][index][cell] = int(raw.strip()) if cell in _INT_CELLS else raw.strip()
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
        return [f"QoS '{q}' has no GPU cards" for q in qos[1:] if not groups.get(q)]

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
    "MAIL_USER", "GOLDEN_QOS", "CPU_PARTITION", "CPU_QOS", "CPU_MEM", "CPU_CPUS",
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
