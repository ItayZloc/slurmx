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
from dataclasses import dataclass

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(REPO, "config.py")
TEMPLATE_DIR = os.path.join(REPO, "config-examples")

DEFAULT_MAIL_DOMAIN = "post.bgu.ac.il"


def default_mail_user() -> str:
    """Prefill for an empty or absent MAIL_USER."""
    return f"{os.environ.get('USER', '')}@{DEFAULT_MAIL_DOMAIN}"


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
    Field("MAIL_USER", "str", "address for SLURM mail notifications"),
    Field("GOLDEN_QOS", "list", "golden QoS list; the first is primary"),
    Field("CPU_PARTITION", "str", "partition for CPU-only jobs"),
    Field("CPU_QOS", "str", "QoS for CPU-only jobs"),
    Field("MAIN_PARTITION", "str", "shared preemptible GPU pool"),
    Field("EXCLUDE_NODES", "list", "nodes to keep jobs off"),
    Field("MAX_MEM_GB", "int", "memory ceiling for a GPU job"),
    Field("CPU_CPUS", "int", "cores for a CPU-only job"),
    Field("CPU_MEM", "str", "memory for a CPU-only job, e.g. 16G"),
    Field("TIME_LIMIT", "str", "default wall clock, D-HH:MM:SS"),
    Field("START_TIMEOUT", "int", "seconds submit_job waits for RUNNING"),
    Field("GPU_DEFINITIONS_BY_QOS", "table", "GPU cards per QoS"),
    Field("GPU_DEFINITIONS", "derived", "the primary QoS's cards"),
)
FIELD_BY_NAME = {f.name: f for f in FIELDS}


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


def load(path: str = CONFIG_PATH) -> ConfigDoc:
    with open(path) as f:
        return ConfigDoc(path, f.read())
