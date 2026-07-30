# `slurmx config` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `slurmx config` subcommand that edits `config.py` as a full-screen curses form, splicing single literals in place so comments and `os.environ.get` fallbacks survive.

**Architecture:** Three layers. `cli/config_model.py` is pure and does all parsing, validation, and writing (`ast` for spans, subprocess for validation, `os.replace` for the write). `cli/config_form.py` holds the curses UI, split the way `cli/watch.py` is: pure `build_rows` / `dispatch` functions with unit tests, plus a thin `curses.wrapper` loop with none. `cli/config_cmd.py` is the argparse entry point that routes TTY to the form and non-TTY to a text dump.

**Tech Stack:** Python 3.12, stdlib only (`ast`, `curses`, `json`, `subprocess`), pytest.

Spec: `docs/superpowers/specs/2026-07-30-slurmx-config-design.md`.

## Global Constraints

- **No new dependencies.** stdlib only, same as the rest of the repo.
- **`cli/config_model.py` and `cli/config_cmd.py` must never import `slurm_mcp` or the top-level `config` module.** Task 9 depends on them being importable when `config.py` does not exist. `from maintenance import _parse_slurm_time` is allowed (stdlib-only module).
- **Never write to the repo's real `config.py` from a test.** Every test builds its own file under `tmp_path`.
- **The curses loop gets no test.** Pure helpers do. Same split as `cli/watch.py`.
- **Non-TTY must never block.** Piped, redirected, or run from an agent's Bash, `slurmx config` prints text and exits.
- **The module is `cli/config_cmd.py`, not `cli/config.py`** — the latter would shadow the top-level `config` module once the repo root is on `sys.path`.
- **`MAIL_USER` default is exactly** `f"{USERNAME}@post.bgu.ac.il"`.
- Repo lives at `~/.claude/mcp-servers/slurmx`, branch `main`. Run tests with `.venv/bin/python -m pytest`.
- Commit after every task. Conventional-commit subject, and end each message with `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `cli/config_model.py` (new, ~250 lines) | Field schema, validators, `ConfigDoc` (parse → spans + provenance, stage edits, render, validate, save). No UI, no curses. |
| `cli/config_form.py` (new, ~330 lines) | `Row`, `FormState`, `build_rows`, `move`, `dispatch` (all pure) plus `run_form` (curses glue). |
| `cli/config_cmd.py` (new, ~90 lines) | `add_arguments` / `run`, the `--show` text dump, template bootstrap. |
| `cli/theme.py` (modify) | Six new `Role` members + their attrs in `init_theme`. |
| `cli/slurmx.py` (modify) | Register the subcommand; degrade to a config-only parser when the `config` module is missing. |
| `tests/test_config_edit.py` (new, ~330 lines) | Everything above except the curses loop. |
| `config-examples/{default,yisroel}.py` (modify) | `MAIL_USER` default. |
| `setup.sh`, `README.md`, `WELCOME.md`, `.gitignore` (modify) | Docs + ignore `config.py.bak`. |

---

### Task 1: Config parsing and byte-identical round-trip

The foundation: read `config.py`, find the exact source span of every editable literal, and be able to write the file back unchanged.

**Files:**
- Create: `cli/config_model.py`
- Test: `tests/test_config_edit.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `FIELDS: tuple[Field, ...]`, `FIELD_BY_NAME: dict[str, Field]`, `Field(name, kind, help, validator)` with `kind in {"str","int","list","table","derived"}`, `Slot(field, span, value, provenance, env_var, as_csv)` where `span` is `tuple[int,int] | None` and `provenance` is one of `"file" | "env-default" | "absent" | "derived" | "unsupported"`, `ConfigDoc(path, text)` with `.slots: dict[str, Slot]`, `.text: str`, `.render() -> str`, `load(path) -> ConfigDoc`, `default_mail_user() -> str`, `CONFIG_PATH`, `TEMPLATE_DIR`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_config_edit.py`:

```python
"""Tests for `slurmx config` — the config model and the pure form helpers.

The curses loop itself has no test (same call as cli/watch.py): everything it
needs is in build_rows/move/dispatch, which are pure.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cli import config_model as cm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES = [
    os.path.join(REPO, "config-examples", "default.py"),
    os.path.join(REPO, "config-examples", "yisroel.py"),
]

# A file with deliberately awkward formatting: no trailing newline on the last
# assignment, inline comments, odd spacing, single quotes.
MANGLED = '''\
import os
USERNAME = os.environ.get("USER", "")
MAIL_USER   =    'someone@example.com'   # inline comment
GOLDEN_QOS = ['alpha' , 'beta']
CPU_PARTITION="cpu"
CPU_QOS = "normal"
MAIN_PARTITION = os.environ.get("SLURM_MAIN_PARTITION", "main")
_EXCLUDE_NODES_DEFAULT = "n1,n2"
EXCLUDE_NODES = [
    n.strip()
    for n in os.environ.get("SLURM_EXCLUDE_NODES", _EXCLUDE_NODES_DEFAULT).split(",")
    if n.strip()
]
MAX_MEM_GB = 80
CPU_CPUS = 4
CPU_MEM = "16G"
TIME_LIMIT = "7-0:00:00"
START_TIMEOUT = 300
GPU_DEFINITIONS_BY_QOS = {
    "alpha": [
        ("a_card", "A Card", 96, 16, "a_part"),
    ],
    "beta": [],
}
GPU_DEFINITIONS = GPU_DEFINITIONS_BY_QOS[GOLDEN_QOS[0]]'''


def write(tmp_path, text, name="config.py"):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


class TestLoad:
    @pytest.mark.parametrize("path", TEMPLATES)
    def test_template_round_trip_is_byte_identical(self, path):
        doc = cm.load(path)
        assert doc.render() == open(path).read()

    def test_mangled_round_trip_is_byte_identical(self, tmp_path):
        doc = cm.load(write(tmp_path, MANGLED))
        assert doc.render() == MANGLED

    def test_plain_literal_span_covers_only_the_value(self, tmp_path):
        doc = cm.load(write(tmp_path, MANGLED))
        start, end = doc.slots["MAX_MEM_GB"].span
        assert MANGLED[start:end] == "80"
        assert doc.slots["MAX_MEM_GB"].value == 80
        assert doc.slots["MAX_MEM_GB"].provenance == "file"

    def test_env_wrapper_span_points_at_the_default(self, tmp_path):
        doc = cm.load(write(tmp_path, MANGLED))
        slot = doc.slots["MAIN_PARTITION"]
        start, end = slot.span
        assert MANGLED[start:end] == '"main"'
        assert slot.provenance == "env-default"
        assert slot.env_var == "SLURM_MAIN_PARTITION"

    def test_exclude_nodes_span_points_at_the_helper_string(self, tmp_path):
        doc = cm.load(write(tmp_path, MANGLED))
        slot = doc.slots["EXCLUDE_NODES"]
        start, end = slot.span
        assert MANGLED[start:end] == '"n1,n2"'
        assert slot.value == ["n1", "n2"]
        assert slot.as_csv is True

    def test_single_quoted_list_parses(self, tmp_path):
        doc = cm.load(write(tmp_path, MANGLED))
        assert doc.slots["GOLDEN_QOS"].value == ["alpha", "beta"]

    def test_derived_fields_have_no_span(self, tmp_path):
        doc = cm.load(write(tmp_path, MANGLED))
        for name in ("USERNAME", "GPU_DEFINITIONS"):
            assert doc.slots[name].span is None
            assert doc.slots[name].provenance == "derived"

    def test_absent_key_is_marked_absent(self, tmp_path):
        text = MANGLED.replace(
            'MAIN_PARTITION = os.environ.get("SLURM_MAIN_PARTITION", "main")\n', ""
        )
        doc = cm.load(write(tmp_path, text))
        assert doc.slots["MAIN_PARTITION"].provenance == "absent"
        assert doc.slots["MAIN_PARTITION"].span is None

    def test_unsupported_shape_is_read_only_not_a_crash(self, tmp_path):
        text = MANGLED.replace("MAX_MEM_GB = 80", "MAX_MEM_GB = 40 + 40")
        doc = cm.load(write(tmp_path, text))
        assert doc.slots["MAX_MEM_GB"].provenance == "unsupported"
        assert doc.slots["MAX_MEM_GB"].span is None

    def test_default_mail_user_uses_the_bgu_domain(self, monkeypatch):
        monkeypatch.setenv("USER", "someone")
        assert cm.default_mail_user() == "someone@post.bgu.ac.il"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd ~/.claude/mcp-servers/slurmx && .venv/bin/python -m pytest tests/test_config_edit.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'cli.config_model'`.

- [ ] **Step 3: Write the implementation**

Create `cli/config_model.py`:

```python
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
from dataclasses import dataclass, field as dataclass_field

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


FIELDS: tuple[Field, ...] = ()          # filled in Task 2
FIELD_BY_NAME: dict[str, Field] = {}    # filled in Task 2


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
        self._staged: dict[str, str] = {}    # field -> replacement literal source
        self._values: dict[str, object] = {}  # field -> staged parsed value
        self._appends: list[str] = []         # new assignments for absent keys
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_config_edit.py -q`
Expected: all `TestLoad` tests pass. `FIELDS` is empty at this point, so `doc.slots` is empty and the span tests will fail — that is why Task 2 defines `FIELDS`. To keep this task green on its own, add `FIELDS` now with the schema below (Task 2 only adds validators to it):

```python
FIELDS = (
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
```

Re-run: all 11 tests pass.

- [ ] **Step 5: Commit**

```bash
cd ~/.claude/mcp-servers/slurmx
git add cli/config_model.py tests/test_config_edit.py
git commit -m "$(cat <<'EOF'
feat(config): parse config.py into editable literal spans

Locates the exact source span of each editable value, including the default
inside an os.environ.get wrapper and the _EXCLUDE_NODES_DEFAULT comma string
behind the EXCLUDE_NODES comprehension, so a write can swap one literal and
leave every other byte alone. render() on a clean doc is byte-identical.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Validators and staged scalar/list edits

**Files:**
- Modify: `cli/config_model.py`
- Test: `tests/test_config_edit.py`

**Interfaces:**
- Consumes: `Field`, `Slot`, `ConfigDoc`, `FIELDS` from Task 1.
- Produces: `ConfigDoc.set(name, raw) -> str | None` (error message or None), `.revert(name) -> None`, `.value(name) -> object`, `.text_value(name) -> str`, `.is_editable(name) -> bool`, `.staged_names() -> set[str]`, `.dirty -> bool` (property).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config_edit.py`:

```python
class TestSet:
    def doc(self, tmp_path):
        return cm.load(write(tmp_path, MANGLED))

    def test_int_edit_replaces_only_the_literal(self, tmp_path):
        doc = self.doc(tmp_path)
        assert doc.set("MAX_MEM_GB", "64") is None
        out = doc.render()
        assert "MAX_MEM_GB = 64\n" in out
        assert "CPU_CPUS = 4\n" in out
        assert out.count("MAX_MEM_GB") == 1

    def test_str_edit_keeps_the_inline_comment(self, tmp_path):
        doc = self.doc(tmp_path)
        assert doc.set("MAIL_USER", "new@post.bgu.ac.il") is None
        assert "\"new@post.bgu.ac.il\"   # inline comment" in doc.render()

    def test_env_default_edit_keeps_the_env_call(self, tmp_path):
        doc = self.doc(tmp_path)
        assert doc.set("MAIN_PARTITION", "gpu") is None
        assert 'os.environ.get("SLURM_MAIN_PARTITION", "gpu")' in doc.render()

    def test_list_edit_renders_a_python_list(self, tmp_path):
        doc = self.doc(tmp_path)
        assert doc.set("GOLDEN_QOS", "alpha, gamma") is None
        assert 'GOLDEN_QOS = ["alpha", "gamma"]' in doc.render()

    def test_csv_list_edit_renders_a_comma_string(self, tmp_path):
        doc = self.doc(tmp_path)
        assert doc.set("EXCLUDE_NODES", "n7, n8") is None
        out = doc.render()
        assert '_EXCLUDE_NODES_DEFAULT = "n7,n8"' in out
        assert "for n in os.environ.get(" in out

    def test_absent_key_is_appended(self, tmp_path):
        text = MANGLED.replace(
            'MAIN_PARTITION = os.environ.get("SLURM_MAIN_PARTITION", "main")\n', ""
        )
        doc = cm.load(write(tmp_path, text))
        assert doc.set("MAIN_PARTITION", "gpu") is None
        out = doc.render()
        assert out.endswith('MAIN_PARTITION = "gpu"\n')
        assert "# --- Added by `slurmx config` ---" in out

    def test_absent_key_edited_twice_appends_once(self, tmp_path):
        text = MANGLED.replace(
            'MAIN_PARTITION = os.environ.get("SLURM_MAIN_PARTITION", "main")\n', ""
        )
        doc = cm.load(write(tmp_path, text))
        doc.set("MAIN_PARTITION", "gpu")
        doc.set("MAIN_PARTITION", "main")
        assert doc.render().count("MAIN_PARTITION =") == 1

    def test_derived_and_unsupported_are_not_editable(self, tmp_path):
        text = MANGLED.replace("MAX_MEM_GB = 80", "MAX_MEM_GB = 40 + 40")
        doc = cm.load(write(tmp_path, text))
        assert doc.is_editable("GPU_DEFINITIONS") is False
        assert doc.is_editable("MAX_MEM_GB") is False
        assert "not editable" in doc.set("MAX_MEM_GB", "64")

    def test_revert_clears_the_stage(self, tmp_path):
        doc = self.doc(tmp_path)
        doc.set("MAX_MEM_GB", "64")
        assert doc.dirty is True
        doc.revert("MAX_MEM_GB")
        assert doc.dirty is False
        assert doc.render() == MANGLED

    def test_value_and_text_value_read_through_the_stage(self, tmp_path):
        doc = self.doc(tmp_path)
        assert doc.value("GOLDEN_QOS") == ["alpha", "beta"]
        assert doc.text_value("GOLDEN_QOS") == "alpha, beta"
        doc.set("GOLDEN_QOS", "solo")
        assert doc.value("GOLDEN_QOS") == ["solo"]
        assert doc.text_value("GOLDEN_QOS") == "solo"

    def test_empty_mail_user_prefills_the_bgu_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("USER", "someone")
        doc = cm.load(write(tmp_path, MANGLED.replace(
            "MAIL_USER   =    'someone@example.com'   # inline comment",
            'MAIL_USER = ""',
        )))
        assert doc.text_value("MAIL_USER") == "someone@post.bgu.ac.il"

    @pytest.mark.parametrize("name,raw,ok", [
        ("MAIL_USER", "a@b.c", True),
        ("MAIL_USER", "nope", False),
        ("MAIL_USER", "  ", False),
        ("GOLDEN_QOS", "a,b", True),
        ("GOLDEN_QOS", "", False),
        ("GOLDEN_QOS", "a b", False),
        ("CPU_PARTITION", "cpu", True),
        ("CPU_PARTITION", "two words", False),
        ("EXCLUDE_NODES", "", True),
        ("EXCLUDE_NODES", "n1,n2", True),
        ("EXCLUDE_NODES", "n 1", False),
        ("MAX_MEM_GB", "64", True),
        ("MAX_MEM_GB", "0", False),
        ("MAX_MEM_GB", "huge", False),
        ("CPU_MEM", "16G", True),
        ("CPU_MEM", "16", True),
        ("CPU_MEM", "16GB", False),
        ("TIME_LIMIT", "7-0:00:00", True),
        ("TIME_LIMIT", "0-12:30:00", True),
        ("TIME_LIMIT", "12:30", False),
        ("START_TIMEOUT", "300", True),
        ("START_TIMEOUT", "-1", False),
    ])
    def test_validators(self, tmp_path, name, raw, ok):
        doc = self.doc(tmp_path)
        err = doc.set(name, raw)
        assert (err is None) is ok, err
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_config_edit.py::TestSet -q`
Expected: FAIL, `AttributeError: 'ConfigDoc' object has no attribute 'set'`.

- [ ] **Step 3: Write the implementation**

Add validators above `FIELDS` in `cli/config_model.py`:

```python
from maintenance import _parse_slurm_time  # stdlib-only module; safe to import


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
```

Attach them in `FIELDS` (replace the tuple from Task 1's Step 4):

```python
FIELDS = (
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
```

Add module-level literal helpers:

```python
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
```

Add to `ConfigDoc`:

```python
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

    def set(self, name: str, raw: str) -> str | None:
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_config_edit.py -q`
Expected: PASS (Task 1 tests plus all of `TestSet`).

- [ ] **Step 5: Commit**

```bash
git add cli/config_model.py tests/test_config_edit.py
git commit -m "$(cat <<'EOF'
feat(config): validate and stage scalar/list edits

Per-field validators reject bad input before anything is staged, so the model
can never hold a value that would not import. An env-wrapped field edits the
default inside os.environ.get, EXCLUDE_NODES edits the comma string behind its
comprehension, and an absent key is appended once no matter how often it is
edited.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: The GPU card table

**Files:**
- Modify: `cli/config_model.py`
- Test: `tests/test_config_edit.py`

**Interfaces:**
- Consumes: everything from Tasks 1-2.
- Produces: `CARD_CELLS: tuple[tuple[str, object], ...]` (cell name, validator) in order `name, display, vram, quota, partition`, `NEW_CARD: tuple`, `ConfigDoc.groups() -> list[tuple[str, list[tuple]]]`, `.set_card(qos, index, cell, raw) -> str | None`, `.add_card(qos) -> None`, `.delete_card(qos, index) -> None`.

- [ ] **Step 1: Write the failing tests**

```python
class TestTable:
    def doc(self, tmp_path):
        return cm.load(write(tmp_path, MANGLED))

    def test_groups_follow_golden_qos_order(self, tmp_path):
        doc = self.doc(tmp_path)
        assert [q for q, _ in doc.groups()] == ["alpha", "beta"]
        assert doc.groups()[0][1] == [("a_card", "A Card", 96, 16, "a_part")]
        assert doc.groups()[1][1] == []

    def test_qos_without_a_group_shows_as_empty(self, tmp_path):
        doc = self.doc(tmp_path)
        doc.set("GOLDEN_QOS", "alpha, gamma")
        assert dict(doc.groups())["gamma"] == []

    def test_group_not_in_golden_qos_still_listed(self, tmp_path):
        doc = self.doc(tmp_path)
        doc.set("GOLDEN_QOS", "alpha")
        assert [q for q, _ in doc.groups()] == ["alpha", "beta"]

    def test_edit_a_cell_rewrites_the_dict_only(self, tmp_path):
        doc = self.doc(tmp_path)
        assert doc.set_card("alpha", 0, 2, "48") is None
        out = doc.render()
        assert '("a_card", "A Card", 48, 16, "a_part"),' in out
        assert "GPU_DEFINITIONS = GPU_DEFINITIONS_BY_QOS[GOLDEN_QOS[0]]" in out
        assert out.count("GPU_DEFINITIONS_BY_QOS = {") == 1

    def test_cell_validators_reject_and_do_not_stage(self, tmp_path):
        doc = self.doc(tmp_path)
        assert "integer" in doc.set_card("alpha", 0, 2, "lots")
        assert "whitespace" in doc.set_card("alpha", 0, 0, "two words")
        assert doc.dirty is False

    def test_quota_zero_is_allowed_vram_zero_is_not(self, tmp_path):
        doc = self.doc(tmp_path)
        assert doc.set_card("alpha", 0, 3, "0") is None
        assert doc.set_card("alpha", 0, 2, "0") is not None

    def test_add_card_appends_a_placeholder(self, tmp_path):
        doc = self.doc(tmp_path)
        doc.add_card("beta")
        assert dict(doc.groups())["beta"] == [cm.NEW_CARD]
        assert '"beta": [' in doc.render()

    def test_delete_card(self, tmp_path):
        doc = self.doc(tmp_path)
        doc.delete_card("alpha", 0)
        assert dict(doc.groups())["alpha"] == []
        assert doc.dirty is True

    def test_table_stays_loadable_after_a_rewrite(self, tmp_path):
        doc = self.doc(tmp_path)
        doc.set_card("alpha", 0, 4, "new_part")
        doc.add_card("beta")
        reloaded = cm.ConfigDoc(doc.path, doc.render())
        assert dict(reloaded.groups())["alpha"][0][4] == "new_part"
        assert dict(reloaded.groups())["beta"] == [cm.NEW_CARD]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_config_edit.py::TestTable -q`
Expected: FAIL, `AttributeError: 'ConfigDoc' object has no attribute 'groups'`.

- [ ] **Step 3: Write the implementation**

Add to `cli/config_model.py` at module level:

```python
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
        for c in cards:
            name = (json.dumps(c[0]) + ",").ljust(w0 + 2)
            disp = (json.dumps(c[1]) + ",").ljust(w1 + 2)
            out.append(f"        ({name}{disp}{c[2]}, {c[3]}, {json.dumps(c[4])}),")
        out.append("    ],")
    out.append("}")
    return "\n".join(out)
```

Add to `ConfigDoc`:

```python
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
```

Extend `dirty` so a table-only edit counts (it already does, because
`_stage_table` writes into `_staged`).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_config_edit.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cli/config_model.py tests/test_config_edit.py
git commit -m "$(cat <<'EOF'
feat(config): edit the GPU card table

Cards are read out of GPU_DEFINITIONS_BY_QOS as (qos, cards) groups in
GOLDEN_QOS order, with an empty group for a QoS that has no cards yet so it can
be filled from the form. A card edit re-renders the dict literal only; the
comment block above it and the derived GPU_DEFINITIONS line are untouched.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Cross-field checks, subprocess validation, save

**Files:**
- Modify: `cli/config_model.py`
- Test: `tests/test_config_edit.py`

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces: `ConfigDoc.cross_field_errors() -> list[str]`, `.warnings() -> list[str]`, `.save() -> str | None`, `validate_file(path) -> str | None`, `BACKUP_SUFFIX = ".bak"`.

- [ ] **Step 1: Write the failing tests**

```python
class TestGate:
    def doc(self, tmp_path):
        return cm.load(write(tmp_path, MANGLED))

    def test_clean_doc_has_no_errors(self, tmp_path):
        doc = self.doc(tmp_path)
        assert doc.cross_field_errors() == []

    def test_primary_qos_without_cards_blocks(self, tmp_path):
        doc = self.doc(tmp_path)
        doc.set("GOLDEN_QOS", "beta, alpha")   # beta's group is empty
        errs = doc.cross_field_errors()
        assert any("beta" in e and "no GPU cards" in e for e in errs)

    def test_missing_primary_group_blocks(self, tmp_path):
        doc = self.doc(tmp_path)
        doc.set("GOLDEN_QOS", "gamma")
        assert any("gamma" in e for e in doc.cross_field_errors())

    def test_secondary_qos_without_cards_only_warns(self, tmp_path):
        doc = self.doc(tmp_path)
        doc.set("GOLDEN_QOS", "alpha, gamma")
        assert doc.cross_field_errors() == []
        assert any("gamma" in w for w in doc.warnings())

    def test_duplicate_card_names_block(self, tmp_path):
        doc = self.doc(tmp_path)
        doc.add_card("alpha")
        doc.set_card("alpha", 1, 0, "a_card")
        assert any("duplicate" in e for e in doc.cross_field_errors())

    def test_save_writes_backup_and_replaces(self, tmp_path):
        path = write(tmp_path, MANGLED)
        doc = cm.load(path)
        doc.set("MAX_MEM_GB", "64")
        assert doc.save() is None
        assert "MAX_MEM_GB = 64" in open(path).read()
        assert open(path + ".bak").read() == MANGLED
        assert not os.path.exists(path + ".tmp")

    def test_save_clears_dirty_and_reloads_state(self, tmp_path):
        doc = cm.load(write(tmp_path, MANGLED))
        doc.set("MAX_MEM_GB", "64")
        doc.save()
        assert doc.dirty is False
        assert doc.value("MAX_MEM_GB") == 64

    def test_save_rejects_a_file_that_would_not_import(self, tmp_path):
        # A table rewrite that drops the primary QoS makes the derived line raise.
        path = write(tmp_path, MANGLED)
        doc = cm.load(path)
        doc.delete_card("alpha", 0)
        doc._table.pop("alpha")
        doc._stage_table()
        err = doc.save()
        assert err is not None and "alpha" in err
        assert open(path).read() == MANGLED       # original untouched
        assert not os.path.exists(path + ".tmp")
        assert not os.path.exists(path + ".bak")

    def test_validate_file_rejects_a_syntax_error(self, tmp_path):
        bad = write(tmp_path, "MAIL_USER = (", name="bad.py")
        assert "SyntaxError" in cm.validate_file(bad)

    def test_validate_file_rejects_a_short_card_tuple(self, tmp_path):
        text = MANGLED.replace('("a_card", "A Card", 96, 16, "a_part")',
                               '("a_card", "A Card", 96)')
        assert "5 fields" in cm.validate_file(write(tmp_path, text, name="short.py"))

    @pytest.mark.parametrize("path", TEMPLATES)
    def test_templates_validate(self, path):
        assert cm.validate_file(path) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_config_edit.py::TestGate -q`
Expected: FAIL, `AttributeError: module 'cli.config_model' has no attribute 'validate_file'`.

- [ ] **Step 3: Write the implementation**

Add imports `shutil`, `subprocess`, `sys` to `cli/config_model.py`, then:

```python
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
    raise SystemExit("%s: %s" % (type(e).__name__, e))
missing = [k for k in required if k not in ns]
if missing:
    raise SystemExit("missing keys: " + ", ".join(missing))
if not ns["GPU_DEFINITIONS"]:
    raise SystemExit("GPU_DEFINITIONS is empty: the primary QoS has no cards")
for qos, cards in ns["GPU_DEFINITIONS_BY_QOS"].items():
    for c in cards:
        if len(c) != 5:
            raise SystemExit("%s: card %r needs 5 fields" % (qos, tuple(c)))
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
```

Add to `ConfigDoc`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_config_edit.py -q`
Expected: PASS. Then confirm nothing else broke: `.venv/bin/python -m pytest tests/ -q -k "not live"` → 221 existing tests still pass.

- [ ] **Step 5: Commit**

```bash
git add cli/config_model.py tests/test_config_edit.py
git commit -m "$(cat <<'EOF'
feat(config): gate saves on the checks that would brick the CLI

Two conditions block a write: a primary QoS with no cards (the derived
GPU_DEFINITIONS line raises KeyError at import, which takes down every
subcommand) and duplicate card names in one group (GPU_BY_NAME silently drops
one). A secondary QoS with no cards only warns. The candidate is exec'd in a
subprocess before the backup is written, so a rejected save leaves the
directory untouched.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: `slurmx config --show` and the subcommand wiring

First user-visible deliverable: `slurmx config --show` prints the resolved config. The form comes in Tasks 6-8.

**Files:**
- Create: `cli/config_cmd.py`
- Modify: `cli/slurmx.py:18-26` (imports) and `cli/slurmx.py:103-106` (registration)
- Test: `tests/test_config_edit.py`

**Interfaces:**
- Consumes: `cli.config_model` (Tasks 1-4).
- Produces: `cli.config_cmd.add_arguments(parser)`, `cli.config_cmd.run(args)`, `cli.config_cmd.show_text(doc) -> str`.

- [ ] **Step 1: Write the failing tests**

```python
from cli import config_cmd


class TestShow:
    def test_show_text_lists_every_field_with_provenance(self, tmp_path):
        doc = cm.load(write(tmp_path, MANGLED))
        out = config_cmd.show_text(doc)
        assert "MAIL_USER" in out and "someone@example.com" in out
        assert "MAIN_PARTITION" in out and "env-default" in out
        assert "GPU_DEFINITIONS" in out and "derived" in out
        assert "alpha" in out and "a_card" in out and "96" in out

    def test_show_text_marks_an_active_env_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SLURM_MAIN_PARTITION", "gpu")
        doc = cm.load(write(tmp_path, MANGLED))
        assert "SLURM_MAIN_PARTITION=gpu" in config_cmd.show_text(doc)

    def test_show_text_renders_an_empty_list_readably(self, tmp_path):
        doc = cm.load(write(tmp_path, MANGLED.replace('"n1,n2"', '""')))
        assert "(none)" in config_cmd.show_text(doc)

    def test_run_show_prints_and_does_not_touch_the_file(self, tmp_path, capsys):
        path = write(tmp_path, MANGLED)
        args = type("A", (), {"show": True, "path": path})()
        config_cmd.run(args)
        assert "MAIL_USER" in capsys.readouterr().out
        assert open(path).read() == MANGLED

    def test_non_tty_routes_to_text(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr("sys.stdout.isatty", lambda: False, raising=False)
        args = type("A", (), {"show": False, "path": write(tmp_path, MANGLED)})()
        config_cmd.run(args)
        assert "GOLDEN_QOS" in capsys.readouterr().out


class TestParser:
    def test_config_is_registered(self):
        from cli import slurmx as slurmx_cli
        parser = slurmx_cli.build_parser()
        args = parser.parse_args(["config", "--show"])
        assert args.show is True
        assert args._run is config_cmd.run
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_config_edit.py::TestShow -q`
Expected: FAIL, `ImportError: cannot import name 'config_cmd' from 'cli'`.

- [ ] **Step 3: Write the implementation**

Create `cli/config_cmd.py`:

```python
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


def _display(doc, name: str) -> str:
    slot = doc.slots[name]
    if slot.field.kind == "list":
        return doc.text_value(name) or "(none)"
    if slot.provenance == "derived":
        cards = sum(len(c) for _, c in doc.groups()) if name == "GPU_DEFINITIONS" \
            else 0
        if name == "GPU_DEFINITIONS":
            primary = (doc.value("GOLDEN_QOS") or [""])[0]
            return f"{len(dict(doc.groups()).get(primary, []))} cards ({primary})"
        return os.environ.get("USER", "")
    return doc.text_value(name) or "(unset)"


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
        line = f"  {f.name.ljust(NAME_W)}{_display(doc, f.name)}"
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
```

`_display`'s dead `cards` local is a leftover; delete it while writing the file
so the function reads:

```python
def _display(doc, name: str) -> str:
    slot = doc.slots[name]
    if slot.field.kind == "list":
        return doc.text_value(name) or "(none)"
    if name == "GPU_DEFINITIONS":
        primary = (doc.value("GOLDEN_QOS") or [""])[0]
        return f"{len(dict(doc.groups()).get(primary, []))} cards ({primary})"
    if name == "USERNAME":
        return os.environ.get("USER", "")
    return doc.text_value(name) or "(unset)"
```

Register it in `cli/slurmx.py` — add the import beside the others:

```python
from cli import config_cmd as config_mod
```

and the subparser, placed after `cancel` and before `setup`:

```python
    _add_subcommand(
        subparsers, "config", config_mod,
        help="Edit config.py in a terminal form (--show prints it as text).",
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_config_edit.py -q`
Expected: PASS.

Then eyeball the real thing (read-only, safe):

```bash
cd ~/.claude/mcp-servers/slurmx && .venv/bin/python -m cli.slurmx config --show
```

Expected: your live config values, `MAIN_PARTITION` tagged `env-default`, six cards under `GPU cards · yisroel`, no errors or warnings.

- [ ] **Step 5: Commit**

```bash
git add cli/config_cmd.py cli/slurmx.py tests/test_config_edit.py
git commit -m "$(cat <<'EOF'
feat(cli): add `slurmx config --show`

Prints every field with its provenance (file / env-default / absent / derived),
flags an env var that is currently overriding a default, and lists the GPU
cards per QoS. Non-TTY routes here automatically, so an agent's Bash can never
land on a blocking UI.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Form rows

**Files:**
- Create: `cli/config_form.py`
- Modify: `cli/theme.py:23-35` (Role members) and `cli/theme.py:125-137` (attr map)
- Test: `tests/test_config_edit.py`

**Interfaces:**
- Consumes: `cli.config_model` (Tasks 1-4).
- Produces: `Row(kind, spans, field=None, qos=None, index=None, selectable=False)` with `kind in {"blank","field","group","thead","card","add"}`, `FormState(doc, path, cursor=0, folds=set(), cell=0, editing=None, edit_pos=0, status="", confirm=None, done=False)`, `build_rows(state) -> list[Row]`.
- New `theme.Role` members: `CFG_NAME`, `CFG_VALUE`, `CFG_TAG`, `CFG_EDITED`, `CFG_THEAD`, `CFG_ERROR`.

- [ ] **Step 1: Write the failing tests**

```python
from cli import config_form as cf


def state_for(tmp_path, text=MANGLED):
    path = write(tmp_path, text)
    return cf.FormState(doc=cm.load(path), path=path)


def flat(rows):
    return ["".join(t for t, _ in r.spans) for r in rows]


class TestRows:
    def test_every_scalar_field_gets_a_row(self, tmp_path):
        rows = cf.build_rows(state_for(tmp_path))
        fields = [r.field for r in rows if r.kind == "field"]
        assert fields == [f.name for f in cm.FIELDS if f.kind != "table"]

    def test_table_renders_a_group_header_per_qos(self, tmp_path):
        rows = cf.build_rows(state_for(tmp_path))
        groups = [r for r in rows if r.kind == "group"]
        assert [g.qos for g in groups] == ["alpha", "beta"]
        assert "GPU cards · alpha (1)" in "".join(t for t, _ in groups[0].spans)

    def test_single_qos_header_omits_the_qos_name(self, tmp_path):
        st = state_for(tmp_path)
        st.doc.set("GOLDEN_QOS", "alpha")
        st.doc._table = {"alpha": [list(cm.NEW_CARD)]}
        rows = [r for r in cf.build_rows(st) if r.kind == "group"]
        assert "GPU cards (1)" in "".join(t for t, _ in rows[0].spans)

    def test_unfolded_group_shows_a_header_cards_and_add(self, tmp_path):
        rows = cf.build_rows(state_for(tmp_path))
        kinds = [r.kind for r in rows if r.qos == "alpha"]
        assert kinds == ["group", "thead", "card", "add"]

    def test_folded_group_hides_its_cards(self, tmp_path):
        st = state_for(tmp_path)
        st.folds.add("alpha")
        rows = [r for r in cf.build_rows(st) if r.qos == "alpha"]
        assert [r.kind for r in rows] == ["group"]
        assert "▸" in "".join(t for t, _ in rows[0].spans)

    def test_empty_group_still_offers_add(self, tmp_path):
        rows = cf.build_rows(state_for(tmp_path))
        assert [r.kind for r in rows if r.qos == "beta"] == ["group", "thead", "add"]

    def test_derived_rows_are_not_selectable_and_are_tagged(self, tmp_path):
        rows = {r.field: r for r in cf.build_rows(state_for(tmp_path)) if r.kind == "field"}
        assert rows["GPU_DEFINITIONS"].selectable is False
        assert "derived" in "".join(t for t, _ in rows["GPU_DEFINITIONS"].spans)

    def test_staged_field_is_tagged_edited(self, tmp_path):
        st = state_for(tmp_path)
        st.doc.set("MAX_MEM_GB", "64")
        row = next(r for r in cf.build_rows(st) if r.field == "MAX_MEM_GB")
        assert "edited" in "".join(t for t, _ in row.spans)
        assert "64" in "".join(t for t, _ in row.spans)

    def test_editing_row_shows_the_buffer_and_a_caret(self, tmp_path):
        st = state_for(tmp_path)
        st.cursor = next(i for i, r in enumerate(cf.build_rows(st))
                         if r.field == "MAX_MEM_GB")
        st.editing = "12"
        st.edit_pos = 2
        row = cf.build_rows(st)[st.cursor]
        assert "12▏" in "".join(t for t, _ in row.spans)

    def test_selected_card_cell_is_marked(self, tmp_path):
        st = state_for(tmp_path)
        st.cursor = next(i for i, r in enumerate(cf.build_rows(st)) if r.kind == "card")
        st.cell = 2
        spans = cf.build_rows(st)[st.cursor].spans
        assert any(role is cf.Role.CFG_EDITED and "96" in t for t, role in spans)

    def test_theme_roles_exist_and_map(self):
        from cli import theme
        for name in ("CFG_NAME", "CFG_VALUE", "CFG_TAG", "CFG_EDITED",
                     "CFG_THEAD", "CFG_ERROR"):
            assert hasattr(theme.Role, name)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_config_edit.py::TestRows -q`
Expected: FAIL, `ImportError: cannot import name 'config_form' from 'cli'`.

- [ ] **Step 3: Write the implementation**

Add to `cli/theme.py`, inside `class Role`:

```python
    CFG_NAME = auto()       # config field name column
    CFG_VALUE = auto()      # config field value column
    CFG_TAG = auto()        # derived / default / env tag
    CFG_EDITED = auto()     # staged value, or the selected card cell
    CFG_THEAD = auto()      # card table column header
    CFG_ERROR = auto()      # blocking error in the status bar
```

and to the dict returned by `init_theme`:

```python
        Role.CFG_NAME: cp(6),
        Role.CFG_VALUE: 0,
        Role.CFG_TAG: cp(5),
        Role.CFG_EDITED: cp(3) | soften,
        Role.CFG_THEAD: cp(1) | soften,
        Role.CFG_ERROR: cp(4) | soften,
```

Create `cli/config_form.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_config_edit.py -q`
Expected: PASS. Also `.venv/bin/python -m pytest tests/test_slurm_mcp.py -q -k "not live"` to confirm the new `Role` members did not break `test_init_theme_no_color_returns_empty`.

- [ ] **Step 5: Commit**

```bash
git add cli/config_form.py cli/theme.py tests/test_config_edit.py
git commit -m "$(cat <<'EOF'
feat(config): lay out the form as role-tagged rows

build_rows is pure, so the whole layout is unit tested without a terminal:
scalar rows with their provenance tag, one fold group per QoS (the name is
omitted when there is only one), a card table with a selected-cell highlight,
and an `+ add card` row even on an empty group.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Movement, editing, and the confirm latches

**Files:**
- Modify: `cli/config_form.py`
- Test: `tests/test_config_edit.py`

**Interfaces:**
- Consumes: Task 6.
- Produces: `move(state, key) -> FormState`, `dispatch(state, key) -> FormState` (both mutate and return `state`), `SAVE_KEYS`, `QUIT_KEYS`.

- [ ] **Step 1: Write the failing tests**

```python
import curses


class TestDispatch:
    def cursor_field(self, st, name):
        st.cursor = next(i for i, r in enumerate(cf.build_rows(st)) if r.field == name)
        return st

    def test_down_skips_unselectable_rows(self, tmp_path):
        st = state_for(tmp_path)
        st.cursor = 0
        cf.dispatch(st, curses.KEY_DOWN)
        assert cf.build_rows(st)[st.cursor].selectable is True

    def test_down_then_up_returns(self, tmp_path):
        st = self.cursor_field(state_for(tmp_path), "MAX_MEM_GB")
        start = st.cursor
        cf.dispatch(st, ord("j"))
        cf.dispatch(st, ord("k"))
        assert st.cursor == start

    def test_g_and_shift_g_jump(self, tmp_path):
        st = state_for(tmp_path)
        cf.dispatch(st, ord("G"))
        assert cf.build_rows(st)[st.cursor].selectable is True
        cf.dispatch(st, ord("g"))
        assert st.cursor == 1

    def test_enter_on_a_field_starts_editing_prefilled(self, tmp_path):
        st = self.cursor_field(state_for(tmp_path), "MAX_MEM_GB")
        cf.dispatch(st, ord("\n"))
        assert st.editing == "80"
        assert st.edit_pos == 2

    def test_typing_then_enter_stages_the_value(self, tmp_path):
        st = self.cursor_field(state_for(tmp_path), "MAX_MEM_GB")
        cf.dispatch(st, ord("\n"))
        for _ in range(2):
            cf.dispatch(st, curses.KEY_BACKSPACE)
        for ch in "64":
            cf.dispatch(st, ord(ch))
        cf.dispatch(st, ord("\n"))
        assert st.editing is None
        assert st.doc.value("MAX_MEM_GB") == 64

    def test_invalid_value_keeps_the_editor_open_with_a_reason(self, tmp_path):
        st = self.cursor_field(state_for(tmp_path), "MAX_MEM_GB")
        cf.dispatch(st, ord("\n"))
        st.editing, st.edit_pos = "huge", 4
        cf.dispatch(st, ord("\n"))
        assert st.editing == "huge"
        assert "integer" in st.status
        assert st.doc.dirty is False

    def test_escape_cancels_the_edit(self, tmp_path):
        st = self.cursor_field(state_for(tmp_path), "MAX_MEM_GB")
        cf.dispatch(st, ord("\n"))
        st.editing = "64"
        cf.dispatch(st, 27)
        assert st.editing is None
        assert st.doc.dirty is False

    def test_r_reverts_one_field(self, tmp_path):
        st = self.cursor_field(state_for(tmp_path), "MAX_MEM_GB")
        st.doc.set("MAX_MEM_GB", "64")
        cf.dispatch(st, ord("r"))
        assert st.doc.dirty is False
        assert st.doc.value("MAX_MEM_GB") == 80

    def test_enter_on_a_group_toggles_the_fold(self, tmp_path):
        st = state_for(tmp_path)
        st.cursor = next(i for i, r in enumerate(cf.build_rows(st)) if r.kind == "group")
        cf.dispatch(st, ord("\n"))
        assert "alpha" in st.folds
        cf.dispatch(st, ord("\n"))
        assert "alpha" not in st.folds

    def test_left_right_move_the_card_cell(self, tmp_path):
        st = state_for(tmp_path)
        st.cursor = next(i for i, r in enumerate(cf.build_rows(st)) if r.kind == "card")
        cf.dispatch(st, curses.KEY_RIGHT)
        assert st.cell == 1
        for _ in range(10):
            cf.dispatch(st, curses.KEY_RIGHT)
        assert st.cell == len(cm.CARD_CELLS) - 1
        for _ in range(10):
            cf.dispatch(st, curses.KEY_LEFT)
        assert st.cell == 0

    def test_enter_on_a_card_edits_the_selected_cell(self, tmp_path):
        st = state_for(tmp_path)
        st.cursor = next(i for i, r in enumerate(cf.build_rows(st)) if r.kind == "card")
        st.cell = 2
        cf.dispatch(st, ord("\n"))
        assert st.editing == "96"
        st.editing, st.edit_pos = "48", 2
        cf.dispatch(st, ord("\n"))
        assert dict(st.doc.groups())["alpha"][0][2] == 48

    def test_a_adds_a_card_to_the_group_under_the_cursor(self, tmp_path):
        st = state_for(tmp_path)
        st.cursor = next(i for i, r in enumerate(cf.build_rows(st)) if r.kind == "card")
        cf.dispatch(st, ord("a"))
        assert len(dict(st.doc.groups())["alpha"]) == 2

    def test_enter_on_add_row_adds_a_card(self, tmp_path):
        st = state_for(tmp_path)
        st.cursor = next(i for i, r in enumerate(cf.build_rows(st))
                         if r.kind == "add" and r.qos == "beta")
        cf.dispatch(st, ord("\n"))
        assert dict(st.doc.groups())["beta"] == [cm.NEW_CARD]

    def test_delete_needs_two_presses(self, tmp_path):
        st = state_for(tmp_path)
        st.cursor = next(i for i, r in enumerate(cf.build_rows(st)) if r.kind == "card")
        cf.dispatch(st, ord("d"))
        assert st.confirm == "delete"
        assert len(dict(st.doc.groups())["alpha"]) == 1
        cf.dispatch(st, ord("d"))
        assert dict(st.doc.groups())["alpha"] == []
        assert st.confirm is None

    def test_any_other_key_clears_the_delete_latch(self, tmp_path):
        st = state_for(tmp_path)
        st.cursor = next(i for i, r in enumerate(cf.build_rows(st)) if r.kind == "card")
        cf.dispatch(st, ord("d"))
        cf.dispatch(st, ord("j"))
        assert st.confirm is None
        cf.dispatch(st, ord("d"))
        assert len(dict(st.doc.groups())["alpha"]) == 1

    def test_quit_when_clean_is_immediate(self, tmp_path):
        st = state_for(tmp_path)
        cf.dispatch(st, ord("q"))
        assert st.done is True

    def test_quit_when_dirty_needs_two_presses(self, tmp_path):
        st = self.cursor_field(state_for(tmp_path), "MAX_MEM_GB")
        st.doc.set("MAX_MEM_GB", "64")
        cf.dispatch(st, ord("q"))
        assert st.done is False and st.confirm == "quit"
        cf.dispatch(st, ord("q"))
        assert st.done is True

    def test_s_saves_and_reports(self, tmp_path):
        st = self.cursor_field(state_for(tmp_path), "MAX_MEM_GB")
        st.doc.set("MAX_MEM_GB", "64")
        cf.dispatch(st, ord("s"))
        assert "saved" in st.status
        assert "/mcp" in st.status
        assert "MAX_MEM_GB = 64" in open(st.path).read()
        assert st.doc.dirty is False

    def test_s_refuses_a_blocked_config(self, tmp_path):
        st = state_for(tmp_path)
        st.doc.set("GOLDEN_QOS", "gamma")
        cf.dispatch(st, ord("s"))
        assert "gamma" in st.status
        assert not os.path.exists(st.path + ".bak")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_config_edit.py::TestDispatch -q`
Expected: FAIL, `AttributeError: module 'cli.config_form' has no attribute 'dispatch'`.

- [ ] **Step 3: Write the implementation**

Append to `cli/config_form.py`:

```python
import curses

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
        state.editing = str(card[state.cell])
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

    if key not in (ord("d"),) and state.confirm == "delete":
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
    elif key in (curses.KEY_RIGHT,) and row is not None and row.kind == "card":
        state.cell = min(state.cell + 1, len(cm.CARD_CELLS) - 1)
    elif key in (curses.KEY_LEFT,) and row is not None and row.kind == "card":
        state.cell = max(state.cell - 1, 0)
    else:
        move(state, key)
    return state
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_config_edit.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cli/config_form.py tests/test_config_edit.py
git commit -m "$(cat <<'EOF'
feat(config): key handling for the form

One pure reducer holds the whole key contract, so it is testable without a
terminal: movement that skips unselectable rows, inline editing that refuses to
commit an invalid value, per-field revert, fold toggles, card add/delete behind
a two-press latch, and a quit that needs confirming while dirty.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: The curses loop

**Files:**
- Modify: `cli/config_form.py`
- Test: none (loop is glue; see Global Constraints)

**Interfaces:**
- Consumes: Tasks 6-7.
- Produces: `run_form(path: str) -> None`, raising `curses.error` when the terminal cannot host curses (`cli/config_cmd.run` already catches it).

- [ ] **Step 1: Write the implementation**

Append to `cli/config_form.py`:

```python
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
                    stdscr.addnstr(y + 1, 0, "▸", 1, theme.get(theme_mod.Role.CFG_THEAD, 0))
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


def run_form(path: str) -> None:
    """Open the form on `path`. Blocks until the user quits.

    Raises curses.error when the terminal can't host curses (TERM=dumb, no tty);
    cli/config_cmd.run falls back to the text dump, same as slurmx status.
    """
    state = FormState(doc=cm.load(path), path=path)
    try:
        curses.wrapper(_loop, state)
    except KeyboardInterrupt:
        pass
```

- [ ] **Step 2: Verify the whole suite still passes**

Run: `.venv/bin/python -m pytest tests/ -q -k "not live"`
Expected: 221 pre-existing + the new tests, all passing.

- [ ] **Step 3: Drive it by hand**

```bash
cd ~/.claude/mcp-servers/slurmx
cp config.py /tmp/config_probe.py
.venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from cli import config_form
config_form.run_form('/tmp/config_probe.py')
"
```

Check by hand, in order: arrow keys skip the two `derived` rows; `Enter` on
`MAX_MEM_GB` prefills `80` and typing replaces it; `Esc` cancels; `Enter` on the
group header folds and unfolds; `←→` walks the five card cells; `a` then `d`,`d`
adds and removes a card; the status bar counts unsaved changes; `s` writes and
prints the `/mcp` reminder; `q` after an edit needs two presses. Then:

```bash
diff /tmp/config_probe.py config.py     # only the lines you changed
head -20 /tmp/config_probe.py           # comments intact
rm /tmp/config_probe.py /tmp/config_probe.py.bak
```

- [ ] **Step 4: Commit**

```bash
git add cli/config_form.py
git commit -m "$(cat <<'EOF'
feat(config): curses loop for the config form

Thin glue over build_rows and dispatch: draw, scroll to keep the cursor
visible, read a key, hand it to the reducer. Raises curses.error on a terminal
that can't host it so the caller falls back to the text dump, matching
slurmx status.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Fresh-clone bootstrap

Without this, `slurmx config` is unusable in the one situation it is most needed: a clone with no `config.py`, where every subcommand dies at import.

**Files:**
- Modify: `cli/config_cmd.py` (`_bootstrap`)
- Modify: `cli/slurmx.py:18-26`, `55-116`
- Modify: `setup.sh:24-30`
- Test: `tests/test_config_edit.py`

**Interfaces:**
- Consumes: Tasks 1-8.
- Produces: `cli.config_cmd.templates() -> list[tuple[str, str]]` (label, path), `cli.config_cmd._bootstrap(path, choose) -> str | None` where `choose` is a callable taking the prompt text and returning the typed line (defaults to `input`), `cli.slurmx.CONFIG_ONLY_HINT`.

- [ ] **Step 1: Write the failing tests**

```python
import importlib


class TestBootstrap:
    def test_templates_lists_both_examples(self):
        labels = [l for l, _ in config_cmd.templates()]
        assert "default" in labels and "yisroel" in labels

    def test_bootstrap_copies_the_chosen_template(self, tmp_path):
        path = str(tmp_path / "config.py")
        err = config_cmd._bootstrap(path, choose=lambda prompt: "1")
        assert err is None
        assert os.path.exists(path)
        assert "GPU_DEFINITIONS_BY_QOS" in open(path).read()

    def test_bootstrap_rejects_a_bad_choice(self, tmp_path):
        path = str(tmp_path / "config.py")
        err = config_cmd._bootstrap(path, choose=lambda prompt: "9")
        assert err is not None
        assert not os.path.exists(path)

    def test_bootstrap_aborts_on_empty_input(self, tmp_path):
        path = str(tmp_path / "config.py")
        assert config_cmd._bootstrap(path, choose=lambda prompt: "") is not None
        assert not os.path.exists(path)


class TestDegradedParser:
    class _Block:
        def find_spec(self, name, path=None, target=None):
            if name == "config":
                raise ModuleNotFoundError("No module named 'config'")
            return None

    def test_parser_offers_only_config_setup_update_without_config_py(self):
        blocker = self._Block()
        sys.meta_path.insert(0, blocker)
        saved = {k: v for k, v in sys.modules.items()
                 if k == "config" or k.startswith(("cli.", "slurm_mcp"))}
        for k in list(saved):
            sys.modules.pop(k, None)
        try:
            slurmx_cli = importlib.import_module("cli.slurmx")
            importlib.reload(slurmx_cli)
            parser = slurmx_cli.build_parser()
            names = set(parser._subparsers._group_actions[0].choices)
            assert names == {"config", "setup", "update"}
            assert "slurmx config" in slurmx_cli.CONFIG_ONLY_HINT
        finally:
            sys.meta_path.remove(blocker)
            for k in list(sys.modules):
                if k.startswith(("cli.", "slurm_mcp")) or k == "config":
                    sys.modules.pop(k, None)
            sys.modules.update(saved)

    def test_full_parser_has_every_subcommand(self):
        from cli import slurmx as slurmx_cli
        importlib.reload(slurmx_cli)
        parser = slurmx_cli.build_parser()
        names = set(parser._subparsers._group_actions[0].choices)
        assert {"status", "submit", "config", "cancel", "setup", "update"} <= names
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_config_edit.py::TestBootstrap tests/test_config_edit.py::TestDegradedParser -q`
Expected: FAIL, `AttributeError: module 'cli.config_cmd' has no attribute 'templates'`.

- [ ] **Step 3: Write the implementation**

Replace `_bootstrap` in `cli/config_cmd.py`:

```python
def templates() -> list[tuple[str, str]]:
    """(label, path) for each config-examples/*.py, default first."""
    names = sorted(n for n in os.listdir(cm.TEMPLATE_DIR)
                   if n.endswith(".py") and not n.startswith("_"))
    names.sort(key=lambda n: (n != "default.py", n))
    return [(n[:-3], os.path.join(cm.TEMPLATE_DIR, n)) for n in names]


_TEMPLATE_BLURB = {
    "default": "blank template, fill in your own QoS and cards",
    "yisroel": "Yisroel's lab, pre-filled QoS and golden quotas",
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
```

`run` already calls `_bootstrap` and exits on error. Add one line after a
successful bootstrap so the form opens on `MAIL_USER` — replace the body of
`run` up to `doc = cm.load(path)` with:

```python
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
        print(show_text(cm.load(path)))
```

and give `run_form` the parameter in `cli/config_form.py`:

```python
def run_form(path: str, start_field: str | None = None) -> None:
    """Open the form on `path`. Blocks until the user quits.

    Raises curses.error when the terminal can't host curses (TERM=dumb, no tty);
    cli/config_cmd.run falls back to the text dump, same as slurmx status.
    """
    state = FormState(doc=cm.load(path), path=path)
    if start_field:
        rows = build_rows(state)
        for i, row in enumerate(rows):
            if row.field == start_field and row.selectable:
                state.cursor = i
                break
    try:
        curses.wrapper(_loop, state)
    except KeyboardInterrupt:
        pass
```

Restructure `cli/slurmx.py`. Replace the import block (lines 18-26) with:

```python
from cli import config_cmd as config_mod

# Every other subcommand reaches slurm_mcp, which does `from config import ...`
# at import time. On a checkout with no config.py that is a hard
# ModuleNotFoundError before argparse ever runs, so `slurmx --help` dies too.
# Degrade instead: offer the three subcommands that don't need a config, with
# `slurmx config` among them so the file can actually be created.
CONFIG_ONLY_HINT = (
    "config.py is missing, so only `slurmx config`, `slurmx setup`, and "
    "`slurmx update` are available. Run `slurmx config` to create it."
)

try:
    from cli import status as status_mod
    from cli import submit as submit_mod
    from cli import select_gpu as select_gpu_mod
    from cli import history as history_mod
    from cli import job_status as job_status_mod
    from cli import wait as wait_mod
    from cli import log as log_mod
    from cli import diagnose as diagnose_mod
    from cli import cancel as cancel_mod
    HAVE_CONFIG = True
except ModuleNotFoundError as e:
    if e.name != "config":
        raise
    HAVE_CONFIG = False
```

In `build_parser`, guard the config-dependent registrations. The `config`,
`setup`, and `update` registrations stay unconditional; wrap the other nine in
`if HAVE_CONFIG:` and append the hint to the parser description when it is
False:

```python
def build_parser() -> argparse.ArgumentParser:
    description = (
        "Cluster CLI: submit, monitor, and manage SLURM jobs. "
        "Use `slurmx <subcommand> --help` for per-command options."
    )
    if not HAVE_CONFIG:
        description = CONFIG_ONLY_HINT
    parser = argparse.ArgumentParser(prog="slurmx", description=description)
    subparsers = parser.add_subparsers(
        dest="subcommand", metavar="<subcommand>", required=True,
    )
    if HAVE_CONFIG:
        _add_subcommand(subparsers, "status", status_mod, aliases=("s",),
                        help="Live SLURM dashboard (scrollable; one-shot text when piped or --once).")
        # ... the other eight, unchanged ...
    _add_subcommand(
        subparsers, "config", config_mod,
        help="Edit config.py in a terminal form (--show prints it as text).",
    )
    _add_script_subcommand(subparsers, "setup", "setup.sh",
                           help="Run the project setup script (uv sync + symlink CLIs into ~/.local/bin/).")
    _add_script_subcommand(subparsers, "update", "update.sh",
                           help="Fast-forward git pull; re-runs uv sync if dependencies changed.")
    return parser
```

Update `setup.sh` lines 24-30:

```bash
# --- config.py warning ---
if [ ! -f "$REPO/config.py" ]; then
    echo
    echo "WARNING: config.py is missing. Create it with:"
    echo "    slurmx config"
fi
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/ -q -k "not live"`
Expected: PASS.

Then verify the degraded path for real, against a tree with no `config.py`:

```bash
SB=$(mktemp -d) && cd ~/.claude/mcp-servers/slurmx
for f in * ; do [ "$f" = config.py ] || ln -s "$PWD/$f" "$SB/$f"; done
cd "$SB" && ~/.claude/mcp-servers/slurmx/.venv/bin/python -c "
import sys; sys.path.insert(0, '$SB')
sys.modules.pop('config', None)
import importlib, importlib.abc
class B(importlib.abc.MetaPathFinder):
    def find_spec(self, n, p=None, t=None):
        if n == 'config': raise ModuleNotFoundError(\"No module named 'config'\")
sys.meta_path.insert(0, B())
from cli import slurmx
slurmx.build_parser().print_help()
"
rm -rf "$SB"
```

Expected: help text listing only `config`, `setup`, `update`, with the hint as
the description.

- [ ] **Step 5: Commit**

```bash
git add cli/config_cmd.py cli/config_form.py cli/slurmx.py tests/test_config_edit.py setup.sh
git commit -m "$(cat <<'EOF'
feat(cli): make `slurmx config` work on a checkout with no config.py

Every other subcommand reaches slurm_mcp, which imports `config` at module
load, so a missing config.py used to kill `slurmx --help` before argparse ran.
The dispatcher now degrades to config/setup/update with a hint, and `slurmx
config` offers the config-examples templates, copies the one you pick, and
opens the form on MAIL_USER.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: MAIL_USER default and docs

**Files:**
- Modify: `config-examples/default.py:15`, `config-examples/yisroel.py:13`
- Modify: `README.md`, `WELCOME.md`, `.gitignore`
- Test: `tests/test_config_edit.py`

**Interfaces:**
- Consumes: Tasks 1-9.
- Produces: nothing new.

- [ ] **Step 1: Write the failing tests**

```python
class TestTemplateDefaults:
    @pytest.mark.parametrize("path", TEMPLATES)
    def test_mail_user_defaults_to_the_bgu_address(self, path, monkeypatch):
        monkeypatch.setenv("USER", "someone")
        monkeypatch.delenv("SLURM_MAIL_USER", raising=False)
        ns = {}
        exec(compile(open(path).read(), path, "exec"), ns)
        assert ns["MAIL_USER"] == "someone@post.bgu.ac.il"

    @pytest.mark.parametrize("path", TEMPLATES)
    def test_env_still_overrides_mail_user(self, path, monkeypatch):
        monkeypatch.setenv("SLURM_MAIL_USER", "other@example.com")
        ns = {}
        exec(compile(open(path).read(), path, "exec"), ns)
        assert ns["MAIL_USER"] == "other@example.com"

    @pytest.mark.parametrize("path", TEMPLATES)
    def test_no_template_hardcodes_a_personal_address(self, path):
        assert "itayzloc" not in open(path).read()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_config_edit.py::TestTemplateDefaults -q`
Expected: FAIL — `default.py` yields `""` and `yisroel.py` yields the hardcoded
address.

- [ ] **Step 3: Write the implementation**

In `config-examples/default.py`, replace line 15:

```python
# --- Personal ---
# Defaults to $USER@post.bgu.ac.il; override per-shell with SLURM_MAIL_USER.
MAIL_USER = os.environ.get("SLURM_MAIL_USER", f"{USERNAME}@post.bgu.ac.il")
```

In `config-examples/yisroel.py`, replace line 13 with the same two lines
(dropping the `TODO: fill in your email` comment on line 12, since there is
nothing left to fill in), and update the module docstring's last line from
`Copy to config.py and fill in MAIL_USER.` to `Copy to config.py — MAIL_USER
defaults to $USER@post.bgu.ac.il.`

In `.gitignore`, add below the existing `config.py` line:

```
config.py.bak
config.py.tmp
```

In `README.md`, rewrite the Configuration section's opening so it leads with the
subcommand:

```markdown
## Configuration

Run `slurmx config` — a terminal form over `config.py`. It creates the file from
a template on first run, validates every field before it writes, and keeps your
comments and `os.environ.get` fallbacks intact (it replaces one literal, not the
file). `slurmx config --show` prints the resolved values instead, and that is
what you get automatically when the output is piped.

| Field | What to fill in |
|-------|----------------|
| `MAIL_USER` | Your cluster email for SLURM notifications. Defaults to `$USER@post.bgu.ac.il`. |
| `GOLDEN_QOS` | List of your QoS, e.g. `["yisroel"]` or `["yisroel", "shared"]`. First entry is primary for job submission. |
| `GPU_DEFINITIONS_BY_QOS` | Dict keyed by QoS name; each value is a list of `(name, display_name, vram_gb, golden_quota, golden_partition)` tuples for that QoS. |

A save writes `config.py.bak` first, so the previous version is always one `mv`
away. The form refuses to write a config whose primary QoS has no GPU cards:
`GPU_DEFINITIONS = GPU_DEFINITIONS_BY_QOS[GOLDEN_QOS[0]]` would raise at import
and take down every subcommand.
```

Keep the two paragraphs that follow (env override, gitignored + `config_defaults.py`).

Add to the README's CLI block, after the `slurmx cancel` line:

```bash
slurmx config                              # edit config.py in a terminal form
slurmx config --show                       # print resolved config as text
```

Add the same two lines to the README installation snippet in place of step 2's
`cp` commands:

```bash
# 2. Configure (creates config.py from a template, then opens the form)
cd ~/.claude/mcp-servers/slurmx
./setup.sh          # first, so the venv exists
slurmx config
```

In `WELCOME.md`, add to the CLI COMMANDS list after `slurmx cancel`:

```
                             slurmx config                edit config.py in a form
                                                          (--show prints it as text)
```

and change NEXT STEPS item 1 from `Verify config.py has your MAIL_USER and the
right GOLDEN_QOS list.` to `Run \`slurmx config\` to check MAIL_USER and your
GOLDEN_QOS list.`

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/ -q -k "not live"`
Expected: PASS, including `tests/test_slurm_mcp.py::TestConfigDefaults`.

Confirm the live config was not touched:

```bash
cd ~/.claude/mcp-servers/slurmx && git status --short config.py
```

Expected: no output (`config.py` is gitignored and unmodified).

- [ ] **Step 5: Commit and push**

```bash
git add config-examples/default.py config-examples/yisroel.py .gitignore \
        README.md WELCOME.md tests/test_config_edit.py
git commit -m "$(cat <<'EOF'
feat(config): default MAIL_USER to $USER@post.bgu.ac.il, document slurmx config

Both templates now derive MAIL_USER from $USER instead of shipping empty
(default.py) or one person's address (yisroel.py, which every labmate who
copied it inherited). SLURM_MAIL_USER still overrides. README and WELCOME lead
with `slurmx config` for setup, and the backup files it writes are gitignored.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
git push origin main
```

---

## Self-Review

**Spec coverage.** Every spec section maps to a task: UX contract → 5-8; key map
→ 7; field schema → 2; card cells → 3; `cli/config_model.py` → 1-4;
`cli/config_form.py` → 6-8; `cli/config_cmd.py` → 5, 9; save gate → 4; the third
blocker (empty primary group) → 4 (`test_primary_qos_without_cards_blocks`);
fresh-clone bootstrap → 9; MAIL_USER default → 10; the test list → spread across
1-10; docs → 10. The spec's group-header wording rule is covered by Task 6's
`test_single_qos_header_omits_the_qos_name`.

**Naming consistency.** `ConfigDoc.set/revert/value/text_value/is_editable/
staged_names/dirty/groups/set_card/add_card/delete_card/cross_field_errors/
warnings/save`, `cm.load`, `cm.validate_file`, `cm.FIELDS`, `cm.CARD_CELLS`,
`cm.NEW_CARD`, `cf.Row`, `cf.FormState`, `cf.build_rows`, `cf.move`,
`cf.dispatch`, `cf.run_form`, `config_cmd.show_text/templates/_bootstrap/
add_arguments/run` are each defined in one task and used with the same spelling
after. `run_form` gains `start_field` in Task 9; Task 8 defines it without, and
Task 9 restates the full function rather than referring back.

**Known ordering wrinkle.** Task 1's Step 4 has to introduce `FIELDS` for its own
tests to pass, and Task 2's Step 3 replaces that tuple with the same entries plus
validators. That is deliberate: it keeps each task independently green.
