# `slurmx config` — terminal config editor

Design for a `slurmx config` subcommand that edits `config.py` from the terminal
as a full-screen curses form, in the same visual family as `slurmx status`.

Status: approved 2026-07-30, implemented the same day.

**Amended after implementation (2026-07-30).** The field schema below is the
as-approved one; three changes landed on top of it:

1. `CPU_PARTITION`, `CPU_QOS` and `MAIN_PARTITION` are no longer config keys at
   all. They are cluster facts, identical for every user, so they moved into
   `config_defaults.py` and are read from nowhere else — a copy left in a
   gitignored `config.py` is ignored. They are not displayed anywhere. The one
   exception is an upgrade guard: if a pre-existing `config.py` assigns one with
   a value that *differs* from the fixed one, `warnings()` reports it and names
   the env var that restores the old behaviour, so the change can't be silent.
2. New personal key `MAIL_TYPE` (list of `sbatch --mail-type` events, default
   `["END", "FAIL"]`), replacing the hardcoded `--mail-type=ALL`. An empty list
   or `["NONE"]`, or an empty `MAIL_USER`, drops both mail lines from the script.
   It is the one field that is not typed: the vocabulary is closed, so the form
   shows a fold-out checklist (`⏎` opens, space or `⏎` ticks an event, `r`
   reverts the lot). Ticking `NONE` or `ALL` clears every other event, and
   ticking a specific event clears those two.
3. The card table's fourth column is "golden tickets", not "quota" (display
   only; `GPUType.golden_quota` keeps its name).

## Why

`config.py` is per-user and gitignored. Today you edit it by hand: `cp
config-examples/default.py config.py`, open an editor, hope you got the tuple
arity right in `GPU_DEFINITIONS_BY_QOS`. There is no validation until the next
`slurmx` invocation, and a typo takes down every subcommand at import time
(`from config import ...` runs before argparse). On a fresh clone with no
`config.py` at all, even `slurmx --help` raises `ModuleNotFoundError: No module
named 'config'`; `setup.sh` only prints a warning with two `cp` lines.

## Constraints discovered before designing

1. **`config.py` is code, not data.** Values come in four shapes:
   - plain literal: `MAX_MEM_GB = 80`
   - env wrapper: `MAIN_PARTITION = os.environ.get("SLURM_MAIN_PARTITION", "main")`
   - comprehension over an env wrapper: `EXCLUDE_NODES = [n.strip() for n in
     os.environ.get("SLURM_EXCLUDE_NODES", _EXCLUDE_NODES_DEFAULT).split(",") ...]`
   - derived: `GPU_DEFINITIONS = GPU_DEFINITIONS_BY_QOS[GOLDEN_QOS[0]]`

   Regenerating the file from a template destroys hand comments and env
   fallbacks, so writes must splice in place.

2. **A bad `GOLDEN_QOS[0]` bricks the whole CLI.** The derived line raises
   `KeyError` when the primary QoS has no group in `GPU_DEFINITIONS_BY_QOS`, and
   that import runs for every subcommand.

3. **House pattern for curses** (`cli/watch.py`): pure `build_*` / `clamp_*`
   helpers get unit tests, the curses loop is thin glue, colors live in
   `cli/theme.py`.

4. **Non-TTY must not block** (`cli/status.py`): piped, redirected, or run from
   an agent's Bash, a subcommand prints one-shot text and exits.

## UX contract

    slurmx config              # curses form (interactive terminal)
    slurmx config --show       # resolved values as plain text, no form
    slurmx config | cat        # same as --show (non-TTY auto-routing)

Layout, as approved:

    ┌─ slurmx config ─────────────────────────── config.py ─┐
    │                                                       │
    │   MAIL_USER        itayzloc@post.bgu.ac.il            │
    │   GOLDEN_QOS       yisroel, shared            edited  │
    │   CPU_PARTITION    cpu                                │
    │   MAIN_PARTITION   main                     default   │
    │ ▸ MAX_MEM_GB       64▏                        edited  │
    │   CPU_MEM          16G                                │
    │   TIME_LIMIT       7-0:00:00                          │
    │   EXCLUDE_NODES    (none)                             │
    │                                                       │
    │ ▾ GPU cards (6)                                       │
    │     name          vram  quota  golden partition       │
    │     rtx_pro_6000    96     16  rtx_pro_6000           │
    │     rtx_6000        48     12  rtx6000                │
    │     + add card                                        │
    │                                                       │
    ├───────────────────────────────────────────────────────┤
    │ ↑↓ move  ⏎ edit  a add  d delete  s save  q quit      │
    │ 2 unsaved changes · MAX_MEM_GB must be an int         │
    └───────────────────────────────────────────────────────┘

### Key map

| Key | Action |
|-----|--------|
| `↑` `↓` `j` `k`, PgUp/PgDn, `g` `G` | move the cursor |
| `Enter` | field row → inline edit; group header → fold/unfold; `+ add card` → append a row |
| `←` `→` | on a card row, select the cell (name / display / vram / quota / partition) |
| `Esc` | cancel the inline edit, keep the old value |
| `a` | add a card to the group under the cursor |
| `d` | delete the card under the cursor (`d` twice to confirm) |
| `r` | revert the field under the cursor to its on-disk value |
| `s` | save |
| `q`, Ctrl-C | quit (when dirty: `q` twice to discard) |

Inline editing is a single-line in-place entry prefilled with the current value
rendered as text (`yisroel, shared` for lists, `96` for ints). Supports
Backspace, Delete, `←` `→`, Home, End. `curs_set(1)` while editing, `0`
otherwise. No readline dependency.

### Field schema

| Field | Type | Validator | Notes |
|-------|------|-----------|-------|
| `USERNAME` | str | — | read-only, tagged `derived` |
| `MAIL_USER` | str | non-empty, contains `@` | empty/absent prefills `$USER@post.bgu.ac.il` |
| `GOLDEN_QOS` | list[str] | ≥1 token, no whitespace in a token | first entry is primary |
| `CPU_PARTITION` | str | non-empty, no whitespace | |
| `CPU_QOS` | str | non-empty, no whitespace | |
| `MAIN_PARTITION` | str | non-empty, no whitespace | env wrapper |
| `EXCLUDE_NODES` | list[str] | 0+ tokens, no whitespace in a token | comprehension |
| `MAX_MEM_GB` | int | `> 0` | |
| `CPU_CPUS` | int | `> 0` | |
| `CPU_MEM` | str | `^\d+[KMGT]?$` | SLURM mem format |
| `TIME_LIMIT` | str | parses under `maintenance._parse_slurm_time` | `D-HH:MM:SS` |
| `START_TIMEOUT` | int | `> 0` | seconds |
| `GPU_DEFINITIONS_BY_QOS` | table | per-cell, below | one fold group per QoS key |
| `GPU_DEFINITIONS` | derived | — | read-only, tagged `derived` |

Card cells: `name` non-empty with no whitespace or comma; `display` non-empty;
`vram` int `> 0`; `quota` int `>= 0`; `partition` non-empty, no whitespace.

A QoS key listed in `GOLDEN_QOS` but absent from the table renders as an empty
group with a `+ add card` row, so it is fillable from inside the form.

The group header reads `GPU cards (N)` when there is one QoS and `GPU cards ·
<qos> (N)` when there is more than one, so the single-QoS case (the common one)
stays uncluttered.

## Components

### `cli/config_model.py`

Pure. Imports `ast`, `os`, `re`, `shutil`, `subprocess`. Must **not** import
`slurm_mcp` or the `config` module (section "Fresh-clone bootstrap" depends on
this).

- `FIELDS` — the schema above as an ordered tuple of field descriptors (name,
  kind, validator, help text).
- `load(path) -> ConfigDoc` — read the file text, `ast.parse` it, and record per
  field the **exact source span of the literal to replace**:
  - plain literal → the literal's own span
  - `os.environ.get(ENV, DEFAULT)` → `DEFAULT`'s span, so the env override keeps
    working
  - `EXCLUDE_NODES` comprehension → the `_EXCLUDE_NODES_DEFAULT = ""` span
  - key absent → provenance `absent`, no span; a later edit appends a new
    assignment at the end of the file
  Each field also carries provenance for display: `file`, `env-default`,
  `absent (default)`, `derived`.
- `set(field, raw_text) -> None | error str` — validate, then stage the rendered
  literal text. Invalid input is rejected and nothing is staged.
- `render() -> str` — original bytes with only the staged spans swapped. With
  nothing staged the output is byte-identical to the input.
- `cross_field_errors() -> list[str]` — see the save gate.
- `validate(text) -> None | error str` — exec the candidate text in a
  **subprocess**, assert the required names load and `GPU_DEFINITIONS` resolves.
  A subprocess so a syntax error or `KeyError` can't poison the running form and
  so the check sees a clean interpreter.
- `save(path) -> None | error str` — the sequence in "Save gate".

### `cli/config_form.py`

Curses. Split like `cli/watch.py`:

- `build_rows(doc, cursor, folds, editing) -> list[list[(text, Role)]]` — pure,
  the whole visible buffer as role-tagged spans.
- `move(rows, cursor, key) -> cursor` and `dispatch(state, key) -> state` — pure
  reducers over cursor position, folds, staged edits, and the confirm latches for
  `d` and `q`.
- `run_form(path)` — `curses.wrapper` glue. Raises `curses.error` when the
  terminal can't host curses, matching `watch.run_tui`, so the caller falls back
  to text.

Adds Roles to `cli/theme.py` for: field name, value, `edited` tag, `derived` /
`default` tag, table header, error text in the status bar.

### `cli/config_cmd.py`

Named `config_cmd`, not `config`: `cli/config.py` would shadow the top-level
`config` module for anything doing `import config` after `sys.path` gains the
repo root.

- `add_arguments(parser)` — `--show`.
- `run(args)` — bootstrap if the file is missing, then route: TTY and not
  `--show` → `run_form`, falling back to the text dump on `curses.error`;
  otherwise print the text dump.
- Text dump: one line per field, `name  value  provenance`, GPU table as one row
  per card. Plain, byte-stable, colorized through `cli/_style.py` only when
  stdout is a TTY.

## Save gate

Type errors never reach the model: they are rejected at `Enter` with the reason
in the status bar. Two cross-field conditions block `s` while they hold:

1. `GOLDEN_QOS[0]` has no group in `GPU_DEFINITIONS_BY_QOS` — the derived line
   would raise `KeyError` at import and break every subcommand. A **secondary**
   QoS with no group is a warning, not a block: the file still imports.
2. `GOLDEN_QOS[0]`'s group is empty. That imports fine but leaves
   `GPU_DEFINITIONS == []`, so `select_gpu` can never return a card and every
   GPU submission fails.
3. Duplicate card names inside one group — `GPU_BY_NAME` silently drops one.

Save sequence:

1. `render()` the candidate text.
2. Write it to `config.py.tmp`.
3. `validate()` it in a subprocess. On failure: delete the tmp, keep the form
   open, put the exception in the status bar.
4. `shutil.copy2(config.py, config.py.bak)`.
5. `os.replace(config.py.tmp, config.py)`.
6. Clear dirty. Status bar: saved, backup written, and that a running Claude
   Code session's MCP server still holds the old config until it is restarted or
   reconnected via `/mcp`.

`config.py.bak` is a single rolling backup, overwritten each save. It is
gitignored along with `config.py`.

## Fresh-clone bootstrap

With no `config.py`, `slurmx config` prints a numbered plain-text template
picker (`config-examples/default.py`, `config-examples/yisroel.py`), copies the
choice, then opens the form with the cursor on `MAIL_USER`.

For that path to be reachable, `cli/slurmx.py` catches
`ModuleNotFoundError` for the `config` module around its subcommand imports. In
that state it registers only `config`, `setup`, and `update`, and prints a
one-line hint naming `slurmx config`. This works because `cli/config_model.py`
and `cli/config_cmd.py` never import `slurm_mcp`.

`setup.sh`'s missing-config warning changes from two `cp` lines to `slurmx
config`.

## MAIL_USER default

`config-examples/default.py` currently ships `MAIL_USER =
os.environ.get("SLURM_MAIL_USER", "")` and `config-examples/yisroel.py`
hardcodes one person's address, which every labmate who copies it inherits. Both
become:

```python
MAIL_USER = os.environ.get("SLURM_MAIL_USER", f"{USERNAME}@post.bgu.ac.il")
```

The form applies the same default as the prefill when `MAIL_USER` is empty or
absent. The existing user's live `config.py` is not touched: it already holds a
correct explicit address.

## Tests

New file `tests/test_config_edit.py`:

- **Round-trip** — `load()` then `render()` with nothing staged is byte-identical,
  for both templates and for a hand-mangled file with odd spacing and inline
  comments.
- **Splice kinds** — a plain literal, an env-wrapper default, the
  `EXCLUDE_NODES` comprehension, and an absent key (appended). Each asserts the
  surrounding comments and the env call survive verbatim.
- **Read-only** — `set()` on `USERNAME` or `GPU_DEFINITIONS` is rejected.
- **Validators** — one accept and one reject per field, plus every card cell.
- **Cross-field** — the primary-QoS block, the secondary-QoS warning, and the
  duplicate-card-name block.
- **Save** — backup written, `os.replace` atomicity (no partial file on a
  validation failure), tmp cleaned up on failure.
- **Subprocess validation** — a file whose `GOLDEN_QOS[0]` has no group is
  rejected with the `KeyError` surfaced.
- **`build_rows`** — layout for folded and unfolded groups, the `edited` and
  `derived` tags, the `+ add card` row, an empty group for a QoS with no cards.
- **Cursor / reducer** — movement skips over folded rows, `d` needs two presses,
  `q` needs two when dirty, `r` reverts one field, dirty count tracks staged
  edits.
- **Degraded parser** — with the `config` module blocked, `cli.slurmx`
  `build_parser()` still builds and exposes exactly `config`, `setup`, `update`.

No test drives the curses loop, matching the `cli/watch.py` precedent.
`tests/test_slurm_mcp.py::TestConfigDefaults` must stay green.

## Docs to update

- `README.md` — Configuration section leads with `slurmx config`; CLI list gains
  the subcommand; `MAIL_USER` row notes the new default.
- `WELCOME.md` — CLI list gains the subcommand.
- `setup.sh` — missing-config warning points at `slurmx config`.
- `.gitignore` — add `config.py.bak`.

## Size estimate

~610 lines of implementation (`config_model` ~230, `config_form` ~320,
`config_cmd` ~60), ~300 lines of tests.

## Out of scope

- `slurmx config get/set KEY VALUE` non-interactive writes. `--show` covers
  reading; nothing today needs scripted writes.
- Editing `maintenance.py`'s `WINDOWS`. Different file, different cadence.
- Multi-file or system-wide config. There is one `config.py` per checkout.
