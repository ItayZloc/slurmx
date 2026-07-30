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
MAIL_TYPE = ["END", "FAIL"]
GOLDEN_QOS = ['alpha' , 'beta']
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


_MAIL_LINE = "MAIL_USER   =    'someone@example.com'   # inline comment"
ENV_WRAPPED = MANGLED.replace(
    _MAIL_LINE, 'MAIL_USER = os.environ.get("SLURM_MAIL_USER", "env@example.com")')
FSTRING_DEFAULT = MANGLED.replace(
    _MAIL_LINE,
    'MAIL_USER = os.environ.get("SLURM_MAIL_USER", f"{USERNAME}@post.bgu.ac.il")')


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
        doc = cm.load(write(tmp_path, ENV_WRAPPED))
        slot = doc.slots["MAIL_USER"]
        start, end = slot.span
        assert ENV_WRAPPED[start:end] == '"env@example.com"'
        assert slot.provenance == "env-default"
        assert slot.env_var == "SLURM_MAIL_USER"

    def test_fstring_env_default_is_still_editable(self, tmp_path):
        """The templates ship f"{USERNAME}@post.bgu.ac.il" as MAIL_USER's
        default. An f-string is not a literal, so an earlier version marked the
        field unsupported and the form refused to edit the one key every new
        user must set."""
        doc = cm.load(write(tmp_path, FSTRING_DEFAULT))
        slot = doc.slots["MAIL_USER"]
        assert slot.provenance == "env-default"
        assert doc.is_editable("MAIL_USER") is True
        assert FSTRING_DEFAULT[slot.span[0]:slot.span[1]] == \
            'f"{USERNAME}@post.bgu.ac.il"'
        assert doc.set("MAIL_USER", "me@example.com") is None
        assert 'os.environ.get("SLURM_MAIL_USER", "me@example.com")' in doc.render()

    @pytest.mark.parametrize("path", TEMPLATES)
    def test_every_editable_field_is_editable_in_both_templates(self, path):
        doc = cm.load(path)
        for f in cm.FIELDS:
            if f.kind in ("derived", "table"):
                continue
            assert doc.is_editable(f.name), \
                f"{f.name} is {doc.slots[f.name].provenance} in {path}"

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
        doc = cm.load(write(tmp_path, MANGLED.replace('MAIL_TYPE = ["END", "FAIL"]\n', "")))
        assert doc.slots["MAIL_TYPE"].provenance == "absent"
        assert doc.slots["MAIL_TYPE"].span is None

    def test_unsupported_shape_is_read_only_not_a_crash(self, tmp_path):
        text = MANGLED.replace("MAX_MEM_GB = 80", "MAX_MEM_GB = 40 + 40")
        doc = cm.load(write(tmp_path, text))
        assert doc.slots["MAX_MEM_GB"].provenance == "unsupported"
        assert doc.slots["MAX_MEM_GB"].span is None

    def test_default_mail_user_uses_the_bgu_domain(self, monkeypatch):
        monkeypatch.setenv("USER", "someone")
        assert cm.default_mail_user() == "someone@post.bgu.ac.il"


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
        doc = cm.load(write(tmp_path, ENV_WRAPPED))
        assert doc.set("MAIL_USER", "new@example.com") is None
        assert 'os.environ.get("SLURM_MAIL_USER", "new@example.com")' in doc.render()

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
        text = MANGLED.replace('MAIL_TYPE = ["END", "FAIL"]\n', "")
        doc = cm.load(write(tmp_path, text))
        assert doc.set("MAIL_TYPE", "END, FAIL") is None
        out = doc.render()
        assert out.endswith('MAIL_TYPE = ["END", "FAIL"]\n')
        assert "# --- Added by `slurmx config` ---" in out

    def test_absent_key_edited_twice_appends_once(self, tmp_path):
        text = MANGLED.replace('MAIL_TYPE = ["END", "FAIL"]\n', "")
        doc = cm.load(write(tmp_path, text))
        doc.set("MAIL_TYPE", "END")
        doc.set("MAIL_TYPE", "ALL")
        assert doc.render().count("MAIL_TYPE =") == 1

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
        ("MAIL_TYPE", "END,FAIL", True),
        ("MAIL_TYPE", "end, fail", True),
        ("MAIL_TYPE", "", True),
        ("MAIL_TYPE", "NONE", True),
        ("MAIL_TYPE", "SOMETIMES", False),
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


from cli import config_cmd


class TestShow:
    def test_show_text_lists_every_field_with_provenance(self, tmp_path):
        doc = cm.load(write(tmp_path, MANGLED))
        out = config_cmd.show_text(doc)
        assert "MAIL_USER" in out and "someone@example.com" in out
        assert "MAIL_TYPE" in out and "END, FAIL" in out
        assert "EXCLUDE_NODES" in out and "env-default" in out
        assert "GPU_DEFINITIONS" in out and "derived" in out
        assert "alpha" in out and "a_card" in out and "96" in out

    def test_show_text_marks_an_active_env_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SLURM_MAIL_USER", "over@example.com")
        doc = cm.load(write(tmp_path, ENV_WRAPPED))
        assert "SLURM_MAIL_USER=over@example.com" in config_cmd.show_text(doc)

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


from cli import config_form as cf


def state_for(tmp_path, text=MANGLED):
    path = write(tmp_path, text)
    return cf.FormState(doc=cm.load(path), path=path)


def flat(rows):
    return ["".join(t for t, _ in r.spans) for r in rows]


class TestRows:
    def test_every_scalar_field_gets_a_row(self, tmp_path):
        rows = cf.build_rows(state_for(tmp_path))
        # MAIL_TYPE is a checklist group, not a typed field.
        fields = [r.field for r in rows if r.kind in ("field", "mailgroup")]
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
        rows = cf.build_rows(st)
        assert rows[st.cursor].selectable is True
        assert st.cursor == max(i for i, r in enumerate(rows) if r.selectable)
        cf.dispatch(st, ord("g"))
        # USERNAME is derived and unselectable, so the top is MAIL_USER.
        assert cf.build_rows(st)[st.cursor].field == "MAIL_USER"

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


class TestDerivedDisplay:
    def test_form_and_show_agree_on_derived_values(self, tmp_path, monkeypatch):
        monkeypatch.setenv("USER", "someone")
        st = state_for(tmp_path)
        rows = {r.field: "".join(t for t, _ in r.spans)
                for r in cf.build_rows(st) if r.kind == "field"}
        assert "someone" in rows["USERNAME"]
        assert "1 cards (alpha)" in rows["GPU_DEFINITIONS"]
        assert "(unset)" not in rows["USERNAME"]
        assert "(unset)" not in rows["GPU_DEFINITIONS"]
        out = config_cmd.show_text(st.doc)
        assert "1 cards (alpha)" in out

    def test_derived_card_count_follows_the_stage(self, tmp_path):
        st = state_for(tmp_path)
        st.doc.add_card("alpha")
        assert st.doc.display_value("GPU_DEFINITIONS") == "2 cards (alpha)"


import subprocess
from unittest.mock import patch as mock_modules_patch


def mock_modules(mapping):
    return mock_modules_patch.dict(sys.modules, mapping)


class TestBootstrap:
    def test_templates_lists_both_examples(self):
        labels = [l for l, _ in config_cmd.templates()]
        assert "default" in labels and "yisroel" in labels

    def test_bootstrap_copies_the_chosen_template(self, tmp_path, capsys):
        path = str(tmp_path / "config.py")
        err = config_cmd._bootstrap(path, choose=lambda prompt: "1")
        assert err is None
        assert os.path.exists(path)
        assert "GPU_DEFINITIONS_BY_QOS" in open(path).read()

    def test_bootstrap_rejects_a_bad_choice(self, tmp_path, capsys):
        path = str(tmp_path / "config.py")
        err = config_cmd._bootstrap(path, choose=lambda prompt: "9")
        assert err is not None
        assert not os.path.exists(path)

    def test_bootstrap_aborts_on_empty_input(self, tmp_path, capsys):
        path = str(tmp_path / "config.py")
        assert config_cmd._bootstrap(path, choose=lambda prompt: "") is not None
        assert not os.path.exists(path)


# Runs in a fresh interpreter: blocking the `config` module in-process means
# popping and re-importing every cli.* and slurm_mcp module, which leaves a
# stale cli.slurmx object on the `cli` package and breaks whatever runs next.
# `name="config"` matters — cli/slurmx.py degrades only for that module and
# re-raises anything else, so a nameless error would exercise the wrong branch.
_DEGRADED_PROBE = r'''
import importlib.abc
import sys


class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "config":
            raise ModuleNotFoundError("No module named 'config'", name="config")
        return None


sys.meta_path.insert(0, Block())
from cli import slurmx

parser = slurmx.build_parser()
print(",".join(sorted(parser._subparsers._group_actions[0].choices)))
print("HINT_OK" if "slurmx config" in slurmx.CONFIG_ONLY_HINT else "HINT_BAD")
'''


class TestDegradedParser:
    def test_parser_offers_only_config_setup_update_without_config_py(self):
        proc = subprocess.run([sys.executable, "-c", _DEGRADED_PROBE],
                              cwd=REPO, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        names, hint = proc.stdout.strip().splitlines()
        assert names == "config,setup,update"
        assert hint == "HINT_OK"

    def test_full_parser_has_every_subcommand(self):
        from cli import slurmx as slurmx_cli
        parser = slurmx_cli.build_parser()
        names = set(parser._subparsers._group_actions[0].choices)
        assert {"status", "submit", "config", "cancel", "setup", "update"} <= names


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


class TestTicketsAndFixed:
    def test_table_header_and_errors_say_golden_tickets(self, tmp_path):
        st = state_for(tmp_path)
        head = next("".join(t for t, _ in r.spans)
                    for r in cf.build_rows(st) if r.kind == "thead")
        assert "tickets" in head
        assert "quota" not in head
        assert "golden tickets must be >= 0" in st.doc.set_card("alpha", 0, 3, "-1")

    def test_show_text_says_tickets(self, tmp_path):
        out = config_cmd.show_text(cm.load(write(tmp_path, MANGLED)))
        assert "tickets" in out
        assert "quota" not in out

    def test_fixed_facts_are_shown_but_not_config_keys(self, tmp_path):
        names = {n for n, _v, _h in cm.FIXED_FACTS}
        assert names == {"CPU_PARTITION", "CPU_QOS", "MAIN_PARTITION"}
        # Not in the schema at all, so they can never be staged or written.
        assert not (names & {f.name for f in cm.FIELDS})
        doc = cm.load(write(tmp_path, MANGLED))
        assert names.isdisjoint(doc.slots)

    def test_fixed_rows_render_unselectable(self, tmp_path):
        rows = cf.build_rows(state_for(tmp_path))
        fixed = [r for r in rows if r.kind == "fixed"]
        assert [r.field for r in fixed] == ["CPU_PARTITION", "CPU_QOS", "MAIN_PARTITION"]
        assert all(r.selectable is False for r in fixed)
        head = next(r for r in rows if r.kind == "fixed_head")
        assert "config_defaults.py" in "".join(t for t, _ in head.spans)
        assert "not part of config.py" in "".join(t for t, _ in head.spans)

    def test_fixed_values_come_from_config_defaults(self):
        import config_defaults
        assert dict((n, v) for n, v, _h in cm.FIXED_FACTS) == {
            "CPU_PARTITION": config_defaults.CPU_PARTITION,
            "CPU_QOS": config_defaults.CPU_QOS,
            "MAIN_PARTITION": config_defaults.MAIN_PARTITION,
        }

    def test_save_no_longer_requires_the_fixed_keys(self, tmp_path):
        # MANGLED has none of the three; a save must still go through.
        doc = cm.load(write(tmp_path, MANGLED))
        doc.set("MAX_MEM_GB", "64")
        assert doc.save() is None


class TestMailType:
    def test_events_are_upper_cased_on_the_way_in(self, tmp_path):
        doc = cm.load(write(tmp_path, MANGLED))
        assert doc.set("MAIL_TYPE", "end, fail") is None
        assert doc.value("MAIL_TYPE") == ["END", "FAIL"]
        assert 'MAIL_TYPE = ["END", "FAIL"]' in doc.render()

    def test_empty_is_allowed_and_means_no_mail(self, tmp_path):
        doc = cm.load(write(tmp_path, MANGLED))
        assert doc.set("MAIL_TYPE", "") is None
        assert doc.value("MAIL_TYPE") == []
        assert "MAIL_TYPE = []" in doc.render()

    def test_a_bogus_event_is_rejected(self, tmp_path):
        doc = cm.load(write(tmp_path, MANGLED))
        assert "not a SLURM mail event" in doc.set("MAIL_TYPE", "END,WHENEVER")
        assert doc.dirty is False

    @pytest.mark.parametrize("path", TEMPLATES)
    def test_templates_ship_mail_type(self, path):
        ns = {}
        exec(compile(open(path).read(), path, "exec"), ns)
        assert ns["MAIL_TYPE"] == ["END", "FAIL"]

    def test_config_defaults_falls_back_for_an_older_config(self):
        import importlib
        import types as _types
        import config as real_config
        stale = _types.ModuleType("config")
        try:
            with mock_modules({"config": stale}):
                mod = importlib.reload(sys.modules["config_defaults"])
                assert mod.MAIL_TYPE == ["END", "FAIL"]
        finally:
            sys.modules["config"] = real_config
            importlib.reload(sys.modules["config_defaults"])


class TestNonePartition:
    NONE_PART = MANGLED.replace('("a_card", "A Card", 96, 16, "a_part"),',
                                '("a_card", "A Card", 96, 16, None),')

    def test_a_none_partition_re_renders_as_python_none_not_json_null(self, tmp_path):
        doc = cm.load(write(tmp_path, self.NONE_PART))
        doc.set_card("alpha", 0, 2, "48")
        out = doc.render()
        assert "None)" in out
        assert "null" not in out
        assert cm.ConfigDoc(doc.path, out).groups()[0][1][0][4] is None

    def test_a_none_partition_table_still_saves(self, tmp_path):
        doc = cm.load(write(tmp_path, self.NONE_PART))
        doc.set_card("alpha", 0, 2, "48")
        assert doc.save() is None

    def test_clearing_the_partition_cell_stores_none(self, tmp_path):
        doc = cm.load(write(tmp_path, MANGLED))
        assert doc.set_card("alpha", 0, 4, "") is None
        assert dict(doc.groups())["alpha"][0][4] is None

    def test_a_none_partition_renders_blank_in_the_form(self, tmp_path):
        st = state_for(tmp_path, self.NONE_PART)
        row = next(r for r in cf.build_rows(st) if r.kind == "card")
        assert "None" not in "".join(t for t, _ in row.spans)

    def test_every_card_column_is_wider_than_its_header(self):
        """'tickets' is exactly 7 chars; a 7-wide column ran it into the next
        header with no gap."""
        for cell, width in zip(cm.CARD_CELLS, cf.CARD_W):
            assert width > len(cell.header), f"{cell.header} needs > {len(cell.header)}"


class TestMailChecklist:
    def cursor_kind(self, st, kind, field=None):
        st.cursor = next(i for i, r in enumerate(cf.build_rows(st))
                         if r.kind == kind and (field is None or r.field == field))
        return st

    def test_mail_type_is_a_collapsed_group_by_default(self, tmp_path):
        st = state_for(tmp_path)
        rows = cf.build_rows(st)
        group = next(r for r in rows if r.kind == "mailgroup")
        assert group.field == "MAIL_TYPE"
        assert "▸" in "".join(t for t, _ in group.spans)
        assert "END, FAIL" in "".join(t for t, _ in group.spans)
        assert not [r for r in rows if r.kind == "mailopt"]

    def test_enter_opens_the_checklist_with_every_event(self, tmp_path):
        st = self.cursor_kind(state_for(tmp_path), "mailgroup")
        cf.dispatch(st, ord("\n"))
        rows = cf.build_rows(st)
        opts = [r for r in rows if r.kind == "mailopt"]
        assert [r.field for r in opts] == list(cm.MAIL_EVENTS)
        assert "▾" in "".join(t for t, _ in
                              next(r for r in rows if r.kind == "mailgroup").spans)

    def test_checked_events_render_with_a_tick(self, tmp_path):
        st = self.cursor_kind(state_for(tmp_path), "mailgroup")
        cf.dispatch(st, ord("\n"))
        boxes = {r.field: "".join(t for t, _ in r.spans)
                 for r in cf.build_rows(st) if r.kind == "mailopt"}
        assert "[x]" in boxes["END"] and "[x]" in boxes["FAIL"]
        assert "[ ]" in boxes["BEGIN"] and "[ ]" in boxes["ALL"]
        assert "the job finishes" in boxes["END"]

    def test_space_and_enter_both_toggle_an_event(self, tmp_path):
        for key in (ord(" "), ord("\n")):
            st = self.cursor_kind(state_for(tmp_path), "mailgroup")
            st.mail_open = True
            self.cursor_kind(st, "mailopt", "BEGIN")
            cf.dispatch(st, key)
            assert st.doc.mail_events() == ["BEGIN", "END", "FAIL"]
            cf.dispatch(st, key)
            assert st.doc.mail_events() == ["END", "FAIL"]

    def test_events_are_written_in_canonical_order(self, tmp_path):
        st = state_for(tmp_path)
        st.doc.toggle_mail_event("FAIL")          # leaves END
        st.doc.toggle_mail_event("BEGIN")
        assert st.doc.mail_events() == ["BEGIN", "END"]
        assert 'MAIL_TYPE = ["BEGIN", "END"]' in st.doc.render()

    def test_checking_none_clears_everything_else(self, tmp_path):
        st = state_for(tmp_path)
        st.doc.toggle_mail_event("NONE")
        assert st.doc.mail_events() == ["NONE"]

    def test_checking_all_clears_everything_else(self, tmp_path):
        st = state_for(tmp_path)
        st.doc.toggle_mail_event("ALL")
        assert st.doc.mail_events() == ["ALL"]

    def test_checking_a_specific_event_clears_none_and_all(self, tmp_path):
        st = state_for(tmp_path)
        st.doc.toggle_mail_event("ALL")
        st.doc.toggle_mail_event("TIME_LIMIT")
        assert st.doc.mail_events() == ["TIME_LIMIT"]

    def test_unchecking_the_last_event_means_no_mail(self, tmp_path):
        st = state_for(tmp_path)
        st.doc.toggle_mail_event("END")
        st.doc.toggle_mail_event("FAIL")
        assert st.doc.mail_events() == []
        assert "MAIL_TYPE = []" in st.doc.render()
        group = next(r for r in cf.build_rows(st) if r.kind == "mailgroup")
        assert "(no mail)" in "".join(t for t, _ in group.spans)

    def test_toggling_marks_the_group_edited(self, tmp_path):
        st = state_for(tmp_path)
        st.doc.toggle_mail_event("BEGIN")
        group = next(r for r in cf.build_rows(st) if r.kind == "mailgroup")
        assert "edited" in "".join(t for t, _ in group.spans)

    def test_r_reverts_the_whole_checklist(self, tmp_path):
        st = self.cursor_kind(state_for(tmp_path), "mailgroup")
        st.doc.toggle_mail_event("ALL")
        cf.dispatch(st, ord("r"))
        assert st.doc.dirty is False
        assert st.doc.mail_events() == ["END", "FAIL"]

    def test_every_event_has_a_help_line(self):
        assert set(cm.MAIL_EVENT_HELP) == set(cm.MAIL_EVENTS)

    def test_movement_walks_into_and_out_of_an_open_checklist(self, tmp_path):
        st = self.cursor_kind(state_for(tmp_path), "mailgroup")
        cf.dispatch(st, ord("\n"))
        cf.dispatch(st, curses.KEY_DOWN)
        assert cf.build_rows(st)[st.cursor].kind == "mailopt"
        for _ in range(len(cm.MAIL_EVENTS)):
            cf.dispatch(st, curses.KEY_DOWN)
        assert cf.build_rows(st)[st.cursor].kind == "field"

    def test_a_toggled_checklist_survives_a_save(self, tmp_path):
        st = state_for(tmp_path)
        st.doc.toggle_mail_event("ALL")
        assert st.doc.save() is None
        assert 'MAIL_TYPE = ["ALL"]' in open(st.path).read()
        assert cm.load(st.path).mail_events() == ["ALL"]


class TestCursorSnap:
    def test_first_keypress_from_row_zero_does_not_skip_a_field(self, tmp_path):
        """The cursor starts on the blank row 0. Stepping from an unselectable
        row used to snap AND step, so the first field (MAIL_USER) was skipped."""
        st = state_for(tmp_path)
        assert st.cursor == 0
        cf.dispatch(st, curses.KEY_DOWN)
        assert cf.build_rows(st)[st.cursor].field == "MAIL_USER"
        cf.dispatch(st, curses.KEY_DOWN)
        assert cf.build_rows(st)[st.cursor].field == "MAIL_TYPE"

    def test_up_from_row_zero_also_snaps(self, tmp_path):
        st = state_for(tmp_path)
        cf.dispatch(st, curses.KEY_UP)
        assert cf.build_rows(st)[st.cursor].selectable is True

    def test_run_form_starts_on_the_first_selectable_row(self, tmp_path, monkeypatch):
        seen = {}

        def fake_wrapper(fn, state):
            seen["cursor"] = state.cursor
            seen["field"] = cf.build_rows(state)[state.cursor].field

        monkeypatch.setattr(cf.curses, "wrapper", fake_wrapper)
        cf.run_form(write(tmp_path, MANGLED))
        assert seen["field"] == "MAIL_USER"

    def test_run_form_honours_start_field(self, tmp_path, monkeypatch):
        seen = {}
        monkeypatch.setattr(cf.curses, "wrapper",
                            lambda fn, st: seen.update(
                                field=cf.build_rows(st)[st.cursor].field))
        cf.run_form(write(tmp_path, MANGLED), start_field="GOLDEN_QOS")
        assert seen["field"] == "GOLDEN_QOS"

    def test_the_mail_group_name_column_aligns_with_the_fields(self, tmp_path):
        rows = cf.build_rows(state_for(tmp_path))
        field = next("".join(t for t, _ in r.spans)
                     for r in rows if r.field == "MAIL_USER")
        group = next("".join(t for t, _ in r.spans)
                     for r in rows if r.kind == "mailgroup")
        assert field.index("MAIL_USER") == group.index("MAIL_TYPE")
