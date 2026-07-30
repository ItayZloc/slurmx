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
