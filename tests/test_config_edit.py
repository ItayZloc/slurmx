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
