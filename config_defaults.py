"""Config keys added after the first release, with defaults for older configs.

`config.py` is per-user and gitignored, so every checkout keeps whatever
template it was copied from. A key added to `config-examples/` later is still
missing from every `config.py` already out there, and a plain
`from config import NEW_KEY` turns the next `slurmx update` into an ImportError
that takes down the whole CLI for everyone who cloned earlier. That is exactly
what MAIN_PARTITION did.

New keys go here with a default and are imported from this module instead. Only
keys with no sensible default (MAIL_USER, GOLDEN_QOS) may be imported straight
from `config`; tests/test_slurm_mcp.py::TestConfigDefaults holds that line.
"""

from __future__ import annotations

import os

import config

# Shared, preemptible GPU pool: where non-golden jobs land, and (together with
# the golden partitions) what counts as reachable capacity. Added 2026-07-28.
# The env fallback mirrors config-examples/default.py, so an override works even
# on a config.py that predates the key.
MAIN_PARTITION = getattr(
    config, "MAIN_PARTITION", os.environ.get("SLURM_MAIN_PARTITION", "main")
)
