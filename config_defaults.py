"""Values that don't belong in config.py, and defaults for keys added later.

Two kinds of thing live here, both for the same reason: `config.py` is
gitignored, so it is never updated by `slurmx update` and holds whatever
template it was copied from.

1. **Fixed cluster facts.** Partition and QoS names are properties of the
   cluster, identical for every user, so they are not personal configuration.
   They are defined here in tracked code and read from nowhere else — a stale
   copy left behind in someone's config.py has no effect. An env var still
   overrides one for a single shell.

2. **Personal keys added after the first release.** A plain
   `from config import NEW_KEY` turns the next `slurmx update` into an
   ImportError for everyone who cloned earlier — that is exactly what
   MAIN_PARTITION did. Such keys are read through `_from_config` with a
   default instead. Only keys with no sensible default (MAIL_USER, GOLDEN_QOS)
   may be imported straight from `config`;
   tests/test_slurm_mcp.py::TestConfigDefaults holds that line.
"""

from __future__ import annotations

import os

try:
    import config
except ModuleNotFoundError:
    # A fresh clone has no config.py yet, and `slurmx config` has to run to
    # create it. Anything reading a personal key gets the default until then.
    config = None


def _from_config(name: str, default):
    return getattr(config, name, default) if config is not None else default


# --- Fixed cluster facts (never read from config.py) ----------------------- #

# Where CPU-only jobs land.
CPU_PARTITION = os.environ.get("SLURM_CPU_PARTITION", "cpu")
CPU_QOS = os.environ.get("SLURM_CPU_QOS", "normal")

# Shared, preemptible GPU pool: where non-golden jobs land, and (together with
# the golden partitions) what counts as reachable capacity. Added 2026-07-28.
MAIN_PARTITION = os.environ.get("SLURM_MAIN_PARTITION", "main")


# --- Personal keys added after the first release --------------------------- #

# SLURM mail events, passed through to `#SBATCH --mail-type`. An empty list or
# ["NONE"] drops both mail lines from the script. Added 2026-07-30; before that
# the script hardcoded ALL, which mailed on BEGIN and REQUEUE too.
# MAIL_TYPE_DEFAULT is exported separately so `slurmx config` can show a
# config.py that predates the key as having these ticked — which is what such a
# config actually gets at submit time — without reading the editing user's own
# config.py to find out.
MAIL_TYPE_DEFAULT = ["END", "FAIL"]
MAIL_TYPE = _from_config("MAIL_TYPE", list(MAIL_TYPE_DEFAULT))

# What an omitted `golden_only` becomes, for both submit_job and `slurmx submit`.
# Added 2026-08-03; before that the golden-only default was hardcoded, which is
# what GOLDEN_POLICY_DEFAULT preserves.
#   golden_only  preemption-immune; queue on golden rather than downgrade
#   allow_main   golden first, then the preemptible main pool
#   ask          refuse to guess — the caller has to choose per job
GOLDEN_POLICIES = ("golden_only", "allow_main", "ask")
GOLDEN_POLICY_DEFAULT = "golden_only"
_golden_policy = _from_config("GOLDEN_POLICY", GOLDEN_POLICY_DEFAULT)
# A hand-edited typo falls back to the safest option instead of silently putting
# jobs on the preemptible pool. `slurmx config` warns when that happens.
GOLDEN_POLICY = (_golden_policy if _golden_policy in GOLDEN_POLICIES
                 else GOLDEN_POLICY_DEFAULT)
