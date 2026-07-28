"""
Comprehensive tests for slurm_mcp.py

Tests are split into:
  1. Unit tests with mocked SLURM commands (always runnable)
  2. Live integration tests against the actual cluster (marked with @live)

Run all tests:   python3 -m pytest tests/ -v
Run unit only:   python3 -m pytest tests/ -v -k "not live"
Run live only:   python3 -m pytest tests/ -v -k "live"
"""

import argparse
import os
import sys
import re
import subprocess
from types import SimpleNamespace
import pytest
from unittest.mock import patch, MagicMock

# Ensure we import from the local slurm_mcp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import slurm_mcp
from slurm_mcp import (
    GPU_TYPES, GPU_BY_NAME, GOLDEN_QOS, MAX_MEM_GB, MAIL_USER,
    GPUType, GPUAvailability, Availability, JobResult, JobStatus,
    check_availability, select_gpu, submit_job,
    my_jobs, cancel_jobs, _build_sbatch_script, _run, _run_quiet,
    get_job_status, read_job_log, wait_for_job, _wait_for_running,
    _FINISHED_STATES, _UNRECOVERABLE_REASONS, _QUOTA_REASONS,
    _QOS_QUOTA_REASONS, _USER_QUOTA_REASONS,
)
from slurm_mcp.availability import (
    _SINFO_FIELDS, _GOLDEN_FIELDS, _QUEUE_FIELDS, _node_is_usable,
)


def _fixed_width_row(values, fields) -> str:
    """Build one fixed-width `-O` row, padded/truncated exactly like sinfo/squeue.
    Rows are built from the real field widths so the mocks can't drift out of sync
    with the format strings the code actually sends."""
    return "".join(str(v).ljust(w)[:w] for v, (_, w) in zip(values, fields))


# ============================================================
# Fixtures: simulated SLURM outputs
# ============================================================

# Simulated squeue output for yisroel QoS (fixed-width, 20+60+60+12 chars)
MOCK_SQUEUE_GOLDEN = """\
itayzloc            gres/gpu:rtx_pro_6000:1                                     N/A                                                         RUNNING
itayzloc            gres/gpu:rtx_pro_6000:1                                     N/A                                                         RUNNING
pankajak            gres/gpu:rtx_pro_6000:1                                     N/A                                                         RUNNING
weissroy            N/A                                                         gres/gpu:rtx_6000:1                                         RUNNING
gressel             gres/gpu:rtx_pro_6000:2                                     N/A                                                         RUNNING
itayzloc            gres/gpu:rtx_pro_6000:1                                     N/A                                                         PENDING
"""

def _sinfo_row(node, gres, gres_used, partition="main", state="idle") -> str:
    """One fixed-width `sinfo -N -h -O` row."""
    return _fixed_width_row([node, gres, gres_used, partition, state], _SINFO_FIELDS)


def _golden_row(user, state, tres_job="N/A", tres_node="N/A", reason="") -> str:
    """One fixed-width `squeue --qos X -h -O` row (the pending/running counter)."""
    return _fixed_width_row(
        [user, tres_job, tres_node, state, reason], _GOLDEN_FIELDS
    )


def _queue_row(priority, job_id, user, tres_job="N/A", tres_node="N/A",
               name="job", reason="") -> str:
    """One fixed-width `squeue --qos X -t PENDING -h -O` row (the ordered queue)."""
    return _fixed_width_row(
        [priority, job_id, user, tres_job, tres_node, reason, name], _QUEUE_FIELDS
    )


# Node inventory the cluster assertions below are built on:
#   usable   gtx_1080     4 + 4 + 1 (cs-pheno-01) = 9 total, 3 used
#   usable   rtx_pro_6000 8 + 8 + 4 (golden-only partition) = 20 total, 14 used
#   usable   rtx_6000     4 + 4 = 8 total, 5 used
#   usable   rtx_4090     3 + 3 + 2 (backfill-planned) = 8 total, 3 used
#   usable   rtx_3090     8 + 1 (cs-pheno-01) = 9 total, 3 used
#   usable   rtx_2080     4 total, 0 used
#   offline  rtx_pro_6000 8 (not responding), rtx_4090 4 (down),
#            rtx_3090 8 (drained) + 4 (invalid registration)
#   ignored  rtx_2080     2 on a node only reachable via slurm-bridge
MOCK_SINFO = "\n".join([
    _sinfo_row("cs-1080-01", "gpu:gtx_1080:4", "gpu:gtx_1080:2(IDX:0-1)", state="mixed"),
    _sinfo_row("cs-1080-02", "gpu:gtx_1080:4", "gpu:gtx_1080:0"),
    _sinfo_row("ise-6000p-01", "gpu:rtx_pro_6000:8(S:0-1)", "gpu:rtx_pro_6000:6(IDX:0-5)", state="mixed"),
    _sinfo_row("ise-6000p-02", "gpu:rtx_pro_6000:8(S:0-1)", "gpu:rtx_pro_6000:8(IDX:0-7)", state="allocated"),
    _sinfo_row("ise-6000-01", "gpu:rtx_6000:4", "gpu:rtx_6000:4(IDX:0-3)", state="allocated"),
    _sinfo_row("ise-6000-02", "gpu:rtx_6000:4", "gpu:rtx_6000:1(IDX:0)", state="mixed"),
    _sinfo_row("cs-4090-01", "gpu:rtx_4090:3(S:0)", "gpu:rtx_4090:3(IDX:0-2)", state="allocated"),
    _sinfo_row("cs-4090-02", "gpu:rtx_4090:3(S:0)", "gpu:rtx_4090:0"),
    _sinfo_row("cs-3090-01", "gpu:rtx_3090:8(S:0)", "gpu:rtx_3090:2(IDX:0-1)", state="mixed"),
    _sinfo_row("cs-2080-01", "gpu:rtx_2080:4", "gpu:rtx_2080:0"),
    _sinfo_row("node-down-01", "gpu:rtx_4090:4", "gpu:rtx_4090:0", state="down"),
    _sinfo_row("node-drain-01", "gpu:rtx_3090:8", "gpu:rtx_3090:0", state="drained"),
    # slurmctld can't reach it — its 8 cards are unschedulable even though one
    # is still held by a job that hasn't finished cleaning up.
    _sinfo_row("node-noresp-01", "gpu:rtx_pro_6000:8", "gpu:rtx_pro_6000:1(IDX:1)",
               state="completing*"),
    # Registered with a config slurmctld rejects.
    _sinfo_row("node-inval-01", "gpu:rtx_3090:4", "gpu:rtx_3090:0", state="inval"),
    # '-' = earmarked by the backfill scheduler. Still allocatable.
    _sinfo_row("node-planned-01", "gpu:rtx_4090:2", "gpu:rtx_4090:0", state="mixed-"),
    # Two card types on one node — both have to be counted.
    _sinfo_row("cs-pheno-01", "gpu:rtx_3090:1(S:0),gpu:gtx_1080:1(S:0)",
               "gpu:rtx_3090:1(IDX:0),gpu:gtx_1080:1(IDX:1)", state="allocated"),
    # Reachable only through a golden partition, never through main.
    _sinfo_row("node-golden-01", "gpu:rtx_pro_6000:4", "gpu:rtx_pro_6000:0",
               partition="rtx_pro_6000"),
    # Reachable only through a queue we never submit to — not our capacity.
    _sinfo_row("node-bridge-01", "gpu:rtx_2080:2", "gpu:rtx_2080:0",
               partition="slurm-bridge"),
]) + "\n"

# Same node listed again under a second partition — sinfo -N does this for every
# multi-partition node, and the two rows must fold into one.
MOCK_SINFO_WITH_DUPES = MOCK_SINFO + "\n".join([
    _sinfo_row("cs-1080-01", "gpu:gtx_1080:4", "gpu:gtx_1080:2(IDX:0-1)",
               partition="gpu", state="mixed"),
    _sinfo_row("ise-6000p-01", "gpu:rtx_pro_6000:8(S:0-1)", "gpu:rtx_pro_6000:6(IDX:0-5)",
               partition="gpu", state="mixed"),
]) + "\n"


def mock_run_quiet_factory(squeue_output="", sinfo_output=""):
    """Create a _run_quiet mock that returns different output per command."""
    def mock_run_quiet(cmd):
        if cmd[0] == "squeue":
            return squeue_output
        elif cmd[0] == "sinfo":
            return sinfo_output
        return ""
    return mock_run_quiet


# ============================================================
# Unit Tests: GPU Definitions
# ============================================================

class TestGPUDefinitions:
    def test_all_gpu_types_defined(self):
        assert len(GPU_TYPES) == 6

    def test_gpu_types_sorted_by_vram(self):
        """GPU_TYPES should be ordered by VRAM descending."""
        vrams = [g.vram_gb for g in GPU_TYPES]
        assert vrams == sorted(vrams, reverse=True)

    def test_gpu_by_name_lookup(self):
        assert "rtx_pro_6000" in GPU_BY_NAME
        assert GPU_BY_NAME["rtx_pro_6000"].vram_gb == 96
        assert GPU_BY_NAME["gtx_1080"].vram_gb == 8

    def test_golden_gpu_types(self):
        golden = [g for g in GPU_TYPES if g.golden_quota > 0]
        assert len(golden) == 2
        names = {g.name for g in golden}
        assert names == {"rtx_pro_6000", "rtx_6000"}

    def test_golden_quotas(self):
        assert GPU_BY_NAME["rtx_pro_6000"].golden_quota == 16
        assert GPU_BY_NAME["rtx_6000"].golden_quota == 12

    def test_golden_partitions(self):
        assert GPU_BY_NAME["rtx_pro_6000"].golden_partition == "rtx_pro_6000"
        assert GPU_BY_NAME["rtx_6000"].golden_partition == "rtx6000"

    def test_non_golden_have_no_quota(self):
        # The four smaller cards are not owned (quota 0) but DO carry a dedicated
        # golden partition so golden_only=True can still force-target them.
        expected_partition = {
            "rtx_4090": "rtx4090", "rtx_3090": "rtx3090",
            "rtx_2080": "rtx2080", "gtx_1080": "gtx1080",
        }
        for name, part in expected_partition.items():
            assert GPU_BY_NAME[name].golden_quota == 0
            assert GPU_BY_NAME[name].golden_partition == part

    def test_all_cards_have_golden_partition(self):
        # Every card is golden_only-targetable.
        for g in GPU_TYPES:
            assert g.golden_partition, f"{g.name} missing golden_partition"


# ============================================================
# Unit Tests: check_availability (mocked)
# ============================================================

class TestCheckAvailabilityMocked:
    @patch("slurm_mcp.shell._run_quiet")
    def test_parses_golden_running_jobs(self, mock_rq):
        mock_rq.side_effect = mock_run_quiet_factory(
            squeue_output=MOCK_SQUEUE_GOLDEN, sinfo_output=MOCK_SINFO
        )
        avail = check_availability()

        # rtx_pro_6000 RUNNING: itayzloc 2 + pankajak 1 + gressel 2 = 5
        # rtx_pro_6000 PENDING: itayzloc 1
        assert avail.golden["rtx_pro_6000"].used == 5         # running only
        assert avail.golden["rtx_pro_6000"].running == 5
        assert avail.golden["rtx_pro_6000"].pending == 1
        assert avail.golden["rtx_pro_6000"].free == 16 - 5    # = 11

        # weissroy: 1 rtx_6000 RUNNING, no pending
        assert avail.golden["rtx_6000"].used == 1
        assert avail.golden["rtx_6000"].running == 1
        assert avail.golden["rtx_6000"].pending == 0
        assert avail.golden["rtx_6000"].free == 12 - 1        # = 11

    @patch("slurm_mcp.shell._run_quiet")
    def test_golden_users_tracked(self, mock_rq):
        mock_rq.side_effect = mock_run_quiet_factory(
            squeue_output=MOCK_SQUEUE_GOLDEN, sinfo_output=MOCK_SINFO
        )
        avail = check_availability()

        # `users` is back-compat alias for running_users only
        users = avail.golden["rtx_pro_6000"].users
        assert users["itayzloc"] == 2  # running only
        assert users["pankajak"] == 1
        assert users["gressel"] == 2

        # New: running_users + pending_users separated
        running = avail.golden["rtx_pro_6000"].running_users
        pending = avail.golden["rtx_pro_6000"].pending_users
        assert running == {"itayzloc": 2, "pankajak": 1, "gressel": 2}
        assert pending == {"itayzloc": 1}

        users_6000 = avail.golden["rtx_6000"].users
        assert users_6000["weissroy"] == 1
        assert avail.golden["rtx_6000"].pending_users == {}

    @patch("slurm_mcp.shell._run_quiet")
    def test_pending_jobs_tracked_separately(self, mock_rq):
        """PENDING jobs are tallied into `pending`/`pending_users`, not `used`."""
        mock_rq.side_effect = mock_run_quiet_factory(
            squeue_output=MOCK_SQUEUE_GOLDEN, sinfo_output=MOCK_SINFO
        )
        avail = check_availability()
        assert avail.golden["rtx_pro_6000"].pending == 1
        assert avail.golden["rtx_pro_6000"].pending_users == {"itayzloc": 1}
        # `used` excludes pending — quota is counted from running only
        assert avail.golden["rtx_pro_6000"].used == 5

    @patch("slurm_mcp.shell._run_quiet")
    def test_cluster_total_gpus(self, mock_rq):
        mock_rq.side_effect = mock_run_quiet_factory(
            squeue_output="", sinfo_output=MOCK_SINFO
        )
        avail = check_availability()

        # gtx_1080: 4 + 4 + 1 (cs-pheno-01's second card)
        assert avail.cluster["gtx_1080"].total == 9
        # rtx_pro_6000: 8 + 8 + 4 (golden-only node); not-responding node excluded
        assert avail.cluster["rtx_pro_6000"].total == 20
        # rtx_6000: 4 + 4
        assert avail.cluster["rtx_6000"].total == 8
        # rtx_4090: 3 + 3 + 2 (backfill-planned); down node excluded
        assert avail.cluster["rtx_4090"].total == 8
        # rtx_3090: 8 + 1 (cs-pheno-01); drained and inval nodes excluded
        assert avail.cluster["rtx_3090"].total == 9
        # rtx_2080: 4; the slurm-bridge-only node is not capacity we can use
        assert avail.cluster["rtx_2080"].total == 4

    @patch("slurm_mcp.shell._run_quiet")
    def test_cluster_allocated_gpus(self, mock_rq):
        mock_rq.side_effect = mock_run_quiet_factory(
            squeue_output="", sinfo_output=MOCK_SINFO
        )
        avail = check_availability()

        assert avail.cluster["gtx_1080"].used == 3     # 2 + 0 + 1 (cs-pheno-01)
        assert avail.cluster["rtx_pro_6000"].used == 14  # 6 + 8 + 0 (not-responding excluded)
        assert avail.cluster["rtx_6000"].used == 5     # 4 + 1
        assert avail.cluster["rtx_4090"].used == 3     # 3 + 0 + 0 (down excluded)
        assert avail.cluster["rtx_3090"].used == 3     # 2 + 1 (drained + inval excluded)
        assert avail.cluster["rtx_2080"].used == 0

    @patch("slurm_mcp.shell._run_quiet")
    def test_cluster_free_gpus(self, mock_rq):
        mock_rq.side_effect = mock_run_quiet_factory(
            squeue_output="", sinfo_output=MOCK_SINFO
        )
        avail = check_availability()

        assert avail.cluster["gtx_1080"].free == 6     # 9 - 3
        assert avail.cluster["rtx_pro_6000"].free == 6  # 20 - 14
        assert avail.cluster["rtx_6000"].free == 3      # 8 - 5
        assert avail.cluster["rtx_4090"].free == 5      # 8 - 3
        assert avail.cluster["rtx_3090"].free == 6      # 9 - 3
        assert avail.cluster["rtx_2080"].free == 4      # 4 - 0

    @patch("slurm_mcp.shell._run_quiet")
    def test_down_drained_nodes_excluded(self, mock_rq):
        mock_rq.side_effect = mock_run_quiet_factory(
            squeue_output="", sinfo_output=MOCK_SINFO
        )
        avail = check_availability()
        # node-down-01 has 4 rtx_4090, node-drain-01 has 8 rtx_3090.
        # Neither is capacity, so neither lands in `total`.
        assert "node-down-01" in avail.cluster["rtx_4090"].offline_nodes
        assert "node-drain-01" in avail.cluster["rtx_3090"].offline_nodes

    @patch("slurm_mcp.shell._run_quiet")
    def test_not_responding_node_excluded(self, mock_rq):
        """A node slurmctld can't reach can't run anything, whatever its base
        state says. sinfo flags that with a trailing '*' — this is the case that
        made slurmx report 18 free rtx_pro_6000 while sres reported 11."""
        mock_rq.side_effect = mock_run_quiet_factory(
            squeue_output="", sinfo_output=MOCK_SINFO
        )
        avail = check_availability()
        pro = avail.cluster["rtx_pro_6000"]
        assert pro.total == 20                     # node-noresp-01's 8 are not here
        assert pro.offline == 8
        assert pro.offline_nodes == ["node-noresp-01"]
        # Its 1 in-use card must not inflate `used` either.
        assert pro.used == 14

    @patch("slurm_mcp.shell._run_quiet")
    def test_invalid_registration_node_excluded(self, mock_rq):
        mock_rq.side_effect = mock_run_quiet_factory(
            squeue_output="", sinfo_output=MOCK_SINFO
        )
        avail = check_availability()
        assert "node-inval-01" in avail.cluster["rtx_3090"].offline_nodes
        assert avail.cluster["rtx_3090"].offline == 12   # 8 drained + 4 inval

    @patch("slurm_mcp.shell._run_quiet")
    def test_backfill_planned_node_still_counted(self, mock_rq):
        """'-' means the backfill scheduler has earmarked the node for a future
        job. It still accepts work, so it stays in the total."""
        mock_rq.side_effect = mock_run_quiet_factory(
            squeue_output="", sinfo_output=MOCK_SINFO
        )
        avail = check_availability()
        assert avail.cluster["rtx_4090"].total == 8      # includes node-planned-01
        assert "node-planned-01" not in avail.cluster["rtx_4090"].offline_nodes

    @patch("slurm_mcp.shell._run_quiet")
    def test_unsubmittable_partition_excluded(self, mock_rq):
        """A node we can only reach through some other group's queue is not our
        capacity — and it isn't 'offline' either, it's just not ours."""
        mock_rq.side_effect = mock_run_quiet_factory(
            squeue_output="", sinfo_output=MOCK_SINFO
        )
        avail = check_availability()
        assert avail.cluster["rtx_2080"].total == 4      # node-bridge-01's 2 excluded
        assert avail.cluster["rtx_2080"].offline == 0
        assert avail.cluster["rtx_2080"].offline_nodes == []

    @patch("slurm_mcp.shell._run_quiet")
    def test_golden_partition_only_node_counted(self, mock_rq):
        """Golden partitions are ours too, so a node reachable only there counts."""
        mock_rq.side_effect = mock_run_quiet_factory(
            squeue_output="", sinfo_output=MOCK_SINFO
        )
        avail = check_availability()
        assert avail.cluster["rtx_pro_6000"].total == 20  # 8 + 8 + node-golden-01's 4

    @patch("slurm_mcp.shell._run_quiet")
    def test_multi_gres_node_counts_every_card_type(self, mock_rq):
        """cs-pheno-01 carries one rtx_3090 AND one gtx_1080. Matching only the
        first entry hid 24 real gtx_1080s on this cluster."""
        mock_rq.side_effect = mock_run_quiet_factory(
            squeue_output="", sinfo_output=MOCK_SINFO
        )
        avail = check_availability()
        assert avail.cluster["gtx_1080"].total == 9      # 4 + 4 + 1
        assert avail.cluster["gtx_1080"].used == 3       # 2 + 0 + 1
        assert avail.cluster["rtx_3090"].total == 9      # 8 + 1

    @patch("slurm_mcp.shell._run_quiet")
    def test_healthy_cards_report_no_offline(self, mock_rq):
        mock_rq.side_effect = mock_run_quiet_factory(
            squeue_output="", sinfo_output=MOCK_SINFO
        )
        avail = check_availability()
        for name in ("rtx_6000", "rtx_2080", "gtx_1080"):
            assert avail.cluster[name].offline == 0
            assert avail.cluster[name].offline_nodes == []

    @patch("slurm_mcp.shell._run_quiet")
    def test_node_deduplication(self, mock_rq):
        mock_rq.side_effect = mock_run_quiet_factory(
            squeue_output="", sinfo_output=MOCK_SINFO_WITH_DUPES
        )
        avail = check_availability()
        # cs-1080-01 and ise-6000p-01 each appear under a second partition
        assert avail.cluster["gtx_1080"].total == 9        # not 4+4+4+1
        assert avail.cluster["rtx_pro_6000"].total == 20   # not 8+8+8+4

    @patch("slurm_mcp.shell._run_quiet")
    def test_empty_squeue_output(self, mock_rq):
        mock_rq.side_effect = mock_run_quiet_factory(
            squeue_output="", sinfo_output=MOCK_SINFO
        )
        avail = check_availability()
        assert avail.golden["rtx_pro_6000"].used == 0
        assert avail.golden["rtx_pro_6000"].free == 16
        assert avail.golden["rtx_6000"].used == 0
        assert avail.golden["rtx_6000"].free == 12

    @patch("slurm_mcp.shell._run_quiet")
    def test_empty_sinfo_output(self, mock_rq):
        mock_rq.side_effect = mock_run_quiet_factory(
            squeue_output=MOCK_SQUEUE_GOLDEN, sinfo_output=""
        )
        avail = check_availability()
        # Cluster should all be 0
        for gpu in GPU_TYPES:
            assert avail.cluster[gpu.name].total == 0
            assert avail.cluster[gpu.name].free == 0

    @patch("slurm_mcp.shell._run_quiet")
    def test_tres_per_node_parsing(self, mock_rq):
        """Jobs that use --gpus-per-node show GPU in tres-per-node field."""
        squeue = (
            "weissroy            "
            "N/A                                                         "
            "gres/gpu:rtx_6000:2                                         "
            "RUNNING     \n"
        )
        mock_rq.side_effect = mock_run_quiet_factory(
            squeue_output=squeue, sinfo_output=""
        )
        avail = check_availability()
        assert avail.golden["rtx_6000"].used == 2
        assert avail.golden["rtx_6000"].users["weissroy"] == 2

    @patch("slurm_mcp.shell._run_quiet")
    def test_golden_by_qos_populated(self, mock_rq):
        """The single configured QoS shows up under golden_by_qos."""
        mock_rq.side_effect = mock_run_quiet_factory(
            squeue_output=MOCK_SQUEUE_GOLDEN, sinfo_output=MOCK_SINFO
        )
        avail = check_availability()
        assert slurm_mcp.PRIMARY_QOS in avail.golden_by_qos
        # `golden` alias matches the primary entry in golden_by_qos
        primary = avail.golden_by_qos[slurm_mcp.PRIMARY_QOS]
        assert avail.golden == primary

    @patch("slurm_mcp.shell._run_quiet")
    def test_multi_qos_iterates_squeue(self, mock_rq):
        """With two configured QoS, squeue is called once per QoS and
        golden_by_qos has an entry for each."""
        called_qos = []

        def fake_run_quiet(cmd):
            if cmd[0] == "squeue":
                # The QoS argument is the value after --qos
                if "--qos" in cmd:
                    called_qos.append(cmd[cmd.index("--qos") + 1])
                return MOCK_SQUEUE_GOLDEN
            return ""

        mock_rq.side_effect = fake_run_quiet

        # Patch the bindings actually read by availability.check_availability:
        # both GOLDEN_QOS and GPU_TYPES_BY_QOS are imported as module-level
        # names inside slurm_mcp.availability.
        extra_qos = "secondary"
        extra_types = list(slurm_mcp.GPU_TYPES_BY_QOS[slurm_mcp.PRIMARY_QOS])
        with patch.object(slurm_mcp.availability, "GOLDEN_QOS",
                          [slurm_mcp.PRIMARY_QOS, extra_qos]), \
             patch.dict(slurm_mcp.availability.GPU_TYPES_BY_QOS,
                        {extra_qos: extra_types}):
            avail = check_availability()

        assert called_qos == [slurm_mcp.PRIMARY_QOS, extra_qos]
        assert set(avail.golden_by_qos.keys()) == {
            slurm_mcp.PRIMARY_QOS, extra_qos,
        }


# ============================================================
# Unit Tests: node usability predicate
# ============================================================

class TestNodeIsUsable:
    """`sinfo` StateLong strings, and whether that node can take a new job."""

    @pytest.mark.parametrize("state", [
        "idle", "mixed", "allocated", "completing", "reserved",
        "mixed-",       # earmarked by the backfill scheduler
        "idle~",        # powered down for power save, resumes on demand
        "idle#",        # powering up
        "ALLOCATED",    # casing shouldn't matter
    ])
    def test_usable_states(self, state):
        assert _node_is_usable(state) is True

    @pytest.mark.parametrize("state", [
        "down", "down*", "drained", "draining",
        "inval", "invalid_reg",
        "fail", "failing",
        "unknown", "unk",
        "future",
        "maint",
        "reboot", "reboot_issued", "reboot_requested",
        "powering_down", "powered_down", "power_down",
        "completing*",  # base state fine, but slurmctld can't reach it
        "mixed*",
        "idle*",
    ])
    def test_unusable_states(self, state):
        assert _node_is_usable(state) is False

    @pytest.mark.parametrize("word,flag", [
        ("maint", "mixed$"),            # busy node in a maintenance reservation
        ("maint", "allocated$"),
        ("reboot", "idle@"),            # reboot requested
        ("reboot^", "mixed^"),          # reboot issued
    ])
    def test_flag_form_matches_word_form(self, word, flag):
        """Slurm renders the same condition as a word or a trailing flag
        depending on the base state — 'maint' when idle, 'mixed$' when busy.
        Both have to be rejected, or the answer would depend on whether a job
        happened to be running on the node."""
        assert _node_is_usable(word) is False
        assert _node_is_usable(flag) is False


# ============================================================
# Unit Tests: select_gpu (mocked)
# ============================================================

class TestSelectGPUMocked:
    def _make_avail(self, golden_free=None, cluster_free=None):
        """Helper to build a mock Availability."""
        golden_free = golden_free or {}
        cluster_free = cluster_free or {}
        avail = Availability()
        for gpu in GPU_TYPES:
            if gpu.golden_quota > 0:
                free = golden_free.get(gpu.name, 0)
                avail.golden[gpu.name] = GPUAvailability(
                    gpu.name, gpu.golden_quota, gpu.golden_quota - free, free
                )
            total = cluster_free.get(gpu.name, 0) + 10
            free = cluster_free.get(gpu.name, 0)
            avail.cluster[gpu.name] = GPUAvailability(
                gpu.name, total, total - free, free
            )
        return avail

    @patch("slurm_mcp.availability.check_availability")
    def test_selects_golden_rtx_6000_for_48gb(self, mock_avail):
        mock_avail.return_value = self._make_avail(
            golden_free={"rtx_pro_6000": 5, "rtx_6000": 3},
            cluster_free={"rtx_4090": 10},
        )
        result = select_gpu(48)
        assert result == ("rtx_6000", "rtx6000", "yisroel")

    @patch("slurm_mcp.availability.check_availability")
    def test_selects_golden_rtx_pro_for_96gb(self, mock_avail):
        mock_avail.return_value = self._make_avail(
            golden_free={"rtx_pro_6000": 5, "rtx_6000": 3},
        )
        result = select_gpu(96)
        assert result == ("rtx_pro_6000", "rtx_pro_6000", "yisroel")

    @patch("slurm_mcp.availability.check_availability")
    def test_falls_back_to_golden_pro_when_6000_full(self, mock_avail):
        mock_avail.return_value = self._make_avail(
            golden_free={"rtx_pro_6000": 5, "rtx_6000": 0},
        )
        result = select_gpu(48)
        # rtx_6000 golden is full, should try rtx_pro_6000 golden
        assert result == ("rtx_pro_6000", "rtx_pro_6000", "yisroel")

    @patch("slurm_mcp.availability.check_availability")
    def test_falls_back_to_cluster_when_golden_full(self, mock_avail):
        mock_avail.return_value = self._make_avail(
            golden_free={"rtx_pro_6000": 0, "rtx_6000": 0},
            cluster_free={"rtx_6000": 5},
        )
        result = select_gpu(48)
        assert result == ("rtx_6000", "main", "normal")

    @patch("slurm_mcp.availability.check_availability")
    def test_cluster_picks_smallest_gpu(self, mock_avail):
        mock_avail.return_value = self._make_avail(
            golden_free={"rtx_pro_6000": 0, "rtx_6000": 0},
            cluster_free={"rtx_4090": 5, "rtx_3090": 5, "rtx_pro_6000": 5},
        )
        result = select_gpu(24)
        # rtx_3090 and rtx_4090 both have 24GB, sorted by (vram, name)
        # rtx_3090 < rtx_4090 alphabetically
        assert result[0] in ("rtx_3090", "rtx_4090")
        assert result[1] == "main"
        assert result[2] == "normal"

    @patch("slurm_mcp.availability.check_availability")
    def test_returns_none_when_nothing_available(self, mock_avail):
        mock_avail.return_value = self._make_avail(
            golden_free={"rtx_pro_6000": 0, "rtx_6000": 0},
            cluster_free={},  # all zero
        )
        result = select_gpu(48)
        assert result is None

    def test_returns_none_for_impossible_vram(self):
        result = select_gpu(200)
        assert result is None

    @patch("slurm_mcp.availability.check_availability")
    def test_8gb_prefers_golden(self, mock_avail):
        """Even for 8GB, golden is preferred over public queue."""
        mock_avail.return_value = self._make_avail(
            golden_free={"rtx_pro_6000": 0, "rtx_6000": 3},
            cluster_free={"gtx_1080": 10},
        )
        result = select_gpu(8)
        assert result == ("rtx_6000", "rtx6000", "yisroel")


class TestSelectGPUGoldenOnly:
    """golden_only=True forces a dedicated golden partition, no availability check."""

    def test_golden_only_48gb(self):
        assert select_gpu(48, golden_only=True) == ("rtx_6000", "rtx6000", "yisroel")

    def test_golden_only_96gb(self):
        assert select_gpu(96, golden_only=True) == (
            "rtx_pro_6000", "rtx_pro_6000", "yisroel",
        )

    def test_golden_only_24gb_targets_small_card_partition(self):
        gpu, part, qos = select_gpu(24, golden_only=True)
        assert gpu in ("rtx_3090", "rtx_4090")
        assert part in ("rtx3090", "rtx4090")
        assert qos == "yisroel"

    def test_golden_only_does_not_query_availability(self):
        # golden_only must return BEFORE calling check_availability.
        with patch("slurm_mcp.availability.check_availability",
                   side_effect=AssertionError("should not be called")):
            gpu, part, qos = select_gpu(8, golden_only=True)
        assert qos == "yisroel"
        assert part == "gtx1080"

    def test_golden_only_impossible_vram(self):
        assert select_gpu(200, golden_only=True) is None


# ============================================================
# Unit Tests: _build_sbatch_script
# ============================================================

class TestBuildSbatchScript:
    def test_basic_script(self):
        script = _build_sbatch_script(
            cmd="python train.py",
            partition="rtx_pro_6000",
            qos="yisroel",
            gpu_type="rtx_pro_6000",
            num_gpus=1,
            job_name="test-job",
            output_path="./slurm-test-job-%J.out",
            workdir=None,
        )
        assert "#!/bin/bash" in script
        assert "#SBATCH --partition rtx_pro_6000" in script
        assert "#SBATCH --qos=yisroel" in script
        assert "#SBATCH --gres=gpu:rtx_pro_6000:1" in script
        assert "#SBATCH --nodes=1" in script
        assert "#SBATCH --mem=80G" in script
        assert "#SBATCH --job-name test-job" in script
        assert "#SBATCH --time 7-0:00:00" in script
        assert "#SBATCH --output ./slurm-test-job-%J.out" in script
        assert f"#SBATCH --mail-user={MAIL_USER}" in script
        assert "#SBATCH --mail-type=ALL" in script
        assert "python train.py" in script

    def test_scratch_dir_setup(self):
        script = _build_sbatch_script(
            cmd="python train.py", partition="main", qos="normal",
            gpu_type="rtx_4090", num_gpus=1, job_name="test",
            output_path="out.log", workdir=None,
        )
        assert "export SCRATCH_DIR=/scratch/$USER/$SLURM_JOB_ID" in script
        assert 'mkdir -p "$SCRATCH_DIR"' in script
        assert "trap" in script
        # Scratch setup should come before the command
        scratch_pos = script.index("SCRATCH_DIR")
        cmd_pos = script.index("python train.py")
        assert scratch_pos < cmd_pos

    def test_with_workdir(self):
        script = _build_sbatch_script(
            cmd="python train.py", partition="main", qos="normal",
            gpu_type="rtx_4090", num_gpus=1, job_name="test",
            output_path="out.log",
            workdir="/tmp/test-project",
        )
        assert "cd /tmp/test-project" in script
        cd_pos = script.index("cd /tmp/test-project")
        cmd_pos = script.index("python train.py")
        assert cd_pos < cmd_pos

    def test_multi_gpu(self):
        script = _build_sbatch_script(
            cmd="torchrun train.py", partition="rtx_pro_6000", qos="yisroel",
            gpu_type="rtx_pro_6000", num_gpus=4, job_name="multi",
            output_path="out.log", workdir=None,
        )
        assert "#SBATCH --gres=gpu:rtx_pro_6000:4" in script
        assert "#SBATCH --nodes=1" in script


# ============================================================
# Unit Tests: submit_job (mocked)
# ============================================================

class TestSubmitJobMocked:
    @patch("slurm_mcp.selection.select_gpu")
    def test_dry_run_returns_script(self, mock_select):
        mock_select.return_value = ("rtx_6000", "rtx6000", "yisroel")
        result = submit_job(
            cmd="python train.py",
            vram_gb=48,
            dry_run=True,
        )
        assert result.success is True
        assert result.job_id is None
        assert result.gpu_type == "rtx_6000"
        assert result.partition == "rtx6000"
        assert result.qos == "yisroel"
        assert "python train.py" in result.sbatch_script
        assert "#SBATCH --partition rtx6000" in result.sbatch_script

    @patch("slurm_mcp.selection.select_gpu")
    def test_dry_run_with_workdir(self, mock_select):
        mock_select.return_value = ("rtx_pro_6000", "rtx_pro_6000", "yisroel")
        result = submit_job(
            cmd="python train.py --lr 1e-4",
            vram_gb=96,
            workdir="/tmp/test-project",
            output_dir="/tmp/logs",
            job_name="train-bert",
            dry_run=True,
        )
        assert result.success is True
        assert "cd /tmp/test-project" in result.sbatch_script
        assert "/tmp/logs/slurm-train-bert-%J.out" in result.sbatch_script
        assert "#SBATCH --mem=80G" in result.sbatch_script
        assert "#SBATCH --time 7-0:00:00" in result.sbatch_script
        assert "SCRATCH_DIR" in result.sbatch_script

    def test_manual_gpu_type_override(self):
        result = submit_job(
            cmd="python eval.py",
            vram_gb=48,
            gpu_type="rtx_pro_6000",
            dry_run=True,
        )
        assert result.success is True
        assert result.gpu_type == "rtx_pro_6000"
        assert result.qos == "yisroel"
        assert result.partition == "rtx_pro_6000"

    def test_manual_gpu_type_with_qos_override(self):
        result = submit_job(
            cmd="python eval.py",
            vram_gb=24,
            gpu_type="rtx_4090",
            qos="normal",
            golden_only=False,  # qos override only applies in the fallback path
            dry_run=True,
        )
        assert result.success is True
        assert result.gpu_type == "rtx_4090"
        assert result.qos == "normal"
        assert result.partition == "main"

    def test_manual_gpu_type_invalid(self):
        result = submit_job(
            cmd="echo hi", vram_gb=8, gpu_type="rtx_9999", dry_run=True
        )
        assert result.success is False
        assert "Unknown GPU type" in result.message

    def test_manual_gpu_type_insufficient_vram(self):
        result = submit_job(
            cmd="echo hi", vram_gb=50, gpu_type="rtx_4090", dry_run=True
        )
        assert result.success is False
        assert "24GB VRAM" in result.message
        assert "50GB requested" in result.message

    @patch("slurm_mcp.selection.select_gpu")
    def test_vram_too_high_no_gpu_type_exists(self, mock_select):
        mock_select.return_value = None
        result = submit_job(cmd="echo hi", vram_gb=200, dry_run=True)
        assert result.success is False
        assert "No GPU type has >= 200GB VRAM" in result.message

    @patch("slurm_mcp.selection.select_gpu")
    def test_nothing_available_shows_availability(self, mock_select):
        mock_select.return_value = None
        with patch("slurm_mcp.check_availability") as mock_avail:
            avail = Availability()
            avail.golden["rtx_6000"] = GPUAvailability("rtx_6000", 12, 12, 0)
            avail.golden["rtx_pro_6000"] = GPUAvailability("rtx_pro_6000", 16, 16, 0)
            avail.cluster["rtx_6000"] = GPUAvailability("rtx_6000", 100, 100, 0)
            avail.cluster["rtx_pro_6000"] = GPUAvailability("rtx_pro_6000", 40, 40, 0)
            mock_avail.return_value = avail

            result = submit_job(cmd="echo hi", vram_gb=48, dry_run=True)
            assert result.success is False
            assert "currently free" in result.message
            assert "rtx_6000" in result.message

    @patch("slurm_mcp.selection.select_gpu")
    def test_default_job_name_from_command(self, mock_select):
        mock_select.return_value = ("gtx_1080", "main", "normal")
        result = submit_job(cmd="python train.py --lr 1e-4", vram_gb=8, dry_run=True)
        assert "#SBATCH --job-name python" in result.sbatch_script

    @patch("slurm_mcp.selection.select_gpu")
    def test_custom_job_name(self, mock_select):
        mock_select.return_value = ("gtx_1080", "main", "normal")
        result = submit_job(
            cmd="python train.py", vram_gb=8, job_name="my-training", dry_run=True
        )
        assert "#SBATCH --job-name my-training" in result.sbatch_script

    @patch("slurm_mcp.selection.select_gpu")
    def test_multi_gpu_request(self, mock_select):
        mock_select.return_value = ("rtx_pro_6000", "rtx_pro_6000", "yisroel")
        result = submit_job(
            cmd="torchrun train.py", vram_gb=96, num_gpus=2, dry_run=True
        )
        assert "#SBATCH --gres=gpu:rtx_pro_6000:2" in result.sbatch_script
        assert "#SBATCH --nodes=1" in result.sbatch_script

    def test_num_gpus_cap(self):
        result = submit_job(
            cmd="torchrun train.py", vram_gb=48, num_gpus=3, dry_run=True
        )
        assert result.success is False
        assert "exceeds cluster limit" in result.message

    @patch("slurm_mcp.shell._run")
    @patch("slurm_mcp.selection.select_gpu")
    def test_actual_submit_parses_job_id(self, mock_select, mock_run):
        mock_select.return_value = ("gtx_1080", "main", "normal")
        mock_run.return_value = "Submitted batch job 12345678\n"
        result = submit_job(cmd="echo hi", vram_gb=8, wait_until_running=False)
        assert result.success is True
        assert result.job_id == 12345678

    @patch("slurm_mcp.shell._run")
    @patch("slurm_mcp.selection.select_gpu")
    def test_submit_failure_returns_error(self, mock_select, mock_run):
        mock_select.return_value = ("gtx_1080", "main", "normal")
        mock_run.side_effect = RuntimeError("sbatch: error: invalid partition")
        result = submit_job(cmd="echo hi", vram_gb=8, wait_until_running=False)
        assert result.success is False
        assert "invalid partition" in result.message


# ============================================================
# Unit Tests: submit_job golden_only (mocked)
# ============================================================

class TestSubmitJobGoldenOnly:
    """golden_only=True: force qos=yisroel + dedicated partition, never main."""

    def test_golden_only_96gb_forces_pro_partition(self):
        result = submit_job(cmd="python train.py", vram_gb=96,
                            golden_only=True, dry_run=True)
        assert result.success is True
        assert result.gpu_type == "rtx_pro_6000"
        assert result.partition == "rtx_pro_6000"
        assert result.qos == "yisroel"
        assert "#SBATCH --partition rtx_pro_6000" in result.sbatch_script
        assert "#SBATCH --qos=yisroel" in result.sbatch_script

    def test_golden_only_48gb(self):
        result = submit_job(cmd="python train.py", vram_gb=48,
                            golden_only=True, dry_run=True)
        assert result.partition == "rtx6000"
        assert result.qos == "yisroel"

    def test_golden_only_24gb_small_card(self):
        result = submit_job(cmd="python train.py", vram_gb=24,
                            golden_only=True, dry_run=True)
        assert result.gpu_type in ("rtx_3090", "rtx_4090")
        assert result.partition in ("rtx3090", "rtx4090")
        assert result.qos == "yisroel"

    def test_golden_only_explicit_small_card(self):
        result = submit_job(cmd="python train.py", vram_gb=24, gpu_type="rtx_4090",
                            golden_only=True, dry_run=True)
        assert result.gpu_type == "rtx_4090"
        assert result.partition == "rtx4090"
        assert result.qos == "yisroel"

    def test_golden_only_overrides_qos_arg(self):
        # golden_only wins over an explicit qos override.
        result = submit_job(cmd="python x.py", vram_gb=48, gpu_type="rtx_6000",
                            qos="normal", golden_only=True, dry_run=True)
        assert result.qos == "yisroel"
        assert result.partition == "rtx6000"

    def test_golden_only_ignored_for_cpu(self):
        result = submit_job(cmd="echo hi", vram_gb=0,
                            golden_only=True, dry_run=True)
        assert result.success is True
        assert result.partition == "cpu"
        assert result.qos == "normal"


# ============================================================
# Unit Tests: my_jobs (mocked)
# ============================================================

class TestMyJobsMocked:
    @patch("slurm_mcp.shell._run_quiet")
    def test_parses_jobs(self, mock_rq):
        mock_rq.return_value = (
            "15908232    "
            "seam-train-llama3.1-8b                   "
            "RUNNING     "
            "yisroel     "
            "gres/gpu:rtx_pro_6000:1                           "
            "2:18:55     "
            "ise-6000p-03             "
            "rtx_pro_6000        "
            "None                          \n"
        )
        jobs = my_jobs()
        assert len(jobs) == 1
        assert jobs[0]["job_id"] == "15908232"
        assert jobs[0]["state"] == "RUNNING"
        assert jobs[0]["qos"] == "yisroel"
        assert jobs[0]["partition"] == "rtx_pro_6000"
        assert jobs[0]["reason"] == "None"

    @patch("slurm_mcp.shell._run_quiet")
    def test_parses_pending_reason(self, mock_rq):
        # A PENDING job carries a scheduler reason in the trailing Reason column.
        mock_rq.return_value = (
            "15908299    "
            "attack-sweep-01                          "
            "PENDING     "
            "yisroel     "
            "gres/gpu:rtx_pro_6000:2                           "
            "0:00        "
            "                         "
            "rtx_pro_6000        "
            "QOSMaxGRESPerAccount          \n"
        )
        jobs = my_jobs()
        assert jobs[0]["state"] == "PENDING"
        assert jobs[0]["reason"] == "QOSMaxGRESPerAccount"

    @patch("slurm_mcp.shell._run_quiet")
    def test_empty_returns_empty_list(self, mock_rq):
        mock_rq.return_value = ""
        jobs = my_jobs()
        assert jobs == []


# ============================================================
# Unit Tests: cancel_jobs (mocked)
# ============================================================

class TestCancelJobsMocked:
    @patch("slurm_mcp.shell._run_quiet")
    def test_cancel_specific_ids(self, mock_rq):
        mock_rq.return_value = ""
        count = cancel_jobs(job_ids=[123, 456, 789])
        assert count == 3
        # Should have called scancel 3 times
        calls = [c for c in mock_rq.call_args_list if c[0][0][0] == "scancel"]
        assert len(calls) == 3

    @patch("slurm_mcp.shell._run_quiet")
    def test_cancel_returns_zero_when_nothing(self, mock_rq):
        count = cancel_jobs()
        assert count == 0

    @patch("slurm_mcp.shell._run_quiet")
    def test_cancel_all_counts(self, mock_rq):
        mock_rq.return_value = "job1\njob2\njob3\n"
        count = cancel_jobs(all_jobs=True)
        assert count == 3


# ============================================================
# Unit Tests: _build_sbatch_script — dependency
# ============================================================

class TestBuildSbatchScriptDependency:
    def test_dependency_in_script(self):
        script = _build_sbatch_script(
            cmd="python eval.py", partition="main", qos="normal",
            gpu_type="rtx_4090", num_gpus=1, job_name="eval",
            output_path="out.log", workdir=None,
            dependency="afterok:12345",
        )
        assert "#SBATCH --dependency=afterok:12345" in script

    def test_no_dependency_when_none(self):
        script = _build_sbatch_script(
            cmd="python eval.py", partition="main", qos="normal",
            gpu_type="rtx_4090", num_gpus=1, job_name="eval",
            output_path="out.log", workdir=None,
            dependency=None,
        )
        assert "--dependency" not in script

    def test_multi_job_dependency(self):
        script = _build_sbatch_script(
            cmd="python eval.py", partition="main", qos="normal",
            gpu_type="rtx_4090", num_gpus=1, job_name="eval",
            output_path="out.log", workdir=None,
            dependency="afterok:111:222:333",
        )
        assert "#SBATCH --dependency=afterok:111:222:333" in script

    @patch("slurm_mcp.selection.select_gpu")
    def test_dependency_in_submit_job_dry_run(self, mock_select):
        mock_select.return_value = ("rtx_4090", "main", "normal")
        result = submit_job(
            cmd="python eval.py", vram_gb=24,
            dependency="afterok:99999", dry_run=True,
        )
        assert result.success is True
        assert "#SBATCH --dependency=afterok:99999" in result.sbatch_script


# ============================================================
# Unit Tests: get_job_status (mocked)
# ============================================================

class TestGetJobStatusMocked:
    @patch("slurm_mcp.shell._run_quiet")
    def test_running_job_from_squeue(self, mock_rq):
        # squeue format: JobId:12,State:20,NodeList:25,TimeUsed:15,Reason:40
        mock_rq.return_value = (
            "12345678    "
            "RUNNING             "
            "ise-6000p-01             "
            "1:23:45        "
            "None                                    \n"
        )
        status = get_job_status(12345678)
        assert status.job_id == 12345678
        assert status.state == "RUNNING"
        assert status.node == "ise-6000p-01"
        assert status.elapsed == "1:23:45"
        assert status.reason == ""
        assert status.finished is False

    @patch("slurm_mcp.shell._run_quiet")
    def test_pending_job_with_reason(self, mock_rq):
        mock_rq.return_value = (
            "12345678    "
            "PENDING             "
            "(None)                   "
            "0:00           "
            "QOSMaxGRESPerAccount                    \n"
        )
        status = get_job_status(12345678)
        assert status.state == "PENDING"
        assert status.reason == "QOSMaxGRESPerAccount"
        assert status.finished is False

    @patch("slurm_mcp.shell._run_quiet")
    def test_pending_job_resources_reason(self, mock_rq):
        mock_rq.return_value = (
            "12345678    "
            "PENDING             "
            "(None)                   "
            "0:00           "
            "Resources                               \n"
        )
        status = get_job_status(12345678)
        assert status.state == "PENDING"
        assert status.reason == "Resources"

    @patch("slurm_mcp.shell._run_quiet")
    def test_completed_job_from_sacct(self, mock_rq):
        def side_effect(cmd):
            if cmd[0] == "squeue":
                return ""  # Not in squeue (finished)
            if cmd[0] == "sacct":
                return "12345|COMPLETED|0:0|ise-6000p-01|01:23:45\n12345.batch|COMPLETED|0:0|ise-6000p-01|01:23:45\n"
            return ""
        mock_rq.side_effect = side_effect
        status = get_job_status(12345)
        assert status.state == "COMPLETED"
        assert status.exit_code == 0
        assert status.finished is True
        assert status.node == "ise-6000p-01"

    @patch("slurm_mcp.shell._run_quiet")
    def test_failed_job_from_sacct(self, mock_rq):
        def side_effect(cmd):
            if cmd[0] == "squeue":
                return ""
            if cmd[0] == "sacct":
                return "99999|FAILED|1:0|cs-4090-01|00:05:30\n99999.batch|FAILED|1:0|cs-4090-01|00:05:30\n"
            return ""
        mock_rq.side_effect = side_effect
        status = get_job_status(99999)
        assert status.state == "FAILED"
        assert status.exit_code == 1
        assert status.finished is True

    @patch("slurm_mcp.shell._run_quiet")
    def test_unknown_job(self, mock_rq):
        mock_rq.return_value = ""
        status = get_job_status(99999)
        assert status.state == "UNKNOWN"
        assert status.finished is False

    def test_job_status_to_dict(self):
        status = JobStatus(
            job_id=123, state="RUNNING", exit_code=0,
            node="n1", elapsed="1:00", reason="", finished=False,
        )
        d = status.to_dict()
        assert d["job_id"] == 123
        assert d["state"] == "RUNNING"
        assert d["reason"] == ""


# ============================================================
# Unit Tests: read_job_log (mocked)
# ============================================================

class TestReadJobLog:
    def test_reads_matching_log(self, tmp_path):
        log_file = tmp_path / "slurm-train-12345.out"
        log_file.write_text("epoch 1: loss=0.5\nepoch 2: loss=0.3\n")
        content = read_job_log(12345, output_dir=str(tmp_path), job_name="train")
        assert "epoch 1" in content
        assert "epoch 2" in content

    def test_reads_generic_pattern(self, tmp_path):
        log_file = tmp_path / "slurm-myjob-99999.out"
        log_file.write_text("hello world\n")
        content = read_job_log(99999, output_dir=str(tmp_path))
        assert content == "hello world\n"

    def test_returns_none_when_not_found(self, tmp_path):
        content = read_job_log(99999, output_dir=str(tmp_path))
        assert content is None

    def test_tail_lines(self, tmp_path):
        log_file = tmp_path / "slurm-job-55555.out"
        log_file.write_text("line1\nline2\nline3\nline4\nline5\n")
        content = read_job_log(55555, output_dir=str(tmp_path), tail=2)
        assert content == "line4\nline5"

    def test_reads_simple_slurm_pattern(self, tmp_path):
        """Tests slurm-{job_id}.out pattern."""
        log_file = tmp_path / "slurm-77777.out"
        log_file.write_text("simple log\n")
        content = read_job_log(77777, output_dir=str(tmp_path))
        assert content == "simple log\n"

    def test_reads_non_slurm_prefix_log(self, tmp_path):
        """Tests that logs with prefixes other than 'slurm-' are found
        (e.g. `claude-<name>-<id>.out` written by launch_remote_session)."""
        log_file = tmp_path / "claude-myproj-88888.out"
        log_file.write_text("Remote control session URL: https://claude.ai/code/abc\n")
        content = read_job_log(88888, output_dir=str(tmp_path))
        assert content is not None
        assert "claude.ai/code" in content

    def test_job_id_substring_does_not_match(self, tmp_path):
        """A file whose id-portion is a substring of the requested id must
        NOT match. Guards against `*{id}*.out` style globs that over-match."""
        # 12345 must not pick up 123456789.out
        (tmp_path / "slurm-fake-123456789.out").write_text("wrong job\n")
        content = read_job_log(12345, output_dir=str(tmp_path))
        assert content is None


# ============================================================
# Unit Tests: wait_for_job (mocked)
# ============================================================

class TestWaitForJobMocked:
    @patch("slurm_mcp.monitoring.time.sleep")
    @patch("slurm_mcp.monitoring.get_job_status")
    def test_returns_when_finished(self, mock_status, mock_sleep):
        mock_status.return_value = JobStatus(
            job_id=123, state="COMPLETED", exit_code=0, finished=True,
        )
        result = wait_for_job(123, poll_interval=1)
        assert result.state == "COMPLETED"
        assert result.finished is True
        mock_sleep.assert_not_called()

    @patch("slurm_mcp.monitoring.time.sleep")
    @patch("slurm_mcp.monitoring.get_job_status")
    def test_polls_until_finished(self, mock_status, mock_sleep):
        mock_status.side_effect = [
            JobStatus(job_id=123, state="PENDING"),
            JobStatus(job_id=123, state="RUNNING"),
            JobStatus(job_id=123, state="COMPLETED", exit_code=0, finished=True),
        ]
        result = wait_for_job(123, poll_interval=1)
        assert result.state == "COMPLETED"
        assert mock_sleep.call_count == 2

    @patch("slurm_mcp.monitoring.time.time")
    @patch("slurm_mcp.monitoring.time.sleep")
    @patch("slurm_mcp.monitoring.get_job_status")
    def test_timeout_returns_last_status(self, mock_status, mock_sleep, mock_time):
        mock_time.side_effect = [0, 5, 100]  # start, check, check (past timeout)
        mock_status.return_value = JobStatus(job_id=123, state="PENDING")
        result = wait_for_job(123, poll_interval=1, timeout=10)
        assert result.state == "PENDING"


# ============================================================
# Unit Tests: _wait_for_running (mocked)
# ============================================================

class TestWaitForRunningMocked:
    def _make_job_result(self, job_id=12345):
        return JobResult(
            success=True, job_id=job_id,
            gpu_type="rtx_6000", partition="rtx6000", qos="yisroel",
            message=f"Submitted batch job {job_id}",
            sbatch_script="#!/bin/bash\n",
        )

    @patch("slurm_mcp.submission.time.sleep")
    @patch("slurm_mcp.monitoring.get_job_status")
    def test_immediately_running(self, mock_status, mock_sleep):
        mock_status.return_value = JobStatus(
            job_id=12345, state="RUNNING", node="ise-6000p-01",
        )
        result, outcome = _wait_for_running(self._make_job_result(), timeout=60)
        assert outcome == "running"
        assert result.success is True
        assert "RUNNING" in result.message
        mock_sleep.assert_not_called()

    @patch("slurm_mcp.submission.time.sleep")
    @patch("slurm_mcp.monitoring.get_job_status")
    def test_pending_then_running(self, mock_status, mock_sleep):
        mock_status.side_effect = [
            JobStatus(job_id=12345, state="PENDING", reason="Resources"),
            JobStatus(job_id=12345, state="PENDING", reason="Resources"),
            JobStatus(job_id=12345, state="RUNNING", node="cs-4090-01"),
        ]
        result, outcome = _wait_for_running(self._make_job_result(), timeout=300)
        assert outcome == "running"
        assert result.success is True
        assert "RUNNING" in result.message
        assert mock_sleep.call_count == 2

    @patch("slurm_mcp.submission.time.sleep")
    @patch("slurm_mcp.monitoring.get_job_status")
    @patch("slurm_mcp.shell._run_quiet")
    def test_quota_reason_cancels_job(self, mock_rq, mock_status, mock_sleep):
        mock_status.return_value = JobStatus(
            job_id=12345, state="PENDING", reason="QOSMaxGRESPerAccount",
        )
        result, outcome = _wait_for_running(self._make_job_result(), timeout=300)
        assert outcome == "quota"
        assert result.success is False
        assert "QOSMaxGRESPerAccount" in result.message
        mock_rq.assert_called_once_with(["scancel", "12345"])

    @patch("slurm_mcp.submission.time.sleep")
    @patch("slurm_mcp.monitoring.get_job_status")
    @patch("slurm_mcp.shell._run_quiet")
    def test_user_quota_reason_qos_max_user(self, mock_rq, mock_status, mock_sleep):
        mock_status.return_value = JobStatus(
            job_id=12345, state="PENDING", reason="QOSMaxGRESPerUser",
        )
        result, outcome = _wait_for_running(self._make_job_result(), timeout=300)
        assert outcome == "user_quota"
        assert result.success is False
        assert "per-user limit" in result.message
        assert "QOSMaxGRESPerUser" in result.message
        mock_rq.assert_called_once_with(["scancel", "12345"])

    @patch("slurm_mcp.submission.time.sleep")
    @patch("slurm_mcp.monitoring.get_job_status")
    @patch("slurm_mcp.shell._run_quiet")
    def test_fatal_reason_dependency_never_satisfied(self, mock_rq, mock_status, mock_sleep):
        mock_status.return_value = JobStatus(
            job_id=12345, state="PENDING", reason="DependencyNeverSatisfied",
        )
        result, outcome = _wait_for_running(self._make_job_result(), timeout=300)
        assert outcome == "fatal"
        assert result.success is False
        assert "DependencyNeverSatisfied" in result.message

    @patch("slurm_mcp.submission.time.sleep")
    @patch("slurm_mcp.monitoring.get_job_status")
    def test_job_finished_before_running(self, mock_status, mock_sleep):
        mock_status.return_value = JobStatus(
            job_id=12345, state="FAILED", exit_code=1, finished=True,
        )
        result, outcome = _wait_for_running(self._make_job_result(), timeout=300)
        assert outcome == "finished"
        assert result.success is False
        assert "FAILED" in result.message

    @patch("slurm_mcp.submission.time.sleep")
    @patch("slurm_mcp.monitoring.get_job_status")
    def test_job_cancelled_before_running(self, mock_status, mock_sleep):
        mock_status.return_value = JobStatus(
            job_id=12345, state="CANCELLED", finished=True,
        )
        result, outcome = _wait_for_running(self._make_job_result(), timeout=300)
        assert outcome == "finished"
        assert result.success is False
        assert "CANCELLED" in result.message

    @patch("slurm_mcp.submission.time.time")
    @patch("slurm_mcp.submission.time.sleep")
    @patch("slurm_mcp.monitoring.get_job_status")
    def test_timeout_does_not_cancel_job(self, mock_status, mock_sleep, mock_time):
        """On timeout with benign reason, job stays queued (not cancelled)."""
        mock_time.side_effect = [0, 301]  # start, past timeout
        mock_status.return_value = JobStatus(
            job_id=12345, state="PENDING", reason="Resources",
        )
        result, outcome = _wait_for_running(self._make_job_result(), timeout=300)
        assert outcome == "still_pending"
        assert result.success is True  # job is still alive
        assert "still pending" in result.message
        assert "remains queued" in result.message

    @patch("slurm_mcp.submission.time.time")
    @patch("slurm_mcp.submission.time.sleep")
    @patch("slurm_mcp.monitoring.get_job_status")
    def test_timeout_includes_reason_in_message(self, mock_status, mock_sleep, mock_time):
        mock_time.side_effect = [0, 301]
        mock_status.return_value = JobStatus(
            job_id=12345, state="PENDING", reason="Priority",
        )
        result, outcome = _wait_for_running(self._make_job_result(), timeout=300)
        assert outcome == "still_pending"
        assert "Priority" in result.message

    @patch("slurm_mcp.shell._run")
    @patch("slurm_mcp.submission.time.sleep")
    @patch("slurm_mcp.monitoring.get_job_status")
    @patch("slurm_mcp.selection.select_gpu")
    def test_submit_job_with_wait_until_running(self, mock_select, mock_status, mock_sleep, mock_run):
        """Integration test: submit_job with wait_until_running=True."""
        mock_select.return_value = ("rtx_6000", "rtx6000", "yisroel")
        mock_run.return_value = "Submitted batch job 12345\n"
        mock_status.side_effect = [
            JobStatus(job_id=12345, state="PENDING", reason="Resources"),
            JobStatus(job_id=12345, state="RUNNING", node="ise-6000-01"),
        ]
        result = submit_job(
            cmd="python train.py", vram_gb=48,
            wait_until_running=True,
        )
        assert result.success is True
        assert "RUNNING" in result.message
        assert result.job_id == 12345

    @patch("slurm_mcp.shell._run")
    @patch("slurm_mcp.submission.time.sleep")
    @patch("slurm_mcp.monitoring.get_job_status")
    @patch("slurm_mcp.shell._run_quiet")
    @patch("slurm_mcp.selection.select_gpu")
    def test_submit_job_quota_hit_falls_back_to_normal(self, mock_select, mock_rq, mock_status, mock_sleep, mock_run):
        """Quota hit on golden -> cancel, resubmit on normal QoS, then runs."""
        mock_select.return_value = ("rtx_6000", "rtx6000", "yisroel")
        # First sbatch (golden), then second sbatch (fallback)
        mock_run.side_effect = [
            "Submitted batch job 12345\n",
            "Submitted batch job 12346\n",
        ]
        mock_status.side_effect = [
            # First job: golden quota full
            JobStatus(job_id=12345, state="PENDING", reason="QOSMaxGRESPerAccount"),
            # Second job (fallback): runs
            JobStatus(job_id=12346, state="RUNNING", node="cs-6000-01"),
        ]
        result = submit_job(
            cmd="python train.py", vram_gb=48,
            golden_only=False,  # fallback is opt-in now (default is golden-only)
            wait_until_running=True,
        )
        assert result.success is True
        assert result.job_id == 12346
        assert result.qos == "normal"
        assert result.partition == "main"

    @patch("slurm_mcp.shell._run")
    @patch("slurm_mcp.submission.time.sleep")
    @patch("slurm_mcp.monitoring.get_job_status")
    @patch("slurm_mcp.shell._run_quiet")
    @patch("slurm_mcp.selection.select_gpu")
    def test_submit_job_user_quota_does_not_fallback(self, mock_select, mock_rq, mock_status, mock_sleep, mock_run):
        """QOSMaxGRESPerUser should NOT trigger fallback — it's a per-user limit across all QoS."""
        mock_select.return_value = ("rtx_6000", "rtx6000", "yisroel")
        mock_run.return_value = "Submitted batch job 12345\n"
        mock_status.return_value = JobStatus(
            job_id=12345, state="PENDING", reason="QOSMaxGRESPerUser",
        )
        result = submit_job(
            cmd="python train.py", vram_gb=48,
            golden_only=False,  # even with fallback allowed, per-user quota must not fall back
            wait_until_running=True,
        )
        assert result.success is False
        assert "per-user limit" in result.message
        # Should only have submitted ONCE (no fallback)
        assert mock_run.call_count == 1

    @patch("slurm_mcp.shell._run")
    @patch("slurm_mcp.selection.select_gpu")
    def test_submit_job_without_wait_still_works(self, mock_select, mock_run):
        """submit_job without wait_until_running should return immediately."""
        mock_select.return_value = ("rtx_6000", "rtx6000", "yisroel")
        mock_run.return_value = "Submitted batch job 12345\n"
        result = submit_job(cmd="python train.py", vram_gb=48, wait_until_running=False)
        assert result.success is True
        assert result.job_id == 12345

    @patch("slurm_mcp.submission.time.time")
    @patch("slurm_mcp.submission.time.sleep")
    @patch("slurm_mcp.monitoring.get_job_status")
    @patch("slurm_mcp.shell._run_quiet")
    def test_golden_only_quota_does_not_cancel(self, mock_rq, mock_status, mock_sleep, mock_time):
        """golden_only=True: a quota reason leaves the job queued (no scancel)."""
        mock_time.side_effect = [0, 301]  # start, past timeout
        mock_status.return_value = JobStatus(
            job_id=12345, state="PENDING", reason="QOSMaxGRESPerAccount",
        )
        result, outcome = _wait_for_running(
            self._make_job_result(), timeout=300, golden_only=True,
        )
        assert outcome == "still_pending"
        assert result.success is True
        mock_rq.assert_not_called()  # never scancel a golden_only job for quota

    @patch("slurm_mcp.shell._run")
    @patch("slurm_mcp.submission.time.time")
    @patch("slurm_mcp.submission.time.sleep")
    @patch("slurm_mcp.monitoring.get_job_status")
    @patch("slurm_mcp.shell._run_quiet")
    def test_submit_job_golden_only_no_fallback(self, mock_rq, mock_status, mock_sleep, mock_time, mock_run):
        """golden_only=True: golden quota full -> stays queued, NO normal fallback."""
        mock_time.side_effect = [0, 301]
        mock_run.return_value = "Submitted batch job 12345\n"
        mock_status.return_value = JobStatus(
            job_id=12345, state="PENDING", reason="QOSMaxGRESPerAccount",
        )
        result = submit_job(
            cmd="python train.py", vram_gb=48, golden_only=True,
            wait_until_running=True,
        )
        assert result.qos == "yisroel"
        assert result.partition == "rtx6000"
        assert result.job_id == 12345
        assert mock_run.call_count == 1  # golden only, no fallback resubmit


# ============================================================
# Unit Tests: golden_queue + golden_queues + render (mocked)
# ============================================================

class TestGoldenQueue:
    @patch("slurm_mcp.shell._run_quiet")
    def test_parses_and_orders(self, mock_rq):
        mock_rq.return_value = "\n".join([
            _queue_row(100, "20000002", "alice", tres_node="gres/gpu:rtx_6000:1", name="job-b"),
            _queue_row(100, "20000001", "alice", tres_node="gres/gpu:rtx_6000:1", name="job-a"),
            _queue_row(200, "20000005", "bob", tres_node="gres/gpu:rtx_pro_6000:2", name="big-job"),
        ]) + "\n"
        q = slurm_mcp.golden_queue("yisroel")
        # bob (priority 200) first; then alice's two by ascending job id.
        assert [r["job_id"] for r in q] == ["20000005", "20000001", "20000002"]
        assert q[0]["user"] == "bob"
        assert q[0]["gpu_type"] == "rtx_pro_6000"
        assert q[0]["gpu_count"] == 2
        assert q[1]["name"] == "job-a"

    @patch("slurm_mcp.shell._run_quiet")
    def test_job_scoped_gpu_request_is_listed(self, mock_rq):
        """--gpus / --gres put the request in tres-per-job, --gpus-per-node puts it
        in tres-per-node. Reading only tres-per-node dropped the former from the
        queue list while the counter beside it still counted them."""
        mock_rq.return_value = "\n".join([
            _queue_row(100, "1", "alice", tres_job="gres/gpu:rtx_pro_6000:1"),
            _queue_row(100, "2", "bob", tres_node="gres/gpu:rtx_pro_6000:1"),
        ]) + "\n"
        q = slurm_mcp.golden_queue("yisroel")
        assert [(r["user"], r["gpu_type"], r["gpu_count"]) for r in q] == [
            ("alice", "rtx_pro_6000", 1),
            ("bob", "rtx_pro_6000", 1),
        ]

    @patch("slurm_mcp.shell._run_quiet")
    def test_cpu_only_job_carries_no_gpu_type(self, mock_rq):
        mock_rq.return_value = _queue_row(100, "1", "alice") + "\n"
        q = slurm_mcp.golden_queue("yisroel")
        assert q[0]["gpu_type"] == ""
        assert q[0]["gpu_count"] == 0

    @patch("slurm_mcp.shell._run_quiet")
    def test_array_task_ids_order(self, mock_rq):
        mock_rq.return_value = "\n".join([
            _queue_row(100, "123_2", "u", tres_node="gres/gpu:rtx_6000:1"),
            _queue_row(100, "123_1", "u", tres_node="gres/gpu:rtx_6000:1"),
        ]) + "\n"
        q = slurm_mcp.golden_queue("yisroel")
        assert [r["job_id"] for r in q] == ["123_1", "123_2"]

    @patch("slurm_mcp.shell._run_quiet")
    def test_empty(self, mock_rq):
        mock_rq.return_value = ""
        assert slurm_mcp.golden_queue("yisroel") == []


class TestPendingCounterMatchesQueue:
    """The '(N pending)' counter and the 'Pending (next first)' list below it come
    from two separate squeue calls. They must agree, whichever way a job spelled
    its GPU request."""

    def _mock(self, rows):
        """Serve the counter query and the ordered-queue query from one job list.

        Rows are (user, tres_job, tres_node, state) or (..., state, reason)."""
        rows = [r if len(r) == 5 else (*r, "") for r in rows]

        def run_quiet(cmd):
            if cmd[0] != "squeue":
                return ""
            if "-t" in cmd:  # the ordered PENDING queue
                return "\n".join(
                    _queue_row(100, str(i), u, tj, tn, reason=rsn)
                    for i, (u, tj, tn, st, rsn) in enumerate(rows) if st == "PENDING"
                ) + "\n"
            return "\n".join(
                _golden_row(u, st, tj, tn, rsn) for u, tj, tn, st, rsn in rows
            ) + "\n"
        return run_quiet

    @patch("slurm_mcp.shell._run_quiet")
    def test_counter_equals_listed_gpus(self, mock_rq):
        # 11 via --gpus-per-node, 2 via --gpus: the exact shape that read
        # "13 pending" above a list adding up to 11.
        rows = [("lenga", "N/A", "gres/gpu:rtx_pro_6000:1", "PENDING")] * 11
        rows += [
            ("benchayi", "gres/gpu:rtx_pro_6000:1", "N/A", "PENDING"),
            ("amonfadi", "gres/gpu:rtx_pro_6000:1", "N/A", "PENDING"),
        ]
        mock_rq.side_effect = self._mock(rows)

        avail = check_availability()
        queue = slurm_mcp.golden_queue("yisroel")

        counted = avail.golden["rtx_pro_6000"].pending
        listed = sum(r["gpu_count"] for r in queue if r["gpu_type"] == "rtx_pro_6000")
        assert counted == 13
        assert listed == counted

    @patch("slurm_mcp.shell._run_quiet")
    def test_rendered_card_and_list_agree(self, mock_rq):
        from cli import render
        rows = [
            ("alice", "N/A", "gres/gpu:rtx_pro_6000:2", "PENDING"),
            ("bob", "gres/gpu:rtx_pro_6000:3", "N/A", "PENDING"),
        ]
        mock_rq.side_effect = self._mock(rows)
        avail = check_availability()
        out = render.render_golden_all(
            avail, queues={"yisroel": slurm_mcp.golden_queue("yisroel")}
        )
        assert "5 pending" in out
        assert "alice: 2 GPU(s)" in out
        assert "bob: 3 GPU(s)" in out

    @patch("slurm_mcp.shell._run_quiet")
    def test_they_still_agree_with_a_dependency_in_the_mix(self, mock_rq):
        rows = [
            ("alice", "N/A", "gres/gpu:rtx_pro_6000:2", "PENDING", "Priority"),
            ("lenga", "N/A", "gres/gpu:rtx_pro_6000:1", "PENDING", "Dependency"),
        ]
        mock_rq.side_effect = self._mock(rows)

        avail = check_availability()
        queue = slurm_mcp.golden_queue("yisroel")

        card = avail.golden["rtx_pro_6000"]
        listed = sum(r["gpu_count"] for r in queue if r["gpu_type"] == "rtx_pro_6000")
        assert (card.pending, card.blocked) == (2, 1)
        assert listed == card.pending


class TestBlockedPendingJobs:
    """A job pending on a dependency (or a hold) keeps its priority slot but the
    scheduler skips it, so it will not take the next card that frees. Counting it
    as queue depth overstates how many GPUs are ahead of you."""

    def _mock(self, rows):
        def run_quiet(cmd):
            if cmd[0] != "squeue":
                return ""
            if "-t" in cmd:
                return "\n".join(
                    _queue_row(100, str(i), u, tres_node=tn, reason=rsn, name=f"j{i}")
                    for i, (u, tn, st, rsn) in enumerate(rows) if st == "PENDING"
                ) + "\n"
            return "\n".join(
                _golden_row(u, st, tres_node=tn, reason=rsn) for u, tn, st, rsn in rows
            ) + "\n"
        return run_quiet

    @pytest.mark.parametrize("reason", [
        "Dependency", "DependencyNeverSatisfied",
        "JobHeldUser", "JobHeldAdmin", "JobHoldMaxRequeue", "BeginTime",
    ])
    def test_blocked_reasons_are_not_queue_depth(self, reason):
        assert slurm_mcp.availability._is_blocked(reason)

    @pytest.mark.parametrize("reason", [
        # These clear when someone's *running* job ends — that IS waiting for a
        # slot, so they stay in the count.
        "Priority", "Resources", "None", "",
        "QOSMaxGRESPerUser", "MaxGRESPerAccount", "QOSMaxMemoryPerUser",
        "JobArrayTaskLimit",   # a throttled array still wants the card
    ])
    def test_capacity_reasons_still_count(self, reason):
        assert not slurm_mcp.availability._is_blocked(reason)

    @patch("slurm_mcp.shell._run_quiet")
    def test_counter_splits_pending_from_blocked(self, mock_rq):
        rows = [
            ("alice", "gres/gpu:rtx_pro_6000:1", "PENDING", "Priority"),
            ("bob",   "gres/gpu:rtx_pro_6000:3", "PENDING", "Dependency"),
            ("carol", "gres/gpu:rtx_pro_6000:2", "RUNNING", "None"),
        ]
        mock_rq.side_effect = self._mock(rows)
        card = check_availability().golden["rtx_pro_6000"]
        assert (card.running, card.pending, card.blocked) == (2, 1, 3)

    @patch("slurm_mcp.shell._run_quiet")
    def test_blocked_job_is_not_an_extra_user_row(self, mock_rq):
        rows = [("bob", "gres/gpu:rtx_pro_6000:3", "PENDING", "Dependency")]
        mock_rq.side_effect = self._mock(rows)
        card = check_availability().golden["rtx_pro_6000"]
        assert card.pending_users == {}

    @patch("slurm_mcp.shell._run_quiet")
    def test_queue_list_skips_blocked_jobs(self, mock_rq):
        rows = [
            ("alice", "gres/gpu:rtx_pro_6000:1", "PENDING", "Priority"),
            ("bob",   "gres/gpu:rtx_pro_6000:1", "PENDING", "Dependency"),
        ]
        mock_rq.side_effect = self._mock(rows)
        assert [r["user"] for r in slurm_mcp.golden_queue("yisroel")] == ["alice"]

    @patch("slurm_mcp.shell._run_quiet")
    def test_card_shows_blocked_count(self, mock_rq):
        from cli import render
        rows = [
            ("alice", "gres/gpu:rtx_pro_6000:1", "PENDING", "Priority"),
            ("bob",   "gres/gpu:rtx_pro_6000:2", "PENDING", "Dependency"),
        ]
        mock_rq.side_effect = self._mock(rows)
        avail = check_availability()
        out = render.render_golden_all(avail)
        assert "1 pending, 2 blocked" in out

    def test_no_blocked_suffix_when_zero(self):
        from cli import render
        avail = Availability()
        avail.golden_by_qos["yisroel"] = {
            "rtx_6000": GPUAvailability("rtx_6000", 12, 12, 0, running=12, pending=4),
        }
        out = render.render_golden_all(avail)
        assert "(12 running, 4 pending)" in out
        assert "blocked" not in out


class TestGoldenQueues:
    def test_only_fetches_when_a_card_is_full(self):
        avail = Availability()
        avail.golden_by_qos["yisroel"] = {
            "rtx_6000": GPUAvailability("rtx_6000", 12, 12, 0),          # full
            "rtx_pro_6000": GPUAvailability("rtx_pro_6000", 16, 12, 4),  # free
        }
        with patch("slurm_mcp.availability.golden_queue",
                   return_value=[{"job_id": "1"}]) as gq:
            out = slurm_mcp.golden_queues(avail)
        gq.assert_called_once_with("yisroel")
        assert out["yisroel"] == [{"job_id": "1"}]

    def test_no_fetch_when_nothing_full(self):
        avail = Availability()
        avail.golden_by_qos["yisroel"] = {
            "rtx_6000": GPUAvailability("rtx_6000", 12, 4, 8),  # free
        }
        with patch("slurm_mcp.availability.golden_queue") as gq:
            out = slurm_mcp.golden_queues(avail)
        assert out == {}
        gq.assert_not_called()

    def test_qos_filter(self):
        avail = Availability()
        avail.golden_by_qos["yisroel"] = {
            "rtx_6000": GPUAvailability("rtx_6000", 12, 12, 0),
        }
        avail.golden_by_qos["other"] = {
            "rtx_6000": GPUAvailability("rtx_6000", 4, 4, 0),
        }
        with patch("slurm_mcp.availability.golden_queue", return_value=[]) as gq:
            out = slurm_mcp.golden_queues(avail, qos_filter="yisroel")
        assert set(out.keys()) == {"yisroel"}
        gq.assert_called_once_with("yisroel")


class TestRenderGoldenQueue:
    def _avail_full(self):
        avail = Availability()
        avail.golden_by_qos["yisroel"] = {
            "rtx_6000": GPUAvailability(
                "rtx_6000", 12, 12, 0, running=12, pending=2,
            ),
        }
        return avail

    def test_full_card_lists_pending_in_order(self):
        # Pending mirrors Running (per-user GPU totals) but ordered; no
        # jobid/jobname.
        from cli import render
        avail = self._avail_full()
        queues = {"yisroel": [
            {"job_id": "1001", "user": "alice", "name": "train-a",
             "gpu_type": "rtx_6000", "gpu_count": 1, "priority": 100},
            {"job_id": "1002", "user": "bob", "name": "train-b",
             "gpu_type": "rtx_6000", "gpu_count": 2, "priority": 100},
        ]}
        out = render.render_golden_all(avail, queues=queues)
        assert "Pending (next first):" in out
        assert "alice: 1 GPU(s)" in out and "bob: 2 GPU(s)" in out
        assert out.index("alice") < out.index("bob")   # dispatch order
        assert "train-a" not in out and "1001" not in out  # no jobid/jobname

    def test_consecutive_same_user_gpus_combined(self):
        # 4 in-line 1-GPU jobs from one user collapse to a single summed row
        # (GPUs, not job count) — the reported case.
        from cli import render
        avail = self._avail_full()
        queues = {"yisroel": [
            {"job_id": str(i), "user": "itayzloc", "name": f"EM-{i}",
             "gpu_type": "rtx_6000", "gpu_count": 1, "priority": 100}
            for i in range(4)
        ]}
        out = render.render_golden_all(avail, queues=queues)
        pending = out.split("Pending")[1]
        assert pending.count("itayzloc:") == 1        # one merged row, not four
        assert "itayzloc: 4 GPU(s)" in pending        # GPUs summed

    def test_repeated_user_keeps_queue_order(self):
        # Interleaving edge case: the same user split by another user must stay
        # at both positions (consecutive-only merge).
        from cli import render
        avail = self._avail_full()
        queues = {"yisroel": [
            {"job_id": "1", "user": "itay", "name": "a",
             "gpu_type": "rtx_6000", "gpu_count": 3, "priority": 100},
            {"job_id": "2", "user": "doron", "name": "b",
             "gpu_type": "rtx_6000", "gpu_count": 2, "priority": 100},
            {"job_id": "3", "user": "itay", "name": "c",
             "gpu_type": "rtx_6000", "gpu_count": 1, "priority": 100},
        ]}
        out = render.render_golden_all(avail, queues=queues)
        assert out.count("itay:") == 2  # not merged across doron
        assert (out.index("itay: 3 GPU(s)") < out.index("doron: 2 GPU(s)")
                < out.index("itay: 1 GPU(s)"))

    def test_not_full_falls_back_to_per_user_aggregate(self):
        from cli import render
        avail = Availability()
        avail.golden_by_qos["yisroel"] = {
            "rtx_6000": GPUAvailability("rtx_6000", 12, 8, 4, running=8, pending=4,
                                        pending_users={"carol": 4}),
        }
        # No queue rows (not full) -> per-user aggregate under a plain "Pending:".
        out = render.render_golden_all(avail, queues={"yisroel": []})
        assert "    Pending:" in out and "carol: 4 GPU(s)" in out
        assert "next first" not in out

    def test_free_card_has_no_pending_section(self):
        from cli import render
        avail = Availability()
        avail.golden_by_qos["yisroel"] = {
            "rtx_6000": GPUAvailability("rtx_6000", 12, 4, 8, running=4, pending=0),
        }
        out = render.render_golden_all(avail, queues={"yisroel": []})
        assert "Pending" not in out

    def test_overflow_truncates(self):
        from cli import render
        avail = self._avail_full()
        rows = [
            {"job_id": str(1000 + i), "user": f"u{i}", "name": f"j{i}",
             "gpu_type": "rtx_6000", "gpu_count": 1, "priority": 100}
            for i in range(20)
        ]
        out = render.render_golden_all(avail, queues={"yisroel": rows})
        assert "and 5 more GPU(s) queued" in out  # 20 rows - LIMIT(15) = 5 GPUs

    def test_limit_none_lists_all(self):
        # The scrollable TUI passes limit=None to show every queued row.
        from cli import render
        avail = self._avail_full()
        rows = [
            {"job_id": str(1000 + i), "user": f"u{i}", "name": f"j{i}",
             "gpu_type": "rtx_6000", "gpu_count": 1, "priority": 100}
            for i in range(20)
        ]
        out = render.render_golden_all(avail, queues={"yisroel": rows}, limit=None)
        assert "more GPU(s) queued" not in out
        assert "u19: 1 GPU(s)" in out  # the 20th row is present, not truncated


class TestRenderClusterWide:
    """The Cluster-Wide section, including the offline-capacity annotation."""

    def _avail(self, **kwargs):
        avail = Availability()
        avail.cluster = {
            "rtx_pro_6000": GPUAvailability(
                gpu_type="rtx_pro_6000", total=48, used=37, free=11, **kwargs
            ),
        }
        return avail

    def test_healthy_card_has_no_annotation(self):
        from cli import render
        out = render.render_cluster_wide(self._avail())
        assert "  rtx_pro_6000: 11/48 free" in out
        assert "offline" not in out

    def test_offline_capacity_is_named(self):
        """The whole point: a total that shrank has a visible reason attached."""
        from cli import render
        out = render.render_cluster_wide(
            self._avail(offline=8, offline_nodes=["ise-6000p-07"])
        )
        assert "  rtx_pro_6000: 11/48 free (8 offline: ise-6000p-07)" in out

    def test_long_offline_node_list_collapses(self):
        """The TUI puts this block in the right-hand column, so the name list is
        capped hard — a long one would push the whole dashboard row off-screen."""
        from cli import render
        nodes = [f"node-{i:02d}" for i in range(9)]
        out = render.render_cluster_wide(self._avail(offline=9, offline_nodes=nodes))
        assert "9 offline: node-00, node-01, +7 more" in out

    def test_fully_offline_card_still_listed(self):
        """total=0 with offline>0 means every card of that type is unreachable —
        worth showing, unlike a card the cluster simply doesn't have."""
        from cli import render
        avail = Availability()
        avail.cluster = {
            "rtx_2080": GPUAvailability(gpu_type="rtx_2080", total=0, used=0, free=0),
            "gtx_1080": GPUAvailability(gpu_type="gtx_1080", total=0, used=0, free=0,
                                        offline=4, offline_nodes=["cs-1080-01"]),
        }
        out = render.render_cluster_wide(avail)
        assert "rtx_2080" not in out
        assert "  gtx_1080: 0/0 free (4 offline: cs-1080-01)" in out


class TestWatchDashboard:
    """Pure helpers behind the live `slurmx status` TUI (cli/watch.py)."""

    def _avail(self):
        avail = Availability()
        avail.golden_by_qos["yisroel"] = {
            "rtx_pro_6000": GPUAvailability(
                "rtx_pro_6000", 16, 16, 0, running=16, pending=30),
        }
        avail.cluster["rtx_pro_6000"] = GPUAvailability("rtx_pro_6000", 56, 56, 0)
        return avail

    def _squeue_text(self, n=30):
        header = "JOBID PARTITION NAME USER ST TIME NODES NODELIST(REASON)"
        rows = [
            f"{1000 + i} rtx6000 attack-{i:02d} itayzloc PD 0:00 1 (MaxGRESPerAccount)"
            for i in range(n)
        ]
        return "\n".join([header, *rows])

    def test_squeue_lines_pass_through_uncapped(self):
        from cli import watch
        lines = watch.build_dashboard_lines(
            self._avail(), self._squeue_text(30), {}, qos=None)
        assert lines[0] == "=== squeue --me ==="
        text = "\n".join(lines)
        for i in range(30):  # every job present, nothing dropped
            assert f"{1000 + i} rtx6000" in text

    def test_no_jobs(self):
        from cli import watch
        lines = watch.build_dashboard_lines(self._avail(), "", {}, qos=None)
        assert any("(no jobs)" in l for l in lines)

    def test_golden_and_cluster_side_by_side(self):
        from cli import watch
        lines = watch.build_dashboard_lines(
            self._avail(), self._squeue_text(2), {}, qos=None)
        # the two section headers share a single row (two columns)
        assert any("Golden Tickets" in l and "Cluster-Wide" in l for l in lines)

    def test_side_by_side_helper(self):
        from cli import watch
        left = ["=== Golden ===", "  card: 0/16", "  q1", "  q2 deep"]
        right = ["=== Cluster ===", "  card: 0/56"]
        out = watch._side_by_side(left, right)
        assert "Golden" in out[0] and "Cluster" in out[0]  # headers align
        assert out[3] == "  q2 deep"                        # deep row left-only
        assert watch._side_by_side(left, []) == left        # empty right guard
        assert watch._side_by_side([], right) == right      # empty left guard

    def test_clamp_scroll_bounds(self):
        from cli import watch
        assert watch.clamp_scroll(-5, 100, 10) == 0     # below zero
        assert watch.clamp_scroll(500, 100, 10) == 90   # past the end
        assert watch.clamp_scroll(5, 100, 10) == 5      # in range
        assert watch.clamp_scroll(7, 5, 10) == 0        # viewport >= total
        assert watch.clamp_scroll(3, 0, 10) == 0        # empty buffer


# ============================================================
# Unit Tests: CLI --json output
# ============================================================

class TestCLIJsonOutput:
    def test_json_dry_run(self):
        # Pin --gpu-type so the result is deterministic — the subprocess can't
        # see a mocked select_gpu, and live golden availability shifts underfoot.
        import subprocess
        result = subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(__file__), "..", "cli", "submit.py"),
             "--gpu-type", "rtx_6000", "--vram", "48", "--dry-run", "--json",
             "--", "python", "train.py"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        import json
        data = json.loads(result.stdout)
        assert data["success"] is True
        assert data["gpu_type"] == "rtx_6000"
        assert "sbatch_script" in data

    def test_json_golden_only_is_default(self):
        """golden-only is the default: dedicated partition + yisroel QoS, no flag."""
        import subprocess, json
        result = subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(__file__), "..", "cli", "submit.py"),
             "--gpu-type", "rtx_4090", "--vram", "24",
             "--dry-run", "--json", "--", "python", "train.py"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["success"] is True
        assert data["partition"] == "rtx4090"
        assert data["qos"] == "yisroel"

    def test_json_allow_main_restores_fallback(self):
        """--allow-main opts a non-owned card back onto main/normal."""
        import subprocess, json
        result = subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(__file__), "..", "cli", "submit.py"),
             "--gpu-type", "rtx_4090", "--vram", "24", "--allow-main",
             "--dry-run", "--json", "--", "python", "train.py"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["success"] is True
        assert data["partition"] == "main"
        assert data["qos"] == "normal"

    def test_after_shorthand_builds_afterok(self):
        """--after 111 222 -> #SBATCH --dependency=afterok:111:222 in the script."""
        import subprocess, json
        result = subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(__file__), "..", "cli", "submit.py"),
             "--gpu-type", "rtx_6000", "--vram", "48", "--after", "111", "222",
             "--dry-run", "--json", "--", "python", "train.py"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "#SBATCH --dependency=afterok:111:222" in data["sbatch_script"]

    def test_after_and_dependency_conflict_errors(self):
        """--after and --dependency together is a user error (non-zero exit)."""
        import subprocess
        result = subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(__file__), "..", "cli", "submit.py"),
             "--gpu-type", "rtx_6000", "--vram", "48",
             "--after", "111", "--dependency", "afterany:222",
             "--dry-run", "--json", "--", "python", "train.py"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode != 0
        assert "not both" in result.stderr


# ============================================================
# Unit Tests: refactored slurm_mcp.diagnose_job / job_history
# ============================================================

class TestDiagnoseAndHistoryRefactor:
    """diagnose_job + job_history logic now lives in slurm_mcp/ (shared by the
    MCP tool and the `slurmx diagnose`/`history` CLIs)."""

    def test_functions_exported(self):
        assert callable(slurm_mcp.diagnose_job)
        assert callable(slurm_mcp.job_history)

    @patch("slurm_mcp.diagnostics.get_job_status")
    def test_diagnose_running_job_short_circuits(self, mock_status):
        mock_status.return_value = JobStatus(job_id=9, state="RUNNING")
        out = slurm_mcp.diagnose_job(9)
        assert "No diagnosis needed" in out
        assert isinstance(out, str)

    @patch("slurm_mcp.diagnostics.read_job_log")
    @patch("slurm_mcp.diagnostics.get_job_status")
    def test_diagnose_classifies_oom(self, mock_status, mock_log):
        mock_status.return_value = JobStatus(job_id=9, state="OUT_OF_MEMORY", exit_code=1)
        mock_log.return_value = "torch.cuda.OutOfMemoryError: CUDA out of memory"
        out = slurm_mcp.diagnose_job(9)
        assert "Classification: OOM" in out

    @patch("slurm_mcp.history.subprocess.run")
    def test_history_empty(self, mock_run):
        mock_run.return_value = SimpleNamespace(returncode=0, stdout="", stderr="")
        out = slurm_mcp.job_history(days=2)
        assert "No jobs found" in out


# ============================================================
# Unit Tests: the 7 new CLI subcommands (parity with MCP tools)
# ============================================================

class TestNewCLISubcommands:
    def test_all_registered_in_parser(self):
        from cli import slurmx
        parser = slurmx.build_parser()
        for argv in (["select-gpu", "--vram", "48"], ["history"], ["job-status", "1"],
                     ["job", "1"], ["wait", "1"], ["log", "1"], ["diagnose", "1"],
                     ["cancel", "1"]):
            ns = parser.parse_args(argv)
            assert hasattr(ns, "_run")

    @patch("slurm_mcp.select_gpu", return_value=None)
    @patch("slurm_mcp.check_availability")
    def test_select_gpu_allow_main_passes_flag(self, mock_av, mock_sel):
        from cli import select_gpu as sg
        from slurm_mcp.types import Availability
        mock_av.return_value = Availability()
        sg.run(argparse.Namespace(vram=48, allow_main=True))
        mock_sel.assert_called_once_with(48, golden_only=False)

    @patch("slurm_mcp.select_gpu", return_value=None)
    @patch("slurm_mcp.check_availability")
    def test_select_gpu_default_is_golden_only(self, mock_av, mock_sel):
        from cli import select_gpu as sg
        from slurm_mcp.types import Availability
        mock_av.return_value = Availability()
        sg.run(argparse.Namespace(vram=48, allow_main=False))
        mock_sel.assert_called_once_with(48, golden_only=True)

    @patch("slurm_mcp.job_history", return_value="HIST")
    def test_history_run(self, mock_hist, capsys):
        from cli import history
        history.run(argparse.Namespace(days=3, state=None, limit=30))
        assert "HIST" in capsys.readouterr().out
        mock_hist.assert_called_once_with(days=3, state=None, limit=30)

    @patch("slurm_mcp.diagnose_job", return_value="DIAG")
    def test_diagnose_run(self, mock_d, capsys):
        from cli import diagnose
        diagnose.run(argparse.Namespace(job_id=1, output_dir="logs", log_lines=50))
        assert "DIAG" in capsys.readouterr().out

    @patch("slurm_mcp.get_job_status")
    def test_job_status_json(self, mock_st, capsys):
        from cli import job_status
        mock_st.return_value = JobStatus(job_id=7, state="RUNNING", node="n1")
        job_status.run(argparse.Namespace(job_id=7, json_output=True))
        assert '"state": "RUNNING"' in capsys.readouterr().out

    @patch("slurm_mcp.cancel_jobs", return_value=2)
    def test_cancel_run(self, mock_c, capsys):
        from cli import cancel
        cancel.run(argparse.Namespace(job_ids=[1, 2], all_jobs=False, pending_only=False))
        assert "Cancelled 2" in capsys.readouterr().out
        mock_c.assert_called_once_with(job_ids=[1, 2], all_jobs=False, pending_only=False)

    def test_cancel_requires_target(self):
        from cli import cancel
        with pytest.raises(SystemExit):
            cancel.run(argparse.Namespace(job_ids=[], all_jobs=False, pending_only=False))

    @patch("slurm_mcp.read_job_log", return_value=None)
    def test_log_missing_exits(self, mock_l):
        from cli import log
        with pytest.raises(SystemExit):
            log.run(argparse.Namespace(job_id=1, output_dir="logs", tail=100))


# ============================================================
# Unit Tests: TUI color layer (cli/theme.py, pure classifier)
# ============================================================

class TestTheme:
    def test_classify_block_golden(self):
        from cli.theme import classify_block, Role
        golden = [
            "=== Golden Tickets (yisroel QoS) ===",
            "  rtx_pro_6000: 0/16 free (16 running)",
            "    Running:",
            "      alice: 10 GPU(s)",
            "    Pending (next first):",
            "      itay: 3 GPU(s)",
        ]
        assert classify_block(golden) == [
            Role.HEADER, Role.CARD_FULL, Role.LABEL, Role.ROW_RUNNING,
            Role.LABEL, Role.ROW_PENDING,
        ]

    def test_classify_block_cluster(self):
        from cli.theme import classify_block, Role
        assert classify_block(["=== Cluster-Wide ===", "  rtx_6000: 5/12 free"]) == [
            Role.HEADER, Role.CARD_FREE,
        ]

    def test_squeue_role(self):
        from cli.theme import squeue_role, Role
        assert squeue_role("  111 rtx6000 train me R 1:00 1 node1") == Role.SQUEUE_RUNNING
        assert squeue_role("  222 rtx6000 eval me PD 0:00 1 (Priority)") == Role.SQUEUE_PENDING
        assert squeue_role("     JOBID PARTITION NAME USER ST TIME") == Role.PLAIN

    def test_init_theme_no_color_returns_empty(self):
        from cli import theme as th
        with patch("curses.has_colors", return_value=False):
            assert th.init_theme() == {}


class TestDashboardSegments:
    """Side-by-side rows keep the golden-left and cluster-right segments as separate
    spans with independent roles — the fix for 'the whole merged row got one color'."""

    def test_merged_row_has_two_independent_roles(self):
        from cli.watch import _side_by_side_spans
        from cli.theme import Role
        # golden "Running:" (LABEL) sits beside a cluster card (CARD_FREE) on the
        # same physical line — each segment keeps its own role.
        rows = _side_by_side_spans(
            ["    Running:"], [Role.LABEL],
            ["  rtx_6000: 5/12 free"], [Role.CARD_FREE],
        )
        assert [role for _, role in rows[0]] == [Role.LABEL, Role.CARD_FREE]

    def test_left_only_overflow_is_single_span(self):
        from cli.watch import _side_by_side_spans
        from cli.theme import Role
        rows = _side_by_side_spans(
            ["  a: 0/8 free", "      deep: 1 GPU(s)"], [Role.CARD_FULL, Role.ROW_RUNNING],
            ["  a: 0/8 free"], [Role.CARD_FULL],
        )
        # row 1 has no right segment -> one span, its own role (not the right's)
        assert len(rows[1]) == 1
        assert rows[1][0][1] == Role.ROW_RUNNING

    def test_spans_flatten_matches_plain_lines(self):
        # Flattening span texts must reproduce the byte-stable plain merge.
        from cli.watch import _side_by_side_spans, _side_by_side
        from cli.theme import Role
        left = ["=== Golden ===", "  a: 0/8 free", "    Running:"]
        right = ["=== Cluster-Wide ===", "  a: 0/8 free"]
        spans = _side_by_side_spans(left, [Role.PLAIN] * 3, right, [Role.PLAIN] * 2)
        flat = ["".join(t for t, _ in r) for r in spans]
        assert flat == _side_by_side(left, right)


# Keys every config.py has carried since the first release, so importing them
# straight from `config` is safe. Anything added later must go through
# config_defaults.py with a fallback — a user's config.py is gitignored and
# never updated by `slurmx update`, so a new hard import breaks their CLI.
_BASELINE_CONFIG_KEYS = {
    "USERNAME", "MAIL_USER", "GOLDEN_QOS", "CPU_PARTITION", "CPU_QOS",
    "CPU_CPUS", "CPU_MEM", "EXCLUDE_NODES", "MAX_MEM_GB", "TIME_LIMIT",
    "START_TIMEOUT", "GPU_DEFINITIONS", "GPU_DEFINITIONS_BY_QOS",
    "CLAUDE_LOG_DIR",
}

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _repo_python_files():
    for sub in (".", "slurm_mcp", "cli", "tests", "bin"):
        d = os.path.join(_REPO_ROOT, sub)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".py"):
                yield os.path.join(d, fn)


class TestConfigDefaults:
    """A key added to the config templates is still absent from every config.py
    already on disk (it's gitignored). MAIN_PARTITION shipped as a hard
    `from config import ...` and broke `slurmx status` on every older clone."""

    def test_stale_config_still_resolves_main_partition(self):
        import importlib
        import types as _types
        import config as real_config

        stale = _types.ModuleType("config")   # a config.py predating the key
        assert not hasattr(stale, "MAIN_PARTITION")
        try:
            with patch.dict(sys.modules, {"config": stale}):
                mod = importlib.reload(sys.modules["config_defaults"])
                assert mod.MAIN_PARTITION == "main"
        finally:
            sys.modules["config"] = real_config
            importlib.reload(sys.modules["config_defaults"])

    def test_stale_config_honours_env_override(self):
        import importlib
        import types as _types
        import config as real_config

        stale = _types.ModuleType("config")
        try:
            with patch.dict(sys.modules, {"config": stale}), \
                 patch.dict(os.environ, {"SLURM_MAIN_PARTITION": "gpu"}):
                mod = importlib.reload(sys.modules["config_defaults"])
                assert mod.MAIN_PARTITION == "gpu"
        finally:
            sys.modules["config"] = real_config
            importlib.reload(sys.modules["config_defaults"])

    def test_real_config_wins_over_the_default(self):
        import config
        if hasattr(config, "MAIN_PARTITION"):
            import config_defaults
            assert config_defaults.MAIN_PARTITION == config.MAIN_PARTITION

    def test_no_module_hard_imports_a_post_release_config_key(self):
        import ast

        offenders = []
        for path in _repo_python_files():
            with open(path) as fh:
                tree = ast.parse(fh.read(), filename=path)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == "config":
                    for alias in node.names:
                        if alias.name not in _BASELINE_CONFIG_KEYS:
                            offenders.append(
                                f"{os.path.relpath(path, _REPO_ROOT)}: {alias.name}"
                            )
        assert not offenders, (
            "these names are imported straight from `config` but are not in the "
            "baseline every existing config.py has; give them a default in "
            f"config_defaults.py instead: {offenders}"
        )

    def test_templates_carry_every_baseline_key(self):
        import ast

        for name in ("default.py", "yisroel.py"):
            path = os.path.join(_REPO_ROOT, "config-examples", name)
            tree = ast.parse(open(path).read(), filename=path)
            assigned = {
                t.id
                for node in tree.body if isinstance(node, ast.Assign)
                for t in node.targets if isinstance(t, ast.Name)
            }
            missing = _BASELINE_CONFIG_KEYS - assigned
            assert not missing, f"config-examples/{name} is missing {missing}"


# ============================================================
# Live Integration Tests (run on actual cluster)
# ============================================================

def is_on_slurm_cluster():
    """Check if we're on a machine with SLURM."""
    try:
        result = subprocess.run(["squeue", "--version"], capture_output=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

live = pytest.mark.skipif(
    not is_on_slurm_cluster(),
    reason="Not on a SLURM cluster"
)


@live
class TestCheckAvailabilityLive:
    def test_returns_availability_object(self):
        avail = check_availability()
        assert isinstance(avail, Availability)

    def test_golden_has_both_types(self):
        avail = check_availability()
        assert "rtx_pro_6000" in avail.golden
        assert "rtx_6000" in avail.golden

    def test_golden_totals_match_quotas(self):
        avail = check_availability()
        assert avail.golden["rtx_pro_6000"].total == 16
        assert avail.golden["rtx_6000"].total == 12

    def test_golden_free_is_non_negative(self):
        avail = check_availability()
        for name, g in avail.golden.items():
            assert g.free >= 0, f"Golden {name} free is negative: {g.free}"
            assert g.used >= 0
            assert g.free == max(0, g.total - g.used)

    def test_cluster_has_all_gpu_types(self):
        avail = check_availability()
        for gpu in GPU_TYPES:
            assert gpu.name in avail.cluster, f"Missing cluster entry for {gpu.name}"

    def test_cluster_totals_positive(self):
        avail = check_availability()
        # At least some GPU types should have positive totals
        total_gpus = sum(c.total for c in avail.cluster.values())
        assert total_gpus > 0, "No GPUs found in cluster"

    def test_cluster_free_non_negative(self):
        avail = check_availability()
        for name, c in avail.cluster.items():
            assert c.free >= 0, f"Cluster {name} free is negative: {c.free}"

    def test_golden_users_are_strings(self):
        avail = check_availability()
        for name, g in avail.golden.items():
            for user, count in g.users.items():
                assert isinstance(user, str)
                assert isinstance(count, int)
                assert count > 0


@live
class TestSelectGPULive:
    def test_8gb_returns_something_or_none(self):
        result = select_gpu(8)
        if result is not None:
            gpu_type, partition, qos = result
            assert gpu_type in GPU_BY_NAME
            assert qos in ("yisroel", "normal")

    def test_200gb_returns_none(self):
        result = select_gpu(200)
        assert result is None

    def test_result_has_valid_partition(self):
        result = select_gpu(24)
        if result is not None:
            gpu_type, partition, qos = result
            if qos == "yisroel":
                assert partition in ("rtx_pro_6000", "rtx6000")
            else:
                assert partition == "main"


@live
class TestSubmitJobLive:
    def test_dry_run_48gb(self):
        result = submit_job(
            cmd="echo hello",
            vram_gb=48,
            dry_run=True,
        )
        # Should succeed (dry run) or fail with "not free" — both are valid
        if result.success:
            assert result.gpu_type in GPU_BY_NAME
            assert GPU_BY_NAME[result.gpu_type].vram_gb >= 48
            assert "echo hello" in result.sbatch_script

    def test_dry_run_with_all_options(self):
        result = submit_job(
            cmd="python train.py --lr 1e-4",
            vram_gb=48,
            job_name="test-all-opts",
            num_gpus=1,
            workdir="/tmp",
            output_dir="/tmp",
            dry_run=True,
        )
        if result.success:
            assert "#SBATCH --job-name test-all-opts" in result.sbatch_script
            assert "#SBATCH --mem=80G" in result.sbatch_script
            assert "#SBATCH --time 7-0:00:00" in result.sbatch_script
            assert "SCRATCH_DIR" in result.sbatch_script
            assert "cd /tmp" in result.sbatch_script
            assert "/tmp/slurm-test-all-opts-%J.out" in result.sbatch_script

    def test_dry_run_manual_gpu_type(self):
        result = submit_job(
            cmd="echo test",
            vram_gb=0,
            gpu_type="rtx_4090",
            dry_run=True,
        )
        assert result.success is True
        assert result.gpu_type == "rtx_4090"
        assert "#SBATCH --gres=gpu:rtx_4090:1" in result.sbatch_script
        assert "#SBATCH --nodes=1" in result.sbatch_script


@live
class TestMyJobsLive:
    def test_returns_list(self):
        jobs = my_jobs()
        assert isinstance(jobs, list)

    def test_jobs_have_expected_fields(self):
        jobs = my_jobs()
        if jobs:
            expected_keys = {"job_id", "name", "state", "qos", "gpu_gres",
                             "runtime", "node", "partition", "reason"}
            assert set(jobs[0].keys()) == expected_keys

    def test_qos_filter(self):
        jobs = my_jobs(qos="yisroel")
        for job in jobs:
            assert job["qos"] == "yisroel"

