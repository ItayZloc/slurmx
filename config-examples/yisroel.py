"""Cluster-specific configuration — Yisroel's lab.

Pre-filled with Yisroel's QoS and golden ticket counts.
Copy to config.py — MAIL_USER defaults to $USER@post.bgu.ac.il.
"""

import os

# --- Auto-detected ---
USERNAME = os.environ.get("USER", "")

# --- Personal ---
# Defaults to $USER@post.bgu.ac.il; override per-shell with SLURM_MAIL_USER.
MAIL_USER = os.environ.get("SLURM_MAIL_USER", f"{USERNAME}@post.bgu.ac.il")
# SLURM mail events, passed to `sbatch --mail-type`. Valid: NONE, BEGIN, END,
# FAIL, REQUEUE, ALL, TIME_LIMIT, TIME_LIMIT_90/80/50, ARRAY_TASKS. An empty
# list or ["NONE"] drops the mail lines from the script entirely.
MAIL_TYPE = ["END", "FAIL"]

# --- Golden QoS ---
# Golden-ticket QoS list. First entry is primary for job submission.
GOLDEN_QOS = ["yisroel"]

# --- Excluded nodes ---
# Nodes to exclude from job placement (sbatch --exclude). Override via
# SLURM_EXCLUDE_NODES="nodeA,nodeB" env var if needed.
_EXCLUDE_NODES_DEFAULT = ""
EXCLUDE_NODES = [
    n.strip()
    for n in os.environ.get("SLURM_EXCLUDE_NODES", _EXCLUDE_NODES_DEFAULT).split(",")
    if n.strip()
]

# --- Resource Defaults ---
MAX_MEM_GB = 80
CPU_CPUS = 4
CPU_MEM = "16G"
TIME_LIMIT = "7-0:00:00"
START_TIMEOUT = 300  # seconds to wait for job to start

# --- GPU Definitions ---
# Each tuple: (name, display_name, vram_gb, golden_tickets, golden_partition)
# golden_tickets and golden_partition are QoS-specific reserved allocations.
GPU_DEFINITIONS_BY_QOS = {
    "yisroel": [
        ("rtx_pro_6000", "RTX 6000 Pro", 96, 16, "rtx_pro_6000"),
        ("rtx_6000",     "RTX 6000",     48, 12, "rtx6000"),
        ("rtx_4090",     "RTX 4090",     24, 0,  None),
        ("rtx_3090",     "RTX 3090",     24, 0,  None),
        ("rtx_2080",     "RTX 2080",     11, 0,  None),
        ("gtx_1080",     "GTX 1080",     8,  0,  None),
    ],
}

# Back-compat: flat list = primary QoS's definitions.
GPU_DEFINITIONS = GPU_DEFINITIONS_BY_QOS[GOLDEN_QOS[0]]
