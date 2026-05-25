"""Cluster-specific configuration — TEMPLATE.

User-specific values are read from environment variables.
Add to your ~/.bashrc:
  export SLURM_MAIL_USER="your@email.com"
  export SLURM_GOLDEN_QOS="yisroel"  # or "yisroel,shared" for multiple QoS
"""

import os

# --- Auto-detected ---
USERNAME = os.environ.get("USER", "")

# --- Personal (from env) ---
MAIL_USER = os.environ.get("SLURM_MAIL_USER", "")

# --- QoS / Partitions ---
# Golden-ticket QoS list. First entry is treated as the primary for job
# submission. Set SLURM_GOLDEN_QOS="a,b" to include multiple.
GOLDEN_QOS = [
    q.strip()
    for q in os.environ.get("SLURM_GOLDEN_QOS", "yisroel").split(",")
    if q.strip()
]
CPU_PARTITION = "cpu"
CPU_QOS = "normal"

# --- Excluded nodes ---
# Nodes to exclude from job placement (sbatch --exclude). Override via
# SLURM_EXCLUDE_NODES="nodeA,nodeB" env var.
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
# Each tuple: (name, display_name, vram_gb, golden_quota, golden_partition)
# golden_quota and golden_partition are QoS-specific reserved allocations.
# Add entries under each QoS the user belongs to.
# TODO: fill in golden_quota and golden_partition for your QoS
GPU_DEFINITIONS_BY_QOS = {
    "yisroel": [
        ("rtx_pro_6000", "RTX 6000 Pro", 96, 0, None),  # TBA
        ("rtx_6000",     "RTX 6000",     48, 0, None),  # TBA
        ("rtx_4090",     "RTX 4090",     24, 0, None),  # TBA
        ("rtx_3090",     "RTX 3090",     24, 0, None),  # TBA
        ("rtx_2080",     "RTX 2080",     11, 0, None),  # TBA
        ("gtx_1080",     "GTX 1080",     8,  0, None),  # TBA
    ],
}

# Back-compat: flat list = primary QoS's definitions.
GPU_DEFINITIONS = GPU_DEFINITIONS_BY_QOS[GOLDEN_QOS[0]]

# --- Claude Job Defaults (auto-populated from USERNAME) ---
CLAUDE_LOG_DIR = f"/home/{USERNAME}/.claude/logs"
