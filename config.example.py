"""Cluster-specific configuration — TEMPLATE.

Copy this to config.py and fill in the personal fields marked with TODO.
Shared cluster values (GPU definitions, QoS, partitions) are pre-filled.
"""

import os

# --- Auto-detected ---
USERNAME = os.environ.get("USER", "")

# --- Personal (TODO: fill in your email) ---
MAIL_USER = ""  # e.g. "username@post.bgu.ac.il"

# --- QoS / Partitions ---
GOLDEN_QOS = "yisroel"
CPU_PARTITION = "cpu"
CPU_QOS = "normal"

# --- Resource Defaults ---
MAX_MEM_GB = 80
CPU_CPUS = 4
CPU_MEM = "16G"
TIME_LIMIT = "7-0:00:00"
START_TIMEOUT = 300  # seconds to wait for job to start

# --- GPU Definitions ---
# Each tuple: (name, display_name, vram_gb, golden_quota, golden_partition)
# golden_quota and golden_partition are QoS-specific reserved allocations
GPU_DEFINITIONS = [
    ("rtx_pro_6000", "RTX 6000 Pro", 96, 16, "rtx_pro_6000"),
    ("rtx_6000",     "RTX 6000",     48, 12, "rtx6000"),
    ("rtx_4090",     "RTX 4090",     24, 0,  None),
    ("rtx_3090",     "RTX 3090",     24, 0,  None),
    ("rtx_2080",     "RTX 2080",     11, 0,  None),
    ("gtx_1080",     "GTX 1080",     8,  0,  None),
]

# --- Claude Job Defaults (auto-populated from USERNAME) ---
CLAUDE_LOG_DIR = f"/home/{USERNAME}/.claude/logs"
