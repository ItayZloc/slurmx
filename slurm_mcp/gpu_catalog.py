"""GPU type catalog — parsed once from config at import time.

Exposes:
  GPU_TYPES         — list of GPUType for the primary QoS (back-compat).
  GPU_BY_NAME       — name -> GPUType for the primary QoS.
  GPU_TYPES_BY_QOS  — qos -> [GPUType] for every configured QoS.
  PRIMARY_QOS       — GOLDEN_QOS[0]; used for submission paths.
"""

from __future__ import annotations

from config import GOLDEN_QOS, GPU_DEFINITIONS, GPU_DEFINITIONS_BY_QOS

from .types import GPUType


PRIMARY_QOS = GOLDEN_QOS[0]

GPU_TYPES = [
    GPUType(name=d[0], display_name=d[1], vram_gb=d[2],
            golden_quota=d[3], golden_partition=d[4])
    for d in GPU_DEFINITIONS
]

GPU_BY_NAME = {g.name: g for g in GPU_TYPES}

GPU_TYPES_BY_QOS = {
    qos: [
        GPUType(name=d[0], display_name=d[1], vram_gb=d[2],
                golden_quota=d[3], golden_partition=d[4])
        for d in defs
    ]
    for qos, defs in GPU_DEFINITIONS_BY_QOS.items()
}
