"""GPU selection — picks the smallest GPU that meets a VRAM requirement,
preferring golden tickets on the primary QoS, then falling back to cluster-wide."""

from __future__ import annotations

from typing import Optional

from . import availability
from .gpu_catalog import GPU_TYPES, PRIMARY_QOS


def select_gpu(vram_gb: int) -> Optional[tuple[str, str, str]]:
    """
    Pick the best available GPU for the given VRAM requirement.

    Strategy:
      1. Filter GPU types where vram >= vram_gb, sort cheapest first
      2. Try golden tickets first (smallest fitting golden GPU)
      3. Fall back to normal QoS cluster-wide (smallest first)
      4. Return None if nothing available

    Returns:
        (gpu_type, partition, qos) or None
    """
    candidates = [g for g in GPU_TYPES if g.vram_gb >= vram_gb]
    candidates.sort(key=lambda g: (g.vram_gb, g.name))

    if not candidates:
        return None

    avail = availability.check_availability()

    golden_candidates = [g for g in candidates if g.golden_quota > 0]
    for gpu in golden_candidates:
        golden = avail.golden.get(gpu.name)
        if golden and golden.free > 0:
            return (gpu.name, gpu.golden_partition, PRIMARY_QOS)

    for gpu in candidates:
        cluster = avail.cluster.get(gpu.name)
        if cluster and cluster.free > 0:
            return (gpu.name, "main", "normal")

    return None
