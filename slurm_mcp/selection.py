"""GPU resource selection for metadata-aware submissions."""

from __future__ import annotations

from dataclasses import dataclass

from config_defaults import MAIN_PARTITION

from . import availability
from .gpu_catalog import GPU_TYPES, PRIMARY_QOS


@dataclass(frozen=True)
class GPUChoice:
    gpu_type: str
    num_gpus: int
    partition: str
    qos: str


def _candidates(total_vram_gb: int, supports_gpu_sharding: bool):
    counts = (1, 2) if supports_gpu_sharding else (1,)
    candidates = [
        (gpu, count)
        for gpu in GPU_TYPES
        for count in counts
        if gpu.vram_gb * count >= total_vram_gb
    ]
    return sorted(
        candidates,
        key=lambda item: (
            item[0].vram_gb * item[1], item[1], item[0].vram_gb, item[0].name,
        ),
    )


def select_resources(
    total_vram_gb: int,
    supports_gpu_sharding: bool,
    preemption_safe: bool,
) -> GPUChoice | None:
    """Choose up to two same-type GPUs using the script's declared policy."""
    candidates = _candidates(total_vram_gb, supports_gpu_sharding)

    if not preemption_safe:
        for gpu, count in candidates:
            if gpu.golden_partition:
                return GPUChoice(gpu.name, count, gpu.golden_partition, PRIMARY_QOS)
        return None

    avail = availability.check_availability()
    for gpu, count in candidates:
        golden = avail.golden.get(gpu.name)
        node_free = avail.node_free.get(gpu.golden_partition, {}).get(gpu.name, 0)
        if golden and golden.free >= count and node_free >= count and gpu.golden_partition:
            return GPUChoice(gpu.name, count, gpu.golden_partition, PRIMARY_QOS)

    for gpu, count in candidates:
        cluster = avail.cluster.get(gpu.name)
        node_free = avail.node_free.get(MAIN_PARTITION, {}).get(gpu.name, 0)
        if cluster and cluster.free >= count and node_free >= count:
            return GPUChoice(gpu.name, count, MAIN_PARTITION, "normal")
    return None


def select_gpu(vram_gb: int, golden_only: bool = False):
    """Backward-compatible advisory one-GPU selection."""
    choice = select_resources(
        vram_gb, supports_gpu_sharding=False, preemption_safe=not golden_only,
    )
    if choice is None:
        return None
    return (choice.gpu_type, choice.partition, choice.qos)
