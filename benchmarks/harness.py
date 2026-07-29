from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class BenchmarkConfig:
    warmup_iterations: int = 25
    samples: int = 20
    iterations_per_sample: int = 100


@dataclass(frozen=True)
class BenchmarkResult:
    kernel_name: str
    median_ms: float
    minimum_ms: float
    p10_ms: float
    p90_ms: float
    maximum_ms: float


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def benchmark_cuda_operation(
    name: str,
    operation: Callable[[], Tensor],
    config: BenchmarkConfig,
) -> BenchmarkResult:
    with torch.inference_mode():
        for _ in range(config.warmup_iterations):
            operation()

        torch.cuda.synchronize()

        sample_times_ms: list[float] = []

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        for _ in range(config.samples):
            start.record()

            for _ in range(config.iterations_per_sample):
                operation()

            end.record()
            end.synchronize()

            sample_times_ms.append(
                start.elapsed_time(end)
                / config.iterations_per_sample
            )

    return BenchmarkResult(
        kernel_name=name,
        median_ms=statistics.median(sample_times_ms),
        minimum_ms=min(sample_times_ms),
        p10_ms=percentile(sample_times_ms, 0.10),
        p90_ms=percentile(sample_times_ms, 0.90),
        maximum_ms=max(sample_times_ms),
    )