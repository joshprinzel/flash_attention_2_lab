from __future__ import annotations

import argparse

import torch

from benchmarks.harness import (
    BenchmarkConfig,
    benchmark_cuda_operation,
)
from benchmarks.kernels import KERNELS
from benchmarks.workloads import (
    AttentionWorkload,
    generate_attention_inputs,
)

DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16
}

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--query-length", type=int, default=512)
    parser.add_argument("--key-length", type=int, default=None)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--dtype", type=str, choices=list(DTYPES), default="float32")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=25)

    args = parser.parse_args()

    key_length = (
        args.key_length
        if args.key_length is not None
        else args.query_length
    )

    workload = AttentionWorkload(
        batch=args.batch,
        heads=args.heads,
        query_length=args.query_length,
        key_length=key_length,
        head_dimension=args.head_dim,
        dtype=DTYPES[args.dtype],
        causal=args.causal,
    )

    inputs = generate_attention_inputs(workload)

    config = BenchmarkConfig(
        warmup_iterations=args.warmup,
        samples=args.samples,
        iterations_per_sample=args.iterations,
    )

    print(f"GPU:      {torch.cuda.get_device_name()}")
    print(f"Workload: {workload.label}")
    print()

    results = []

    for kernel in KERNELS:
        if not kernel.supported(workload):
            print(f"{kernel.name:20s} unsupported")
            continue

        result = benchmark_cuda_operation(
            kernel.name,
            lambda kernel=kernel: kernel.function(inputs, workload),
            config,
        )

        results.append(result)

        print(
            f"{result.kernel_name:20s} "
            f"median={result.median_ms:9.4f} ms  "
            f"p10={result.p10_ms:9.4f} ms  "
            f"p90={result.p90_ms:9.4f} ms"
        )

    baseline = next(
        (
            result
            for result in results
            if result.kernel_name == "pytorch_sdpa"
        ),
        None,
    )

    if baseline is not None:
        print()

        for result in results:
            ratio = result.median_ms / baseline.median_ms
            print(
                f"{result.kernel_name:20s} "
                f"{ratio:8.2f}x SDPA latency"
            )


if __name__ == "__main__":
    main()