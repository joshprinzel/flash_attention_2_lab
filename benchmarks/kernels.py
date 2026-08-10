from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

import flash_attention_cuda

from benchmarks.workloads import AttentionInputs, AttentionWorkload


KernelFunction = Callable[
    [AttentionInputs, AttentionWorkload],
    Tensor,
]


@dataclass(frozen=True)
class KernelSpec:
    name: str
    function: KernelFunction
    supported: Callable[[AttentionWorkload], bool]


def naive_online(
    inputs: AttentionInputs,
    workload: AttentionWorkload,
) -> Tensor:
    return flash_attention_cuda.naive_attention_forward(
        inputs.query,
        inputs.key,
        inputs.value,
    )

def block_online(
    inputs: AttentionInputs,
    workload: AttentionWorkload,
) -> Tensor:
    return flash_attention_cuda.block_attention_forward(
        inputs.query,
        inputs.key,
        inputs.value,
    )

def tiled_simt(
        inputs: AttentionInputs,
        workload: AttentionWorkload
) -> Tensor:
    return flash_attention_cuda.tiled_attention_forward(
        inputs.query,
        inputs.key,
        inputs.value,
    )

def tilewise_simt(
        inputs: AttentionInputs,
        workload: AttentionWorkload
) -> Tensor:
    return flash_attention_cuda.tilewise_attention_forward(
        inputs.query,
        inputs.key,
        inputs.value
    )
def tensorcore_attention(
        inputs: AttentionInputs,
        workload: AttentionWorkload
) -> Tensor:
    return flash_attention_cuda.tensorcore_attention_forward(
        inputs.query,
        inputs.key,
        inputs.value
    )

def tensorcore_attention_bc32(
        inputs: AttentionInputs,
        workload: AttentionWorkload
) -> Tensor:
    return flash_attention_cuda.tensorcore_attention_forward_bc32(
        inputs.query,
        inputs.key,
        inputs.value
    )

def tensorcore_attention_bc32_raw_pv(
    inputs: AttentionInputs,
    workload: AttentionWorkload,
) -> Tensor:
    return (
        flash_attention_cuda
        .tensorcore_attention_forward_bc32_raw_pv(
            inputs.query,
            inputs.key,
            inputs.value,
        )
    )
def tensorcore_attention_bc32_raw_qk_raw_pv(
    inputs: AttentionInputs,
    workload: AttentionWorkload,
) -> Tensor:
    return (
        flash_attention_cuda
        .tensorcore_attention_forward_bc32_raw_qk_raw_pv(
            inputs.query,
            inputs.key,
            inputs.value,
        )
    )

def tensorcore_attention_production_128x64(
    inputs: AttentionInputs,
    workload: AttentionWorkload,
) -> Tensor:
    return (
        flash_attention_cuda
        .tensorcore_attention_forward_production_128x64(
            inputs.query,
            inputs.key,
            inputs.value,
        )
    )
def pytorch_sdpa(
    inputs: AttentionInputs,
    workload: AttentionWorkload,
) -> Tensor:
    return F.scaled_dot_product_attention(
        inputs.query,
        inputs.key,
        inputs.value,
        is_causal=workload.causal,
    )


KERNELS = [
    KernelSpec(
        name="naive_online",
        function=naive_online,
        supported=lambda workload: (
            workload.dtype == torch.float32
            and workload.query_length == workload.key_length
            and workload.head_dimension in {64, 128}
            and not workload.causal
        ),
    ),
    KernelSpec(
        name="pytorch_sdpa",
        function=pytorch_sdpa,
        supported=lambda workload: True,
    ),
    KernelSpec(
        name="block_online",
        function=block_online,
        supported=lambda workload: (
            workload.dtype == torch.float32
            and workload.query_length == workload.key_length
            and workload.head_dimension in {64, 128}
            and not workload.causal
        ),
    ),
    KernelSpec(
        name="tiled_simt",
        function=tiled_simt,
        supported=lambda workload: (
            workload.dtype == torch.float32
            and workload.query_length == workload.key_length
            and workload.head_dimension == 64
            and not workload.causal
        ),
    ),

    KernelSpec(
        name="tilewise_simt",
        function=tilewise_simt,
        supported=lambda workload: (
            workload.dtype == torch.float32
            and workload.query_length == workload.key_length
            and workload.head_dimension == 64
            and not workload.causal
        ),
    ),

    KernelSpec(
        name="tensorcore_attention",
        function=tensorcore_attention,
        supported=lambda workload: (
            workload.dtype == torch.float16
            and workload.query_length == workload.key_length
            and workload.head_dimension == 64
            and workload.query_length % 16 == 0
            and not workload.causal
        ),
    ),
    KernelSpec(
        name="tensorcore_attention_bc32",
        function=tensorcore_attention_bc32,
        supported=lambda workload: (
            workload.dtype == torch.float16
            and workload.query_length
                == workload.key_length
            and workload.head_dimension == 64
            and workload.query_length % 32 == 0
            and not workload.causal
        ),
    ),
    KernelSpec(
        name="tensorcore_bc32_raw_pv",
        function=tensorcore_attention_bc32_raw_pv,
        supported=lambda workload: (
            workload.dtype == torch.float16
            and workload.query_length == workload.key_length
            and workload.head_dimension == 64
            and workload.query_length % 32 == 0
            and not workload.causal
        ),
    ),
    KernelSpec(
        name="tensorcore_bc32_raw_qk_raw_pv",
        function=tensorcore_attention_bc32_raw_qk_raw_pv,
        supported=lambda workload: (
            workload.dtype == torch.float16
            and workload.query_length == workload.key_length
            and workload.head_dimension == 64
            and workload.query_length % 32 == 0
            and not workload.causal
        ),
    ),
    KernelSpec(
        name="tensorcore_production_128x64",
        function=tensorcore_attention_production_128x64,
        supported=lambda workload: (
            workload.dtype == torch.float16
            and workload.query_length == workload.key_length
            and workload.head_dimension == 64
            and workload.query_length % 32 == 0
            and not workload.causal
        ),
    ),
]