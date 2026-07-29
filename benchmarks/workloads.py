from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class AttentionWorkload:
    batch: int
    heads: int
    query_length: int
    key_length: int
    head_dimension: int
    dtype: torch.dtype = torch.float32
    causal: bool = False
    seed: int = 0

    @property
    def query_shape(self) -> tuple[int, int, int, int]:
        return (
            self.batch,
            self.heads,
            self.query_length,
            self.head_dimension,
        )

    @property
    def key_value_shape(self) -> tuple[int, int, int, int]:
        return (
            self.batch,
            self.heads,
            self.key_length,
            self.head_dimension,
        )

    @property
    def label(self) -> str:
        return (
            f"B={self.batch}, H={self.heads}, "
            f"Nq={self.query_length}, Nk={self.key_length}, "
            f"D={self.head_dimension}, "
            f"dtype={str(self.dtype).removeprefix('torch.')}, "
            f"causal={self.causal}"
        )


@dataclass
class AttentionInputs:
    query: Tensor
    key: Tensor
    value: Tensor


def generate_attention_inputs(
    workload: AttentionWorkload,
    *,
    device: torch.device | str = "cuda",
) -> AttentionInputs:
    generator = torch.Generator(device=device)
    generator.manual_seed(workload.seed)

    query = torch.randn(
        workload.query_shape,
        device=device,
        dtype=workload.dtype,
        generator=generator,
    )

    key = torch.randn(
        workload.key_value_shape,
        device=device,
        dtype=workload.dtype,
        generator=generator,
    )

    value = torch.randn(
        workload.key_value_shape,
        device=device,
        dtype=workload.dtype,
        generator=generator,
    )

    return AttentionInputs(
        query=query.contiguous(),
        key=key.contiguous(),
        value=value.contiguous(),
    )