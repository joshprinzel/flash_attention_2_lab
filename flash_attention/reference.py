from __future__ import annotations

import math

import torch
from torch import Tensor


def attention_reference(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        causal: bool = False,
        scale: float | None = None
) -> Tensor:
    """
    Pytorch reference implementation of scaled dot-product attention.

    Expected tensor layouts
        query: [batch, heads, query_length, head_dimension]
        key:   [batch, heads, key_length, head_dimension]
        value: [batch, heads, key_length, head_dimension]

    The score matrix and softmax are computed in FP32 for numerical stability.
    The result is converted back to the input dtype
    """

    _validate_inputs(query, key, value)

    query_length = query.shape[-2]
    key_length = key.shape[-2]
    head_dimension = query.shape[-1]

    attention_scale = scale if scale is not None else 1.0 / math.sqrt(head_dimension)

    #Perform the reference computation in FP32, even when the inputs are FP16
    query_fp32 = query.float()
    key_fp32 = key.float()
    value_fp32 = value.float()

    scores = torch.matmul(
        query_fp32,
        key_fp32.transpose(-2, -1)
    )

    scores *= attention_scale

    if causal:
        causal_mask = torch.ones(
            (query_length, key_length),
            device=query.device,
            dtype = torch.bool,
        ).triu(diagonal=1)

        scores = scores.masked_fill(causal_mask, float("-inf"))

    probabilities = torch.softmax(scores, dim=-1)
    output = torch.matmul(probabilities, value_fp32)

    return output.to(dtype=query.dtype)



def _validate_inputs(query: Tensor, key: Tensor, value: Tensor) -> None:
    tensors = {
        "query": query,
        "key": key,
        "value": value,
    }

    for name, tensor in tensors.items():
        if tensor.ndim != 4:
            raise ValueError(
                f"{name} must have shape [batch, heads, sequence, head_dimension], "
                f"but received shape {tuple(tensor.shape)}"
            )

        if not tensor.is_floating_point():
            raise TypeError(
                f"{name} must use a floating-point dtype, but received {tensor.dtype}"
            )

    if query.device != key.device or query.device != value.device:
        raise ValueError("query, key, and value must be on the same device")

    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError("query, key, and value must use the same dtype")

    if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0]:
        raise ValueError("query, key, and value must have the same batch size")

    if query.shape[1] != key.shape[1] or query.shape[1] != value.shape[1]:
        raise ValueError("query, key, and value must have the same number of heads")

    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key must have the same head dimension")

    if key.shape[-2] != value.shape[-2]:
        raise ValueError("key and value must have the same sequence length")