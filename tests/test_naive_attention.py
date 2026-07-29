import pytest
import torch

import flash_attention_cuda


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required",
)
@pytest.mark.parametrize(
    ("batch", "heads", "sequence", "head_dimension"),
    [
        (1, 1, 1, 16),
        (1, 2, 17, 32),
        (2, 4, 64, 64),
    ],
)
def test_naive_attention_launch_geometry(
    batch: int,
    heads: int,
    sequence: int,
    head_dimension: int,
) -> None:
    query = torch.randn(
        batch,
        heads,
        sequence,
        head_dimension,
        device="cuda",
        dtype=torch.float32,
    )
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    output = flash_attention_cuda.naive_attention_forward(
        query,
        key,
        value,
    )

    assert output.shape == query.shape
    assert output.dtype == query.dtype
    assert output.device == query.device
    assert output.is_contiguous()

    torch.testing.assert_close(
        output,
        torch.zeros_like(query),
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required",
)
def test_naive_attention_rejects_float16_for_now() -> None:
    query = torch.randn(
        1,
        1,
        8,
        32,
        device="cuda",
        dtype=torch.float16,
    )

    with pytest.raises(RuntimeError, match="float32"):
        flash_attention_cuda.naive_attention_forward(
            query,
            query,
            query,
        )