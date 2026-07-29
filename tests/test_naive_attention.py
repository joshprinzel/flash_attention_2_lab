import pytest
import torch

import flash_attention_cuda

from flash_attention import attention_reference


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required",
)


@pytest.mark.parametrize(
    ("batch", "heads", "sequence", "head_dimension"),
    [
        (1, 1, 1, 64),
        (1, 2, 17, 64),
        (2, 3, 31, 64),
        (1, 4, 65, 64),
        (1, 2, 17, 128),
    ],
)
def test_naive_online_attention_matches_reference(
    batch: int,
    heads: int,
    sequence: int,
    head_dimension: int,
) -> None:
    torch.manual_seed(0)

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

    actual = flash_attention_cuda.naive_attention_forward(
        query,
        key,
        value,
    )

    expected = attention_reference(
        query,
        key,
        value,
        causal=False,
    )

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-4,
        atol=1e-4,
    )


def test_naive_online_attention_does_not_modify_inputs() -> None:
    query = torch.randn(1, 2, 17, 64, device="cuda")
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    query_before = query.clone()
    key_before = key.clone()
    value_before = value.clone()

    flash_attention_cuda.naive_attention_forward(
        query,
        key,
        value,
    )

    torch.testing.assert_close(query, query_before)
    torch.testing.assert_close(key, key_before)
    torch.testing.assert_close(value, value_before)


def test_naive_online_attention_rejects_unsupported_head_dimension() -> None:
    query = torch.randn(1, 1, 8, 32, device="cuda")
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    with pytest.raises(RuntimeError, match="64 and 128"):
        flash_attention_cuda.naive_attention_forward(
            query,
            key,
            value,
        )