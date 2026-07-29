import math

import pytest
import torch
import torch.nn.functional as F

from flash_attention import attention_reference


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    ("batch", "heads", "sequence", "head_dimension"),
    [
        (1, 1, 1, 16),
        (1, 2, 17, 32),
        (2, 4, 64, 64),
        (1, 8, 65, 128),
    ],
)
def test_reference_matches_pytorch_sdpa(
    batch: int,
    heads: int,
    sequence: int,
    head_dimension: int,
    causal: bool,
) -> None:
    torch.manual_seed(0)

    query = torch.randn(batch, heads, sequence, head_dimension)
    key = torch.randn(batch, heads, sequence, head_dimension)
    value = torch.randn(batch, heads, sequence, head_dimension)

    actual = attention_reference(
        query,
        key,
        value,
        causal=causal,
    )

    expected = F.scaled_dot_product_attention(
        query,
        key,
        value,
        is_causal=causal,
    )

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-5,
        atol=1e-6,
    )


def test_reference_supports_different_query_and_key_lengths() -> None:
    torch.manual_seed(0)

    query = torch.randn(2, 4, 7, 32)
    key = torch.randn(2, 4, 11, 32)
    value = torch.randn(2, 4, 11, 32)

    output = attention_reference(query, key, value)

    assert output.shape == (2, 4, 7, 32)


def test_reference_supports_custom_scale() -> None:
    torch.manual_seed(0)

    query = torch.randn(1, 2, 8, 32)
    key = torch.randn(1, 2, 8, 32)
    value = torch.randn(1, 2, 8, 32)

    scale = 0.25

    actual = attention_reference(
        query,
        key,
        value,
        scale=scale,
    )

    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    expected = torch.matmul(torch.softmax(scores, dim=-1), value)

    torch.testing.assert_close(actual, expected)


def test_default_scale_is_inverse_square_root_head_dimension() -> None:
    torch.manual_seed(0)

    query = torch.randn(1, 1, 4, 64)
    key = torch.randn(1, 1, 4, 64)
    value = torch.randn(1, 1, 4, 64)

    actual = attention_reference(query, key, value)

    scores = torch.matmul(query, key.transpose(-2, -1))
    scores *= 1.0 / math.sqrt(64)

    expected = torch.matmul(torch.softmax(scores, dim=-1), value)

    torch.testing.assert_close(actual, expected)


def test_rejects_invalid_rank() -> None:
    query = torch.randn(8, 64)
    key = torch.randn(8, 64)
    value = torch.randn(8, 64)

    with pytest.raises(ValueError, match="must have shape"):
        attention_reference(query, key, value)


def test_rejects_mismatched_key_value_lengths() -> None:
    query = torch.randn(1, 1, 8, 64)
    key = torch.randn(1, 1, 8, 64)
    value = torch.randn(1, 1, 7, 64)

    with pytest.raises(ValueError, match="same sequence length"):
        attention_reference(query, key, value)