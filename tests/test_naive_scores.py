import math

import pytest
import torch

import flash_attention_cuda


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required",
)


@pytest.mark.parametrize(
    ("batch", "heads", "sequence", "head_dimension"),
    [
        (1, 1, 1, 16),
        (1, 1, 7, 32),
        (1, 2, 17, 64),
        (2, 3, 31, 32),
        (1, 4, 65, 64),
    ],
)
def test_naive_attention_scores_match_pytorch(
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

    actual = flash_attention_cuda.naive_attention_scores(
        query,
        key,
    )

    expected = torch.matmul(
        query,
        key.transpose(-2, -1),
    )

    expected *= 1.0 / math.sqrt(head_dimension)

    assert actual.shape == (
        batch,
        heads,
        sequence,
        sequence,
    )

    assert actual.dtype == torch.float32
    assert actual.device == query.device
    assert actual.is_contiguous()

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-5,
        atol=1e-5,
    )


def test_naive_scores_preserve_batch_head_separation() -> None:
    query = torch.zeros(
        2,
        2,
        3,
        4,
        device="cuda",
        dtype=torch.float32,
    )
    key = torch.zeros_like(query)

    # Give every batch/head pair a distinct constant.
    query[0, 0].fill_(1.0)
    query[0, 1].fill_(2.0)
    query[1, 0].fill_(3.0)
    query[1, 1].fill_(4.0)

    key[0, 0].fill_(5.0)
    key[0, 1].fill_(6.0)
    key[1, 0].fill_(7.0)
    key[1, 1].fill_(8.0)

    actual = flash_attention_cuda.naive_attention_scores(
        query,
        key,
    )

    expected = torch.matmul(
        query,
        key.transpose(-2, -1),
    ) / math.sqrt(4)

    torch.testing.assert_close(actual, expected)


def test_naive_scores_reject_noncontiguous_input() -> None:
    query = torch.randn(
        1,
        2,
        8,
        32,
        device="cuda",
        dtype=torch.float32,
    )

    key = torch.randn_like(query)

    noncontiguous_query = query.transpose(1, 2)

    assert not noncontiguous_query.is_contiguous()

    with pytest.raises(RuntimeError, match="contiguous"):
        flash_attention_cuda.naive_attention_scores(
            noncontiguous_query,
            key,
        )