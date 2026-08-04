import math

import pytest
import torch

import flash_attention_cuda


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required",
)


@pytest.mark.parametrize(
    ("batch", "heads", "sequence"),
    [
        (1, 1, 16),
        (1, 2, 64),
        (1, 4, 80),
        (1, 8, 128),
        (2, 3, 256),
    ],
)
def test_tensorcore_attention_matches_reference(
    batch: int,
    heads: int,
    sequence: int,
) -> None:
    torch.manual_seed(0)

    query = torch.randn(
        batch,
        heads,
        sequence,
        64,
        device="cuda",
        dtype=torch.float16,
    )

    key = torch.randn_like(query)
    value = torch.randn_like(query)

    actual = (
        flash_attention_cuda
        .tensorcore_attention_forward(
            query,
            key,
            value,
        )
    )

    scores = (
        query.float()
        @ key.float().transpose(-2, -1)
    ) / math.sqrt(64)

    expected = (
        torch.softmax(
            scores,
            dim=-1,
        )
        @ value.float()
    )

    assert actual.dtype == torch.float16
    assert actual.shape == expected.shape

    torch.testing.assert_close(
        actual.float(),
        expected,
        rtol=3e-2,
        atol=3e-2,
    )