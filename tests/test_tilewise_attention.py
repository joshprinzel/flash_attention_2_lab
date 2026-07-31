import pytest
import torch

import flash_attention_cuda

from flash_attention import attention_reference


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required",
)


@pytest.mark.parametrize(
    ("batch", "heads", "sequence"),
    [
        (1, 1, 1),
        (1, 1, 15),
        (1, 2, 16),
        (1, 2, 17),
        (1, 4, 31),
        (1, 4, 32),
        (1, 4, 33),
        (2, 3, 65),
        (1, 8, 128),
        (1, 8, 512),
    ],
)
def test_tilewise_attention_matches_reference(
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
        dtype=torch.float32,
    )

    key = torch.randn_like(query)
    value = torch.randn_like(query)

    actual = (
        flash_attention_cuda
        .tilewise_attention_forward(
            query,
            key,
            value,
        )
    )

    expected = attention_reference(
        query,
        key,
        value,
    )

    torch.testing.assert_close(
        actual,
        expected,
        rtol=3e-4,
        atol=3e-4,
    )