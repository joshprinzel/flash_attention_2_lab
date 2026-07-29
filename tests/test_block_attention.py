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
        (1, 8, 128, 64),
        (1, 2, 17, 128),
        (1, 8, 128, 128),
    ],
)
def test_block_attention_matches_reference(
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

    actual = flash_attention_cuda.block_attention_forward(
        query,
        key,
        value,
    )

    expected = attention_reference(
        query,
        key,
        value,
    )

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-4,
        atol=1e-4,
    )