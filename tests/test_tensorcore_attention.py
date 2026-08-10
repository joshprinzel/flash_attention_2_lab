import math

import pytest
import torch
import torch.nn.functional as F
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


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required",
)
@pytest.mark.parametrize(
    ("batch", "heads", "sequence_length"),
    [
        (1, 1, 32),
        (1, 2, 64),
        (1, 4, 96),
        (1, 8, 128),
        (1, 8, 512),
    ],
)
def test_tensorcore_attention_forward_bc32_matches_sdpa(
    batch: int,
    heads: int,
    sequence_length: int,
) -> None:
    torch.manual_seed(0)

    shape = (
        batch,
        heads,
        sequence_length,
        64,
    )

    query = torch.randn(
        shape,
        device="cuda",
        dtype=torch.float16,
    ).contiguous()

    key = torch.randn(
        shape,
        device="cuda",
        dtype=torch.float16,
    ).contiguous()

    value = torch.randn(
        shape,
        device="cuda",
        dtype=torch.float16,
    ).contiguous()

    expected = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
    )

    actual = (
        flash_attention_cuda
        .tensorcore_attention_forward_bc32(
            query,
            key,
            value,
        )
    )

    assert actual.shape == expected.shape
    assert actual.dtype == torch.float16
    assert actual.is_cuda

    torch.testing.assert_close(
        actual,
        expected,
        rtol=3e-2,
        atol=3e-2,
    )


  
    
def test_tensorcore_attention_forward_bc32_rejects_partial_key_tile(
) -> None:
    shape = (1, 1, 48, 64)

    query = torch.randn(
        shape,
        device="cuda",
        dtype=torch.float16,
    )

    key = torch.randn_like(query)
    value = torch.randn_like(query)

    with pytest.raises(
        RuntimeError,
        match="divisible by 32",
    ):
        (
            flash_attention_cuda
            .tensorcore_attention_forward_bc32(
                query,
                key,
                value,
            )
        )


@pytest.mark.parametrize(
    ("batch", "heads", "sequence_length"),
    [
        (1, 1, 32),
        (1, 2, 64),
        (1, 4, 96),
        (1, 8, 128),
        (1, 8, 512),
    ],
)
def test_tensorcore_attention_bc32_raw_pv_matches_sdpa(
    batch: int,
    heads: int,
    sequence_length: int,
) -> None:
    torch.manual_seed(0)

    shape = (
        batch,
        heads,
        sequence_length,
        64,
    )

    query = torch.randn(
        shape,
        device="cuda",
        dtype=torch.float16,
    )

    key = torch.randn_like(query)
    value = torch.randn_like(query)

    expected = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        dropout_p=0.0,
        is_causal=False,
    )

    actual = (
        flash_attention_cuda
        .tensorcore_attention_forward_bc32_raw_pv(
            query,
            key,
            value,
        )
    )

    torch.testing.assert_close(
        actual,
        expected,
        rtol=3e-2,
        atol=3e-2,
    )


    @pytest.mark.parametrize(
        ("batch", "heads", "sequence_length"),
        [
            (1, 1, 32),
            (1, 2, 64),
            (1, 4, 96),
            (1, 8, 128),
            (1, 8, 512),
        ],
    )
    def test_tensorcore_attention_bc32_raw_qk_raw_pv_matches_sdpa(
        batch: int,
        heads: int,
        sequence_length: int,
    ) -> None:
        torch.manual_seed(0)

        shape = (
            batch,
            heads,
            sequence_length,
            64,
        )

        query = torch.randn(
            shape,
            device="cuda",
            dtype=torch.float16,
        )

        key = torch.randn_like(query)
        value = torch.randn_like(query)

        expected = (
            torch.nn.functional
            .scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=0.0,
                is_causal=False,
            )
        )

        actual = (
            flash_attention_cuda
            .tensorcore_attention_forward_bc32_raw_qk_raw_pv(
                query,
                key,
                value,
            )
        )

        torch.testing.assert_close(
            actual,
            expected,
            rtol=3e-2,
            atol=3e-2,
        )


@pytest.mark.parametrize(
    ("batch", "heads", "sequence_length"),
    [
        (1, 1, 64),
        (1, 2, 128),
        (1, 4, 256),
        (1, 8, 512),
        (1, 8, 1024)
    ],
)
def test_tensorcore_attention_production_128x64_matches_sdpa(
    batch: int,
    heads: int,
    sequence_length: int,
) -> None:
    torch.manual_seed(0)

    shape = (
        batch,
        heads,
        sequence_length,
        64,
    )

    query = torch.randn(
        shape,
        device="cuda",
        dtype=torch.float16,
    ).contiguous()

    key = torch.randn(
        shape,
        device="cuda",
        dtype=torch.float16,
    ).contiguous()

    value = torch.randn(
        shape,
        device="cuda",
        dtype=torch.float16,
    ).contiguous()

    expected = (
        torch.nn.functional
        .scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=False,
        )
    )

    actual = (
        flash_attention_cuda
        .tensorcore_attention_forward_production_128x64(
            query,
            key,
            value,
        )
    )

    assert actual.shape == expected.shape
    assert actual.dtype == torch.float16
    assert actual.is_cuda

    torch.testing.assert_close(
        actual,
        expected,
        rtol=3e-2,
        atol=3e-2,
    )