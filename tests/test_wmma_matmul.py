import pytest
import torch

import flash_attention_cuda


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required",
)


@pytest.mark.parametrize("batch_size", [1, 4])
def test_wmma_matmul_identity_block(
    batch_size: int,
) -> None:
    torch.manual_seed(0)

    # A has logical shape M x K = 16 x 64.
    matrix_a = torch.randn(
        batch_size,
        16,
        64,
        device="cuda",
        dtype=torch.float16,
    )

    # B has logical shape K x N = 64 x 16.
    #
    # The first 16 rows form an identity matrix and the remaining
    # 48 rows are zero:
    #
    # B = [I]
    #     [0]
    #
    # Therefore A @ B selects A[:, :, 0:16].
    matrix_b = torch.zeros(
        batch_size,
        64,
        16,
        device="cuda",
        dtype=torch.float16,
    )

    matrix_b[:, :16, :] = torch.eye(
        16,
        device="cuda",
        dtype=torch.float16,
    )

    matrix_b = matrix_b.contiguous()

    actual = (
        flash_attention_cuda
        .wmma_matmul_16x64x16(
            matrix_a,
            matrix_b,
        )
    )

    expected = matrix_a[:, :, :16].float()

    assert actual.dtype == torch.float32
    assert actual.shape == (
        batch_size,
        16,
        16,
    )

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.parametrize("batch_size", [1, 4, 17])
def test_wmma_matmul_matches_reference(
    batch_size: int,
) -> None:
    torch.manual_seed(0)

    matrix_a = torch.randn(
        batch_size,
        16,
        64,
        device="cuda",
        dtype=torch.float16,
    )

    matrix_b = torch.randn(
        batch_size,
        64,
        16,
        device="cuda",
        dtype=torch.float16,
    )

    actual = (
        flash_attention_cuda
        .wmma_matmul_16x64x16(
            matrix_a,
            matrix_b,
        )
    )

    expected = (
        matrix_a.float()
        @ matrix_b.float()
    )

    assert actual.dtype == torch.float32
    assert actual.shape == expected.shape

    torch.testing.assert_close(
        actual,
        expected,
        rtol=5e-3,
        atol=5e-3,
    )