import pytest
import torch

import flash_attention_cuda


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required",
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ],
)
def test_copy_cuda(dtype: torch.dtype) -> None:
    torch.manual_seed(0)

    input_tensor = torch.randn(
        2,
        3,
        17,
        64,
        device="cuda",
        dtype=dtype,
    )

    output = flash_attention_cuda.copy_cuda(input_tensor)

    assert output.device == input_tensor.device
    assert output.dtype == input_tensor.dtype
    assert output.shape == input_tensor.shape
    assert output.data_ptr() != input_tensor.data_ptr()

    torch.testing.assert_close(output, input_tensor)


def test_copy_cuda_rejects_cpu_tensor() -> None:
    input_tensor = torch.randn(8)

    with pytest.raises(RuntimeError, match="CUDA tensor"):
        flash_attention_cuda.copy_cuda(input_tensor)