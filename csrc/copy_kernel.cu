#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace{

    template <typename scalar_t>
    __global__ void copy_kernel(
        const scalar_t* input,
        scalar_t* output,
        int64_t num_elements
    ){
        const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

        if(index < num_elements){
            output[index] = input[index];
        }
    }
} // namespace

torch::Tensor copy_cuda(torch::Tensor input){
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    const c10::cuda::CUDAGuard device_guard(input.device());

    torch::Tensor output = torch::empty_like(input);

    const int64_t num_elements = input.numel();

    if(num_elements == 0) return output;

    constexpr int threads = 256;
    const int blocks = static_cast<int>((num_elements + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        torch::kHalf,
        torch::kBFloat16,
        input.scalar_type(),
        "copy_cuda",
        [&] {
            copy_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                num_elements
            );
        }
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}