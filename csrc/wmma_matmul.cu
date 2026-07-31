#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_fp16.h>
#include <mma.h>

#include <cstdint>
#include <limits>

namespace {

namespace wmma = nvcuda::wmma;

/*
 * Logical GEMM:
 *
 *     A [16 x 64]
 *     B [64 x 16]
 *     C [16 x 16]
 *
 * Each WMMA operation consumes:
 *
 *     A fragment [16 x 16]
 *     B fragment [16 x 16]
 *     C fragment [16 x 16]
 *
 * The reduction dimension is 64, so the kernel executes four
 * 16 x 16 x 16 WMMA operations into the same FP32 accumulator.
 */
constexpr int kMmaM = 16;
constexpr int kMmaN = 16;
constexpr int kMmaK = 16;

constexpr int kReductionDimension = 64;
constexpr int kThreadsPerBlock = 32;

constexpr int kElementsPerA =
    kMmaM * kReductionDimension;  // 16 * 64 = 1024

constexpr int kElementsPerB =
    kReductionDimension * kMmaN;  // 64 * 16 = 1024

constexpr int kElementsPerC =
    kMmaM * kMmaN;  // 16 * 16 = 256


__global__ void wmma_matmul_16x64x16_kernel(
    const half* matrix_a,
    const half* matrix_b,
    float* matrix_c
) {
    /*
     * One CUDA block contains one warp and processes one independent
     * matrix pair from the batch.
     */
    const int64_t matrix_index =
        static_cast<int64_t>(blockIdx.x);

    const int64_t a_matrix_offset =
        matrix_index * kElementsPerA;

    const int64_t b_matrix_offset =
        matrix_index * kElementsPerB;

    const int64_t c_matrix_offset =
        matrix_index * kElementsPerC;

    /*
     * A fragment:
     *
     *     logical shape: 16 x 16
     *     physical layout: row-major
     */
    wmma::fragment<
        wmma::matrix_a,
        kMmaM,
        kMmaN,
        kMmaK,
        half,
        wmma::row_major
    > a_fragment;

    /*
     * B fragment:
     *
     *     logical shape: 16 x 16
     *     physical layout: row-major
     */
    wmma::fragment<
        wmma::matrix_b,
        kMmaM,
        kMmaN,
        kMmaK,
        half,
        wmma::row_major
    > b_fragment;

    /*
     * C fragment:
     *
     *     logical shape: 16 x 16
     *     accumulation type: FP32
     */
    wmma::fragment<
        wmma::accumulator,
        kMmaM,
        kMmaN,
        kMmaK,
        float
    > c_fragment;

    wmma::fill_fragment(
        c_fragment,
        0.0f
    );

    /*
     * Compute:
     *
     * C =
     *     A[:,  0:16] @ B[ 0:16, :]
     *   + A[:, 16:32] @ B[16:32, :]
     *   + A[:, 32:48] @ B[32:48, :]
     *   + A[:, 48:64] @ B[48:64, :]
     */
#pragma unroll
    for (
        int k_offset = 0;
        k_offset < kReductionDimension;
        k_offset += kMmaK
    ) {
        /*
         * A is physically [16, 64], row-major.
         *
         * Adding k_offset selects columns:
         *
         *     A[:, k_offset:k_offset+16]
         *
         * Its physical row stride remains 64.
         */
        const half* a_tile =
            matrix_a +
            a_matrix_offset +
            k_offset;

        /*
         * B is physically [64, 16], row-major.
         *
         * Moving down k_offset rows requires skipping:
         *
         *     k_offset * 16
         *
         * elements.
         */
        const half* b_tile =
            matrix_b +
            b_matrix_offset +
            k_offset * kMmaN;

        wmma::load_matrix_sync(
            a_fragment,
            a_tile,
            kReductionDimension
        );

        wmma::load_matrix_sync(
            b_fragment,
            b_tile,
            kMmaN
        );

        wmma::mma_sync(
            c_fragment,
            a_fragment,
            b_fragment,
            c_fragment
        );
    }

    /*
     * C is physically [16, 16], row-major, so its leading dimension
     * is 16.
     */
    wmma::store_matrix_sync(
        matrix_c + c_matrix_offset,
        c_fragment,
        kMmaN,
        wmma::mem_row_major
    );
}


void validate_wmma_inputs(
    const torch::Tensor& matrix_a,
    const torch::Tensor& matrix_b
) {
    TORCH_CHECK(
        matrix_a.is_cuda() &&
        matrix_b.is_cuda(),
        "A and B must be CUDA tensors"
    );

    TORCH_CHECK(
        matrix_a.is_contiguous() &&
        matrix_b.is_contiguous(),
        "A and B must be contiguous"
    );

    TORCH_CHECK(
        matrix_a.scalar_type() == torch::kFloat16 &&
        matrix_b.scalar_type() == torch::kFloat16,
        "A and B must use float16"
    );

    TORCH_CHECK(
        matrix_a.device() == matrix_b.device(),
        "A and B must be on the same CUDA device"
    );

    TORCH_CHECK(
        matrix_a.dim() == 3 &&
        matrix_b.dim() == 3,
        "A and B must be rank-3 tensors"
    );

    TORCH_CHECK(
        matrix_a.size(0) == matrix_b.size(0),
        "A and B must have the same batch size"
    );

    TORCH_CHECK(
        matrix_a.size(1) == kMmaM &&
        matrix_a.size(2) == kReductionDimension,
        "A must have shape [batch, 16, 64]"
    );

    TORCH_CHECK(
        matrix_b.size(1) == kReductionDimension &&
        matrix_b.size(2) == kMmaN,
        "B must have shape [batch, 64, 16]"
    );
}

}  // namespace


torch::Tensor wmma_matmul_16x64x16(
    torch::Tensor matrix_a,
    torch::Tensor matrix_b
) {
    validate_wmma_inputs(
        matrix_a,
        matrix_b
    );

    const c10::cuda::CUDAGuard device_guard(
        matrix_a.device()
    );

    const int64_t batch_size =
        matrix_a.size(0);

    torch::Tensor matrix_c =
        torch::empty(
            {
                batch_size,
                kMmaM,
                kMmaN,
            },
            matrix_a.options().dtype(
                torch::kFloat32
            )
        );

    if (batch_size == 0) {
        return matrix_c;
    }

    TORCH_CHECK(
        batch_size <=
            static_cast<int64_t>(
                std::numeric_limits<
                    unsigned int
                >::max()
            ),
        "batch size exceeds CUDA grid.x limit"
    );

    const half* matrix_a_ptr =
        reinterpret_cast<const half*>(
            matrix_a.data_ptr<at::Half>()
        );

    const half* matrix_b_ptr =
        reinterpret_cast<const half*>(
            matrix_b.data_ptr<at::Half>()
        );

    wmma_matmul_16x64x16_kernel<<<
        static_cast<unsigned int>(batch_size),
        kThreadsPerBlock,
        0,
        at::cuda::getCurrentCUDAStream()
    >>>(
        matrix_a_ptr,
        matrix_b_ptr,
        matrix_c.data_ptr<float>()
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return matrix_c;
}