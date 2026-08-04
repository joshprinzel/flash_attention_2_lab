#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <mma.h>

#include <cmath>
#include <cstdint>
#include <limits>


namespace {

namespace wmma = nvcuda::wmma;


constexpr int kHeadDimension = 64;

constexpr int kMmaM = 16;
constexpr int kMmaN = 16;
constexpr int kMmaK = 16;

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 4;
constexpr int kThreadsPerBlock =
    kWarpSize * kWarpsPerBlock;

constexpr int kQueryRowsPerWarp = 16;
constexpr int kQueryTileSize =
    kWarpsPerBlock * kQueryRowsPerWarp;

constexpr unsigned int kFullWarpMask = 0xffffffffu;

struct MmaOperandA {
    uint32_t registers[4];
};
struct MmaOperandB {
    uint32_t registers[2];
};
struct MmaAccumulator{
    float registers[4];
};

__device__ __forceinline__ uint32_t shared_address(const void* pointer){
    return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ MmaOperandA load_mma_a_row_major(
    const half* matrix,
    int leading_dimension
){
    const int lane_id = static_cast<int>(threadIdx.x) & 31;

    /*
    ldmatrix.x4 treats the 16x16 matrix as four
    row-major 8x8 matrices:

    matrix 0: rows 0-7, columns 0-7
    matrix 1: rows 0-7, columns 8-15
    matrix 2: rows 8-15, columns 0-7
    matrix 3: rows 8-15, columns 8-15

    Lanes 0-7 provied row addresses for matrix 0,
    lanes 8-15 for matrix 1, and so on.
    */

    const int matrix_index = lane_id >> 3;
    const int row_within_matrix = lane_id & 7;
    const int matrix_row_offset = (matrix_index & 1) * 8;
    const int matrix_column_offset = (matrix_index >> 1) * 8;

    const half* row_pointer = matrix + (
        matrix_row_offset + row_within_matrix
    ) * leading_dimension + matrix_column_offset;

    const uint32_t address = shared_address(row_pointer);

    MmaOperandA fragment;

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];\n"
        : "=r"(fragment.registers[0]),
          "=r"(fragment.registers[1]),
          "=r"(fragment.registers[2]),
          "=r"(fragment.registers[3])
        : "r"(address)
    );
    return fragment;
}

__device__ __forceinline__ MmaOperandB load_mma_b_col_major_from_row_major(
    const half* matrix,
    int leading_dimension
){
    const int lane_id = static_cast<int>(threadIdx.x) & 31;

    /*
    x2 loads two 8x8 matrices:

    matrix 0: row 0-7, columns 0-7
    matrix 1: rows 8-15, columns 0-7
    */

    const int address_lane = lane_id & 15;
    const int matrix_index = address_lane >> 3;
    const int row_within_matrix = address_lane & 7;
    const int row = matrix_index * 8 + row_within_matrix;
    const half* row_pointer = matrix + row * leading_dimension;
    const uint32_t address = shared_address(row_pointer);

    MmaOperandB fragment;

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
        "{%0, %1}, [%2];\n"
        : "=r"(fragment.registers[0]),
          "=r"(fragment.registers[1])
        : "r"(address)
    );
    return fragment;
}

__device__ __forceinline__
void mma_m16n8k16_f16_f32(
    MmaAccumulator& accumulator,
    const MmaOperandA& matrix_a,
    const MmaOperandB& matrix_b
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col."
        "f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(accumulator.registers[0]),
          "+f"(accumulator.registers[1]),
          "+f"(accumulator.registers[2]),
          "+f"(accumulator.registers[3])
        : "r"(matrix_a.registers[0]),
          "r"(matrix_a.registers[1]),
          "r"(matrix_a.registers[2]),
          "r"(matrix_a.registers[3]),
          "r"(matrix_b.registers[0]),
          "r"(matrix_b.registers[1])
    );
}

__device__ __forceinline__ 
MmaAccumulator zero_mma_accumulator(){
    MmaAccumulator accumulator;

    #pragma unroll
    for(int index = 0; index < 4; ++index){
        accumulator.registers[index] = 0.0f;
    }
    return accumulator;
}

__device__ __forceinline__
int mma_accumulator_row(
    int lane_id,
    int register_index
){
    const int group_id = lane_id >> 2;

    return register_index < 2 ? group_id : group_id + 8;
}

__device__ __forceinline__
int mma_accumulator_column(
    int lane_id,
    int register_index
){
    const int thread_in_group = lane_id & 3;
    return thread_in_group * 2 + (register_index & 1);
}


template <int kKeyTileSize, bool kUseRawPv = false>
__global__ void tensorcore_attention_forward_kernel_d64(
    const half* __restrict__ query,
    const half* __restrict__ key,
    const half* __restrict__ value,
    half* __restrict__ output,
    int64_t sequence_length,
    float scale
) {
    static_assert(
        kKeyTileSize == 16 ||
        kKeyTileSize == 32,
        "Only Bc=16 and Bc=32 are supported"
    );

    static_assert(
        kKeyTileSize % kMmaK == 0,
        "Key tile must be divisible by the MMA K dimension"
    );

    static_assert(
        kQueryTileSize ==
            kWarpsPerBlock * kQueryRowsPerWarp
    );

    static_assert(
        !kUseRawPv || kKeyTileSize == 32,
        "The raw PV experiment requires Bc=32"
    );

    const int thread_id =
        static_cast<int>(threadIdx.x);

    const int warp_id =
        thread_id / kWarpSize;

    const int lane_id =
        thread_id % kWarpSize;

    const int64_t batch_head_index =
        static_cast<int64_t>(blockIdx.y);

    const int64_t query_tile_start =
        static_cast<int64_t>(blockIdx.x) *
        kQueryTileSize;

    const int64_t tensor_base =
        batch_head_index *
        sequence_length *
        kHeadDimension;


    /*
     * Shared-memory layout
     *
     * Q is staged once for the entire query block.
     * K and V are replaced on each K/V-loop iteration.
     *
     * Score, probability, and PV storage is private to
     * each warp.
     */

    __shared__ __align__(32)
    half query_shared[
        kQueryTileSize
    ][
        kHeadDimension
    ];

    __shared__ __align__(32)
    half key_shared[
        kKeyTileSize
    ][
        kHeadDimension
    ];

    __shared__ __align__(32)
    half value_shared[
        kKeyTileSize
    ][
        kHeadDimension
    ];

    __shared__ __align__(32)
    float score_shared[
        kWarpsPerBlock
    ][
        kQueryRowsPerWarp
    ][
        kKeyTileSize
    ];

    __shared__ __align__(32)
    half probability_shared[
        kWarpsPerBlock
    ][
        kQueryRowsPerWarp
    ][
        kKeyTileSize
    ];

    /*
    The raw-PV specialization does not matrielize PV
    A one-element placeholder keeps the declaration valid
    while avoiding the 16 KiB allocation.
    */

    constexpr int kPvSharedWarps = kUseRawPv ? 1 : kWarpsPerBlock;
    constexpr int kPvSharedRows = kUseRawPv ? 1 : kQueryRowsPerWarp;
    constexpr int kPvSharedColumns = kUseRawPv ? 1 : kHeadDimension;
    __shared__ __align__(32)
    float pv_shared[
        kPvSharedWarps
    ][
        kPvSharedRows
    ][
        kPvSharedColumns
    ];

    __shared__ __align__(32)
    float previous_scale_shared[
        kWarpsPerBlock
    ][
        kQueryRowsPerWarp
    ];

    __shared__ __align__(32)
    float inverse_sum_shared[
        kWarpsPerBlock
    ][
        kQueryRowsPerWarp
    ];


    /*
     * Each warp owns a 16x64 output tile.
     *
     * A warp owns 1024 FP32 output values. Distributed
     * evenly over 32 lanes, each lane owns 32 values.
     */

    float output_accumulator[32];

#pragma unroll
    for (int item = 0; item < 32; ++item) {
        output_accumulator[item] = 0.0f;
    }


    /*
     * Lanes 0-15 own the online-softmax state for the
     * warp's 16 query rows.
     */

    float running_row_max = -CUDART_INF_F;
    float running_row_sum = 0.0f;


    /*
     * Stage the Br=64 query tile once.
     */

    constexpr int kQueryElements =
        kQueryTileSize * kHeadDimension;

    for (
        int element = thread_id;
        element < kQueryElements;
        element += kThreadsPerBlock
    ) {
        const int local_query_row =
            element / kHeadDimension;

        const int dimension =
            element % kHeadDimension;

        const int64_t global_query_row =
            query_tile_start + local_query_row;

        if (global_query_row < sequence_length) {
            const int64_t global_index =
                tensor_base +
                global_query_row *
                    kHeadDimension +
                dimension;

            query_shared[
                local_query_row
            ][
                dimension
            ] = query[global_index];
        } else {
            query_shared[
                local_query_row
            ][
                dimension
            ] = __float2half_rn(0.0f);
        }
    }

    __syncthreads();


    /*
     * Iterate over K/V tiles.
     */

    for (
        int64_t key_tile_start = 0;
        key_tile_start < sequence_length;
        key_tile_start += kKeyTileSize
    ) {
        /*
         * Cooperatively stage one K and V tile.
         *
         * The current full-tile specialization requires
         * sequence_length % kKeyTileSize == 0, so no key
         * boundary predicate is required here.
         */

        constexpr int kKeyValueElements =
            kKeyTileSize * kHeadDimension;

        for (
            int element = thread_id;
            element < kKeyValueElements;
            element += kThreadsPerBlock
        ) {
            const int local_key_row =
                element / kHeadDimension;

            const int dimension =
                element % kHeadDimension;

            const int64_t global_key_row =
                key_tile_start + local_key_row;

            const int64_t global_index =
                tensor_base +
                global_key_row *
                    kHeadDimension +
                dimension;

            key_shared[
                local_key_row
            ][
                dimension
            ] = key[global_index];

            value_shared[
                local_key_row
            ][
                dimension
            ] = value[global_index];
        }

        __syncthreads();


        /*
         * QK^T
         *
         * Each warp owns:
         *
         *     Q: [16, 64]
         *
         * For Bc=16, it computes one [16,16] score tile.
         * For Bc=32, it computes two adjacent [16,16]
         * score tiles.
         */

#pragma unroll
        for (
            int key_subtile_offset = 0;
            key_subtile_offset < kKeyTileSize;
            key_subtile_offset += kMmaN
        ) {
            wmma::fragment<
                wmma::matrix_a,
                kMmaM,
                kMmaN,
                kMmaK,
                half,
                wmma::row_major
            > query_fragment;

            /*
             * K is physically stored as [Bc, 64]
             * row-major.
             *
             * Loading it as a column-major matrix with
             * leading dimension 64 reinterprets it as
             * the required logical K^T tile [64,16].
             */
            wmma::fragment<
                wmma::matrix_b,
                kMmaM,
                kMmaN,
                kMmaK,
                half,
                wmma::col_major
            > key_fragment;

            wmma::fragment<
                wmma::accumulator,
                kMmaM,
                kMmaN,
                kMmaK,
                float
            > score_fragment;

            wmma::fill_fragment(
                score_fragment,
                0.0f
            );

#pragma unroll
            for (
                int dimension_offset = 0;
                dimension_offset < kHeadDimension;
                dimension_offset += kMmaK
            ) {
                const half* query_tile =
                    &query_shared[
                        warp_id *
                            kQueryRowsPerWarp
                    ][
                        dimension_offset
                    ];

                const half* key_transpose_tile =
                    &key_shared[
                        key_subtile_offset
                    ][
                        dimension_offset
                    ];

                wmma::load_matrix_sync(
                    query_fragment,
                    query_tile,
                    kHeadDimension
                );

                wmma::load_matrix_sync(
                    key_fragment,
                    key_transpose_tile,
                    kHeadDimension
                );

                wmma::mma_sync(
                    score_fragment,
                    query_fragment,
                    key_fragment,
                    score_fragment
                );
            }

            /*
             * Apply 1 / sqrt(D) while the score values
             * are still in the FP32 WMMA accumulator.
             */

#pragma unroll
            for (
                int element = 0;
                element <
                    score_fragment.num_elements;
                ++element
            ) {
                score_fragment.x[element] *= scale;
            }

            /*
             * score_shared has physical row stride Bc.
             *
             * With Bc=32:
             *
             *   subtile 0 -> columns  0-15
             *   subtile 1 -> columns 16-31
             */

            wmma::store_matrix_sync(
                &score_shared[
                    warp_id
                ][
                    0
                ][
                    key_subtile_offset
                ],
                score_fragment,
                kKeyTileSize,
                wmma::mem_row_major
            );
        }

        /*
         * The score storage is warp-private, so only a
         * warp-level synchronization is required before
         * the same warp consumes it.
         */

        __syncwarp(kFullWarpMask);


        /*
         * Tilewise online softmax.
         *
         * Lanes 0-15 each own one query row.
         */

        if (lane_id < kQueryRowsPerWarp) {
            const int local_row = lane_id;

            float tile_row_max = -CUDART_INF_F;

#pragma unroll
            for (
                int column = 0;
                column < kKeyTileSize;
                ++column
            ) {
                tile_row_max = fmaxf(
                    tile_row_max,
                    score_shared[
                        warp_id
                    ][
                        local_row
                    ][
                        column
                    ]
                );
            }

            const float new_row_max =
                fmaxf(
                    running_row_max,
                    tile_row_max
                );

            const float previous_scale =
                running_row_sum == 0.0f
                    ? 0.0f
                    : __expf(
                        running_row_max -
                        new_row_max
                    );

            float tile_row_sum = 0.0f;

#pragma unroll
            for (
                int column = 0;
                column < kKeyTileSize;
                ++column
            ) {
                const float probability =
                    __expf(
                        score_shared[
                            warp_id
                        ][
                            local_row
                        ][
                            column
                        ] -
                        new_row_max
                    );

                tile_row_sum += probability;

                probability_shared[
                    warp_id
                ][
                    local_row
                ][
                    column
                ] = __float2half_rn(
                    probability
                );
            }

            running_row_sum =
                previous_scale *
                    running_row_sum +
                tile_row_sum;

            running_row_max =
                new_row_max;

            previous_scale_shared[
                warp_id
            ][
                local_row
            ] = previous_scale;
        }

        /*
         * Ensure every probability row and previous
         * scale is visible to all lanes in the warp.
         */

        __syncwarp(kFullWarpMask);


        /*
         * PV
         *
         * Each output fragment covers 16 output
         * dimensions.
         *
         * Bc=16:
         *   one MMA reduction step.
         *
         * Bc=32:
         *   two MMA reduction steps accumulated into
         *   the same FP32 fragment.
         */

         if constexpr (kUseRawPv) {
        constexpr int kRawMmaOutputColumns = 8;

        constexpr int kRawOutputSubtiles =
            kHeadDimension /
            kRawMmaOutputColumns;

        /*
        * Raw PV path:
        *
        * P [16, 32] x V [32, 64]
        *
        * Each mma.m16n8k16 instruction produces a
        * [16, 8] output tile. Eight output subtiles cover
        * the complete 64-dimensional output.
        */
    #pragma unroll
        for (
            int output_subtile = 0;
            output_subtile < kRawOutputSubtiles;
            ++output_subtile
        ) {
            const int output_dimension_offset =
                output_subtile *
                kRawMmaOutputColumns;

            MmaAccumulator pv_accumulator =
                zero_mma_accumulator();

            /*
            * Bc=32 requires two K=16 MMA reduction steps:
            *
            * P[:,  0:16] x V[ 0:16, :]
            * P[:, 16:32] x V[16:32, :]
            */
    #pragma unroll
            for (
                int reduction_offset = 0;
                reduction_offset < kKeyTileSize;
                reduction_offset += kMmaK
            ) {
                const half* probability_tile =
                    &probability_shared[
                        warp_id
                    ][
                        0
                    ][
                        reduction_offset
                    ];

                const half* value_tile =
                    &value_shared[
                        reduction_offset
                    ][
                        output_dimension_offset
                    ];

                const MmaOperandA probability_fragment =
                    load_mma_a_row_major(
                        probability_tile,
                        kKeyTileSize
                    );

                const MmaOperandB value_fragment =
                    load_mma_b_col_major_from_row_major(
                        value_tile,
                        kHeadDimension
                    );

                mma_m16n8k16_f16_f32(
                    pv_accumulator,
                    probability_fragment,
                    value_fragment
                );
            }

            /*
            * Preserve raw-MMA accumulator ownership.
            *
            * Each lane owns four values from each [16,8]
            * output subtile. Across eight subtiles:
            *
            * 8 subtiles x 4 values = 32 FP32 values/lane.
            */
    #pragma unroll
            for (
                int register_index = 0;
                register_index < 4;
                ++register_index
            ) {
                const int local_row =
                    mma_accumulator_row(
                        lane_id,
                        register_index
                    );

                const int accumulator_index =
                    output_subtile * 4 +
                    register_index;

                const float previous_scale =
                    previous_scale_shared[
                        warp_id
                    ][
                        local_row
                    ];

                output_accumulator[
                    accumulator_index
                ] =
                    previous_scale *
                        output_accumulator[
                            accumulator_index
                        ] +
                    pv_accumulator.registers[
                        register_index
                    ];
            }
        }
    } else {
        /*
        * Existing WMMA PV path.
        *
        * Each output fragment covers 16 output dimensions.
        */
    #pragma unroll
        for (
            int output_dimension_offset = 0;
            output_dimension_offset < kHeadDimension;
            output_dimension_offset += kMmaN
        ) {
            wmma::fragment<
                wmma::accumulator,
                kMmaM,
                kMmaN,
                kMmaK,
                float
            > pv_fragment;

            wmma::fill_fragment(
                pv_fragment,
                0.0f
            );

    #pragma unroll
            for (
                int reduction_offset = 0;
                reduction_offset < kKeyTileSize;
                reduction_offset += kMmaK
            ) {
                wmma::fragment<
                    wmma::matrix_a,
                    kMmaM,
                    kMmaN,
                    kMmaK,
                    half,
                    wmma::row_major
                > probability_fragment;

                wmma::fragment<
                    wmma::matrix_b,
                    kMmaM,
                    kMmaN,
                    kMmaK,
                    half,
                    wmma::row_major
                > value_fragment;

                const half* probability_tile =
                    &probability_shared[
                        warp_id
                    ][
                        0
                    ][
                        reduction_offset
                    ];

                const half* value_tile =
                    &value_shared[
                        reduction_offset
                    ][
                        output_dimension_offset
                    ];

                wmma::load_matrix_sync(
                    probability_fragment,
                    probability_tile,
                    kKeyTileSize
                );

                wmma::load_matrix_sync(
                    value_fragment,
                    value_tile,
                    kHeadDimension
                );

                wmma::mma_sync(
                    pv_fragment,
                    probability_fragment,
                    value_fragment,
                    pv_fragment
                );
            }

            wmma::store_matrix_sync(
                &pv_shared[
                    warp_id
                ][
                    0
                ][
                    output_dimension_offset
                ],
                pv_fragment,
                kHeadDimension,
                wmma::mem_row_major
            );
        }

        /*
        * Only the WMMA path stores PV into shared memory.
        * Synchronize before reading that data back.
        */
        __syncwarp(kFullWarpMask);

        /*
        * Merge the materialized WMMA PV result into the
        * persistent online-softmax numerator.
        */
    #pragma unroll
        for (int item = 0; item < 32; ++item) {
            const int linear_index =
                lane_id +
                item * kWarpSize;

            const int local_row =
                linear_index /
                kHeadDimension;

            const int dimension =
                linear_index %
                kHeadDimension;

            const float previous_scale =
                previous_scale_shared[
                    warp_id
                ][
                    local_row
                ];

            output_accumulator[item] =
                previous_scale *
                    output_accumulator[item] +
                pv_shared[
                    warp_id
                ][
                    local_row
                ][
                    dimension
                ];
        }
    }

    /*
    * All warps must finish consuming the current K/V tile
    * before any thread overwrites shared K and V storage.
    */
    __syncthreads();
}


    /*
     * Publish the final reciprocal softmax denominator.
     */

    if (lane_id < kQueryRowsPerWarp) {
        inverse_sum_shared[
            warp_id
        ][
            lane_id
        ] = 1.0f / running_row_sum;
    }

    __syncwarp(kFullWarpMask);


    /*
     * Normalize the persistent FP32 numerator and write
     * the final FP16 output.
     */

     if constexpr(kUseRawPv){
        constexpr int kRawMmaOutputColumns = 8; 
        constexpr int kRawOutputSubtiles = kHeadDimension / kRawMmaOutputColumns;

        #pragma unroll
        for(int output_subtiles = 0; output_subtiles < kRawOutputSubtiles; output_subtiles++){
            #pragma unroll
            for(int register_index = 0; register_index < 4; register_index++){
                const int local_row = mma_accumulator_row(lane_id, register_index);
                const int dimension = output_subtiles * kRawMmaOutputColumns + 
                mma_accumulator_column(lane_id, register_index);

                const int accumulator_index = output_subtiles * 4 + register_index;
                const int64_t global_query_row = query_tile_start + warp_id * kQueryRowsPerWarp + local_row;

                if(global_query_row < sequence_length){
                    const float normalized_output = output_accumulator[accumulator_index] *
                    inverse_sum_shared[warp_id][local_row];

                    const int64_t global_index = tensor_base + global_query_row * kHeadDimension + dimension;

                    output[global_index] = __float2half_rn(normalized_output);
                }
            }
        }


     }else{
        #pragma unroll
        for (int item = 0; item < 32; ++item) {
            const int linear_index =
                lane_id +
                item * kWarpSize;

            const int local_row =
                linear_index /
                kHeadDimension;

            const int dimension =
                linear_index %
                kHeadDimension;

            const int64_t global_query_row =
                query_tile_start +
                warp_id *
                    kQueryRowsPerWarp +
                local_row;

            if (global_query_row < sequence_length) {
                const float normalized_output =
                    output_accumulator[item] *
                    inverse_sum_shared[
                        warp_id
                    ][
                        local_row
                    ];

                const int64_t global_index =
                    tensor_base +
                    global_query_row *
                        kHeadDimension +
                    dimension;

                output[global_index] =
                    __float2half_rn(
                        normalized_output
                    );
            }
        }
     }


}


template <int kKeyTileSize>
void validate_tensorcore_attention_inputs(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value
) {
    TORCH_CHECK(
        query.is_cuda(),
        "query must be a CUDA tensor"
    );

    TORCH_CHECK(
        key.is_cuda(),
        "key must be a CUDA tensor"
    );

    TORCH_CHECK(
        value.is_cuda(),
        "value must be a CUDA tensor"
    );

    TORCH_CHECK(
        query.device() == key.device() &&
        query.device() == value.device(),
        "query, key, and value must be on the same device"
    );

    TORCH_CHECK(
        query.scalar_type() == at::kHalf,
        "query must have dtype torch.float16"
    );

    TORCH_CHECK(
        key.scalar_type() == at::kHalf,
        "key must have dtype torch.float16"
    );

    TORCH_CHECK(
        value.scalar_type() == at::kHalf,
        "value must have dtype torch.float16"
    );

    TORCH_CHECK(
        query.dim() == 4,
        "query must have shape [B, H, N, D]"
    );

    TORCH_CHECK(
        key.dim() == 4,
        "key must have shape [B, H, N, D]"
    );

    TORCH_CHECK(
        value.dim() == 4,
        "value must have shape [B, H, N, D]"
    );

    TORCH_CHECK(
        query.sizes() == key.sizes(),
        "query and key must have identical shapes"
    );

    TORCH_CHECK(
        query.sizes() == value.sizes(),
        "query and value must have identical shapes"
    );

    TORCH_CHECK(
        query.is_contiguous(),
        "query must be contiguous"
    );

    TORCH_CHECK(
        key.is_contiguous(),
        "key must be contiguous"
    );

    TORCH_CHECK(
        value.is_contiguous(),
        "value must be contiguous"
    );

    TORCH_CHECK(
        query.size(3) == kHeadDimension,
        "head dimension must equal 64"
    );

    TORCH_CHECK(
        query.size(2) % kKeyTileSize == 0,
        "sequence length must be divisible by ",
        kKeyTileSize
    );
}


template <int kKeyTileSize, bool kUseRawPv = false>
torch::Tensor tensorcore_attention_forward_impl(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value
) {
    validate_tensorcore_attention_inputs<
        kKeyTileSize
    >(
        query,
        key,
        value
    );

    const c10::cuda::CUDAGuard device_guard(
        query.device()
    );

    torch::Tensor output =
        torch::empty_like(query);

    const int64_t batch_size =
        query.size(0);

    const int64_t num_heads =
        query.size(1);

    const int64_t sequence_length =
        query.size(2);

    if (
        batch_size == 0 ||
        num_heads == 0 ||
        sequence_length == 0
    ) {
        return output;
    }

    const int64_t num_query_tiles =
        (
            sequence_length +
            kQueryTileSize -
            1
        ) /
        kQueryTileSize;

    const int64_t num_batch_heads =
        batch_size * num_heads;

    TORCH_CHECK(
        num_query_tiles <=
            static_cast<int64_t>(
                std::numeric_limits<
                    unsigned int
                >::max()
            ),
        "too many query tiles for CUDA grid.x"
    );

    TORCH_CHECK(
        num_batch_heads <= 65535,
        "batch * heads exceeds CUDA grid.y limit"
    );

    const float scale =
        1.0f /
        std::sqrt(
            static_cast<float>(
                kHeadDimension
            )
        );

    const dim3 grid(
        static_cast<unsigned int>(
            num_query_tiles
        ),
        static_cast<unsigned int>(
            num_batch_heads
        ),
        1
    );

    const dim3 block(
        kThreadsPerBlock,
        1,
        1
    );

    const half* query_ptr =
        reinterpret_cast<const half*>(
            query.data_ptr<at::Half>()
        );

    const half* key_ptr =
        reinterpret_cast<const half*>(
            key.data_ptr<at::Half>()
        );

    const half* value_ptr =
        reinterpret_cast<const half*>(
            value.data_ptr<at::Half>()
        );

    half* output_ptr =
        reinterpret_cast<half*>(
            output.data_ptr<at::Half>()
        );

    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(
            query.get_device()
        );

    tensorcore_attention_forward_kernel_d64<
        kKeyTileSize, kUseRawPv
    ><<<
        grid,
        block,
        0,
        stream
    >>>(
        query_ptr,
        key_ptr,
        value_ptr,
        output_ptr,
        sequence_length,
        scale
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}

}  // namespace


/*
 * Existing Bc=16 baseline.
 */
torch::Tensor tensorcore_attention_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value
) {
    return tensorcore_attention_forward_impl<16>(
        query,
        key,
        value
    );
}


/*
 * New Bc=32 experiment.
 */
torch::Tensor tensorcore_attention_forward_bc32(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value
) {
    return tensorcore_attention_forward_impl<32>(
        query,
        key,
        value
    );
}


torch::Tensor tensorcore_attention_forward_bc32_raw_pv(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value
) {
    return tensorcore_attention_forward_impl<
        32,
        true
    >(
        query,
        key,
        value
    );
}