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

/*
Experiment 9: Production-style CTA geometry.

9A keeps BlockN=32 while introducing
BlockM=128 with four warps
*/

constexpr int kProductionBlockM = 128;
constexpr int kProductionBlockN = 128;

constexpr int kProductionWarpsPerBlock = 4;
constexpr int kProductionThreadsPerBlock = kProductionWarpsPerBlock * kWarpSize;
constexpr int kProductionMmaRowsPerWarp = kProductionBlockM / (kProductionWarpsPerBlock * kMmaM);

static_assert(kProductionMmaRowsPerWarp == 2);

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

__device__ __forceinline__ 
void copy_global_to_shared_async_16(
    half* shared_destination,
    const half* global_source
){
    const uint32_t shared_destination_address = shared_address(shared_destination);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :
        : "r"(shared_destination_address),
          "l"(global_source)
        : "memory"
    );
}

__device__ __forceinline__
void commit_async_copy_group() {
    asm volatile(
        "cp.async.commit_group;\n"
        :
        :
        : "memory"
    );
}

__device__ __forceinline__
void wait_for_async_copy_group() {
    asm volatile(
        "cp.async.wait_group 0;\n"
        :
        :
        : "memory"
    );
}

__device__ __forceinline__
void wait_for_async_copy_group_1() {
    asm volatile(
        "cp.async.wait_group 1;\n"
        :
        :
        : "memory"
    );
}

__device__ __forceinline__ uint32_t pack_two_floats_to_half2(
    float first,
    float second
){
    union{
        half2 half_values;
        uint32_t bits;
    } packed;

    packed.half_values = __floats2half2_rn(first, second);
    return packed.bits;
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

__device__ __forceinline__ MmaOperandB load_mma_b_k_transpose_from_row_major(
    const half* matrix,
    int leading_dimension
){
    const int lane_id = static_cast<int>(threadIdx.x) & 31;

    const int address_lane = lane_id & 15;
    const int matrix_index = address_lane >> 3;
    const int row_within_matrix = address_lane & 7;

    const int column_offset = matrix_index * 8;
    const half* row_pointer = matrix + row_within_matrix * leading_dimension + column_offset;
    const uint32_t address = shared_address(row_pointer);

    MmaOperandB fragment;

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
        "{%0, %1}, [%2];\n"
        : "=r"(fragment.registers[0]),
        "=r"(fragment.registers[1])
        : "r"(address)
    );
    return fragment;
}

__device__ __forceinline__
int swizzled_value_chunk(
    int row,
    int logical_chunk
){
    return logical_chunk ^ (row & 7);
}

__device__ __forceinline__
MmaOperandB load_mma_b_col_major_from_swizzled_value(
    const half* matrix,
    int reduction_offset,
    int logical_column_offset
){
    const int lane_id = static_cast<int>(threadIdx.x) & 31;
    const int address_lane = lane_id & 15;

    const int logical_row =
        reduction_offset + address_lane;

    const int logical_chunk =
        logical_column_offset >> 3;

    const int physical_chunk =
        logical_chunk ^ (address_lane & 7);

    const int half_offset =
        (logical_row << 6) +
        (physical_chunk << 3);

    const uint32_t address =
        shared_address(matrix + half_offset);

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
MmaOperandB load_mma_b_k_transpose_from_swizzled_row_major(
    const half* matrix,
    int key_row_offset,
    int dimension_offset
) {
    const int lane_id =
        static_cast<int>(threadIdx.x) & 31;

    const int address_lane =
        lane_id & 15;

    const int row_within_tile =
        address_lane & 7;

    const int matrix_index =
        address_lane >> 3;

    const int logical_row =
        key_row_offset + row_within_tile;

    const int logical_chunk =
        (dimension_offset >> 3) + matrix_index;

    const int physical_chunk =
        logical_chunk ^ row_within_tile;

    const int half_offset =
        (logical_row << 6) +
        (physical_chunk << 3);

    const uint32_t address =
        shared_address(matrix + half_offset);

    MmaOperandB fragment;

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
        "{%0, %1}, [%2];\n"
        : "=r"(fragment.registers[0]),
          "=r"(fragment.registers[1])
        : "r"(address)
    );

    return fragment;
}

__device__ __forceinline__
MmaOperandA load_mma_a_from_swizzled_row_major_64(
    const half* matrix,
    int query_row_offset,
    int dimension_offset
) {
    const int lane_id =
        static_cast<int>(threadIdx.x) & 31;

    const int matrix_index =
        lane_id >> 3;

    const int row_within_matrix =
        lane_id & 7;

    const int row_offset =
        (matrix_index & 1) * 8 +
        row_within_matrix;

    const int column_offset =
        (matrix_index >> 1) * 8;

    const int logical_row =
        query_row_offset +
        row_offset;

    const int logical_chunk =
        (dimension_offset + column_offset) >> 3;

    const int physical_chunk =
        logical_chunk ^
        (logical_row & 7);

    const int half_offset =
        (logical_row << 6) +
        (physical_chunk << 3);

    const uint32_t address =
        shared_address(
            matrix + half_offset
        );

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

__device__ __forceinline__ MmaOperandA make_probability_mma_a(
    const MmaAccumulator& first_score_subtile,
    const MmaAccumulator& second_score_subtile
){
    MmaOperandA fragment;

    fragment.registers[0] = pack_two_floats_to_half2(first_score_subtile.registers[0], first_score_subtile.registers[1]);
    fragment.registers[1] = pack_two_floats_to_half2(first_score_subtile.registers[2], first_score_subtile.registers[3]);

    fragment.registers[2] = pack_two_floats_to_half2(second_score_subtile.registers[0], second_score_subtile.registers[1]);
    fragment.registers[3] = pack_two_floats_to_half2(second_score_subtile.registers[2], second_score_subtile.registers[3]);

    return fragment;
}
template<int kScoreSubtiles>
__device__ __forceinline__
void normalize_score_tile_in_place(
    MmaAccumulator (&score_accumulators)[kScoreSubtiles],
    float scale,
    int lane_id,
    float& running_row_max,
    float& running_row_sum,
    float& previous_scale_for_lane
) {
    static_assert(kScoreSubtiles % 2 == 0);

    float upper_max = -CUDART_INF_F;
    float lower_max = -CUDART_INF_F;

#pragma unroll
    for (
        int score_subtile = 0;
        score_subtile < kScoreSubtiles;
        ++score_subtile
    ) {
        upper_max = fmaxf(
            upper_max,
            fmaxf(
                score_accumulators[score_subtile].registers[0] * scale,
                score_accumulators[score_subtile].registers[1] * scale
            )
        );

        lower_max = fmaxf(
            lower_max,
            fmaxf(
                score_accumulators[score_subtile].registers[2] * scale,
                score_accumulators[score_subtile].registers[3] * scale
            )
        );
    }

    upper_max = fmaxf(
        upper_max,
        __shfl_xor_sync(kFullWarpMask, upper_max, 1, 4)
    );
    upper_max = fmaxf(
        upper_max,
        __shfl_xor_sync(kFullWarpMask, upper_max, 2, 4)
    );

    lower_max = fmaxf(
        lower_max,
        __shfl_xor_sync(kFullWarpMask, lower_max, 1, 4)
    );
    lower_max = fmaxf(
        lower_max,
        __shfl_xor_sync(kFullWarpMask, lower_max, 2, 4)
    );

    const int canonical_source_lane =
        (lane_id & 7) * 4;

    const float canonical_upper_max =
        __shfl_sync(
            kFullWarpMask,
            upper_max,
            canonical_source_lane
        );

    const float canonical_lower_max =
        __shfl_sync(
            kFullWarpMask,
            lower_max,
            canonical_source_lane
        );

    float tile_row_max = -CUDART_INF_F;

    if (lane_id < 8) {
        tile_row_max = canonical_upper_max;
    } else if (lane_id < 16) {
        tile_row_max = canonical_lower_max;
    }

    float new_row_max_for_lane = -CUDART_INF_F;
    previous_scale_for_lane = 0.0f;

    if (lane_id < kQueryRowsPerWarp) {
        new_row_max_for_lane =
            fmaxf(
                running_row_max,
                tile_row_max
            );

        previous_scale_for_lane =
            running_row_sum == 0.0f
                ? 0.0f
                : __expf(
                    running_row_max -
                    new_row_max_for_lane
                );

        running_row_max =
            new_row_max_for_lane;
    }

    const int group_id =
        lane_id >> 2;

    const float upper_new_row_max =
        __shfl_sync(
            kFullWarpMask,
            new_row_max_for_lane,
            group_id
        );

    const float lower_new_row_max =
        __shfl_sync(
            kFullWarpMask,
            new_row_max_for_lane,
            group_id + 8
        );

    float upper_probability_sum = 0.0f;
    float lower_probability_sum = 0.0f;

#pragma unroll
    for (
        int score_subtile = 0;
        score_subtile < kScoreSubtiles;
        ++score_subtile
    ) {
        MmaAccumulator& score_accumulator =
            score_accumulators[score_subtile];

        score_accumulator.registers[0] =
            __expf(
                score_accumulator.registers[0] * scale -
                upper_new_row_max
            );

        score_accumulator.registers[1] =
            __expf(
                score_accumulator.registers[1] * scale -
                upper_new_row_max
            );

        score_accumulator.registers[2] =
            __expf(
                score_accumulator.registers[2] * scale -
                lower_new_row_max
            );

        score_accumulator.registers[3] =
            __expf(
                score_accumulator.registers[3] * scale -
                lower_new_row_max
            );

        upper_probability_sum +=
            score_accumulator.registers[0] +
            score_accumulator.registers[1];

        lower_probability_sum +=
            score_accumulator.registers[2] +
            score_accumulator.registers[3];
    }

    upper_probability_sum +=
        __shfl_xor_sync(
            kFullWarpMask,
            upper_probability_sum,
            1,
            4
        );
    upper_probability_sum +=
        __shfl_xor_sync(
            kFullWarpMask,
            upper_probability_sum,
            2,
            4
        );

    lower_probability_sum +=
        __shfl_xor_sync(
            kFullWarpMask,
            lower_probability_sum,
            1,
            4
        );
    lower_probability_sum +=
        __shfl_xor_sync(
            kFullWarpMask,
            lower_probability_sum,
            2,
            4
        );

    const float canonical_upper_sum =
        __shfl_sync(
            kFullWarpMask,
            upper_probability_sum,
            canonical_source_lane
        );

    const float canonical_lower_sum =
        __shfl_sync(
            kFullWarpMask,
            lower_probability_sum,
            canonical_source_lane
        );

    float tile_row_sum = 0.0f;

    if (lane_id < 8) {
        tile_row_sum = canonical_upper_sum;
    } else if (lane_id < 16) {
        tile_row_sum = canonical_lower_sum;
    }

    if (lane_id < kQueryRowsPerWarp) {
        running_row_sum =
            previous_scale_for_lane *
                running_row_sum +
            tile_row_sum;
    }
}
template<int kKeyTileSize>
__device__ __forceinline__
void stage_tile_async_16(
    half* shared_tile,
    const half* global_tensor,
    int64_t tile_start,
    int64_t tensor_base,
    int shared_stride,
    int thread_id,
    int thread_count
){
    constexpr int kHalfElementsPerAsyncCopy = 8;
    constexpr int kCopiesPerRow = kHeadDimension / kHalfElementsPerAsyncCopy;
    constexpr int kTileCopies = kKeyTileSize * kCopiesPerRow;

    #pragma unroll 1
    for(int copy = thread_id; copy < kTileCopies; copy += thread_count){
        const int local_row = copy / kCopiesPerRow;
        const int copy_within_row = copy % kCopiesPerRow;
        const int dimension = copy_within_row * kHalfElementsPerAsyncCopy;

        const int64_t global_row = tile_start + local_row;
        const int64_t global_index = tensor_base + global_row * kHeadDimension + dimension;

        copy_global_to_shared_async_16(&shared_tile[local_row * shared_stride + dimension], &global_tensor[global_index]);
    }

}



template<int kKeyTileSize>
__device__ __forceinline__
void stage_tile_async_16_swizzled_64(
    half* shared_tile,
    const half* global_value,
    int64_t tile_start,
    int64_t tensor_base,
    int thread_id,
    int thread_count
) {
    constexpr int kHalfElementsPerAsyncCopy = 8;
    constexpr int kCopiesPerRow =
        kHeadDimension / kHalfElementsPerAsyncCopy;
    constexpr int kTileCopies =
        kKeyTileSize * kCopiesPerRow;

#pragma unroll 1
    for (
        int copy = thread_id;
        copy < kTileCopies;
        copy += thread_count
    ) {
        const int local_row =
            copy >> 3;

        const int logical_chunk =
            copy & 7;

        const int logical_dimension =
            logical_chunk << 3;

        const int physical_chunk =
            logical_chunk ^ (local_row & 7);

        const int physical_dimension =
            physical_chunk << 3;

        const int64_t global_index =
            tensor_base +
            ((tile_start + local_row) << 6) +
            logical_dimension;

        const int shared_offset =
            (local_row << 6) +
            physical_dimension;

        copy_global_to_shared_async_16(
            &shared_tile[shared_offset],
            &global_value[global_index]
        );
    }
}


template <int kKeyTileSize, bool kUseRawPv = false, bool kUseRawQk = false>
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

    constexpr int kKernelWarpsPerBlock = (kUseRawQk && kUseRawPv) ? 4 : kWarpsPerBlock;
    constexpr int kKernelThreadsPerBlock = kWarpSize * kKernelWarpsPerBlock;
    constexpr int kKernelQueryTileSize = kKernelWarpsPerBlock * kQueryRowsPerWarp;

    static_assert(
        kKeyTileSize % kMmaK == 0,
        "Key tile must be divisible by the MMA K dimension"
    );

    static_assert(
        kKernelQueryTileSize ==
            kKernelWarpsPerBlock * kQueryRowsPerWarp
    );

    static_assert(
        !kUseRawPv || kKeyTileSize == 32,
        "The raw PV experiment requires Bc=32"
    );

    static_assert(
        !kUseRawQk || kKeyTileSize == 32,
        "The raw QK experiment requires Bc=32"
    );

    static_assert(
        !kUseRawQk || kUseRawPv,
        "The raw Qk experiment requires the raw PV path"
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
        kKernelQueryTileSize;

    const int64_t tensor_base =
        batch_head_index *
        sequence_length *
        kHeadDimension;

    constexpr int kSharedHeadStride = kHeadDimension + 8; // 72 
    constexpr int kValueSharedStride = (kUseRawQk && kUseRawPv) ? kHeadDimension : kSharedHeadStride;
    constexpr int kKeySharedStride = (kUseRawQk && kUseRawPv) ? kHeadDimension : kSharedHeadStride;
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
        kKernelQueryTileSize
    ][
        kSharedHeadStride
    ];

    constexpr int kKeySharedStages = (kUseRawQk && kUseRawPv) ? 2 : 1;

    __shared__ __align__(128)
    half key_shared[kKeySharedStages]
    [
        kKeyTileSize
    ][
        kKeySharedStride
    ];

    __shared__ __align__(128)
    half value_shared[
        kKeyTileSize
    ][
        kValueSharedStride
    ];

    constexpr int kScoreSharedWarps = kUseRawQk ? 1 : kKernelWarpsPerBlock;
    constexpr int kScoreSharedRows = kUseRawQk ? 1 : kQueryRowsPerWarp;
    constexpr int kScoreSharedColumns = kUseRawQk ? 1 : kKeyTileSize;

    __shared__ __align__(32)
    float score_shared[
        kScoreSharedWarps
    ][
        kScoreSharedRows
    ][
        kScoreSharedColumns
    ];

    constexpr int kProbabilitySharedWarps = kUseRawQk ? 1 : kKernelWarpsPerBlock;
    constexpr int kProbabilitySharedRows = kUseRawQk ? 1 : kQueryRowsPerWarp;
    constexpr int kProbabilitySharedColumns = kUseRawQk ? 1 : kKeyTileSize;
    __shared__ __align__(32)
    half probability_shared[
        kProbabilitySharedWarps
    ][
        kProbabilitySharedRows
    ][
        kProbabilitySharedColumns
    ];

    /*
    The raw-PV specialization does not matrielize PV
    A one-element placeholder keeps the declaration valid
    while avoiding the 16 KiB allocation.
    */

    constexpr int kPvSharedWarps = kUseRawPv ? 1 : kKernelWarpsPerBlock;
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
        kKernelWarpsPerBlock
    ][
        kQueryRowsPerWarp
    ];

    __shared__ __align__(32)
    float inverse_sum_shared[
        kKernelWarpsPerBlock
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
        kKernelQueryTileSize * kHeadDimension;

    for (
        int element = thread_id;
        element < kQueryElements;
        element += kKernelThreadsPerBlock
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

    if constexpr(kUseRawQk && kUseRawPv){
        /*
        Prime pipeline with K0 and V0.
        */

        stage_tile_async_16_swizzled_64<kKeyTileSize>(
            &key_shared[0][0][0],
            key,
            0,
            tensor_base,
            thread_id,
            kKernelThreadsPerBlock
        );
        stage_tile_async_16_swizzled_64<kKeyTileSize>(
            &value_shared[0][0],
            value,
            0,
            tensor_base,
            thread_id,
            kKernelThreadsPerBlock
        );

        commit_async_copy_group();
        wait_for_async_copy_group();
        __syncthreads();
    }


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

         const int key_stage = (kUseRawPv && kUseRawQk) ? ((key_tile_start / kKeyTileSize) & 1) : 0;

        const int64_t next_key_tile_start = key_tile_start + kKeyTileSize;
        const bool has_next_tile = next_key_tile_start < sequence_length;

        if constexpr(kUseRawQk && kUseRawPv){
            /*
            K[current] and V[current] are already ready.

            Launch K[next] into the alternate K buffer.
            It can now move while we compute the current tile
            */

            if(has_next_tile){
                const int next_key_stage = key_stage ^ 1;
                stage_tile_async_16_swizzled_64<kKeyTileSize>(
                    &key_shared[next_key_stage][0][0],
                    key,
                    next_key_tile_start,
                    tensor_base,
                    thread_id,
                    kKernelThreadsPerBlock
                );

                commit_async_copy_group();
            }
        }else{
                /*
                Preserve the existing non-pipelined behavior for
                legacy specialization.
                */

                stage_tile_async_16<kKeyTileSize>(
                    &key_shared[0][0][0],
                    key,
                    key_tile_start,
                    tensor_base,
                    kSharedHeadStride,
                    thread_id,
                    kKernelThreadsPerBlock
                );

                stage_tile_async_16<kKeyTileSize>(
                    &value_shared[0][0],
                    value,
                    key_tile_start,
                    tensor_base,
                    kSharedHeadStride,
                    thread_id,
                    kKernelThreadsPerBlock
                );

                commit_async_copy_group();
                wait_for_async_copy_group();
                __syncthreads();
            }


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

         MmaOperandA probability_fragments_from_registers[2];
         if constexpr (kUseRawQk){
            constexpr int kRawMmaOutputColumns = 8;

            constexpr int kRawScoresSubtiles = kKeyTileSize / kRawMmaOutputColumns;

            MmaAccumulator score_accumulators[kRawScoresSubtiles];
            #pragma unroll
            for(int score_subtile = 0; score_subtile < kRawScoresSubtiles; ++score_subtile){
                score_accumulators[score_subtile] = zero_mma_accumulator();
            }

            #pragma unroll
            for(int dimension_offset = 0; dimension_offset < kHeadDimension; dimension_offset += kMmaK){
                const half* query_tile = &query_shared[warp_id * kQueryRowsPerWarp][dimension_offset];

                const MmaOperandA query_fragment = load_mma_a_row_major(query_tile, kSharedHeadStride);

                #pragma unroll
                for(int score_subtile = 0; score_subtile < kRawScoresSubtiles; ++score_subtile){
                    const int key_offset = score_subtile * kRawMmaOutputColumns;

                    const MmaOperandB key_fragment = load_mma_b_k_transpose_from_swizzled_row_major(&key_shared[key_stage][0][0], key_offset, dimension_offset);
                    mma_m16n8k16_f16_f32(score_accumulators[score_subtile], query_fragment, key_fragment);
                }
            }

            float upper_max = -CUDART_INF_F;
            float lower_max = -CUDART_INF_F;

            #pragma unroll
            for(int score_subtile=0; score_subtile < kRawScoresSubtiles; score_subtile++){
                upper_max = fmaxf(upper_max, fmaxf(score_accumulators[
                    score_subtile
                    ].registers[0] * scale,
                    score_accumulators[score_subtile].registers[1] * scale
                ));

                lower_max = fmaxf(lower_max,
                fmaxf(
                    score_accumulators[score_subtile].registers[2] * scale,
                    score_accumulators[score_subtile].registers[3] * scale
                ));
            }

            upper_max = fmaxf(upper_max, __shfl_xor_sync(
                kFullWarpMask,
                upper_max,
                1,
                4
            ));

            upper_max = fmaxf(upper_max, __shfl_xor_sync(
                kFullWarpMask,
                upper_max,
                2,
                4
            ));

            lower_max = fmaxf(lower_max, __shfl_xor_sync(kFullWarpMask,lower_max, 1, 4));
            lower_max = fmaxf(lower_max, __shfl_xor_sync(kFullWarpMask, lower_max, 2, 4));

            const int upper_source_lane = (lane_id & 7) * 4;
            const float canonical_upper_max = __shfl_sync(kFullWarpMask, upper_max, upper_source_lane);

            const float canonical_lower_max = __shfl_sync(kFullWarpMask, lower_max, upper_source_lane);

            float tile_row_max_from_raw = -CUDART_INF_F;

            if(lane_id < 8){
                tile_row_max_from_raw = canonical_upper_max;
            }else if(lane_id < 16){
                tile_row_max_from_raw = canonical_lower_max;
            }

            float new_row_max_for_lane = -CUDART_INF_F;
            float previous_scale_for_lane = 0.0f;

            if(lane_id < kQueryRowsPerWarp){
                new_row_max_for_lane = fmaxf(running_row_max, tile_row_max_from_raw);

                previous_scale_for_lane = running_row_sum == 0.0f ? 0.0f
                : __expf(running_row_max - new_row_max_for_lane);

                running_row_max = new_row_max_for_lane;

                previous_scale_shared[warp_id][lane_id] = previous_scale_for_lane;
            }

            const int group_id = lane_id >> 2;
            const float upper_new_row_max = __shfl_sync(kFullWarpMask, new_row_max_for_lane, group_id);
            const float lower_new_row_max = __shfl_sync(kFullWarpMask, new_row_max_for_lane, group_id + 8);

            
            

            float upper_probability_sum = 0.0f;
            float lower_probability_sum = 0.0f;

            #pragma unroll
            for(int score_subtile = 0; score_subtile < kRawScoresSubtiles; ++score_subtile){
                MmaAccumulator& score_accumulator = score_accumulators[score_subtile];

                score_accumulator.registers[0] = __expf(score_accumulator.registers[0] * scale - upper_new_row_max);
                score_accumulator.registers[1] = __expf(score_accumulator.registers[1] * scale - upper_new_row_max);
                score_accumulator.registers[2] = __expf(score_accumulator.registers[2] * scale - lower_new_row_max);
                score_accumulator.registers[3] = __expf(score_accumulator.registers[3] * scale - lower_new_row_max);

                upper_probability_sum += score_accumulator.registers[0] + score_accumulator.registers[1];
                lower_probability_sum += score_accumulator.registers[2] + score_accumulator.registers[3];
            }

            upper_probability_sum += __shfl_xor_sync(kFullWarpMask, upper_probability_sum, 1, 4);
            upper_probability_sum += __shfl_xor_sync(kFullWarpMask, upper_probability_sum, 2, 4);
            lower_probability_sum += __shfl_xor_sync(kFullWarpMask, lower_probability_sum, 1, 4);
            lower_probability_sum += __shfl_xor_sync(kFullWarpMask, lower_probability_sum, 2, 4);

            const int canonical_sum_source_lane = (lane_id & 7) * 4;
            const float canonical_upper_sum = __shfl_sync(kFullWarpMask, upper_probability_sum, canonical_sum_source_lane);
            const float canonical_lower_sum = __shfl_sync(kFullWarpMask, lower_probability_sum, canonical_sum_source_lane);

            float tile_row_sum_from_raw = 0.0f;

            if(lane_id < 8){
                tile_row_sum_from_raw = canonical_upper_sum;
            }else if(lane_id < 16){
                tile_row_sum_from_raw = canonical_lower_sum;
            }

            if(lane_id < kQueryRowsPerWarp){
                running_row_sum = previous_scale_for_lane * running_row_sum + tile_row_sum_from_raw;
            }

            probability_fragments_from_registers[0] = make_probability_mma_a(score_accumulators[0], score_accumulators[1]);
            probability_fragments_from_registers[1] = make_probability_mma_a(score_accumulators[2], score_accumulators[3]);

            


            
         }else{
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
                        &key_shared[key_stage][
                            key_subtile_offset
                        ][
                            dimension_offset
                        ];

                    wmma::load_matrix_sync(
                        query_fragment,
                        query_tile,
                        kSharedHeadStride
                    );

                    wmma::load_matrix_sync(
                        key_fragment,
                        key_transpose_tile,
                        kSharedHeadStride
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
        if constexpr(!kUseRawQk){
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
    }
        /*
         * Ensure every probability row and previous
         * scale is visible to all lanes in the warp.
         */

        __syncwarp(kFullWarpMask);

        if constexpr(kUseRawPv && kUseRawQk){
            /*
            Tile 0's V was fully loaded in the pipeline
            Later V tiles are launched async at the 
            bottom of the previous iteration
            */

            if(key_tile_start > 0){
                if(has_next_tile){
                    /*
                    Outstanding groups:

                    older: V[current]
                    newer: K[next]

                    Wait until only the newest group may remain.
                    Therefore V[current] is ready while K[next]
                    may continue transferring during PV.
                    */

                    wait_for_async_copy_group_1();
                }else{
                    /*
                    Last tile:

                    There is no K[next], so V[current] is the 
                    only outstanding group. wait_group 1 would 
                    be allowed to leave it outstanding.
                   
                    */

                    wait_for_async_copy_group();

                }
                __syncthreads();
            }
        }


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
            Load the two P[16x16] reduction fragments once.
            They are reused by every output-dim subtile

            */

            MmaOperandA probability_fragments[2];

            if constexpr (kUseRawQk) {
                probability_fragments[0] =
                    probability_fragments_from_registers[0];

                probability_fragments[1] =
                    probability_fragments_from_registers[1];
            } else {
            #pragma unroll
                for(int reduction_subtile = 0;
                    reduction_subtile < 2;
                    ++reduction_subtile) {

                    const int reduction_offset =
                        reduction_subtile * kMmaK;

                    const half* probability_tile =
                        &probability_shared[
                            warp_id
                        ][
                            0
                        ][
                            reduction_offset
                        ];

                    probability_fragments[reduction_subtile] =
                        load_mma_a_row_major(
                            probability_tile,
                            kKeyTileSize
                        );
                }
            }
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
                int reduction_subtile = 0;
                reduction_subtile < 2;
                reduction_subtile++
            ) {
                const int reduction_offset = reduction_subtile * kMmaK;

                const MmaOperandA& probability_fragment = probability_fragments[reduction_subtile];

                MmaOperandB value_fragment;

                if constexpr (kUseRawQk && kUseRawPv) {
                    value_fragment =
                        load_mma_b_col_major_from_swizzled_value(
                            &value_shared[0][0],
                            reduction_offset,
                            output_dimension_offset
                        );
                } else {
                    const half* value_tile =
                        &value_shared[
                            reduction_offset
                        ][
                            output_dimension_offset
                        ];

                    value_fragment =
                        load_mma_b_col_major_from_row_major(
                            value_tile,
                            kValueSharedStride
                        );
                }

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
                    kSharedHeadStride
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
    Everyone must finish consuming V[current]
    before value_shared is reused

    K[next] has been moving async during the 
    current Qk/softmax/PV work.
    */

   __syncthreads();

    if constexpr (kUseRawQk && kUseRawPv) {
        if (has_next_tile) {

            /*
            * K[next] has been moving during current compute.
            * Finish it before the next iteration uses it.
            */
            wait_for_async_copy_group();

            /*
            * K[next] was cooperatively loaded by the CTA.
            * Make it visible to all consumers.
            */
            __syncthreads();

            /*
            * Start V[next], but DO NOT wait for it.
            *
            * It will remain in flight as the next
            * iteration starts QK + softmax.
            */
            stage_tile_async_16_swizzled_64<kKeyTileSize>(
                &value_shared[0][0],
                value,
                next_key_tile_start,
                tensor_base,
                thread_id,
                kKernelThreadsPerBlock
            );

            commit_async_copy_group();
        }
    }
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
            for(int register_pair = 0; register_pair < 2; ++register_pair){
                const int register_index = register_pair * 2;

                /*
                (0,1) belong to the same upper row
                (2,3) belong to the same lower row
                */

                const int local_row = mma_accumulator_row(lane_id, register_index);

                /*
                register_index is 0 or 2. so this gives
                the even column of the contiguous pair
                */

                const int dimension = output_subtiles * kRawMmaOutputColumns + mma_accumulator_column(lane_id, register_index);
                const int accumulator_index = output_subtiles * 4 + register_index;

                const int64_t global_query_row = query_tile_start + warp_id * kQueryRowsPerWarp + local_row;

                if(global_query_row < sequence_length){
                    const float inverse_sum = inverse_sum_shared[warp_id][local_row];

                    const float first = output_accumulator[accumulator_index] * inverse_sum;
                    const float second = output_accumulator[accumulator_index + 1] * inverse_sum;

                    const half2 packed_output = __floats2half2_rn(first, second);

                    const int64_t global_index = tensor_base + global_query_row * kHeadDimension + dimension;

                    *reinterpret_cast<half2*>(&output[global_index]) = packed_output;
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

__global__ void tensorcore_attention_forward_kernel_d64_production(
    const half* __restrict__ query,
    const half* __restrict__ key,
    const half* __restrict__ value,
    half* __restrict__ output,
    int64_t sequence_length,
    float scale
) {
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
        kProductionBlockM;

    const int64_t tensor_base =
        batch_head_index *
        sequence_length *
        kHeadDimension;


    /*
     * 128x128x64 production tile.
     *
     * Q = 16 KiB
     * K = 16 KiB
     * V = 16 KiB
     *
     * Total static shared = 48 KiB.
     *
     * All three operands use the compact
     * 64-wide XOR-swizzled shared layout.
     */
    __shared__ __align__(128)
    half query_shared[
        kProductionBlockM
    ][
        kHeadDimension
    ];

    __shared__ __align__(128)
    half key_shared[
        kProductionBlockN
    ][
        kHeadDimension
    ];

    __shared__ __align__(128)
    half value_shared[
        kProductionBlockN
    ][
        kHeadDimension
    ];


    /*
     * Two 16-row M slices per warp.
     *
     * warp 0:   0..15,  64..79
     * warp 1:  16..31,  80..95
     * warp 2:  32..47,  96..111
     * warp 3:  48..63, 112..127
     */
    const int first_query_row =
        warp_id * kMmaM;

    const int second_query_row =
        first_query_row +
        kProductionWarpsPerBlock *
            kMmaM;


    /*
     * Stage the entire 128x64 Q tile once.
     *
     * Current production path assumes sequence
     * length is divisible by 128, so every Q tile
     * is complete.
     */
    stage_tile_async_16_swizzled_64<
        kProductionBlockM
    >(
        &query_shared[0][0],
        query,
        query_tile_start,
        tensor_base,
        thread_id,
        kProductionThreadsPerBlock
    );

    commit_async_copy_group();
    wait_for_async_copy_group();

    __syncthreads();


    /*
     * Full 128-column score geometry.
     *
     * 16 x m16n8 score fragments cover N=128.
     * 8 x 16-column probability fragments cover
     * the PV reduction dimension.
     */
    constexpr int kProductionRawMmaOutputColumns =
        8;

    constexpr int kProductionScoreSubtiles =
        kProductionBlockN /
        kProductionRawMmaOutputColumns;
    // 16

    constexpr int kProductionOutputSubtiles =
        kHeadDimension /
        kProductionRawMmaOutputColumns;
    // 8

    constexpr int kProductionProbabilityFragments =
        kProductionBlockN /
        kMmaK;
    // 8

    static_assert(
        kProductionScoreSubtiles ==
        kProductionProbabilityFragments * 2
    );


    /*
     * Persistent FP32 output numerators.
     *
     * Each lane owns 32 FP32 values for each
     * 16-row M slice.
     */
    
    MmaAccumulator first_output_accumulator[kProductionOutputSubtiles];
    MmaAccumulator second_output_accumulator[kProductionOutputSubtiles];

    #pragma unroll
    for(int output_subtile = 0; output_subtile < kProductionOutputSubtiles; ++output_subtile){
        first_output_accumulator[output_subtile] = zero_mma_accumulator();
        second_output_accumulator[output_subtile] = zero_mma_accumulator();
    }


    /*
     * Independent online-softmax state for the
     * two 16-row slices owned by this warp.
     *
     * Lanes 0..15 own row-level max/sum state.
     */
    float first_running_row_max =
        -CUDART_INF_F;

    float first_running_row_sum =
        0.0f;

    float second_running_row_max =
        -CUDART_INF_F;

    float second_running_row_sum =
        0.0f;


    /*
     * MMA accumulator row ownership.
     *
     * Registers 0/1 correspond to one row from
     * lanes 0..7. Registers 2/3 correspond to
     * the row eight positions below it.
     */
    const int upper_output_row =
        lane_id >> 2;

    const int lower_output_row =
        upper_output_row + 8;


    /*
     * Stream through physical K/V tiles of 128 rows.
     *
     * Each iteration performs:
     *
     *   QK over all 128 K rows
     *   one online-softmax update over 128 scores
     *   PV over all 128 probability values
     *   one persistent O update
     *
     * No 32-column compute-window approximation.
     */
    for (
        int64_t key_tile_start = 0;
        key_tile_start < sequence_length;
        key_tile_start += kProductionBlockN
    ) {

        /*
         * Stage the full 128x64 K/V tiles.
         */
        stage_tile_async_16_swizzled_64<
            kProductionBlockN
        >(
            &key_shared[0][0],
            key,
            key_tile_start,
            tensor_base,
            thread_id,
            kProductionThreadsPerBlock
        );

        stage_tile_async_16_swizzled_64<
            kProductionBlockN
        >(
            &value_shared[0][0],
            value,
            key_tile_start,
            tensor_base,
            thread_id,
            kProductionThreadsPerBlock
        );

        commit_async_copy_group();
        wait_for_async_copy_group();

        __syncthreads();


        /*
         * Full 128-column QK score state.
         *
         * Two independent score banks because each
         * warp owns two 16-row Q slices.
         */
        MmaAccumulator
            first_score_accumulators[
                kProductionScoreSubtiles
            ];

        MmaAccumulator
            second_score_accumulators[
                kProductionScoreSubtiles
            ];


#pragma unroll
        for (
            int score_subtile = 0;
            score_subtile <
                kProductionScoreSubtiles;
            ++score_subtile
        ) {
            first_score_accumulators[
                score_subtile
            ] =
                zero_mma_accumulator();

            second_score_accumulators[
                score_subtile
            ] =
                zero_mma_accumulator();
        }


        /*
         * QK^T.
         *
         * Q fragments are reused across all 16
         * N subtiles for a given K=16 dimension.
         *
         * Each K fragment is reused across both
         * M slices.
         */
#pragma unroll
        for (
            int dimension_offset = 0;
            dimension_offset < kHeadDimension;
            dimension_offset += kMmaK
        ) {
            const MmaOperandA
                first_query_fragment =
                    load_mma_a_from_swizzled_row_major_64(
                        &query_shared[0][0],
                        first_query_row,
                        dimension_offset
                    );

            const MmaOperandA
                second_query_fragment =
                    load_mma_a_from_swizzled_row_major_64(
                        &query_shared[0][0],
                        second_query_row,
                        dimension_offset
                    );


#pragma unroll
            for (
                int score_subtile = 0;
                score_subtile <
                    kProductionScoreSubtiles;
                ++score_subtile
            ) {
                const int key_offset =
                    score_subtile *
                    kProductionRawMmaOutputColumns;

                const MmaOperandB
                    key_fragment =
                        load_mma_b_k_transpose_from_swizzled_row_major(
                            &key_shared[0][0],
                            key_offset,
                            dimension_offset
                        );

                mma_m16n8k16_f16_f32(
                    first_score_accumulators[
                        score_subtile
                    ],
                    first_query_fragment,
                    key_fragment
                );

                mma_m16n8k16_f16_f32(
                    second_score_accumulators[
                        score_subtile
                    ],
                    second_query_fragment,
                    key_fragment
                );
            }
        }


        /*
         * Full 128-column online softmax.
         *
         * The helper now instantiates with
         * kProductionScoreSubtiles == 16.
         *
         * Therefore max/sum/rescaling happens once
         * for all 128 score columns.
         */

        float first_previous_scale_for_lane =
            0.0f;

        float second_previous_scale_for_lane =
            0.0f;


        normalize_score_tile_in_place<
            kProductionScoreSubtiles
        >(
            first_score_accumulators,
            scale,
            lane_id,
            first_running_row_max,
            first_running_row_sum,
            first_previous_scale_for_lane
        );

        normalize_score_tile_in_place<
            kProductionScoreSubtiles
        >(
            second_score_accumulators,
            scale,
            lane_id,
            second_running_row_max,
            second_running_row_sum,
            second_previous_scale_for_lane
        );


        /*
         * Broadcast the online-rescale coefficient
         * from each logical row owner to the lanes
         * that own that row's output registers.
         */
        const float
            first_upper_previous_scale =
                __shfl_sync(
                    kFullWarpMask,
                    first_previous_scale_for_lane,
                    upper_output_row
                );

        const float
            first_lower_previous_scale =
                __shfl_sync(
                    kFullWarpMask,
                    first_previous_scale_for_lane,
                    lower_output_row
                );

        const float
            second_upper_previous_scale =
                __shfl_sync(
                    kFullWarpMask,
                    second_previous_scale_for_lane,
                    upper_output_row
                );

        const float
            second_lower_previous_scale =
                __shfl_sync(
                    kFullWarpMask,
                    second_previous_scale_for_lane,
                    lower_output_row
                );
        
        #pragma unroll
        for(int output_subtile = 0; output_subtile < kProductionOutputSubtiles; ++output_subtile){
            #pragma unroll 
            for(int register_index = 0; register_index < 4; ++register_index){
                const bool is_upper_row = register_index < 2;

                const float first_previous_scale = is_upper_row ? first_upper_previous_scale : first_lower_previous_scale;
                const float second_previous_scale = is_upper_row ? second_upper_previous_scale : second_lower_previous_scale;

                first_output_accumulator[output_subtile].registers[register_index] *= first_previous_scale;
                second_output_accumulator[output_subtile].registers[register_index] *= second_previous_scale;
            }
        }

        MmaOperandA first_probability_fragments[kProductionProbabilityFragments];
        MmaOperandA second_probability_fragments[kProductionProbabilityFragments];

        #pragma unroll
        for(int probability_fragment = 0; probability_fragment < kProductionProbabilityFragments; ++probability_fragment){
            first_probability_fragments[probability_fragment] = make_probability_mma_a(
                first_score_accumulators[probability_fragment*2],
                first_score_accumulators[probability_fragment*2 + 1]
            );

            second_probability_fragments[probability_fragment] = make_probability_mma_a(
                second_score_accumulators[probability_fragment * 2],
                second_score_accumulators[probability_fragment * 2 + 1]
            );
        }


        /*
         * PV.
         *
         * For every 8-column output D subtile,
         * reduce across all eight 16-row portions
         * of the 128-wide probability tile.
         *
         * Each V fragment feeds both M slices.
         */
#pragma unroll
        for (
            int output_subtile = 0;
            output_subtile <
                kProductionOutputSubtiles;
            ++output_subtile
        ) {
            const int output_dimension_offset =
                output_subtile *
                kProductionRawMmaOutputColumns;


#pragma unroll
            for (
                int reduction_subtile = 0;
                reduction_subtile <
                    kProductionProbabilityFragments;
                ++reduction_subtile
            ) {
                const int reduction_offset =
                    reduction_subtile *
                    kMmaK;

                const MmaOperandB
                    value_fragment =
                        load_mma_b_col_major_from_swizzled_value(
                            &value_shared[0][0],
                            reduction_offset,
                            output_dimension_offset
                        );

                mma_m16n8k16_f16_f32(
                    first_output_accumulator[output_subtile],
                    first_probability_fragments[
                        reduction_subtile
                    ],
                    value_fragment
                );

                mma_m16n8k16_f16_f32(
                    second_output_accumulator[output_subtile],
                    second_probability_fragments[
                        reduction_subtile
                    ],
                    value_fragment
                );
            }
        }


        /*
         * All warps must finish consuming this K/V
         * tile before the next iteration overwrites
         * shared memory.
         */
        __syncthreads();
    }

    /*
    K/V dead after the final KV iteration.
    Reuse the 128x64 V shared tile as the output
    epilogue scratch buffer
    */

    half* output_shared = &value_shared[0][0];


    /*
     * Final normalization.
     */
    const float first_upper_inverse_sum =
        1.0f /
        __shfl_sync(
            kFullWarpMask,
            first_running_row_sum,
            upper_output_row
        );

    const float first_lower_inverse_sum =
        1.0f /
        __shfl_sync(
            kFullWarpMask,
            first_running_row_sum,
            lower_output_row
        );

    const float second_upper_inverse_sum =
        1.0f /
        __shfl_sync(
            kFullWarpMask,
            second_running_row_sum,
            upper_output_row
        );

    const float second_lower_inverse_sum =
        1.0f /
        __shfl_sync(
            kFullWarpMask,
            second_running_row_sum,
            lower_output_row
        );
    
    /*
    Production-style output

    Phase 1:
        Normalize MMa-owned FP32 output regs,
        convert to FP16, and scatter them into the
        reusable 128x64 shared tile

    The scratch tile uses the same 8-half XOR 
    swizzle as Q/K/V:

        physical_chunk = logical_chunk ^ (row & 7)
    
    This converts awkward MMA register ownership
    into a complete logical output tile
    */

    #pragma unroll
    for(int output_subtile = 0; output_subtile < kProductionOutputSubtiles; ++output_subtile){
        #pragma unroll
        for(int register_pair = 0; register_pair < 2; ++register_pair){
            const int register_index = register_pair * 2;

            const int local_row = mma_accumulator_row(
                lane_id,
                register_index
            );
            const int logical_dimension = output_subtile * kProductionRawMmaOutputColumns +
                mma_accumulator_column(
                    lane_id,
                    register_index
                );
            
            const bool is_upper_row = register_index < 2;
            const float first_inverse_sum = is_upper_row ? first_upper_inverse_sum : first_lower_inverse_sum;
            const float second_inverse_sum = is_upper_row ? second_upper_inverse_sum : second_lower_inverse_sum;


            /*
            First M Slice
            */

            {
                const int output_row = first_query_row + local_row;
                
                const float first_value = first_output_accumulator[output_subtile].registers[register_index] * first_inverse_sum;
                const float second_value = first_output_accumulator[output_subtile].registers[register_index + 1] * first_inverse_sum;

                const half2 packed_output = __floats2half2_rn(first_value, second_value);

                const int logical_chunk = logical_dimension >> 3;
                const int physical_chunk = logical_chunk ^ (output_row & 7);

                const int physical_dimension = (physical_chunk << 3) + (logical_dimension & 7);

                *reinterpret_cast<half2*>(&output_shared[(output_row << 6) + physical_dimension]) = packed_output;
            }
            /*
            Second M Slice
            */
            {
                const int output_row = second_query_row + local_row;
                
                const float first_value = second_output_accumulator[output_subtile].registers[register_index] * second_inverse_sum;
                const float second_value = second_output_accumulator[output_subtile].registers[register_index + 1] * second_inverse_sum;

                const half2 packed_output = __floats2half2_rn(first_value, second_value);

                const int logical_chunk = logical_dimension >> 3;
                const int physical_chunk = logical_chunk ^ (output_row & 7);

                const int physical_dimension = (physical_chunk << 3) + (logical_dimension & 7);

                *reinterpret_cast<half2*>(&output_shared[(output_row << 6) + physical_dimension]) = packed_output;
            }
        }
    }

    __syncthreads();

    /*
    * Phase 2:
    *
    * Repartition ownership for global memory.
    *
    * One warp writes one complete logical output
    * row at a time:
    *
    *   lane  0 -> cols  0..1
    *   lane  1 -> cols  2..3
    *   ...
    *   lane 31 -> cols 62..63
    *
    * Therefore the warp collectively writes one
    * contiguous 128-byte output row.
    *
    * Four warps process four rows concurrently;
    * 32 iterations cover all 128 rows.
    */

    #pragma unroll
    for(int row_iteration = 0; row_iteration < kProductionBlockM / kProductionWarpsPerBlock; row_iteration++){
        const int output_row = row_iteration * kProductionWarpsPerBlock + warp_id;

        const int logical_dimension = lane_id * 2;
        const int logical_chunk = logical_dimension >> 3;

        const int physical_chunk = logical_chunk ^ (output_row & 7);
        const int physical_dimension = (physical_chunk << 3) + (logical_dimension & 7);

        const half2 packed_output = *reinterpret_cast<const half2*>(&output_shared[(output_row << 6) + physical_dimension]);

        const int64_t global_query_row = query_tile_start + output_row;

        if(global_query_row < sequence_length){
            const int64_t global_index = tensor_base + global_query_row * kHeadDimension + logical_dimension;

            *reinterpret_cast<half2*>(&output[global_index]) = packed_output;
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


template <int kKeyTileSize, bool kUseRawPv = false, bool kUseRawQk = false>
torch::Tensor tensorcore_attention_forward_impl(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value
) {

    constexpr int kKernelWarpsPerBlock =
    (kUseRawQk && kUseRawPv) ? 4 : kWarpsPerBlock;

    constexpr int kKernelThreadsPerBlock =
        kWarpSize * kKernelWarpsPerBlock;

    constexpr int kKernelQueryTileSize =
        kKernelWarpsPerBlock * kQueryRowsPerWarp;
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
            kKernelQueryTileSize -
            1
        ) /
        kKernelQueryTileSize;

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
        kKernelThreadsPerBlock,
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
        kKeyTileSize, kUseRawPv, kUseRawQk
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

torch::Tensor tensorcore_attention_forward_production_impl(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value
) {
    /*
     * 9A uses BlockN = 32, so K/V currently require
     * sequence length divisible by 32.
     */
    validate_tensorcore_attention_inputs<
        kProductionBlockN
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

    /*
     * Production 9A processes 128 query rows / CTA.
     */
    const int64_t num_query_tiles =
        (
            sequence_length +
            kProductionBlockM -
            1
        ) /
        kProductionBlockM;

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
        kProductionThreadsPerBlock,
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

    tensorcore_attention_forward_kernel_d64_production<<<
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

torch::Tensor tensorcore_attention_forward_bc32_raw_qk_raw_pv(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value
) {
    return tensorcore_attention_forward_impl<
        32,
        true,
        true
    >(
        query,
        key,
        value
    );
}

torch::Tensor tensorcore_attention_forward_production_128x128(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value
) {
    return tensorcore_attention_forward_production_impl(
        query,
        key,
        value
    );
}