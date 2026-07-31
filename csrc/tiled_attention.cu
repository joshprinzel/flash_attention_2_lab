#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace{
    constexpr int kHeadDimension = 64;
    constexpr int kQueryTileSize = 16;
    constexpr int kKeyTileSize = 32;
    constexpr int kThreadsPerBlock = 128;
    constexpr int kWarpsPerBlock = 4;
    constexpr int kRowsPerWarp = 4;
    constexpr int kWarpSize = 32;
    constexpr unsigned int kFullWarpMask = 0xffffffffu;

    static_assert(
        kQueryTileSize == kWarpsPerBlock * kRowsPerWarp
    );


    __device__ inline float warp_reduce_sum(float value){
        #pragma unroll
        for(int offset = kWarpSize / 2; offset > 0; offset >>= 1){
            value += __shfl_down_sync(
                kFullWarpMask,
                value,
                offset
            );
        }
        return value;
    }

    __global__ void tiled_attention_forward_kernel_d64(
        const float* query,
        const float* key,
        const float* value,
        float* output,
        int64_t sequence_length,
        float scale
    ){
        /*
            Grid:
                blockIdx.x -> query tile
                blockIdx.y -> flattened [batch, head]
            
            Block: 
                128 threads = 4 warps
                each warp owns 4 query rows
        */

        const int thread_id = threadIdx.x;
        const int lane_id = thread_id % kWarpSize;
        const int warp_id = thread_id / kWarpSize;

        const int64_t batch_head = static_cast<int64_t>(blockIdx.y);
        const int64_t query_tile_start = static_cast<int64_t>(blockIdx.x) * kQueryTileSize;
        const int64_t batch_head_offset = batch_head * sequence_length * kHeadDimension;

        /*
        Each lane owns 2 dims:

            lane_id
            lane_id + 32
        */

        const int dimension_0 = lane_id;
        const int dimension_1 = lane_id + 32;

        __shared__ float key_shared[kKeyTileSize * kHeadDimension];
        __shared__ float value_shared[kKeyTileSize * kHeadDimension];

        float query_0[kRowsPerWarp];
        float query_1[kRowsPerWarp];

        float output_accumulator_0[kRowsPerWarp];
        float output_accumulator_1[kRowsPerWarp];

        /*
        Only lane zero requires the persistent softmax stats
        Other lanes receive the update factors through warp shuffles
        */

        float row_max[kRowsPerWarp];
        float row_sum[kRowsPerWarp];

        #pragma unroll
        for(int row = 0; row < kRowsPerWarp; row++){
            const int64_t query_position = query_tile_start + warp_id * kRowsPerWarp + row;

            if(query_position < sequence_length){
                const int64_t query_offset = batch_head_offset + query_position * kHeadDimension;
                query_0[row] = query[query_offset + dimension_0];
                query_1[row] = query[query_offset + dimension_1];
            }else{
                query_0[row] = 0.0f;
                query_1[row] = 0.0f;
            }

            output_accumulator_0[row] = 0.0f;
            output_accumulator_1[row] = 0.0f;

            row_max[row] = -INFINITY;
            row_sum[row] = 0.0f;
        }

        for(int64_t key_tile_start = 0; key_tile_start < sequence_length; key_tile_start += kKeyTileSize){
            /*
            Load on K tile and one V tile

            Flattened shared-memory index should give coalesced global
            accesses since neighboring threads load neighboring 
            head-dimension elements
            */

            constexpr int tile_elements = kKeyTileSize * kHeadDimension;

            for(int tile_index = thread_id; tile_index < tile_elements; tile_index += kThreadsPerBlock){
                const int key_row_in_tile = tile_index / kHeadDimension;

                const int dimension = tile_index % kHeadDimension;

                const int64_t key_position = key_tile_start + key_row_in_tile;

                if(key_position < sequence_length){
                    const int64_t global_offset = batch_head_offset + key_position * kHeadDimension + dimension;
                    key_shared[tile_index] = key[global_offset];
                    value_shared[tile_index] = value[global_offset];
                }else{
                    key_shared[tile_index] = 0.0f;
                    value_shared[tile_index] = 0.0f;
                }
            }
            __syncthreads();

            const int valid_keys = static_cast<int>(
                min(
                    static_cast<int64_t>(kKeyTileSize),
                    sequence_length - key_tile_start
                )
            );

            for(int key_row_in_tile = 0; key_row_in_tile < valid_keys; key_row_in_tile++){
                const int shared_key_offset = key_row_in_tile * kHeadDimension;

                const float key_0 = key_shared[shared_key_offset + dimension_0];
                const float key_1 = key_shared[shared_key_offset + dimension_1];
                const float value_0 = value_shared[shared_key_offset + dimension_0];
                const float value_1 = value_shared[shared_key_offset + dimension_1];

                #pragma unroll
                for(int row = 0; row < kRowsPerWarp; row++){
                    const int64_t query_position = query_tile_start + warp_id * kRowsPerWarp + row;

                    if(query_position >= sequence_length){
                        continue;
                    }

                    float partial_dot_product = query_0[row] * key_0 + query_1[row] * key_1;
                    const float dot_product = warp_reduce_sum(partial_dot_product);

                    float previous_scale = 0.0f;
                    float current_scale = 0.0f;

                    if(lane_id == 0){
                        const float score = dot_product * scale;
                        const float new_row_max = fmaxf(row_max[row], score);

                        previous_scale = expf(row_max[row] - new_row_max);
                        current_scale = expf(score - new_row_max);

                        row_sum[row] = previous_scale * row_sum[row] + current_scale;
                        row_max[row] = new_row_max;
                    }
                    previous_scale = __shfl_sync(
                        kFullWarpMask,
                        previous_scale,
                        0
                    );
                    current_scale = __shfl_sync(
                        kFullWarpMask,
                        current_scale,
                        0
                    );

                    output_accumulator_0[row] = previous_scale * output_accumulator_0[row] + current_scale * value_0;
                    output_accumulator_1[row] = previous_scale * output_accumulator_1[row] + current_scale * value_1;
                }
            }
            /*
            All warps must finish reading the current K/V tile before
            any thread overwrites shared memory with the next tile.
            */

            __syncthreads();
        }
        for(int row = 0; row < kRowsPerWarp; row++){
            const int64_t query_position = query_tile_start + warp_id * kRowsPerWarp + row;
            if(query_position >= sequence_length){
                continue;
            }

            const float normalization = __shfl_sync(kFullWarpMask, row_sum[row], 0);

            const int64_t output_offset = batch_head_offset + query_position * kHeadDimension;

            output[
                output_offset + dimension_0
            ] = output_accumulator_0[row] / normalization;

            output[
                output_offset + dimension_1
            ] = output_accumulator_1[row] / normalization;
        }
    }
    void validate_tiled_attention_inputs(
        const torch::Tensor& query,
        const torch::Tensor& key,
        const torch::Tensor& value
    ) {
        TORCH_CHECK(
            query.is_cuda() &&
            key.is_cuda() &&
            value.is_cuda(),
            "query, key, and value must be CUDA tensors"
        );

        TORCH_CHECK(
            query.is_contiguous() &&
            key.is_contiguous() &&
            value.is_contiguous(),
            "query, key, and value must be contiguous"
        );

        TORCH_CHECK(
            query.dim() == 4 &&
            key.dim() == 4 &&
            value.dim() == 4,
            "query, key, and value must be rank 4"
        );

        TORCH_CHECK(
            query.scalar_type() == torch::kFloat32 &&
            key.scalar_type() == torch::kFloat32 &&
            value.scalar_type() == torch::kFloat32,
            "the initial tiled kernel requires float32 inputs"
        );

        TORCH_CHECK(
            query.device() == key.device() &&
            query.device() == value.device(),
            "query, key, and value must be on the same CUDA device"
        );

        TORCH_CHECK(
            query.sizes() == key.sizes() &&
            query.sizes() == value.sizes(),
            "the initial tiled kernel requires identical Q, K, and V shapes"
        );

        TORCH_CHECK(
            query.size(3) == kHeadDimension,
            "the initial tiled kernel supports head dimension 64"
        );
    }

} // namespace

torch::Tensor tiled_attention_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value
) {
    validate_tiled_attention_inputs(
        query,
        key,
        value
    );

    const c10::cuda::CUDAGuard device_guard(
        query.device()
    );

    torch::Tensor output = torch::empty_like(query);

    const int64_t batch_size = query.size(0);
    const int64_t num_heads = query.size(1);
    const int64_t sequence_length = query.size(2);

    if(sequence_length == 0){
        return output;
    }

    const int64_t query_tiles = (
        sequence_length + kQueryTileSize - 1
    ) / kQueryTileSize;

    const int64_t batch_heads = batch_size * num_heads;

    TORCH_CHECK(
        query_tiles <= static_cast<int64_t>(std::numeric_limits<unsigned int>::max()),
        "too many query tiles for CUDA grid"
    );

    TORCH_CHECK(
        batch_heads <= 65535,
        "batch x heads exceeds CUDA grid.y limit"
    );

    const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDimension));

    const dim3 grid(
        static_cast<unsigned int>(query_tiles),
        static_cast<unsigned int>(batch_heads)
    );

    tiled_attention_forward_kernel_d64<<<
    grid, kThreadsPerBlock, 0, at::cuda::getCurrentCUDAStream()
    >>>(
        query.data_ptr<float>(),
        key.data_ptr<float>(),
        value.data_ptr<float>(),
        output.data_ptr<float>(),
        sequence_length,
        scale
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}