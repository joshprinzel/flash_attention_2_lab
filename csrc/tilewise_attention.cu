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

    constexpr int kWarpSize = 32;
    constexpr int kWarpsPerBlock = 4;
    constexpr int kRowsPerWarp = 4;
    constexpr int kThreadsPerBlock =
        kWarpSize * kWarpsPerBlock;

    constexpr int kKeySharedStride =
        kKeyTileSize + 1;

    constexpr unsigned kFullWarpMask =
        0xffffffffu;

    static_assert(
        kQueryTileSize ==
        kWarpsPerBlock * kRowsPerWarp
    );

    __device__ inline float warp_allreduce_sum( float value){
        #pragma unroll
        for(int offset = kWarpSize/2; offset > 0; offset >>= 1){
            value += __shfl_down_sync(kFullWarpMask, value, offset);
        }
        return __shfl_sync(kFullWarpMask, value, 0);
    }

    __device__ inline float warp_allreduce_max(float value){
        #pragma unroll
        for(int offset = kWarpSize/2; offset > 0; offset >>= 1){
            value = fmaxf(value, __shfl_down_sync(kFullWarpMask, value, offset));
        }
        return __shfl_sync(kFullWarpMask, value, 0);
    }


    __global__ void tilewise_attention_forward_kernel_d64(
        const float* query,
        const float* key,
        const float* value,
        float* output,
        int64_t sequence_length,
        float scale
    ){
        const int thread_id = threadIdx.x;
        const int lane_id = thread_id % kWarpSize;
        const int warp_id = thread_id / kWarpSize;

        const int64_t batch_head = static_cast<int64_t>(blockIdx.y);
        const int64_t query_tile_start = static_cast<int64_t>(blockIdx.x) * kQueryTileSize;
        const int64_t batch_head_offset = batch_head * sequence_length * kHeadDimension;

        /*
        One warp owns a complete D=64 output row.

        Each lane owns two dimensions:
            lane_id
            lane_id + 32
        */

        const int dimension_0 = lane_id;
        const int dimension_1 = lane_id + kWarpSize;

        /*
        K is stored transposed:

            [head_dimension][key_position]

        This lets all lanes read the same head dim for 
        different keys without a 32 way shared memory bank conflict

        The +1 padding prevents bank alignment between successive
        head dim rows        
        */

        __shared__ float key_shared[kHeadDimension][kKeySharedStride];

        /*
        V remains row-major because lanes later read adjacent output
        dimension from the same value row
        */

        __shared__ float value_shared[kKeyTileSize][kHeadDimension];

        /*One probability per warp*/
        __shared__ float probability_shared[kWarpsPerBlock][kRowsPerWarp][kKeyTileSize];

        float query_0[kRowsPerWarp];
        float query_1[kRowsPerWarp];

        float output_accumulator_0[kRowsPerWarp];
        float output_accumulator_1[kRowsPerWarp];

        float row_max[kRowsPerWarp];
        float row_sum[kRowsPerWarp];

        #pragma unroll
        for(int row = 0; row < kRowsPerWarp; ++row){
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
            constexpr int tile_elements = kKeyTileSize * kHeadDimension;

            /*
            Cooperative KV staging
            */

            for(int tile_index = thread_id; tile_index < tile_elements; tile_index += kThreadsPerBlock){
                const int key_row_in_tile = tile_index / kHeadDimension;
                const int dimension = tile_index % kHeadDimension;

                const int64_t key_position = key_tile_start + key_row_in_tile;

                float key_element = 0.0f;
                float value_element = 0.0f;

                if(key_position < sequence_length){
                    const int64_t global_offset = batch_head_offset +
                    key_position * kHeadDimension + dimension;

                    key_element = key[global_offset];
                    value_element = value[global_offset];
                }

                key_shared[dimension][key_row_in_tile] = key_element;
                value_shared[key_row_in_tile][dimension] = value_element;
            }
            __syncthreads();

            int valid_keys = static_cast<int>(sequence_length - key_tile_start);

            if(valid_keys > kKeyTileSize){
                valid_keys = kKeyTileSize;
            }
            float previous_output_scale[kRowsPerWarp];

            /*
            Compute a complete [4 x 32] score tile per warp.

            One lane corresponds to one key column.
            */

            #pragma unroll
            for(int row = 0; row < kRowsPerWarp; row++){
                const int64_t query_position = query_tile_start + warp_id * kRowsPerWarp + row;

                /*
                This branch is uniform across the warp
                */

                if(query_position >= sequence_length){
                    continue;
                }
                float score = 0.0f;

                /*
                Each lane computes one complete Q dot K scores

                Query fragments are broadcast from their owning lanes.
                K is transposed in shared memory, so lane 1 reads key 1
                */

                #pragma unroll
                for(int source_lane = 0; source_lane < kWarpSize; source_lane++){
                    const float query_element_0 = __shfl_sync(
                        kFullWarpMask,
                        query_0[row],
                        source_lane
                    );

                    const float query_element_1 = __shfl_sync(
                        kFullWarpMask,
                        query_1[row],
                        source_lane
                    );

                    score = fmaf(query_element_0, key_shared[source_lane][lane_id], score);
                    score = fmaf(query_element_1, key_shared[source_lane + kWarpSize][lane_id], score);
                }

                if(lane_id < valid_keys){
                    score *= scale;
                }else{
                    score = -INFINITY;
                }

                const float tile_row_max = warp_allreduce_max(score);
                const float new_row_max = fmaxf(row_max[row], tile_row_max);
                const float previous_scale = expf(row_max[row] - new_row_max);
                const float probability = lane_id < valid_keys ? expf(score - new_row_max) : 0.0f;

                const float tile_row_sum = warp_allreduce_sum(probability);

                probability_shared[warp_id][row][lane_id] = probability;
                previous_output_scale[row] = previous_scale;

                row_sum[row] = previous_scale * row_sum[row] + tile_row_sum;
                row_max[row] = new_row_max;
            }

            __syncthreads();

            /*
            Compute the complete tile contribution:

            P_tile @ V_tile

            Each lane owns two output dimensions
            */

            #pragma unroll
            for(int row = 0; row < kRowsPerWarp; row++){
                const int64_t query_position = query_tile_start + warp_id * kRowsPerWarp + row;

                if(query_position >= sequence_length){
                    continue;
                }

                float tile_output_0 = 0.0f;
                float tile_output_1 = 0.0f;

                for(int key_row_in_tile = 0; key_row_in_tile < valid_keys; key_row_in_tile++){
                    const float probability = probability_shared[warp_id][row][key_row_in_tile];

                    tile_output_0 = fmaf(probability, value_shared[key_row_in_tile][dimension_0], tile_output_0);
                    tile_output_1 = fmaf(probability, value_shared[key_row_in_tile][dimension_1], tile_output_1);

                }

                /*
                Merge the complete K/V tile once
                */

                output_accumulator_0[row] = previous_output_scale[row] * output_accumulator_0[row] + tile_output_0;
                output_accumulator_1[row] = previous_output_scale[row] * output_accumulator_1[row] + tile_output_1;
            }
            __syncthreads();
        }
        #pragma unroll
        for(int row = 0; row < kRowsPerWarp; row++){
            const int64_t query_position = query_tile_start + warp_id * kRowsPerWarp + row;

            if(query_position >= sequence_length){
                continue;
            }

            const float inverse_row_sum = 1.0f / row_sum[row];

            const int64_t output_offset = batch_head_offset + query_position * kHeadDimension;

            output[output_offset + dimension_0] = output_accumulator_0[row] * inverse_row_sum;
            output[output_offset + dimension_1] = output_accumulator_1[row] * inverse_row_sum;
        }
    }

    void validate_tilewise_attention_inputs(
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
            "tilewise attention currently requires float32"
        );

        TORCH_CHECK(
            query.device() == key.device() &&
            query.device() == value.device(),
            "query, key, and value must be on the same CUDA device"
        );

        TORCH_CHECK(
            query.sizes() == key.sizes() &&
            query.sizes() == value.sizes(),
            "tilewise attention currently requires identical Q, K, and V shapes"
        );

        TORCH_CHECK(
            query.size(3) == kHeadDimension,
            "tilewise attention currently supports head dimension 64"
        );
    }
} // namespace

torch::Tensor tilewise_attention_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value
){
    validate_tilewise_attention_inputs(query, key, value);

    const c10::cuda::CUDAGuard device_guard(query.device());

    torch::Tensor output = torch::empty_like(query);

    const int64_t batch_size = query.size(0);
    const int64_t num_heads = query.size(1);
    const int64_t sequence_length = query.size(2);

    if(sequence_length == 0) return output;

    const int64_t num_query_tiles = (sequence_length + kQueryTileSize - 1) / kQueryTileSize;
    const int64_t num_batch_heads = batch_size * num_heads;

    TORCH_CHECK(
        num_query_tiles <= static_cast<int64_t>(std::numeric_limits<unsigned int>::max()),
        "too many query tiles for CUDA grid.x"
    );

    TORCH_CHECK(
        num_batch_heads <= 65535,
        "batch * num_heads exceeds CUDA grid.y limit"
    );

    const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDimension));

    const dim3 grid(
        static_cast<unsigned int>(num_query_tiles),
        static_cast<unsigned int>(num_batch_heads)
    );

    tilewise_attention_forward_kernel_d64<<<
        grid,
        kThreadsPerBlock,
        0,
        at::cuda::getCurrentCUDAStream()
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