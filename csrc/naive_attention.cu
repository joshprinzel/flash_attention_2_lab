#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cmath>
#include <cstdint>


namespace {

__device__ float warp_reduce_sum(float value){
    for(int offset = warpSize / 2; offset > 0; offset >>= 1){
        value += __shfl_down_sync(
            0xffffffff,
            value,
            offset
        );
    }
    return value;
}

template <int NUM_WARPS>
__device__ float block_reduce_sum(
    float value,
    float* warp_sums
){
    const int lane_id = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;

    value = warp_reduce_sum(value);
    if(lane_id == 0){
        warp_sums[warp_id] = value;
    }

    __syncthreads();

    float block_sum = 0.0f;
    if(warp_id == 0){
        block_sum = lane_id < NUM_WARPS ? warp_sums[lane_id] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
    }
    return block_sum;
}

template<int HEAD_DIM>
__global__ void block_online_attention_kernel(
    const float* query,
    const float* key, 
    const float* value,
    float* output,
    int64_t num_query_rows,
    int64_t sequence_length,
    float scale
){
    constexpr int WARP_SIZE = 32;
    constexpr int NUM_WARPS = HEAD_DIM / WARP_SIZE;
    const int64_t query_row = static_cast<int64_t>(blockIdx.x);

    if(query_row >= num_query_rows){
        return;
    }

    const int dimension = threadIdx.x;
    const int64_t batch_head = query_row / sequence_length;
    const int64_t query_offset = query_row * HEAD_DIM;

    __shared__ float query_shared[HEAD_DIM];
    __shared__ float warp_sums[NUM_WARPS];

    __shared__ float row_max_shared;
    __shared__ float row_sum_shared;
    __shared__ float previous_scale_shared;
    __shared__ float current_scale_shared;

    query_shared[dimension] = query[query_offset + dimension];

    if(dimension == 0){
        row_max_shared = -INFINITY;
        row_sum_shared = 0.0f;
    }
    float output_accumulator = 0.0f;
    __syncthreads();

    for(int64_t key_position = 0; key_position < sequence_length; key_position++){
        const int64_t key_row = batch_head * sequence_length + key_position;

        const int64_t key_offset = key_row * HEAD_DIM;

        const float partial_dot_product = query_shared[dimension] * key[key_offset + dimension];
        const float dot_product = block_reduce_sum<NUM_WARPS>(partial_dot_product, warp_sums);

        if(dimension == 0){
            const float score = dot_product * scale;
            const float new_row_max = fmaxf(row_max_shared, score);

            previous_scale_shared = expf(row_max_shared - new_row_max);
            current_scale_shared = expf(score - new_row_max);

            row_sum_shared = previous_scale_shared * row_sum_shared + current_scale_shared;
            row_max_shared = new_row_max;
        }
        __syncthreads();

        output_accumulator = previous_scale_shared * output_accumulator + current_scale_shared * value[key_offset + dimension];
        __syncthreads();
    }
    output[query_offset + dimension] = output_accumulator / row_sum_shared;
}
template <int HEAD_DIM>
__global__ void naive_online_attention_kernel(
    const float* query,
    const float* key, 
    const float* value,
    float* output,
    int64_t num_query_rows,
    int64_t sequence_length,
    float scale
) {
    const int64_t query_row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if(query_row >= num_query_rows){
        return;
    }

    const int64_t batch_head = query_row / sequence_length;
    const int64_t query_offset = query_row * HEAD_DIM;

    float row_max = -INFINITY;
    float row_sum = 0.0f;

    float output_accumulator[HEAD_DIM];

    #pragma unroll
    for(int dimension = 0; dimension < HEAD_DIM; ++dimension){
        output_accumulator[dimension] = 0.0f;
    }

    for(int64_t key_position = 0; key_position < sequence_length; key_position++){
        const int64_t key_row = batch_head * sequence_length + key_position;
        const int64_t key_offset = key_row * HEAD_DIM;
        float score = 0.0f;

        #pragma unroll
        for(int dimension = 0; dimension < HEAD_DIM; dimension++){
            score += query[query_offset + dimension] * key[key_offset + dimension];
        }
        score *= scale;

        const float new_row_max = fmaxf(row_max, score);
        const float previous_scale = expf(row_max - new_row_max);
        const float current_scale = expf(score - new_row_max);

        row_sum = previous_scale * row_sum + current_scale;

        #pragma unroll
        for(int dimension = 0; dimension < HEAD_DIM; dimension++){
            output_accumulator[dimension] = previous_scale * output_accumulator[dimension] + current_scale * value[key_offset + dimension];
        }
        row_max = new_row_max;
    }

    const float inverse_row_sum = 1.0f / row_sum;

    #pragma unroll
    for(int dimension = 0; dimension < HEAD_DIM; dimension++){
        output[query_offset + dimension] = output_accumulator[dimension] * inverse_row_sum;
    }

    
}

__global__ void naive_attention_scores_kernel(
    const float* query,
    const float* key,
    float* scores,
    int64_t num_score_elements,
    int64_t sequence_length,
    int64_t head_dimension,
    float scale
){
    const int64_t score_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(score_index >= num_score_elements){
        return;
    } 

    const int64_t key_position = score_index % sequence_length;
    const int64_t query_row = score_index / sequence_length;
    const int64_t batch_head = query_row / sequence_length;
    const int64_t query_offset = query_row * head_dimension;
    const int64_t key_row = batch_head * sequence_length + key_position;
    const int64_t key_offset = key_row * head_dimension;
    
    float dot_product = 0.0f;

    for(int64_t dimension = 0; dimension < head_dimension; dimension++){
        dot_product += query[query_offset + dimension] * key[key_offset + dimension];
    }
    scores[score_index] = dot_product * scale;
}

void validate_naive_attention_inputs(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value
) {
    TORCH_CHECK(query.is_cuda(), "query must be a CUDA tensor");
    TORCH_CHECK(key.is_cuda(), "key must be a CUDA tensor");
    TORCH_CHECK(value.is_cuda(), "value must be a CUDA tensor");

    TORCH_CHECK(query.is_contiguous(), "query must be contiguous");
    TORCH_CHECK(key.is_contiguous(), "key must be contiguous");
    TORCH_CHECK(value.is_contiguous(), "value must be contiguous");

    TORCH_CHECK(query.dim() == 4, "query must be rank 4");
    TORCH_CHECK(key.dim() == 4, "key must be rank 4");
    TORCH_CHECK(value.dim() == 4, "value must be rank 4");

    TORCH_CHECK(
        query.scalar_type() == torch::kFloat32,
        "query must use float32"
    );
    TORCH_CHECK(
        key.scalar_type() == torch::kFloat32,
        "key must use float32"
    );
    TORCH_CHECK(
        value.scalar_type() == torch::kFloat32,
        "value must use float32"
    );

    TORCH_CHECK(
        query.device() == key.device() &&
        query.device() == value.device(),
        "query, key, and value must be on the same CUDA device"
    );

    TORCH_CHECK(
        query.sizes() == key.sizes() &&
        query.sizes() == value.sizes(),
        "the initial naive kernel requires query, key, and value "
        "to have identical shapes"
    );
}

}  // namespace

torch::Tensor naive_attention_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value
) {
    validate_naive_attention_inputs(query, key, value);

    const c10::cuda::CUDAGuard device_guard(query.device());

    torch::Tensor output = torch::empty_like(query);

    const int64_t batch_size = query.size(0);
    const int64_t num_heads = query.size(1);
    const int64_t query_length = query.size(2);
    const int64_t head_dimension = query.size(3);

    const int64_t num_query_rows =
        batch_size * num_heads * query_length;

    if (num_query_rows == 0 || head_dimension == 0) {
        return output;
    }

    const float scale =
    1.0f / std::sqrt(static_cast<float>(head_dimension));

    constexpr int threads_per_block = 128;

    const int blocks = static_cast<int>(
        (num_query_rows + threads_per_block - 1) /
        threads_per_block
    );

    switch (head_dimension) {
        case 64:
            naive_online_attention_kernel<64><<<
                blocks,
                threads_per_block,
                0,
                at::cuda::getCurrentCUDAStream()
            >>>(
                query.data_ptr<float>(),
                key.data_ptr<float>(),
                value.data_ptr<float>(),
                output.data_ptr<float>(),
                num_query_rows,
                query_length,
                scale
            );
            break;

        case 128:
            naive_online_attention_kernel<128><<<
                blocks,
                threads_per_block,
                0,
                at::cuda::getCurrentCUDAStream()
            >>>(
                query.data_ptr<float>(),
                key.data_ptr<float>(),
                value.data_ptr<float>(),
                output.data_ptr<float>(),
                num_query_rows,
                query_length,
                scale
            );
            break;

        default:
            TORCH_CHECK(
                false,
                "naive_attention_forward currently supports "
                "head dimensions 64 and 128"
            );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}

torch::Tensor naive_attention_scores(
    torch::Tensor query,
    torch::Tensor key
) {
    TORCH_CHECK(query.is_cuda(), "query must be a CUDA tensor");
    TORCH_CHECK(key.is_cuda(), "key must be a CUDA tensor");

    TORCH_CHECK(query.is_contiguous(), "query must be contiguous");
    TORCH_CHECK(key.is_contiguous(), "key must be contiguous");

    TORCH_CHECK(query.dim() == 4, "query must be rank 4");
    TORCH_CHECK(key.dim() == 4, "key must be rank 4");

    TORCH_CHECK(
        query.scalar_type() == torch::kFloat32,
        "query must use float32"
    );
    TORCH_CHECK(
        key.scalar_type() == torch::kFloat32,
        "key must use float32"
    );

    TORCH_CHECK(
        query.device() == key.device(),
        "query and key must be on the same CUDA device"
    );

    TORCH_CHECK(
        query.sizes() == key.sizes(),
        "the initial score kernel requires query and key "
        "to have identical shapes"
    );

    const c10::cuda::CUDAGuard device_guard(query.device());

    const int64_t batch_size = query.size(0);
    const int64_t num_heads = query.size(1);
    const int64_t sequence_length = query.size(2);
    const int64_t head_dimension = query.size(3);

    torch::Tensor scores = torch::empty(
        {
            batch_size,
            num_heads,
            sequence_length,
            sequence_length,
        },
        query.options()
    );

    const int64_t num_score_elements =
        batch_size *
        num_heads *
        sequence_length *
        sequence_length;

    if (num_score_elements == 0) {
        return scores;
    }

    const float scale =
        1.0f / std::sqrt(static_cast<float>(head_dimension));

    constexpr int threads_per_block = 256;

    const int blocks = static_cast<int>(
        (num_score_elements + threads_per_block - 1) /
        threads_per_block
    );

    naive_attention_scores_kernel<<<
        blocks,
        threads_per_block,
        0,
        at::cuda::getCurrentCUDAStream()
    >>>(
        query.data_ptr<float>(),
        key.data_ptr<float>(),
        scores.data_ptr<float>(),
        num_score_elements,
        sequence_length,
        head_dimension,
        scale
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return scores;
}

torch::Tensor block_attention_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value
) {
    validate_naive_attention_inputs(
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
    const int64_t head_dimension = query.size(3);

    const int64_t num_query_rows =
        batch_size *
        num_heads *
        sequence_length;

    if (num_query_rows == 0) {
        return output;
    }

    const float scale =
        1.0f /
        std::sqrt(
            static_cast<float>(head_dimension)
        );

    switch (head_dimension) {
        case 64:
            block_online_attention_kernel<64><<<
                static_cast<int>(num_query_rows),
                64,
                0,
                at::cuda::getCurrentCUDAStream()
            >>>(
                query.data_ptr<float>(),
                key.data_ptr<float>(),
                value.data_ptr<float>(),
                output.data_ptr<float>(),
                num_query_rows,
                sequence_length,
                scale
            );
            break;

        case 128:
            block_online_attention_kernel<128><<<
                static_cast<int>(num_query_rows),
                128,
                0,
                at::cuda::getCurrentCUDAStream()
            >>>(
                query.data_ptr<float>(),
                key.data_ptr<float>(),
                value.data_ptr<float>(),
                output.data_ptr<float>(),
                num_query_rows,
                sequence_length,
                scale
            );
            break;

        default:
            TORCH_CHECK(
                false,
                "block_attention_forward currently supports "
                "head dimensions 64 and 128"
            );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}