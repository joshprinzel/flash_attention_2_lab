#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cmath>
#include <cstdint>


namespace {

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