#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cmath>
#include <cstdint>


namespace {

__global__ void naive_attention_zero_kernel(
    float* output,
    int64_t num_query_rows,
    int64_t head_dimension
) {
    // One CUDA thread owns one complete query row.
    const int64_t row =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (row >= num_query_rows) {
        return;
    }

    const int64_t output_row_offset = row * head_dimension;

    for (int64_t dimension = 0; dimension < head_dimension; ++dimension) {
        output[output_row_offset + dimension] = 0.0f;
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

    constexpr int threads_per_block = 128;

    const int blocks = static_cast<int>(
        (num_query_rows + threads_per_block - 1) /
        threads_per_block
    );

    naive_attention_zero_kernel<<<
        blocks,
        threads_per_block,
        0,
        at::cuda::getCurrentCUDAStream()
    >>>(
        output.data_ptr<float>(),
        num_query_rows,
        head_dimension
    );

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