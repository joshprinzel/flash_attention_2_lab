#include <torch/extension.h>

torch::Tensor copy_cuda(torch::Tensor input);

torch::Tensor naive_attention_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value
);

torch::Tensor naive_attention_scores(
    torch::Tensor query,
    torch::Tensor key
);

torch::Tensor block_attention_forward(
    torch::Tensor query,
    torch::Tensor key, 
    torch::Tensor value
);

torch::Tensor tiled_attention_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value
);

torch::Tensor tilewise_attention_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value
);

torch::Tensor wmma_matmul_16x64x16(
    torch::Tensor matrix_a,
    torch::Tensor matrix_b
);

torch::Tensor tensorcore_attention_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value
);

torch::Tensor tensorcore_attention_forward_bc32(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module){
    module.def(
        "copy_cuda",
        &copy_cuda,
        "Copy a CUDA tensor"
    );

    module.def(
        "naive_attention_forward",
        &naive_attention_forward,
        "Naive attention forward pass"
    );

    module.def(
        "naive_attention_scores",
        &naive_attention_scores,
        "Compute naive scaled attention scores"
    );

    module.def(
        "block_attention_forward",
        &block_attention_forward,
        "Block-cooperative online attention"
    );

    module.def(
        "tiled_attention_forward",
        &tiled_attention_forward,
        "Tiled SIMT online attention forward pass"
    );

    module.def(
        "tilewise_attention_forward",
        &tilewise_attention_forward,
        "Tilewise SIMT attention forward pass"
    );

    module.def(
        "wmma_matmul_16x64x16",
        &wmma_matmul_16x64x16,
        "WMMA FP16 16x64 by 64x16 matrix multiplication with FP32 accumulation"
    );

    module.def(
        "tensorcore_attention_forward",
        &tensorcore_attention_forward,
        "Fused FP16 Tensor Core attention forward pass"
    );

    module.def(
        "tensorcore_attention_forward_bc32",
        &tensorcore_attention_forward_bc32,
        "Fused FP16 Tensor core attention forward with Bc=32"
    );
}