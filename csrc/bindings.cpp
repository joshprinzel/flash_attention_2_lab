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
}