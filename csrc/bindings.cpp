#include <torch/extension.h>

torch::Tensor copy_cuda(torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module){
    module.def(
        "copy_cuda",
        &copy_cuda,
        "Copy a CUDA tensor"
    );
}