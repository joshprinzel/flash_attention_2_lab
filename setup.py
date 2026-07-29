from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="flash_attention_cuda",
    ext_modules=[
        CUDAExtension(
            name="flash_attention_cuda",
            sources=[
                "csrc/bindings.cpp",
                "csrc/copy_kernel.cu",
            ],
            extra_compile_args={
                "cxx": [
                    "-O3",
                    "-std=c++17",
                ],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "--use_fast_math",
                    "-lineinfo",
                ],
            },
        )
    ],
    cmdclass={
        "build_ext": BuildExtension.with_options(
            use_ninja=True,
        ),
    },
)