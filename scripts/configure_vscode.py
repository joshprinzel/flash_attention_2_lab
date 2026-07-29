from __future__ import annotations

import json
import shutil
import sys
import sysconfig
from pathlib import Path

import torch
from torch.utils.cpp_extension import include_paths


# Assumes this file is:
#   FlashAttention_lab/scripts/configure_vscode.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
VSCODE_DIRECTORY = PROJECT_ROOT / ".vscode"


def normalized_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def unique_paths(paths: list[str | Path]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    for path in paths:
        normalized = normalized_path(path)

        if normalized not in seen:
            seen.add(normalized)
            result.append(normalized)

    return result


def pytorch_cxx11_abi() -> int:
    """
    Return the libstdc++ ABI mode used to compile the installed PyTorch.
    """
    if hasattr(torch, "compiled_with_cxx11_abi"):
        return int(torch.compiled_with_cxx11_abi())

    # Fallback for older PyTorch releases.
    return int(torch._C._GLIBCXX_USE_CXX11_ABI)


def main() -> int:
    compiler = shutil.which("g++")
    nvcc = shutil.which("nvcc")

    if compiler is None:
        raise RuntimeError("Could not locate g++ in PATH")

    if nvcc is None:
        raise RuntimeError(
            "Could not locate nvcc in PATH. "
            "Ensure the CUDA toolkit bin directory is on PATH."
        )

    python_include = sysconfig.get_path("include")

    if python_include is None:
        raise RuntimeError("Could not determine the Python include directory")

    torch_include_paths = include_paths(device_type="cuda")

    include_directories = unique_paths(
        [
            PROJECT_ROOT / "csrc",
            python_include,
            *torch_include_paths,
        ]
    )

    c_cpp_properties = {
        "configurations": [
            {
                "name": "WSL",
                # g++ is the host compiler used by nvcc.
                "compilerPath": compiler,
                "compilerArgs": [
                    "-pthread",
                ],
                "includePath": [
                    "${workspaceFolder}/**",
                    *include_directories,
                ],
                "defines": [
                    # Must match CUDAExtension(name=...) in setup.py.
                    "TORCH_EXTENSION_NAME=flash_attention_cuda",
                    f"_GLIBCXX_USE_CXX11_ABI={pytorch_cxx11_abi()}",
                    "__CUDACC__",
                ],
                "cStandard": "c17",
                "cppStandard": "c++17",
                "intelliSenseMode": "linux-gcc-x64",
                "browse": {
                    "path": [
                        "${workspaceFolder}",
                        *include_directories,
                    ],
                    "limitSymbolsToIncludedHeaders": True,
                },
            }
        ],
        "version": 4,
    }

    settings = {
        "C_Cpp.default.compilerPath": compiler,
        "C_Cpp.default.cppStandard": "c++17",
        "C_Cpp.default.intelliSenseMode": "linux-gcc-x64",
        "C_Cpp.intelliSenseEngine": "default",
        "C_Cpp.errorSquiggles": "enabled",
        "python.defaultInterpreterPath": sys.executable,
        "python.analysis.extraPaths": [
            "${workspaceFolder}",
        ],
        "files.associations": {
            "*.cu": "cuda-cpp",
            "*.cuh": "cuda-cpp",
        },
    }

    extensions = {
        "recommendations": [
            "ms-vscode.cpptools",
            "ms-python.python",
            "ms-vscode-remote.remote-wsl",
            "nvidia.nsight-vscode-edition",
        ]
    }

    VSCODE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    files = {
        VSCODE_DIRECTORY / "c_cpp_properties.json": c_cpp_properties,
        VSCODE_DIRECTORY / "settings.json": settings,
        VSCODE_DIRECTORY / "extensions.json": extensions,
    }

    for path, contents in files.items():
        path.write_text(
            json.dumps(contents, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {path.relative_to(PROJECT_ROOT)}")

    print()
    print(f"Host compiler: {compiler}")
    print(f"NVCC:          {nvcc}")
    print(f"Python:        {sys.executable}")
    print(f"PyTorch:       {torch.__version__}")
    print(f"PyTorch CUDA:  {torch.version.cuda}")
    print(f"C++11 ABI:     {pytorch_cxx11_abi()}")
    print("Include directories:")

    for include_directory in include_directories:
        print(f"  {include_directory}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())