"""Build script for afd_ipc_cpp extension using torch.utils.cpp_extension.

Usage:
    python setup.py build_ext --inplace
    # or
    pip install -e .

The compiled .so will be placed in this directory for direct import.
"""

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = os.path.dirname(os.path.abspath(__file__))
CSRC = os.path.join(ROOT, "..", "..", "..", "..", "..", "sgl-kernel", "csrc", "afd_ipc")

setup(
    name="afd_ipc_cpp",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name="afd_ipc_cpp",
            sources=[
                os.path.join(CSRC, "afd_ipc.cpp"),
                os.path.join(CSRC, "afd_ipc_kernels.cu"),
                os.path.join(CSRC, "afd_ipc_pybind.cpp"),
            ],
            include_dirs=[CSRC],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "--expt-relaxed-constexpr",
                         "-gencode=arch=compute_80,code=sm_80",
                         "-gencode=arch=compute_86,code=sm_86",
                         "-gencode=arch=compute_89,code=sm_89",
                         "-gencode=arch=compute_90,code=sm_90"],
            },
            libraries=["cudart", "pthread", "rt"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
