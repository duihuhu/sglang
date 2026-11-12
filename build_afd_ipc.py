#!/usr/bin/env python3
"""Build the afd_ipc_cpp extension (including FusedPipeline)."""

import os
import sys
import torch
from torch.utils.cpp_extension import load

csrc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "sgl-kernel", "csrc", "afd_ipc")

print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
print(f"Source dir: {csrc_dir}")
print("Compiling afd_ipc_cpp with FusedPipeline...")

mod = load(
    name="afd_ipc_cpp",
    sources=[
        os.path.join(csrc_dir, "afd_ipc.cpp"),
        os.path.join(csrc_dir, "afd_ipc_kernels.cu"),
        os.path.join(csrc_dir, "afd_ipc_pybind.cpp"),
        os.path.join(csrc_dir, "afd_pipeline_driver.cpp"),
        os.path.join(csrc_dir, "afd_fused_pipeline.cpp"),
    ],
    extra_include_paths=[csrc_dir],
    extra_cflags=["-O3", "-std=c++17"],
    extra_cuda_cflags=[
        "-O3", "--expt-relaxed-constexpr",
        "-gencode=arch=compute_80,code=sm_80",
    ],
    extra_ldflags=["-lpthread", "-lrt"],
    verbose=True,
)
print("\n=== Build OK! ===")
print(f"FusedPipeline available: {hasattr(mod, 'FusedPipeline')}")
print(f"Module file: {mod.__file__}")
