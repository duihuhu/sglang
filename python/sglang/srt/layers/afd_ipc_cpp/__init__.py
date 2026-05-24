"""JIT-compiled afd_ipc_cpp extension loader.

Provides lazy compilation of the C++ IPC library on first import.
Falls back to the pre-built .so if available.
"""

import os
import logging

logger = logging.getLogger(__name__)

_module = None
_CSRC_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "..", "..",
    "sgl-kernel", "csrc", "afd_ipc"
)


def get_module():
    """Load or JIT-compile the afd_ipc_cpp extension."""
    global _module
    if _module is not None:
        return _module

    # Try pre-built module first
    try:
        import afd_ipc_cpp
        _module = afd_ipc_cpp
        logger.info("[afd_ipc] Using pre-built afd_ipc_cpp module")
        return _module
    except ImportError:
        pass

    # JIT compile
    from torch.utils.cpp_extension import load
    logger.info("[afd_ipc] JIT compiling afd_ipc_cpp...")

    _module = load(
        name="afd_ipc_cpp",
        sources=[
            os.path.join(_CSRC_DIR, "afd_ipc.cpp"),
            os.path.join(_CSRC_DIR, "afd_ipc_kernels.cu"),
            os.path.join(_CSRC_DIR, "afd_ipc_pybind.cpp"),
        ],
        extra_include_paths=[_CSRC_DIR],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3", "--expt-relaxed-constexpr",
            "-gencode=arch=compute_80,code=sm_80",
            "-gencode=arch=compute_86,code=sm_86",
            "-gencode=arch=compute_89,code=sm_89",
            "-gencode=arch=compute_90,code=sm_90",
        ],
        extra_ldflags=["-lpthread", "-lrt"],
        verbose=False,
    )
    logger.info("[afd_ipc] JIT compilation complete")
    return _module
