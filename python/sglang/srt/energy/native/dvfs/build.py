"""Build the optional NVML wrapper without loading NVML or touching GPUs.

Example::

    python -m sglang.srt.energy.native.dvfs.build --output /tmp/libdvfs_ctrl.so

Requires a C++ compiler, nvml.h and the NVIDIA management library at build time.
The default output is a user cache, so installed packages may be read-only.
"""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from . import LIBRARY_NAME, default_library_path


def build_library(
    output=None, *, cxx=None, cuda_home=None, nvml_include=None, nvml_lib=None
) -> Path:
    """Compile the packaged source and return its explicit output path.

    Compiler flags honor CPPFLAGS, CXXFLAGS and LDFLAGS. Explicit NVML include
    and library directories take priority over CUDA_HOME/CUDA_PATH defaults.
    NVML_LIB_DIR may point to CUDA's stub directory for build-only machines;
    stubs must not be added to the runtime library search path.
    """
    target = Path(output or default_library_path()).expanduser().resolve()
    if target.suffix != ".so":
        raise ValueError("--output must name a shared library ending in .so")
    source = Path(__file__).resolve().with_name("dvfs_ctrl.cpp")
    cuda = cuda_home or os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if not cuda and Path("/usr/local/cuda/include/nvml.h").is_file():
        cuda = "/usr/local/cuda"
    include = nvml_include or os.environ.get("NVML_INCLUDE_DIR")
    lib_dir = nvml_lib or os.environ.get("NVML_LIB_DIR")
    if cuda:
        cuda = Path(cuda).expanduser()
        include = include or cuda / "include"
        if not lib_dir:
            for candidate in (cuda / "lib64", cuda / "lib"):
                if any(candidate.glob("libnvidia-ml.so*")):
                    lib_dir = candidate
                    break
    command = shlex.split(cxx or os.environ.get("CXX", "c++"))
    if not command:
        raise ValueError("CXX must specify a compiler")
    command += ["-shared", "-fPIC", "-O2", "-std=c++11", "-Wall"]
    command += shlex.split(os.environ.get("CPPFLAGS", ""))
    command += shlex.split(os.environ.get("CXXFLAGS", ""))
    if include:
        command += ["-I", str(Path(include).expanduser().resolve())]
    command += [str(source), "-o", str(target)]
    if lib_dir:
        command += ["-L", str(Path(lib_dir).expanduser().resolve())]
    command += shlex.split(os.environ.get("LDFLAGS", ""))
    command += ["-lnvidia-ml"]
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)
    return target


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=default_library_path(),
        help=f"Output .so path (default: %(default)s; filename usually {LIBRARY_NAME})",
    )
    parser.add_argument("--cxx", help="C++ compiler command (default: CXX or c++)")
    parser.add_argument(
        "--cuda-home", help="CUDA toolkit root (default: CUDA_HOME/CUDA_PATH)"
    )
    parser.add_argument(
        "--nvml-include", help="Directory containing nvml.h (or NVML_INCLUDE_DIR)"
    )
    parser.add_argument(
        "--nvml-lib", help="Directory containing libnvidia-ml.so (or NVML_LIB_DIR)"
    )
    args = parser.parse_args(argv)
    try:
        output = build_library(
            args.output,
            cxx=args.cxx,
            cuda_home=args.cuda_home,
            nvml_include=args.nvml_include,
            nvml_lib=args.nvml_lib,
        )
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(
            1,
            f"DVFS build failed: {exc}\nInstall a C++ compiler and NVML development headers/library, "
            "or set --nvml-include and --nvml-lib.\n",
        )
    print(f"Built {output}")
    print(f"export DVFS_CTRL_LIB={shlex.quote(str(output))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
