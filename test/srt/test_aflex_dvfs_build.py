"""CPU-only tests for optional DVFS compilation, discovery and packaging.

Load implementation files directly to keep tests independent of serving extras.
No native library is loaded and no GPU clocks are queried or modified.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "python/sglang/srt/energy/native/dvfs"
PACKAGE = "sglang.srt.energy.native.dvfs"


def load_module(name, path, monkeypatch, *, package=False):
    spec = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=[str(path.parent)] if package else None
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def modules(monkeypatch, tmp_path):
    monkeypatch.delenv("DVFS_CTRL_LIB", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    for name in (
        "CXX",
        "CPPFLAGS",
        "CXXFLAGS",
        "LDFLAGS",
        "CUDA_HOME",
        "CUDA_PATH",
        "NVML_INCLUDE_DIR",
        "NVML_LIB_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    package = load_module(PACKAGE, NATIVE / "__init__.py", monkeypatch, package=True)
    build = load_module(PACKAGE + ".build", NATIVE / "build.py", monkeypatch)
    wrapper = load_module(
        "_aflex_dvfs_test_wrapper",
        ROOT / "python/sglang/srt/layers/dvfs.py",
        monkeypatch,
    )
    return package, build, wrapper


def test_explicit_library_override(modules, monkeypatch, tmp_path):
    _, _, wrapper = modules
    library = tmp_path / "custom.so"
    library.touch()
    monkeypatch.setenv("DVFS_CTRL_LIB", str(library))
    assert wrapper._find_so() == str(library)


def test_invalid_override_does_not_fall_back(modules, monkeypatch, tmp_path):
    _, _, wrapper = modules
    monkeypatch.setenv("DVFS_CTRL_LIB", str(tmp_path / "missing.so"))
    with pytest.raises(FileNotFoundError, match="DVFS_CTRL_LIB"):
        wrapper._find_so()


def test_override_expands_user(modules, monkeypatch, tmp_path):
    _, _, wrapper = modules
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "custom.so").touch()
    monkeypatch.setenv("DVFS_CTRL_LIB", "~/custom.so")
    assert wrapper._find_so() == str(tmp_path / "custom.so")


def test_user_cache_discovery(modules):
    package, _, wrapper = modules
    library = package.default_library_path()
    library.parent.mkdir(parents=True)
    library.touch()
    assert wrapper._find_so() == str(library)


def test_packaged_library_precedes_cache(modules, monkeypatch, tmp_path):
    package, _, wrapper = modules
    fake_file = tmp_path / "srt/layers/dvfs.py"
    monkeypatch.setattr(wrapper, "__file__", str(fake_file))
    packaged = tmp_path / "srt/energy/native/dvfs/libdvfs_ctrl.so"
    packaged.parent.mkdir(parents=True)
    packaged.touch()
    cache = package.default_library_path()
    cache.parent.mkdir(parents=True)
    cache.touch()
    assert wrapper._find_so() == str(packaged)


def test_system_discovery_ignores_benchmark(modules, monkeypatch, tmp_path):
    _, _, wrapper = modules
    monkeypatch.chdir(tmp_path)
    old = tmp_path / "benchmark/test_motivation/hucc/dvfs/libdvfs_ctrl.so"
    old.parent.mkdir(parents=True)
    old.touch()
    monkeypatch.setattr(
        wrapper.ctypes.util, "find_library", lambda name: "system-dvfs.so"
    )
    assert wrapper._find_so() == "system-dvfs.so"
    monkeypatch.setattr(wrapper.ctypes.util, "find_library", lambda name: None)
    assert wrapper._find_so() == "libdvfs_ctrl.so"


def test_load_error_is_actionable(modules, monkeypatch):
    _, _, wrapper = modules
    monkeypatch.setattr(wrapper, "_find_so", lambda: "missing.so")
    monkeypatch.setattr(
        wrapper.ctypes, "CDLL", Mock(side_effect=OSError("missing dependency"))
    )
    with pytest.raises(OSError, match="DVFS_CTRL_LIB"):
        wrapper._load_lib()
    assert wrapper._lib is None


def test_build_explicit_paths_and_flags(modules, monkeypatch, tmp_path):
    _, build, _ = modules
    run = Mock()
    monkeypatch.setattr(build.subprocess, "run", run)
    monkeypatch.setenv("CXXFLAGS", "-g -Wextra")
    monkeypatch.setenv("CPPFLAGS", "-DTEST_BUILD=1")
    monkeypatch.setenv("LDFLAGS", "-Wl,--as-needed")
    output = tmp_path / "output with spaces/custom.so"
    result = build.build_library(
        output,
        cxx="ccache c++",
        nvml_include=tmp_path / "headers",
        nvml_lib=tmp_path / "libraries",
    )
    command = run.call_args.args[0]
    assert command[:2] == ["ccache", "c++"]
    assert command[command.index("-o") + 1] == str(output)
    assert command[command.index("-I") + 1] == str(tmp_path / "headers")
    assert command[command.index("-L") + 1] == str(tmp_path / "libraries")
    assert command[-1] == "-lnvidia-ml"
    assert "-DTEST_BUILD=1" in command
    assert "-Wl,--as-needed" in command
    assert str(NATIVE / "dvfs_ctrl.cpp") in command
    assert result == output
    assert output.parent.is_dir()
    assert run.call_args.kwargs == {"check": True}


def test_build_cuda_and_env_defaults(modules, monkeypatch, tmp_path):
    package, build, _ = modules
    cuda = tmp_path / "cuda"
    (cuda / "lib64").mkdir(parents=True)
    (cuda / "lib64/libnvidia-ml.so").touch()
    monkeypatch.setenv("CUDA_HOME", str(cuda))
    run = Mock()
    monkeypatch.setattr(build.subprocess, "run", run)
    assert build.build_library() == package.default_library_path()
    command = run.call_args.args[0]
    assert command[command.index("-I") + 1] == str(cuda / "include")
    assert command[command.index("-L") + 1] == str(cuda / "lib64")
    assert not any("rpath" in flag for flag in command)


def test_build_invalid_output_and_failure(modules, monkeypatch, tmp_path, capsys):
    _, build, _ = modules
    with pytest.raises(ValueError, match="ending in .so"):
        build.build_library(tmp_path / "invalid")
    monkeypatch.setattr(
        build.subprocess, "run", Mock(side_effect=FileNotFoundError("c++"))
    )
    with pytest.raises(SystemExit) as exc:
        build.main(["--output", str(tmp_path / "missing.so")])
    assert exc.value.code == 1
    assert "NVML development headers" in capsys.readouterr().err


def test_cli_help_without_serving_dependencies(tmp_path):
    # Namespace stubs bypass unrelated top-level serving dependency imports.
    bootstrap = "import runpy, sys, types; "
    for name, relative in (
        ("sglang", "python/sglang"),
        ("sglang.srt", "python/sglang/srt"),
        ("sglang.srt.energy", "python/sglang/srt/energy"),
        ("sglang.srt.energy.native", "python/sglang/srt/energy/native"),
    ):
        bootstrap += f"m=types.ModuleType({name!r}); m.__path__=[{str(ROOT / relative)!r}]; sys.modules[{name!r}]=m; "
    bootstrap += f"sys.argv=['build', '--help']; runpy.run_module({(PACKAGE + '.build')!r}, run_name='__main__')"
    result = subprocess.run(
        [sys.executable, "-c", bootstrap], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "--output" in result.stdout
    assert "--nvml-include" in result.stdout


def test_packaging_has_sources_and_optional_dependencies():
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    config = tomllib.loads((ROOT / "python/pyproject.toml").read_text())
    extras = config["project"]["optional-dependencies"]["aflex"]
    assert {"pandas", "scikit-learn", "lightgbm"} <= set(extras)
    assert "lightgbm" not in config["project"]["dependencies"]
    data = config["tool"]["setuptools"]["package-data"][PACKAGE]
    assert set(data) == {"dvfs_ctrl.cpp", "Makefile"}
    exclusions = config["tool"]["setuptools"]["exclude-package-data"]
    assert "*" not in exclusions
    assert all(
        name == "sglang" or name.startswith("sglang.srt.energy") for name in exclusions
    )
    assert all(pattern.startswith("srt/energy/") for pattern in exclusions["sglang"])
    excludes = exclusions[PACKAGE]
    assert "*.so" in excludes and "*.pkl" in excludes
    assert not list(NATIVE.glob("*.so"))
    assert not list(NATIVE.glob("*.pkl"))
