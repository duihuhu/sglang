import ast
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="stage-a-cpu-only")


def _builder_call(path: Path, builder_name: str) -> ast.Call:
    tree = ast.parse(path.read_text())
    builder_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == builder_name)
            or (isinstance(node.func, ast.Attribute) and node.func.attr == builder_name)
        )
    ]
    if len(builder_calls) != 1:
        raise AssertionError(
            f"Expected one {builder_name} call in {path}, found {len(builder_calls)}"
        )
    return builder_calls[0]


def _string_keyword_values(
    path: Path, builder_name: str, keyword_name: str
) -> set[str]:
    call = _builder_call(path, builder_name)
    value = next(
        (keyword.value for keyword in call.keywords if keyword.arg == keyword_name), None
    )
    if not isinstance(value, (ast.List, ast.Tuple)) or not all(
        isinstance(element, ast.Constant) and isinstance(element.value, str)
        for element in value.elts
    ):
        raise TypeError(f"Expected a literal {keyword_name} list in {path}")
    return {element.value for element in value.elts}


def _source_basenames(path: Path, builder_name: str) -> set[str]:
    sources = next(
        (
            keyword.value
            for keyword in _builder_call(path, builder_name).keywords
            if keyword.arg == "sources"
        ),
        None,
    )
    if not isinstance(sources, (ast.List, ast.Tuple)):
        raise TypeError(f"Expected a literal sources list in {path}")

    basenames = set()
    for source in sources.elts:
        if not (
            isinstance(source, ast.Call)
            and isinstance(source.func, ast.Attribute)
            and source.func.attr == "join"
            and source.args
            and isinstance(source.args[-1], ast.Constant)
            and isinstance(source.args[-1].value, str)
        ):
            raise AssertionError(
                f"Unsupported source expression in {path}: {ast.dump(source)}"
            )
        basenames.add(Path(source.args[-1].value).name)
    return basenames


class TestAfdIpcBuildSources(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        repo_root = Path(__file__).resolve().parents[4]
        cls.ipc_package = repo_root / "python/sglang/srt/layers/afd_ipc_cpp"
        cls.pybind_source = (
            repo_root / "sgl-kernel/csrc/afd_ipc/afd_ipc_pybind.cpp"
        )

    def test_setup_and_jit_loader_sources_match(self):
        setup_sources = _source_basenames(
            self.ipc_package / "setup.py", "CUDAExtension"
        )
        loader_sources = _source_basenames(
            self.ipc_package / "__init__.py", "load"
        )

        self.assertSetEqual(setup_sources, loader_sources)

    def test_cuda_address_range_uses_driver_api(self):
        source = self.pybind_source.read_text()

        self.assertNotIn("cudaMemGetAddressRange", source)
        self.assertIn("#include <cuda.h>", source)
        self.assertIn("cuInit(0)", source)
        self.assertIn("cuMemGetAddressRange(", source)
        self.assertIn("cuGetErrorString(", source)

    def test_cuda_driver_is_linked(self):
        setup_libraries = _string_keyword_values(
            self.ipc_package / "setup.py", "CUDAExtension", "libraries"
        )
        jit_ldflags = _string_keyword_values(
            self.ipc_package / "__init__.py", "load", "extra_ldflags"
        )

        self.assertIn("cuda", setup_libraries)
        self.assertIn("-lcuda", jit_ldflags)


if __name__ == "__main__":
    unittest.main(verbosity=3)
