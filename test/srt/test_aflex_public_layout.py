"""CPU-only public-layout contracts; never import the serving runtime."""

import ast
import fnmatch
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

try:
    import tomllib
except ImportError:
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / "python"
ENERGY = PYTHON / "sglang/srt/energy"
NATIVE = ENERGY / "native/dvfs"
PREDICTION = ENERGY / "prediction"
HARNESS = ROOT / "benchmark/test_aflex"


def _metadata():
    return tomllib.loads((PYTHON / "pyproject.toml").read_text(encoding="utf-8"))


@pytest.mark.parametrize("filename", ["dvfs_ctrl.cpp", "Makefile", "build.py"])
def test_native_build_inputs_are_declared_package_data(filename):
    source = NATIVE / filename
    assert source.is_file(), f"Missing rebuildable native input: {source}"
    setuptools = _metadata()["tool"]["setuptools"]
    included, excluded = [], []
    for section, matches in (
        ("package-data", included),
        ("exclude-package-data", excluded),
    ):
        for package, patterns in setuptools.get(section, {}).items():
            base = PYTHON if package == "*" else PYTHON.joinpath(*package.split("."))
            if source.is_relative_to(base):
                relative = source.relative_to(base).as_posix()
                matches.extend(p for p in patterns if fnmatch.fnmatchcase(relative, p))
    assert included, f"No package-data declaration covers {source.relative_to(PYTHON)}"
    assert not excluded, f"Native input excluded from distribution: {excluded}"


def _dependency_names(requirements):
    # Read metadata without importing optional dependencies.
    return {
        re.sub(
            r"[-_.]+", "-", re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", req).group(0)
        ).lower()
        for req in requirements
    }


def test_aflex_optional_dependencies():
    project = _metadata()["project"]
    extras = _dependency_names(project["optional-dependencies"]["aflex"])
    assert {
        "numpy",
        "scipy",
        "nvidia-ml-py",
        "pandas",
        "scikit-learn",
        "lightgbm",
    } <= extras
    assert not {"pandas", "scikit-learn", "lightgbm"} & _dependency_names(
        project["dependencies"]
    )


@pytest.mark.parametrize(
    "relative, ignored",
    [
        ("benchmark/test_aflex/configs/sample.jsonl", False),
        ("benchmark/test_aflex/examples/sample.jsonl", False),
        ("benchmark/test_aflex/workloads/synthetic/sample.jsonl", False),
        ("benchmark/test_aflex/results/run.json", True),
        ("benchmark/test_aflex/logs/run.txt", True),
        ("benchmark/test_aflex/models/model.bin", True),
        ("benchmark/test_aflex/profiles/profile.json", True),
        ("benchmark/test_aflex/workloads/generated/trace.jsonl", True),
        ("benchmark/test_aflex/workloads/real/trace.jsonl", True),
        ("benchmark/test_aflex/assets/real/trace.json", True),
        ("benchmark/test_aflex/assets/models/model.bin", True),
        ("benchmark/test_aflex/assets/profiles/profile.json", True),
        ("energy/profiling/results/run.json", True),
        ("energy/profiling/profiles/profile.json", True),
        ("energy/prediction/models/model.bin", True),
        ("python/sglang/srt/energy/prediction/models/model.bin", True),
        ("python/sglang/srt/energy/prediction/model.pkl", True),
        ("python/sglang/srt/energy/prediction/model.joblib", True),
        ("python/sglang/srt/energy/native/dvfs/libdvfs_ctrl.so", True),
        ("python/sglang/srt/energy/native/dvfs/dvfs_ctrl.cpp", False),
        ("python/sglang/srt/energy/native/dvfs/build.py", False),
    ],
)
def test_public_fixtures_and_generated_asset_ignore_rules(relative, ignored):
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("Ignore contracts require a Git checkout and git executable")
    result = subprocess.run(
        [
            "git",
            "-c",
            "core.excludesFile=/dev/null",
            "check-ignore",
            "--no-index",
            "--",
            relative,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode in (0, 1), result.stderr
    assert (
        result.returncode == 0
    ) == ignored, (
        f"{relative}: expected ignored={ignored}; git output={result.stdout!r}"
    )


def test_migrated_runtime_has_no_ignored_benchmark_dependencies():
    if not PREDICTION.is_dir():
        pytest.skip("Prediction package migration is not present yet")
    obsolete = {
        "test_motivation",
        "AFlex_bench",
        "other-gpu-2",
        "hicache",
        "dadf_experiments",
        "megascale-pdaf-two-node-addon",
    }
    violations = []
    for source in sorted(ENERGY.rglob("*.py")):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        docstrings = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(
                node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            )
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        }
        for node in ast.walk(tree):
            values = []
            if isinstance(node, ast.Import):
                values = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                values = [node.module or ""]
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if id(node) not in docstrings:
                    values = [node.value]
            for value in values:
                components = set(re.split(r"[/\\.\s]+", value))
                if components & obsolete:
                    violations.append(
                        f"{source.relative_to(ROOT)}:{node.lineno}: {value!r}"
                    )
    assert not violations, "Runtime references ignored experiment trees:\n" + "\n".join(
        violations
    )


def test_root_readme_relative_file_links_resolve():
    if not (HARNESS.is_dir() and PREDICTION.is_dir()):
        pytest.skip(
            "Public harness and prediction directories are still being migrated"
        )
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    # Ignore fenced examples; inspect inline/image links and reference definitions.
    readme = re.sub(r"```.*?```|~~~.*?~~~", "", readme, flags=re.DOTALL)
    targets = re.findall(r"!?\[[^\]\n]*\]\(\s*(<[^>]+>|[^\s)]+)", readme)
    targets += re.findall(
        r"^\s{0,3}\[[^\]\n]+\]:\s*(<[^>]+>|\S+)", readme, re.MULTILINE
    )
    relative_links = []
    for target in targets:
        parsed = urlsplit(target.strip("<>"))
        if (
            parsed.scheme
            or parsed.netloc
            or not parsed.path
            or parsed.path.startswith("/")
        ):
            continue
        relative_links.append(unquote(parsed.path))
    assert (
        relative_links
    ), "Public README should link to local reproduction instructions"
    missing = [link for link in relative_links if not (ROOT / link).exists()]
    assert not missing, f"Broken README relative links: {missing}"
