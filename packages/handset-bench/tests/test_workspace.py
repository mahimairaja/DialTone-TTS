"""Workspace wiring, and the guard that keeps the dependency graph acyclic."""

import importlib
import pathlib


def test_handset_bench_imports():
    assert importlib.import_module("handset_bench") is not None


def test_dialtone_imports():
    assert importlib.import_module("dialtone") is not None


def test_dialtone_can_import_the_shared_codec():
    """dialtone depends on handset-bench. This direction is allowed and required."""
    codec = importlib.import_module("handset_bench.codec")
    assert hasattr(codec, "phone_line")


def test_handset_bench_never_imports_dialtone():
    """The dependency graph must stay acyclic."""
    root = pathlib.Path(__file__).parent.parent / "src" / "handset_bench"
    offenders = []
    for path in root.rglob("*.py"):
        source = path.read_text()
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import dialtone", "from dialtone")):
                offenders.append(f"{path.name}: {stripped}")
    assert offenders == [], f"handset-bench must never import dialtone: {offenders}"
