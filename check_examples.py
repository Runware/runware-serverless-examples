#!/usr/bin/env python3
"""Fails when an example does not have the shape every example has.

Adding one should be dropping in a directory, not remembering what the last one
happened to include. An example is only useful to someone who never opens the
documentation if it carries its own README, so that counts as part of the shape.

Usage: python3 check_examples.py
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUIRED_FILES = ("model.py", "test_model.py", "README.md")


def examples() -> list[Path]:
    return sorted(p for p in ROOT.iterdir() if p.is_dir() and not p.name.startswith("."))


def problems_in(example: Path) -> list[str]:
    found = [f"{example.name}: no {name}" for name in REQUIRED_FILES if not (example / name).exists()]

    if (example / "model.py").exists():
        source = (example / "model.py").read_text(encoding="utf-8")
        if not source.lstrip().startswith('"""'):
            found.append(f"{example.name}: model.py opens with no docstring saying what it is")
        if "@serve" not in source:
            found.append(f"{example.name}: model.py registers no class with @serve")

    if (example / "README.md").exists():
        readme = (example / "README.md").read_text(encoding="utf-8")
        if "runware serverless deploy" not in readme:
            found.append(f"{example.name}: README.md shows no deploy command")

    if not (ROOT / "README.md").read_text(encoding="utf-8").count(f"]({example.name}/)"):
        found.append(f"{example.name}: the README table does not link to it")

    return found


def main() -> int:
    found = [problem for example in examples() for problem in problems_in(example)]

    for problem in found:
        print(f"  {problem}")
    if found:
        print(f"\n{len(found)} problem(s). Every example declares the same things.")
        return 1
    print(f"{len(examples())} example(s), each with the shape every example has.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
