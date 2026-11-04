#!/usr/bin/env python3
"""Fails when an example does not have the shape every example has.

Adding one should be dropping in a directory, not remembering what the last one
happened to include. An example is only useful to someone who never opens the
documentation if it carries its own README, so that counts as part of the shape.

There are two shapes, because the platform takes two kinds of source. A directory
with a `container.yaml` is a container example and is judged against the container
shape; everything else is code. A container example has nothing local to run, so it
declares no tests.

Both shapes carry a `runware.yaml`, which holds the deploy options the README passes
as flags. The two have to agree, or a one-command deploy lands somewhere other than
where the README says it does.

Usage: python3 check_examples.py
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

CODE_FILES = ("model.py", "test_model.py", "README.md", "runware.yaml")
CONTAINER_FILES = ("Dockerfile", "container.yaml", "README.md", "runware.yaml")
CONTAINER_KEYS = ("configVersion", "port", "probes", "endpoints")
MANIFEST_KEYS = ("configVersion", "id", "source", "compute")

# The platform's own AppId rule, which fixes the length at 6 to 30 characters.
# An id outside it cannot be deployed at all, and the README's command fails on
# the first call rather than at the end, so the shape check is the right place
# to catch it. `echo` was that id for a while, four characters against a
# minimum of six.
APP_ID = re.compile(r"^[a-z][a-z0-9-]{4,28}[a-z0-9]$")

# Each deploy flag and the manifest key that carries it. Values are not compared,
# only presence: a flag added to one and not the other is the drift worth catching,
# and reading the values would mean a YAML parser in a check that has no dependencies.
FLAG_KEYS = {
    "--volume": "volumes",
    "--secret": "secrets",
    "--env-var": "envVars",
    "--gpus-per-worker": "gpusPerWorker",
    "--max-workers": "maxWorkers",
    "--min-workers": "minWorkers",
    "--fallback-gpu-type": "fallbackGpuType",
}


def examples() -> list[Path]:
    return sorted(p for p in ROOT.iterdir() if p.is_dir() and not p.name.startswith("."))


def missing(example: Path, files: tuple[str, ...]) -> list[str]:
    return [f"{example.name}: no {name}" for name in files if not (example / name).exists()]


def code_problems(example: Path) -> list[str]:
    found = missing(example, CODE_FILES)

    if (example / "model.py").exists():
        source = (example / "model.py").read_text(encoding="utf-8")
        if not source.lstrip().startswith('"""'):
            found.append(f"{example.name}: model.py opens with no docstring saying what it is")
        if "@serve" not in source:
            found.append(f"{example.name}: model.py registers no class with @serve")

    return found


def container_problems(example: Path) -> list[str]:
    found = missing(example, CONTAINER_FILES)

    config = (example / "container.yaml").read_text(encoding="utf-8")
    for key in CONTAINER_KEYS:
        if not re.search(rf"^{key}:", config, re.MULTILINE):
            found.append(f"{example.name}: container.yaml declares no {key}")

    return found


def declared_key(manifest: str, key: str) -> bool:
    return bool(re.search(rf"^\s*{key}:", manifest, re.MULTILINE))


def manifest_problems(example: Path) -> list[str]:
    manifest = (example / "runware.yaml").read_text(encoding="utf-8")
    found = [
        f"{example.name}: runware.yaml declares no {key}"
        for key in MANIFEST_KEYS
        if not declared_key(manifest, key)
    ]

    # The id is the app the example deploys, not the directory it lives in, so
    # what matters is that the platform would accept it and that the README
    # deploys that same app. `echo` is the example whose id cannot be its own
    # name, since AppId starts at six characters.
    declared_id = re.search(r"^id:\s*(\S+)$", manifest, re.MULTILINE)
    app_id = declared_id.group(1) if declared_id else ""
    if declared_id and not APP_ID.match(app_id):
        found.append(
            f"{example.name}: runware.yaml has an id the platform would refuse, "
            f"{app_id!r} (6 to 30 characters, lowercase letters, digits and hyphens, "
            "starting with a letter and ending with a letter or digit)"
        )

    if not (example / "README.md").exists():
        return found

    # The command is written over several lines, joined with a trailing backslash.
    readme = re.sub(r"\\\n\s*", " ", (example / "README.md").read_text(encoding="utf-8"))
    command = re.search(r"runware serverless deploy[^\n`]*", readme)
    if command is None:
        return found

    if app_id and f"--id {app_id}" not in command.group(0):
        found.append(f"{example.name}: runware.yaml names the app {app_id} and README.md deploys another")

    for flag, key in FLAG_KEYS.items():
        passed, declared = flag in command.group(0), declared_key(manifest, key)
        # A secret can be attached at create or afterwards, and an example whose
        # subject is secrets shows the longer way on purpose.
        if key == "secrets" and "secrets attach" in readme:
            passed = True
        if passed and not declared:
            found.append(f"{example.name}: README.md passes {flag} and runware.yaml has no {key}")
        if declared and not passed:
            found.append(f"{example.name}: runware.yaml declares {key} and README.md passes no {flag}")

    return found


def problems_in(example: Path) -> list[str]:
    container = (example / "container.yaml").exists()
    found = container_problems(example) if container else code_problems(example)

    if (example / "runware.yaml").exists():
        found += manifest_problems(example)

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
