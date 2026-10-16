"""Print a setup report and exit non-zero if a hard requirement is missing.

Usage (from the backend/ directory, with the virtual environment active):
    python scripts/check_environment.py
"""

from __future__ import annotations

import importlib.util
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIN_PYTHON = (3, 12)
MAX_PYTHON_EXCLUSIVE = (3, 15)


def _node_ok(version: str) -> bool:
    match = re.match(r"v?(\d+)\.(\d+)", version)
    if not match:
        return False
    major, minor = int(match[1]), int(match[2])
    return (major == 20 and minor >= 19) or (major == 22 and minor >= 12) or major > 22


def main() -> int:
    problems: list[str] = []
    print(f"Platform: {platform.platform()}")

    py = sys.version_info[:2]
    py_ok = MIN_PYTHON <= py < MAX_PYTHON_EXCLUSIVE
    print(f"Python {sys.version.split()[0]} ({sys.executable}): {'OK' if py_ok else 'UNSUPPORTED'}")
    if not py_ok:
        problems.append("Python must be >=3.12 and <3.15. Recreate .venv with Python 3.12.")
    if sys.prefix == sys.base_prefix:
        problems.append("No virtual environment is active. Run: source .venv/bin/activate")

    node = shutil.which("node")
    if node:
        version = subprocess.run([node, "--version"], capture_output=True, text=True).stdout
        version = version.strip()
        node_ok = _node_ok(version)
        print(f"Node {version}: {'OK' if node_ok else 'UNSUPPORTED'}")
        if not node_ok:
            problems.append("Node must be ^20.19 or >=22.12 (Vite requirement).")
    else:
        print("Node: not found")
        problems.append("Install Node.js 22 LTS for the frontend.")
    print(f"npm: {shutil.which('npm') or 'not found'}")

    required = ("fastapi", "uvicorn", "cv2", "numpy", "pydantic_settings", "multipart")
    optional = ("openai", "reportlab", "pytest", "torch", "torchvision")
    for package in required:
        found = importlib.util.find_spec(package) is not None
        print(f"{package}: {'installed' if found else 'MISSING'}")
        if not found:
            problems.append(
                f"Missing {package}. Run: python -m pip install -e '.[dev,llm,reports]'"
            )
    for package in optional:
        found = importlib.util.find_spec(package) is not None
        print(f"{package}: {'installed' if found else 'not installed (optional)'}")

    print(f".env: {'present' if (ROOT / '.env').exists() else 'absent (defaults = demo mode)'}")
    has_modules = (ROOT.parent / "frontend" / "node_modules").exists()
    print(f"frontend/node_modules: {'present' if has_modules else 'absent (run npm ci)'}")

    if problems:
        print("\nProblems:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nEnvironment OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
