from __future__ import annotations

import importlib.util
import platform
import shutil
import sys


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Node: {shutil.which('node') or 'not found'}")
    print(f"npm: {shutil.which('npm') or 'not found'}")
    for package in ("fastapi", "cv2", "numpy", "torch", "torchvision", "pytest"):
        status = "installed" if importlib.util.find_spec(package) else "not installed"
        print(f"{package}: {status}")


if __name__ == "__main__":
    main()
