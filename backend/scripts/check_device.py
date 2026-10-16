import platform
import sys

from crop_ai.device import get_device_info


def _memory() -> str:
    try:
        import psutil

        return f"{psutil.virtual_memory().available / 1024**3:.1f} GiB available"
    except ImportError:
        return "unavailable (install psutil for this optional check)"


if __name__ == "__main__":
    info = get_device_info()
    print(f"Python: {sys.version.split()[0]} ({platform.platform()})")
    try:
        import torch

        print(f"PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch: not installed")
    print(f"Selected device: {info.selected}")
    print(f"CUDA available: {info.cuda_available}")
    print(f"MPS available: {info.mps_available}")
    print(f"PyTorch available: {info.torch_available}")
    print(f"System memory: {_memory()}")
