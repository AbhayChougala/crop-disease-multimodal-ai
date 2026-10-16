from __future__ import annotations

import os
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DeviceInfo:
    selected: str
    cuda_available: bool
    mps_available: bool
    torch_available: bool

    def as_dict(self) -> dict[str, str | bool]:
        return asdict(self)


def get_device_info() -> DeviceInfo:
    """Select CUDA, then Apple MPS, then CPU without requiring torch at startup."""
    try:
        import torch
    except ImportError:
        return DeviceInfo("cpu", False, False, False)

    cuda_available = bool(torch.cuda.is_available())
    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(mps_backend and mps_backend.is_available())
    requested = os.getenv("CROP_AI_DEVICE", "auto").lower()
    if requested not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError("CROP_AI_DEVICE must be auto, cpu, mps, or cuda.")
    automatic = "cuda" if cuda_available else "mps" if mps_available else "cpu"
    if requested == "cuda" and not cuda_available:
        raise RuntimeError("CROP_AI_DEVICE=cuda was requested but CUDA is unavailable.")
    if requested == "mps" and not mps_available:
        raise RuntimeError("CROP_AI_DEVICE=mps was requested but MPS is unavailable.")
    selected = automatic if requested == "auto" else requested
    return DeviceInfo(selected, cuda_available, mps_available, True)


def get_device():
    """Return a torch.device when torch exists, otherwise the string ``cpu``."""
    info = get_device_info()
    try:
        import torch
    except ImportError:
        return "cpu"
    return torch.device(info.selected)
