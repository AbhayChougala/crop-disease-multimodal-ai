from crop_ai.device import get_device_info

if __name__ == "__main__":
    info = get_device_info()
    print(f"Selected device: {info.selected}")
    print(f"CUDA available: {info.cuda_available}")
    print(f"MPS available: {info.mps_available}")
    print(f"PyTorch available: {info.torch_available}")
