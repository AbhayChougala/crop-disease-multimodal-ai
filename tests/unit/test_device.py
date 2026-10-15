from crop_ai.device import get_device_info


def test_device_falls_back_to_a_supported_name() -> None:
    info = get_device_info()
    assert info.selected in {"cuda", "mps", "cpu"}
    if not info.torch_available:
        assert info.selected == "cpu"
        assert info.cuda_available is False
        assert info.mps_available is False
