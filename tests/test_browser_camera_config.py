from app.config import load_config


def test_browser_camera_config_loads() -> None:
    config = load_config("configs/browser_camera.yaml")
    assert config.source.source_type == "browser_upload"
    assert config.source.browser_camera_port == 8020
    assert config.source.browser_public_host == "192.168.8.176"
    assert config.source.browser_capture_width == 640
    assert config.source.browser_capture_height == 360
