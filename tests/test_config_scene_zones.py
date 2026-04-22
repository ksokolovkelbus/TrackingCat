import logging

from app.config import load_config, load_raw_config
from app.models import ZoneType
from app.zone_editor import ZoneEditor, build_normalized_rect_zone


def test_old_pixel_format_is_read_as_pixels(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
scene_zones:
  enabled: true
  zones:
    - name: table
      enabled: true
      zone_type: surface
      shape_type: rect
      x1: 40
      y1: 60
      x2: 140
      y2: 180
""".strip(),
        encoding="utf-8",
    )

    config = load_config(str(config_path))

    assert config.scene_zones.zones[0].coordinates_mode == "pixels"


def test_zone_editor_saves_normalized_zones_to_yaml(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("scene_zones:\n  enabled: true\n  zones: []\n", encoding="utf-8")

    config = load_config(str(config_path))
    editor = ZoneEditor(
        config=config,
        config_path=str(config_path),
        logger=logging.getLogger("test_zone_editor"),
    )
    editor._scene_zones.zones = [
        build_normalized_rect_zone(
            name="table",
            zone_type=ZoneType.SURFACE,
            start_point=(20, 30),
            end_point=(100, 90),
            frame_shape=(100, 200, 3),
        ),
    ]
    editor._save_zones((100, 200, 3))

    raw = load_raw_config(str(config_path))
    zone_payload = raw["scene_zones"]["zones"][0]

    assert raw["scene_zones"]["coordinates_mode"] == "normalized"
    assert raw["scene_zones"]["zone_editor_enabled"] is False
    assert zone_payload["coordinates_mode"] == "normalized"
    assert zone_payload["x1"] == 0.1
    assert zone_payload["y1"] == 0.3
    assert zone_payload["x2"] == 0.5
    assert zone_payload["y2"] == 0.9


def test_zone_editor_reload_keeps_editor_runtime_enabled_but_not_yaml_flag(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
scene_zones:
  enabled: true
  zone_editor_enabled: false
  zones: []
""".strip(),
        encoding="utf-8",
    )

    config = load_config(str(config_path))
    editor = ZoneEditor(
        config=config,
        config_path=str(config_path),
        logger=logging.getLogger("test_zone_editor"),
    )

    editor._reload_zones()

    assert editor._scene_zones.zone_editor_enabled is True
    raw = load_raw_config(str(config_path))
    assert raw["scene_zones"]["zone_editor_enabled"] is False
