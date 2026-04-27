from app.models import AppConfig, PanTiltControlConfig
from app.pan_tilt import PanTiltControlOverlay, PanTiltState


def test_pantilt_default_config_values() -> None:
    config = AppConfig()
    assert isinstance(config.pan_tilt, PanTiltControlConfig)
    assert config.pan_tilt.enabled is False
    assert config.pan_tilt.default_step_degrees == 3


def test_pantilt_overlay_buttons_cover_expected_actions() -> None:
    overlay = PanTiltControlOverlay(PanTiltControlConfig(enabled=True))
    state = PanTiltState(step_degrees=3, connected=True)
    buttons = overlay.build_buttons((720, 1280, 3), state)
    actions = {button.action for button in buttons}
    assert {"up", "down", "left", "right", "center", "laser-toggle", "refresh", "stop"}.issubset(actions)
    active_speed = [button for button in buttons if button.action == "speed:medium"]
    assert active_speed and active_speed[0].toggled is True




def test_explicit_button_mapping() -> None:
    from app.pan_tilt import PanTiltController
    import logging

    class TestController(PanTiltController):
        def __init__(self):
            super().__init__(PanTiltControlConfig(enabled=True, button_up_direction="left", button_down_direction="right", button_left_direction="down", button_right_direction="up"), logging.getLogger("test"))
            self.last = None
        def _request_json(self, path, params=None):
            self.last = (path, params)
            return {"pan_angle": 90, "tilt_angle": 90, "step_degrees": 3, "connected": True}

    controller = TestController()
    controller.move("right")
    assert controller.last == ("/api/move", {"pan_delta": 0, "tilt_delta": -3})


def test_speed_mode_buttons_present() -> None:
    overlay = PanTiltControlOverlay(PanTiltControlConfig(enabled=True))
    state = PanTiltState(step_degrees=3, speed_mode="fast", connected=True)
    buttons = overlay.build_buttons((720, 1280, 3), state)
    actions = {button.action for button in buttons}
    assert {"speed:slow", "speed:medium", "speed:fast"}.issubset(actions)
