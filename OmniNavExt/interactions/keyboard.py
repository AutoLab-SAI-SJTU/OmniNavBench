import importlib
import time

import numpy as np
import omni

from OmniNav.core.util.interaction import BaseInteraction


def _get_carb_module():
    """Delay carb import until SimulationApp is initialized."""
    module = globals().get('_carb_module')
    if module is None:
        module = importlib.import_module('carb')
        globals()['_carb_module'] = module
    return module


class KeyboardController:
    _RELEASE_GRACE_SEC = 0.2

    def __init__(self):
        self.command = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self._pressed_keys = set()
        self._release_deadlines = {}
        self._key_input = None
        self._app_window = None
        self._keyboard = None
        self._sub_id = None
        self.subscribe()

    def subscribe(self):
        """
        subscribe to keyboard events
        """
        # subscribe to keyboard events
        carb = _get_carb_module()
        self._app_window = omni.appwindow.get_default_app_window()  # noqa
        if self._app_window is None:
            raise RuntimeError("[KeyboardController] No app window available")
        self._keyboard = self._app_window.get_keyboard()
        if self._keyboard is None:
            raise RuntimeError("[KeyboardController] No keyboard device available")
        self._key_input = carb.input.acquire_input_interface()  # noqa
        self._sub_id = self._key_input.subscribe_to_keyboard_events(self._keyboard, self._sub_keyboard_event)

    def _tracked_keys(self, carb):
        return (
            carb.input.KeyboardInput.I,
            carb.input.KeyboardInput.K,
            carb.input.KeyboardInput.J,
            carb.input.KeyboardInput.L,
            carb.input.KeyboardInput.U,
            carb.input.KeyboardInput.O,
        )

    def _canonicalize_key(self, carb, raw_input):
        if raw_input in self._tracked_keys(carb):
            return raw_input
        key_name = getattr(raw_input, "name", None)
        if isinstance(key_name, str):
            key_name = key_name.upper()
            for key in self._tracked_keys(carb):
                if getattr(key, "name", None) == key_name:
                    return key
        return None

    def _update_command(self):
        carb = _get_carb_module()
        now = time.monotonic()
        expired_keys = [
            key
            for key, deadline in self._release_deadlines.items()
            if now >= deadline
        ]
        for key in expired_keys:
            self._release_deadlines.pop(key, None)
            self._pressed_keys.discard(key)

        self.command = np.array(
            [
                1.0 if carb.input.KeyboardInput.I in self._pressed_keys else 0.0,
                1.0 if carb.input.KeyboardInput.K in self._pressed_keys else 0.0,
                1.0 if carb.input.KeyboardInput.J in self._pressed_keys else 0.0,
                1.0 if carb.input.KeyboardInput.L in self._pressed_keys else 0.0,
                1.0 if carb.input.KeyboardInput.U in self._pressed_keys else 0.0,
                1.0 if carb.input.KeyboardInput.O in self._pressed_keys else 0.0,
            ],
            dtype=float,
        )

    def _sub_keyboard_event(self, event, *args, **kwargs):
        """subscribe to keyboard events, map to str"""
        carb = _get_carb_module()
        key = self._canonicalize_key(carb, event.input)
        if key is not None and event.type in (
            carb.input.KeyboardEventType.KEY_PRESS,
            carb.input.KeyboardEventType.KEY_REPEAT,
            carb.input.KeyboardEventType.CHAR,
        ):
            self._pressed_keys.add(key)
            self._release_deadlines.pop(key, None)
        if key is not None and event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._release_deadlines[key] = time.monotonic() + self._RELEASE_GRACE_SEC

        self._update_command()
        return False


@BaseInteraction.register('Keyboard')
class KeyboardInteraction(BaseInteraction):
    """Get keyboard input event(i, k, j, l, u, o)"""

    def __init__(self):
        super().__init__()
        self._type = 'Keyboard'
        self.controller = KeyboardController()

    def get_input(self) -> np.ndarray:
        """
        Read input of Keyboard.
        Returns:
            np.ndarray, len == 6, representing (i, k, j, l, u, o) key pressed or not.
        """
        self.controller._update_command()
        return self.controller.command
