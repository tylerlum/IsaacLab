# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""General keyboard control."""

import weakref
from dataclasses import dataclass
from typing import Any, Callable, Literal

import carb
import omni


@dataclass
class KeyboardCommand:
    key: carb.input.KeyboardInput  # e.g., carb.input.KeyboardInput.R
    func: Callable
    args: list[Any]
    type: Literal[
        carb.input.KeyboardEventType.KEY_PRESS, carb.input.KeyboardEventType.KEY_RELEASE
    ] = carb.input.KeyboardEventType.KEY_PRESS
    description: str = ""

    def __post_init__(self):
        assert isinstance(self.key, carb.input.KeyboardInput), (
            f"Invalid key: {self.key}, must be in carb.input.KeyboardInput.{[x for x in dir(carb.input.KeyboardInput) if not x.startswith('_') and x.isupper()]}."
        )

        assert self.type in [
            carb.input.KeyboardEventType.KEY_PRESS,
            carb.input.KeyboardEventType.KEY_RELEASE,
        ], f"Invalid command type: {self.type}"


class GeneralKeyboard:
    """A keyboard controller general commands.

    This class is designed to provide a keyboard controller for general commands>
    It uses the Omniverse keyboard interface to listen to keyboard events and maps them to general commands.

    The official documentation for the keyboard interface: `Carb Keyboard Interface <https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html>`__.

    """

    def __init__(self, commands: list[KeyboardCommand]):
        """Initialize the keyboard layer."""
        self._commands = commands

        # acquire omniverse interfaces
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        # note: Use weakref on callbacks to ensure that this object can be deleted when its destructor is called.
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(
                event, *args
            ),
        )

    def __del__(self):
        """Release the keyboard interface."""
        self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
        self._keyboard_sub = None

    def __str__(self) -> str:
        """Returns: A string containing the information of joystick."""
        msg = f"General Keyboard Controller: {self.__class__.__name__}\n"
        msg += f"\tKeyboard name: {self._input.get_keyboard_name(self._keyboard)}\n"
        msg += "\t----------------------------------------------\n"
        for command in self._commands:
            msg += f"\t{command.key} - {command.type} - {command.description}\n"
        return msg

    """
    Internal helpers.
    """

    def _on_keyboard_event(self, event, *args, **kwargs):
        """Subscriber callback to when kit is updated.

        Reference:
            https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html
        """
        DEBUG = False

        # apply the command when pressed
        for command in self._commands:
            if DEBUG:
                print(
                    f"command.type: {command.type}, event.type: {event.type}, command.key: {command.key}, event.input: {event.input}"
                )

            if command.type == event.type and command.key == event.input:
                command.func(*command.args)

        # since no error, we are fine :)
        return True
