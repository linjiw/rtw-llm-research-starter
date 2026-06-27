"""Small compatibility helpers for TRL config API drift."""
from __future__ import annotations

from inspect import signature
from typing import Any


def supported_config_kwargs(config_cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return only keyword args accepted by a TRL config class."""
    params = signature(config_cls.__init__).parameters
    return {key: value for key, value in kwargs.items() if key in params}


def set_first_supported_kwarg(
    config_cls: type,
    kwargs: dict[str, Any],
    candidates: list[str],
    value: Any,
) -> None:
    """Set the first supported key from a list of renamed TRL parameters."""
    params = signature(config_cls.__init__).parameters
    for key in candidates:
        if key in params:
            kwargs[key] = value
            return
