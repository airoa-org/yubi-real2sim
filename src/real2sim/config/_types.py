# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Schema value validators and the public configuration error type."""

from __future__ import annotations

from difflib import get_close_matches
from typing import Any, Mapping, Sequence

class ConfigError(ValueError):
    """Raised when a rollout YAML does not satisfy the strict schema."""

    def __init__(
        self,
        message: str,
        *,
        path: str | None = None,
        allowed: Any = None,
        got: Any = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.path = path
        if isinstance(allowed, Mapping):
            allowed = sorted(str(name) for name in allowed)
        elif isinstance(allowed, (set, frozenset)):
            allowed = sorted(str(name) for name in allowed)
        self.allowed = allowed
        self.got = got
        self.suggestions: tuple[str, ...] = ()


def _unknown_key_error(
    message: str,
    *,
    path: str,
    allowed: Sequence[str] | set[str] | frozenset[str],
    got: Sequence[str],
) -> ConfigError:
    allowed_values = sorted(str(item) for item in allowed)
    error = ConfigError(
        message,
        path=path,
        allowed=allowed_values,
        got=tuple(got),
    )
    suggestions: list[str] = []
    for key in got:
        suggestions.extend(get_close_matches(key, allowed_values, n=3, cutoff=0.6))
    error.suggestions = tuple(dict.fromkeys(suggestions))
    return error

def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ConfigError(f"{path} must be a mapping with string keys", path=path)
    return dict(value)

def _string(value: Any, path: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{path} must be a string", path=path)
    result = value.strip()
    if not result and not allow_empty:
        raise ConfigError(f"{path} must not be empty", path=path)
    if "\0" in result or "\n" in result or "\r" in result:
        raise ConfigError(f"{path} must not contain NUL or newlines", path=path)
    return result

def _optional_string(value: Any, path: str) -> str | None:
    if value is None:
        return None
    return _string(value, path)

def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{path} must be a YAML boolean", path=path)
    return value

def _integer(value: Any, path: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{path} must be an integer", path=path)
    if value < minimum or (maximum is not None and value > maximum):
        upper = f" and <= {maximum}" if maximum is not None else ""
        raise ConfigError(f"{path} must be >= {minimum}{upper}", path=path)
    return value

def _optional_integer(value: Any, path: str, *, minimum: int = 0) -> int | None:
    if value is None:
        return None
    return _integer(value, path, minimum=minimum)

def _number(value: Any, path: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path} must be numeric", path=path)
    result = float(value)
    if not (float("-inf") < result < float("inf")):
        raise ConfigError(f"{path} must be finite", path=path)
    if minimum is not None and result < minimum:
        raise ConfigError(f"{path} must be >= {minimum}", path=path)
    return result

def _optional_number(value: Any, path: str, *, minimum: float | None = None) -> float | None:
    if value is None:
        return None
    return _number(value, path, minimum=minimum)

def _choice(value: Any, path: str, choices: Sequence[str]) -> str:
    result = _string(value, path)
    if result not in choices:
        raise ConfigError(
            f"{path} must be one of: {', '.join(sorted(choices))}", path=path
        )
    return result

def _number_list(
    value: Any,
    path: str,
    *,
    length: int,
    minimum: float | None = None,
) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ConfigError(f"{path} must contain exactly {length} numbers", path=path)
    return [
        _number(item, f"{path}[{index}]", minimum=minimum)
        for index, item in enumerate(value)
    ]

def _rgb_color_transform(value: Any, path: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ConfigError(f"{path} must contain exactly 3 numbers", path=path)
    result = []
    for index, item in enumerate(value):
        parsed = _number(item, f"{path}[{index}]")
        if not 0.0 < parsed <= 4.0:
            raise ConfigError(f"{path} values must be in (0, 4]", path=path)
        result.append(parsed)
    return result

def _string_list(value: Any, path: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list):
        raise ConfigError(f"{path} must be a YAML list", path=path)
    result = [_string(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if not allow_empty and not result:
        raise ConfigError(f"{path} must not be empty", path=path)
    if len(set(result)) != len(result):
        raise ConfigError(f"{path} must not contain duplicates", path=path)
    return result
