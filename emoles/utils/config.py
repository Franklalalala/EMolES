"""Minimal runtime config container used by EMolES auto initialization."""

from __future__ import annotations

import inspect
from typing import Any, Iterable


_GLOBAL_ALL_ASKED_FOR_KEYS = set()


class Config(dict):
    def __init__(self, config: dict | None = None, allow_list: Iterable[str] | None = None):
        super().__init__(config or {})
        self._allow_list = list(allow_list or self.keys())

    @classmethod
    def from_function(cls, func, remove_kwargs: bool = True) -> "Config":
        signature = inspect.signature(func)
        config = {}
        allow = []
        for name, parameter in signature.parameters.items():
            if name == "self":
                continue
            if parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                if remove_kwargs:
                    continue
            allow.append(name)
            if parameter.default is not inspect.Parameter.empty:
                config[name] = parameter.default
        return cls(config=config, allow_list=allow)

    @classmethod
    def from_class(cls, builder, remove_kwargs: bool = True) -> "Config":
        return cls.from_function(builder.__init__, remove_kwargs=remove_kwargs)

    @staticmethod
    def as_dict(config: "Config") -> dict:
        return dict(config)

    def allow_list(self) -> list[str]:
        return list(self._allow_list)

    def update(self, data: dict | None = None, **kwargs) -> list[str]:  # type: ignore[override]
        source = {}
        if data:
            source.update(data)
        source.update(kwargs)

        matched = []
        for key in self._allow_list:
            if key in source:
                self[key] = source[key]
                matched.append(key)
        return matched

    def update_w_prefix(self, data: dict | None = None, prefix: str = "") -> dict[str, str]:
        if not data:
            return {}

        matched = {}
        prefix = f"{prefix}_"
        for key in self._allow_list:
            prefixed_key = f"{prefix}{key}"
            if prefixed_key in data:
                self[key] = data[prefixed_key]
                matched[key] = prefixed_key
        return matched
