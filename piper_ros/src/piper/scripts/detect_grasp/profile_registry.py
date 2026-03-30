#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Category -> unified profile registry with deep merge utilities."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

try:
    from .protocol import default_profile_dict
except ImportError:  # script mode
    from protocol import default_profile_dict


def deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge_dict(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


class ProfileRegistry:
    """Stores profile templates and returns merged profile for a category."""

    def __init__(
        self,
        *,
        default_profile: dict[str, Any] | None = None,
        category_profiles: dict[str, dict[str, Any]] | None = None,
    ):
        self._default_profile = deepcopy(default_profile or default_profile_dict())
        self._category_profiles = deepcopy(category_profiles or {})

    def get(self, category: str, *, override: dict[str, Any] | None = None) -> dict[str, Any]:
        base = deepcopy(self._default_profile)
        category_patch = self._category_profiles.get(category, {})
        profile = deep_merge_dict(base, category_patch)
        if override:
            profile = deep_merge_dict(profile, override)
        profile["category"] = category
        return profile

    def set_category_profile(self, category: str, patch: dict[str, Any]) -> None:
        self._category_profiles[category] = deepcopy(patch)

    def set_default_profile(self, profile: dict[str, Any]) -> None:
        self._default_profile = deepcopy(profile)
