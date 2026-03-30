"""Workflow entry points for cabinet shelf detection."""

from .detect_2d import detect
from .detect_3d import detect_3d
from .detect_experiment5 import detect_experiment5

__all__ = ["detect", "detect_3d", "detect_experiment5"]
