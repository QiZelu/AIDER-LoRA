"""Frozen encoders used by the AIDER baseline."""

from .blip import BLIPTextVideoEncoder
from .slowfast import SlowFastEncoder
from .vital import VITALVisionEncoder

__all__ = [
    "BLIPTextVideoEncoder",
    "SlowFastEncoder",
    "VITALVisionEncoder",
]
