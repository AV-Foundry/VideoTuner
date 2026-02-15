"""Encoder type enumeration for VideoTuner."""

from __future__ import annotations

from enum import Enum


class EncoderType(str, Enum):
    """Supported video encoder types."""

    X265 = "x265"
    X264 = "x264"

    @property
    def codec_name(self) -> str:
        """Human-readable codec name for display."""
        return {"x265": "HEVC", "x264": "H.264"}[self.value]

    @property
    def bitstream_extension(self) -> str:
        """File extension for raw bitstream output."""
        return {"x265": ".hevc", "x264": ".264"}[self.value]

    @property
    def supports_hdr_metadata(self) -> bool:
        """Whether this encoder supports HDR10 metadata flags."""
        return self == EncoderType.X265
