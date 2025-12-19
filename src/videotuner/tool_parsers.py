"""Centralized parsing utilities for external tool output.

This module provides shared regex patterns and helper functions for parsing
output from external tools like FFmpeg, x265, mkvmerge, SSIMULACRA2, and
VapourSynth scripts.
"""

from __future__ import annotations

import re

# =============================================================================
# Common Patterns
# =============================================================================

# Matches float values like: 1.5, -2.3, +0.5, 100, -50
FLOAT_PATTERN = r"[-+]?\d*\.?\d+"

# Matches percentage values like: 50%, 100%
PERCENT_PATTERN = r"(\d+)%"


# =============================================================================
# Progress Patterns - for parsing tool output during execution
# =============================================================================

# FFmpeg frame counter: "frame=  123"
FFMPEG_FRAME_RE = re.compile(r"frame=\s*(\d+)")

# x265 frame output: "frame 123:" or "frame 123 "
X265_FRAME_RE = re.compile(r"frame\s+(\d+)(?::|\s)", re.IGNORECASE)

# x265 encoded summary: "encoded 123 frames"
X265_ENCODED_RE = re.compile(r"encoded\s+(\d+)\s+frames", re.IGNORECASE)

# x265 progress bar: "[  5.0%]   50 / 1000 Frames"
X265_PROGRESS_RE = re.compile(
    r"\[\s*(?P<pct>\d+(?:\.\d+)?)%\s*\]\s*(?P<done>\d+)\s*/\s*(?P<total>\d+)\s+Frames",
    re.IGNORECASE,
)

# mkvmerge percentage: "Progress: 50%"
MKVMERGE_PCT_RE = re.compile(r"(\d{1,3})%")

# SSIMULACRA2 progress line: "50.0% 500 / 1000"
SSIM_LINE_RE = re.compile(
    r"(?P<percent>\d+(?:\.\d+)?)%\s+(?P<done>\d+)\s*/\s*(?P<total>\d+)",
    re.IGNORECASE,
)

# ffmsindex progress: "Indexing, please wait... 50%"
FFMSINDEX_PROGRESS_RE = re.compile(r"(\d+)%", re.IGNORECASE)

# AutoCrop progress: "AutoCrop progress: 5/10"
AUTOCROP_PROGRESS_RE = re.compile(r"AutoCrop progress: (\d+)/(\d+)", re.IGNORECASE)

# ANSI escape sequences for terminal colors/formatting
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


# =============================================================================
# VapourSynth Output Patterns
# =============================================================================

# Crop values output: "CROP_VALUES:left,right,top,bottom"
CROP_VALUES_RE = re.compile(r"CROP_VALUES:(\d+),(\d+),(\d+),(\d+)")


# =============================================================================
# Dict Extraction Helpers - for parsing JSON output from tools
# =============================================================================


def get_str(d: dict[str, object], key: str) -> str | None:
    """Extract string value from dict, returning None if not a string.

    Args:
        d: Dictionary to extract from
        key: Key to look up

    Returns:
        String value or None if key doesn't exist or value isn't a string
    """
    val = d.get(key)
    return str(val) if isinstance(val, str) else None


def get_int(d: dict[str, object], key: str) -> int | None:
    """Extract int value from dict, handling int/float/str types.

    Args:
        d: Dictionary to extract from
        key: Key to look up

    Returns:
        Integer value or None if key doesn't exist or value can't be converted
    """
    val = d.get(key)
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val)
    if isinstance(val, str):
        try:
            return int(float(val))
        except ValueError:
            return None
    return None


def get_float(d: dict[str, object], key: str) -> float | None:
    """Extract float value from dict, handling int/float/str types.

    Args:
        d: Dictionary to extract from
        key: Key to look up

    Returns:
        Float value or None if key doesn't exist or value can't be converted
    """
    val = d.get(key)
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val)
        except ValueError:
            return None
    return None


# =============================================================================
# Specialized Parsers
# =============================================================================


def parse_fraction(rate: str) -> float:
    """Parse a fraction string like '30000/1001' into a float.

    Also handles plain numeric strings.

    Args:
        rate: String containing either a fraction (num/den) or plain number

    Returns:
        Parsed float value, or 0.0 if parsing fails
    """
    if "/" in rate:
        num, den = rate.split("/", 1)
        try:
            n = float(num)
            d = float(den)
            return 0.0 if d == 0 else n / d
        except ValueError:
            return 0.0
    try:
        return float(rate)
    except ValueError:
        return 0.0


def parse_crop_values(output: str) -> tuple[int, int, int, int] | None:
    """Parse VapourSynth CROP_VALUES output.

    Looks for a line like "CROP_VALUES:left,right,top,bottom" in the output.

    Args:
        output: Full output text from VapourSynth script

    Returns:
        Tuple of (left, right, top, bottom) crop values, or None if not found
    """
    match = CROP_VALUES_RE.search(output)
    if not match:
        return None
    return (
        int(match.group(1)),
        int(match.group(2)),
        int(match.group(3)),
        int(match.group(4)),
    )


def clean_ansi(line: str) -> str:
    """Remove ANSI escape sequences from a line of text.

    Args:
        line: Text potentially containing ANSI escape codes

    Returns:
        Clean text with escape codes removed and whitespace stripped
    """
    if not line:
        return ""
    return ANSI_ESCAPE_RE.sub("", line).strip()
