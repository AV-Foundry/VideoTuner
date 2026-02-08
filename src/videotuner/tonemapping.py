"""Unified tonemapping helpers for HDR→SDR conversion.

Provides GPU detection and FFmpeg filter chain construction for tonemapping.
Used by both VMAF assessment and autocrop to ensure consistent HDR handling.

GPU path uses libplacebo with bt.2390 (Vulkan required).
CPU path uses zscale + tonemap with hable (no GPU required).
"""

from __future__ import annotations

import logging
import subprocess
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def has_vulkan_support(ffmpeg_bin: str = "ffmpeg") -> bool:
    """Probe whether FFmpeg can initialise a Vulkan device.

    The result is cached for the lifetime of the process so the probe
    runs at most once.

    Args:
        ffmpeg_bin: Path to the ffmpeg binary.

    Returns:
        True if Vulkan initialisation succeeds, False otherwise.
    """
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-v",
        "quiet",
        "-init_hw_device",
        "vulkan",
        "-f",
        "lavfi",
        "-i",
        "nullsrc=s=16x16:d=0.04",
        "-frames:v",
        "1",
        "-f",
        "null",
        "-",
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        supported = result.returncode == 0
    except FileNotFoundError:
        logger.warning(
            "ffmpeg not found at '%s'; assuming no Vulkan support", ffmpeg_bin
        )
        supported = False
    except subprocess.TimeoutExpired:
        logger.warning("Vulkan probe timed out; assuming no Vulkan support")
        supported = False

    if supported:
        logger.info("Vulkan GPU detected — using libplacebo (bt.2390) for tonemapping")
    else:
        logger.info("No Vulkan GPU — using CPU tonemapping (zscale + hable)")

    return supported


def build_tonemap_chain(width: int, height: int, *, use_gpu: bool) -> str:
    """Build an FFmpeg filter chain string for HDR→SDR tonemapping.

    Peak detection is always disabled for reproducible, frame-perfect results.

    Args:
        width: Target output width in pixels.
        height: Target output height in pixels.
        use_gpu: If True, use libplacebo via Vulkan; otherwise use zscale + hable.

    Returns:
        FFmpeg filter chain string (comma-separated filters).
    """
    if use_gpu:
        # libplacebo via Vulkan — bt.2390, peak detection off
        # gamut_mapping defaults to 'perceptual'
        return (
            f"libplacebo=w={width}:h={height}"
            ":colorspace=bt709:color_primaries=bt709:color_trc=bt709"
            ":range=limited:tonemapping=bt.2390:peak_detect=0"
        )

    # CPU fallback — zscale + tonemap (hable)
    # npl=100: SDR reference level (100 nits) for PQ linearisation
    # desat=2: FFmpeg default, desaturates very bright colours naturally
    # peak=0: auto-detect signal peak from HDR metadata
    return (
        "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
        f"tonemap=hable:desat=2:peak=0,zscale=t=bt709:m=bt709:r=tv,"
        f"scale={width}:{height}:flags=bicubic"
    )
