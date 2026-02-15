"""
Centralized x264 encoder parameter building and validation.

This module provides a single source of truth for format-specific x264 parameters
that are auto-detected from source video and should not be specified in user profiles.
"""

from __future__ import annotations

import logging

from .encoding_utils import is_hdr_video
from .media import VideoInfo, get_bit_depth_from_pix_fmt

logger = logging.getLogger(__name__)

# Global x264 parameters that are auto-detected from source video by default.
# These can be overridden in user profiles if needed.
# Note: x264 does NOT support HDR10 metadata flags (hdr10, hdr10-opt,
# master-display, max-cll), repeat-headers, aud, or hrd.
GLOBAL_X264_PARAMS = {
    "colorprim",
    "transfer",
    "colormatrix",
    "chromaloc",
    "output-depth",
    "range",
}


def build_global_x264_params(
    video_info: VideoInfo,
    is_lossless: bool = False,
    chroma_location: int | None = None,
    skip_params: set[str] | None = None,
) -> list[str]:
    """
    Build global x264 parameters from video metadata in CLI format for x264.

    These parameters are auto-detected from the source video and include:
    - Color space parameters (colorprim, transfer, colormatrix, range)
    - Format compatibility (output-depth, chromaloc)

    Unlike x265, x264 does NOT support:
    - HDR10 metadata (--hdr10, --master-display, --max-cll)
    - Streaming compatibility flags (--repeat-headers, --aud, --hrd)

    Args:
        video_info: MediaInfo object from ffprobe
        is_lossless: If True, adds --qp 0 for lossless encoding
        chroma_location: Chroma sample location (0-5), auto-detected if None
        skip_params: Set of parameter names to skip (for profile overrides)

    Returns:
        List of x264 CLI arguments (e.g., ["--colorprim", "bt709", "--qp", "0"])
    """
    x264_params: list[str] = []
    skip = skip_params or set()

    # Lossless encoding: x264 uses --qp 0 (not --lossless like x265)
    if is_lossless:
        x264_params.extend(["--qp", "0"])

    # Detect if content is HDR
    color_trc = video_info.color_trc
    is_hdr = is_hdr_video(color_trc)

    logger.debug("HDR detection: color_trc='%s', is_hdr=%s", color_trc, is_hdr)

    # Determine output bit depth from source pixel format
    # x264 only supports 8-bit and 10-bit (cap 12-bit sources)
    if "output-depth" not in skip:
        output_depth = get_bit_depth_from_pix_fmt(video_info.pix_fmt)
        if output_depth > 10:
            logger.warning(
                "x264 does not support %d-bit encoding, capping to 10-bit",
                output_depth,
            )
            output_depth = 10
        x264_params.extend(["--output-depth", str(output_depth)])

    # Map color primaries (same mapping as x265)
    if "colorprim" not in skip and video_info.color_primaries:
        colorprim_map = {
            "BT.709": "bt709",
            "BT.2020": "bt2020",
            "BT.470M": "bt470m",
            "BT.601 NTSC": "smpte170m",
            "BT.601 PAL": "bt470bg",
        }
        primaries_val = video_info.color_primaries
        colorprim = colorprim_map.get(
            primaries_val,
            primaries_val.lower().replace(".", "").replace(" ", ""),
        )
        x264_params.extend(["--colorprim", colorprim])

    # Map transfer characteristics (same mapping as x265)
    if "transfer" not in skip and color_trc:
        transfer_map = {
            "PQ": "smpte2084",
            "HLG": "arib-std-b67",
            "BT.709": "bt709",
            "BT.601": "bt470m",
            "SMPTE 170M": "smpte170m",
        }
        transfer = transfer_map.get(
            color_trc,
            color_trc.lower().replace(".", "").replace(" ", ""),
        )
        x264_params.extend(["--transfer", transfer])

    # Map color matrix (same logic as x265)
    if "colormatrix" not in skip:
        colormatrix = None
        color_space = video_info.color_space
        if color_space:
            colormatrix_map = {
                "BT.709": "bt709",
                "BT.2020 non-constant": "bt2020nc",
                "BT.2020 constant": "bt2020c",
                "BT.601": "smpte170m",
                "BT.470 System B/G": "bt470bg",
                "bt709": "bt709",
                "bt2020nc": "bt2020nc",
                "bt2020c": "bt2020c",
                "smpte170m": "smpte170m",
                "bt470bg": "bt470bg",
            }
            colormatrix = colormatrix_map.get(color_space)

        # Fallback: infer from color primaries if matrix is unknown
        if colormatrix is None and video_info.color_primaries:
            primaries_lower = video_info.color_primaries.lower()
            if "bt.2020" in primaries_lower or primaries_lower == "bt2020":
                colormatrix = "bt2020nc"
            elif "bt.709" in primaries_lower or primaries_lower == "bt709":
                colormatrix = "bt709"
            elif "bt.601" in primaries_lower or primaries_lower == "bt601":
                colormatrix = "smpte170m"

        logger.debug(
            "Color matrix detection: color_space='%s', colormatrix='%s'",
            video_info.color_space,
            colormatrix,
        )

        if colormatrix:
            x264_params.extend(["--colormatrix", colormatrix])

    # Preserve color range
    # x264 uses "tv"/"pc" naming (not "limited"/"full" like x265)
    if "range" not in skip and video_info.color_range:
        range_val = "tv" if video_info.color_range == "tv" else "pc"
        x264_params.extend(["--range", range_val])

    # Preserve chroma location
    if "chromaloc" not in skip and chroma_location is not None:
        x264_params.extend(["--chromaloc", str(chroma_location)])

    # Note: x264 does NOT support HDR10 mastering display or MaxCLL metadata.
    # HDR sources are rejected at the pipeline level before reaching this point.
    if is_hdr:
        logger.debug("HDR source detected but x264 does not support HDR metadata flags")

    return x264_params
