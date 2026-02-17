"""
Centralized x265 encoder parameter building and validation.

This module provides a single source of truth for format-specific x265 parameters
that are auto-detected from source video but can be overridden in user profiles.
"""

from __future__ import annotations

import logging

from .encoding_utils import is_hdr_video
from .media import VideoInfo, get_bit_depth_from_pix_fmt

logger = logging.getLogger(__name__)

# Global x265 parameters that are auto-detected from source video by default.
# These can be overridden in user profiles if needed.
GLOBAL_X265_PARAMS = {
    "colorprim",
    "transfer",
    "colormatrix",
    "hdr10",
    "hdr10-opt",
    "master-display",
    "max-cll",
    "chromaloc",
    "output-depth",
    "repeat-headers",
    "aud",
    "hrd",
    "range",
}

# Valid x265 preset values
VALID_PRESETS = (
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "veryslow",
    "placebo",
)


def build_global_x265_params(
    video_info: VideoInfo,
    is_lossless: bool = False,
    chroma_location: int | None = None,
    skip_params: set[str] | None = None,
) -> list[str]:
    """
    Build global x265 parameters from video metadata in CLI format for x265.exe.

    These parameters are auto-detected from the source video and include:
    - Color space parameters (colorprim, transfer, colormatrix, range)
    - HDR metadata (hdr10, master-display, max-cll)
    - Format compatibility (output-depth, chromaloc, repeat-headers, aud)

    Args:
        video_info: MediaInfo object from ffprobe
        is_lossless: If True, adds --lossless flag
        chroma_location: Chroma sample location (0-5), auto-detected if None
        skip_params: Set of parameter names to skip (for profile overrides)

    Returns:
        List of x265 CLI arguments (e.g., ["--colorprim", "bt2020", "--hdr10"])
    """
    x265_params: list[str] = []
    skip = skip_params or set()

    # Lossless encoding
    if is_lossless:
        x265_params.append("--lossless")

    # Universal parameters for all encodes
    if "aud" not in skip and "no-aud" not in skip:
        x265_params.append("--aud")  # Access Unit Delimiters (boolean flag)

    # HRD (Hypothetical Reference Decoder) information for VBV compliance
    # Only enable for non-lossless encodes (lossless has no rate control)
    if not is_lossless and "hrd" not in skip and "no-hrd" not in skip:
        x265_params.append("--hrd")

    # Detect if content is HDR (PQ/SMPTE 2084 or HLG/ARIB STD-B67 transfer)
    color_trc = video_info.color_trc
    is_hdr = is_hdr_video(color_trc)

    logger.debug("HDR detection: color_trc='%s', is_hdr=%s", color_trc, is_hdr)

    # Determine output bit depth from source pixel format
    if "output-depth" not in skip:
        output_depth = get_bit_depth_from_pix_fmt(video_info.pix_fmt)
        x265_params.extend(["--output-depth", str(output_depth)])

    # Add repeat-headers: required for HDR, disabled for SDR
    if "repeat-headers" not in skip and "no-repeat-headers" not in skip:
        if is_hdr:
            x265_params.append("--repeat-headers")
        else:
            x265_params.append("--no-repeat-headers")

    # Add hdr10: enable for HDR content, disable for SDR
    if "hdr10" not in skip and "no-hdr10" not in skip:
        if is_hdr:
            x265_params.append("--hdr10")
        else:
            x265_params.append("--no-hdr10")

    # Add hdr10-opt for HDR content (separate check since it's a distinct param)
    if "hdr10-opt" not in skip and "no-hdr10-opt" not in skip:
        if is_hdr:
            x265_params.append("--hdr10-opt")

    # Map color primaries
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
        x265_params.extend(["--colorprim", colorprim])

    # Map transfer characteristics
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
        x265_params.extend(["--transfer", transfer])

    # Map color matrix
    # Infer from primaries if color_space is not a standard matrix identifier
    if "colormatrix" not in skip:
        colormatrix = None
        color_space = video_info.color_space
        if color_space:
            colormatrix_map = {
                # ffprobe uppercase values (from MediaInfo)
                "BT.709": "bt709",
                "BT.2020 non-constant": "bt2020nc",
                "BT.2020 constant": "bt2020c",
                "BT.601": "smpte170m",
                "BT.470 System B/G": "bt470bg",
                # ffprobe lowercase values (from JSON output)
                "bt709": "bt709",
                "bt2020nc": "bt2020nc",
                "bt2020c": "bt2020c",
                "smpte170m": "smpte170m",
                "bt470bg": "bt470bg",
            }
            colormatrix = colormatrix_map.get(color_space)

        # Fallback: infer from color primaries if matrix is unknown
        if colormatrix is None and video_info.color_primaries:
            # Handle both uppercase and lowercase primaries
            primaries_lower = video_info.color_primaries.lower()
            if "bt.2020" in primaries_lower or primaries_lower == "bt2020":
                colormatrix = "bt2020nc"  # Non-constant luminance is standard for UHD
            elif "bt.709" in primaries_lower or primaries_lower == "bt709":
                colormatrix = "bt709"
            elif "bt.601" in primaries_lower or primaries_lower == "bt601":
                colormatrix = "smpte170m"

        logger.debug(
            f"Color matrix detection: color_space='{video_info.color_space}', colormatrix='{colormatrix}'"
        )

        if colormatrix:
            x265_params.extend(["--colormatrix", colormatrix])

    # Preserve color range
    if "range" not in skip and video_info.color_range:
        range_val = "limited" if video_info.color_range == "tv" else "full"
        x265_params.extend(["--range", range_val])

    # Preserve chroma location
    if "chromaloc" not in skip and chroma_location is not None:
        x265_params.extend(["--chromaloc", str(chroma_location)])

    # Preserve HDR mastering display metadata if present
    if "master-display" not in skip and (
        video_info.mastering_display_color_primaries
        and video_info.mastering_display_luminance
    ):
        from .utils import parse_master_display_metadata

        master_display = parse_master_display_metadata(
            video_info.mastering_display_color_primaries,
            video_info.mastering_display_luminance,
        )
        if master_display:
            x265_params.extend(["--master-display", master_display])
            logger.debug("Adding mastering display metadata: %s", master_display)

    # Preserve MaxCLL/MaxFALL if present
    if "max-cll" not in skip and (
        video_info.maximum_content_light_level
        and video_info.maximum_frameaverage_light_level
    ):
        max_cll_val = video_info.maximum_content_light_level.replace(
            " cd/m2", ""
        ).strip()
        max_fall_val = video_info.maximum_frameaverage_light_level.replace(
            " cd/m2", ""
        ).strip()
        x265_params.extend(["--max-cll", f"{max_cll_val},{max_fall_val}"])
        logger.debug("Adding MaxCLL/MaxFALL: %s,%s", max_cll_val, max_fall_val)

    return x265_params
