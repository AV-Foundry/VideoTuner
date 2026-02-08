from __future__ import annotations

import logging
import shlex
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from .media import VideoInfo
    from .profiles import Profile

from .encoding_utils import (
    CropValues,
    EncoderPaths,
    SamplingParams,
    build_sampling_vpy_script,
    build_x265_command,
    calculate_sample_count,
    calculate_usable_range,
    create_temp_encode_paths,
    is_hdr_video,
    mux_and_cleanup,
    resolve_absolute_path,
    run_x265_encode,
    write_vpy_script,
    VapourSynthEnv,
)
from .media import VideoFormat
from .profiles import Profile, create_multipass_profile
from .tonemapping import has_vulkan_support, build_tonemap_chain
from .tool_parsers import CROPDETECT_RE
from .utils import ensure_dir, log_separator, run_capture

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CropConfig:
    """Configuration for cropdetect behavior.

    Groups the enable flag with optional pre-calculated crop values.
    This ensures these related parameters are always passed together.

    Attributes:
        enabled: Whether cropdetect is enabled
        values: Pre-calculated crop values (only used if enabled is True)
    """

    enabled: bool = False
    values: CropValues | None = None

    @classmethod
    def disabled(cls) -> "CropConfig":
        """Create a disabled crop configuration."""
        return cls(enabled=False, values=None)

    @classmethod
    def with_values(cls, values: CropValues) -> "CropConfig":
        """Create an enabled crop configuration with specific values."""
        return cls(enabled=True, values=values)

    @classmethod
    def auto(cls) -> "CropConfig":
        """Create an enabled crop configuration that will auto-detect values."""
        return cls(enabled=True, values=None)


def mux_hevc_to_mkv(
    hevc_path: Path,
    output_path: Path,
    *,
    mkvmerge_bin: str = "mkvmerge",
    cwd: Path | None = None,
    line_handler: Callable[[str], bool] | None = None,
) -> None:
    """Mux a raw HEVC bitstream into an MKV container via mkvmerge."""
    log = logging.getLogger(__name__)

    if not hevc_path.exists():
        raise FileNotFoundError(f"HEVC bitstream not found: {hevc_path}")

    mux_args = [
        mkvmerge_bin,
        "--gui-mode",
        "--stop-after-video-ends",
        "-o",
        str(output_path),
        str(hevc_path),
    ]

    log.debug("mkvmerge mux cmd: %s", " ".join(shlex.quote(str(c)) for c in mux_args))
    _ = run_capture(mux_args, cwd=cwd, line_callback=line_handler)


def build_ffms2_index(
    source_path: Path,
    cache_file: Path,
    *,
    cwd: Path | None = None,
    line_handler: Callable[[str], bool] | None = None,
) -> bool:
    """
    Build FFMS2 index for a source video file using ffmsindex.exe.

    This function creates an FFMS2 index file that can be shared across all
    encodes from the same source, significantly speeding up subsequent operations.

    Args:
        source_path: Path to the source video file
        cache_file: Path where the index file should be created
        cwd: Working directory (for locating vapoursynth-portable)
        line_handler: Optional callback for progress parsing

    Returns:
        True if indexing succeeded, False if ffmsindex.exe not found or failed

    Raises:
        RuntimeError: If indexing fails critically
    """
    log = logging.getLogger(__name__)

    # If index already exists, skip
    if cache_file.exists():
        log.debug("FFMS2 index already exists: %s", cache_file)
        return True

    # VapourSynth portable paths
    vs_env = VapourSynthEnv.from_cwd(cwd)
    ffmsindex_exe = vs_env.vs_plugin_dir / "ffmsindex.exe"

    if not ffmsindex_exe.exists():
        log.debug(
            "ffmsindex.exe not found, FFMS2 will build index automatically (no progress display)"
        )
        return False

    # Ensure cache directory exists
    _ = ensure_dir(cache_file.parent)

    # Build absolute path for source
    abs_source_path = resolve_absolute_path(source_path, cwd)

    index_args = [str(ffmsindex_exe), str(abs_source_path), str(cache_file)]

    try:
        log.info("Building FFMS2 index for source...")
        _ = run_capture(
            index_args,
            cwd=cwd,
            line_callback=line_handler,
        )
        log.info("FFMS2 index created successfully")
        return True
    except Exception as e:
        log.warning("ffmsindex failed, FFMS2 will build index automatically: %s", e)
        # If ffmsindex fails, FFMS2 plugin will create index (but without progress)
        if cache_file.exists():
            cache_file.unlink()  # Remove partial index
        return False


def calculate_cropdetect_values(
    source_path: Path,
    start_frame: int,
    num_frames: int,
    fps: float,
    *,
    is_hdr: bool = False,
    interval: int = 80,
    ffmpeg_bin: str = "ffmpeg",
    source_width: int,
    source_height: int,
    cwd: Path | None = None,
    line_handler: Callable[[str], bool] | None = None,
    cropdetect_mode: str = "black",
    cropdetect_limit: int | None = None,
    cropdetect_round: int = 2,
    cropdetect_mv_threshold: int | None = None,
    cropdetect_low: float | None = None,
    cropdetect_high: float | None = None,
) -> CropValues:
    """Calculate crop values using FFmpeg cropdetect with timestamp seeking.

    Seeks to individual sample points across the middle 80% of the video
    instead of decoding every frame, making it much faster for long videos.
    For HDR content, a tonemapping filter is inserted before cropdetect
    (GPU libplacebo if available, CPU hable otherwise).

    Args:
        source_path: Input source video path.
        start_frame: First frame to analyse (0-indexed).
        num_frames: Total number of frames in the video.
        fps: Video framerate for sampling frequency.
        is_hdr: Whether the source is HDR (PQ/HLG).
        interval: Seconds between sampled frames.
        ffmpeg_bin: Path to the ffmpeg binary.
        source_width: Width of the source video in pixels.
        source_height: Height of the source video in pixels.
        cwd: Working directory for execution.
        line_handler: Optional callback for progress parsing.
        cropdetect_mode: Detection mode (``"black"`` or ``"mvedges"``).
        cropdetect_limit: Black pixel threshold 0-255 (None = FFmpeg default 24).
        cropdetect_round: Crop dimension divisibility (default 2).
        cropdetect_mv_threshold: Motion threshold in pixels (None = FFmpeg default 8).
        cropdetect_low: Canny low threshold 0.0-1.0 (None = FFmpeg default).
        cropdetect_high: Canny high threshold 0.0-1.0 (None = FFmpeg default).

    Returns:
        CropValues with left, right, top, bottom crop values.

    Raises:
        RuntimeError: If ffmpeg binary is not found.
    """
    log = logging.getLogger(__name__)

    # Guard: skip first/last 10 % of frames to avoid intros/outros
    skip = max(1, int(num_frames * 0.10))
    safe_start = start_frame + skip
    safe_end = start_frame + num_frames - skip
    step = max(1, round(fps * interval))

    # Generate sample frame positions and convert to timestamps
    sample_frames = list(range(safe_start, safe_end, step))
    if not sample_frames:
        log.warning("No sample frames to analyse — returning zero crop")
        return CropValues(left=0, right=0, top=0, bottom=0)

    # Build filter chain (no select filter — one frame per seek)
    filters: list[str] = []
    if is_hdr:
        use_gpu = has_vulkan_support(ffmpeg_bin)
        filters.append(
            build_tonemap_chain(source_width, source_height, use_gpu=use_gpu)
        )
    # Normalise to 8-bit YUV so cropdetect's limit (default 24/255) is
    # interpreted consistently regardless of source bit depth.
    filters.append("format=yuv420p")
    # Build cropdetect filter with user-configurable parameters.
    # skip=0: evaluate immediately (default skip=2 discards frames, which
    # produces no output when we only pass a single frame per seek).
    # reset=1: detect per-frame independently.
    cd_parts = [f"round={cropdetect_round}", "reset=1", "skip=0"]
    if cropdetect_mode != "black":
        cd_parts.append(f"mode={cropdetect_mode}")
    if cropdetect_limit is not None:
        cd_parts.append(f"limit={cropdetect_limit}")
    if cropdetect_mv_threshold is not None:
        cd_parts.append(f"mv_threshold={cropdetect_mv_threshold}")
    if cropdetect_low is not None:
        cd_parts.append(f"low={cropdetect_low}")
    if cropdetect_high is not None:
        cd_parts.append(f"high={cropdetect_high}")
    filters.append(f"cropdetect={':'.join(cd_parts)}")
    vf = ",".join(filters)

    abs_source = resolve_absolute_path(source_path, cwd)

    min_left = source_width
    min_right = source_width
    min_top = source_height
    min_bottom = source_height
    any_detected = False

    for i, frame_num in enumerate(sample_frames):
        timestamp = frame_num / fps
        cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(abs_source),
            "-frames:v",
            "1",
            "-vf",
            vf,
            "-an",
            "-sn",
            "-dn",
            "-f",
            "null",
            "-",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
                cwd=str(cwd) if cwd else None,
            )
        except subprocess.TimeoutExpired:
            log.debug("cropdetect timed out at %.3fs, skipping sample", timestamp)
            if line_handler:
                _ = line_handler(f"frame= {i + 1}")
            continue
        except FileNotFoundError as e:
            raise RuntimeError(f"Command not found: {ffmpeg_bin}") from e

        output = result.stderr + "\n" + result.stdout
        matches = cast(list[tuple[str, str, str, str]], CROPDETECT_RE.findall(output))
        for w_s, h_s, x_s, y_s in matches:
            w, h, x, y = int(w_s), int(h_s), int(x_s), int(y_s)
            min_left = min(min_left, x)
            min_top = min(min_top, y)
            min_right = min(min_right, source_width - w - x)
            min_bottom = min(min_bottom, source_height - h - y)
            any_detected = True

        if line_handler:
            _ = line_handler(f"frame= {i + 1}")

    if not any_detected:
        log.warning("cropdetect produced no output — returning zero crop")
        return CropValues(left=0, right=0, top=0, bottom=0)

    # Clamp negatives (can happen if cropdetect returns a larger area than source)
    left_val = max(0, min_left)
    right_val = max(0, min_right)
    top_val = max(0, min_top)
    bottom_val = max(0, min_bottom)

    log.debug(
        "CropDetect values calculated: L=%d R=%d T=%d B=%d",
        left_val,
        right_val,
        top_val,
        bottom_val,
    )
    return CropValues(left=left_val, right=right_val, top=top_val, bottom=bottom_val)


def encode_x265_concatenated_reference(
    source_path: Path,
    output_path: Path,
    interval_frames: int,
    region_frames: int,
    guard_start_frames: int,
    guard_end_frames: int,
    total_frames: int,
    fps: float,
    profile: Profile,
    video_info: VideoInfo,
    *,
    mkvmerge_bin: str = "mkvmerge",
    cwd: Path | None = None,
    temp_dir: Path | None = None,
    line_handler: Callable[[str], bool] | None = None,
    mux_handler: Callable[[str], bool] | None = None,
    perform_mux: bool = True,
    enable_cropdetect: bool = False,
    crop_values: CropValues | None = None,
    metric_label: str | None = None,
) -> Path:
    """
    Encode concatenated lossless reference using periodic sampling.

    Uses VapourSynth's SelectEvery to efficiently sample frames at regular
    intervals, avoiding manual region extraction and splicing.

    Args:
        source_path: Input source video path
        output_path: Output MKV path
        interval_frames: Sample every N frames
        region_frames: Number of consecutive frames per sample
        guard_start_frames: Frames to skip at start (intros/credits)
        guard_end_frames: Frames to skip at end (credits)
        total_frames: Total frames in source video
        fps: Video framerate
        profile: x265 encoding profile (should use preset="ultrafast")
        video_info: MediaInfo from ffprobe
        mkvmerge_bin: Path to mkvmerge binary
        cwd: Working directory
        temp_dir: Directory for temporary files
        line_handler: Optional callback for x265 progress
        mux_handler: Optional callback for mkvmerge progress
        enable_cropdetect: Whether to apply cropdetect
        crop_values: Pre-calculated crop values
        metric_label: Optional label for log messages (e.g., "VMAF", "SSIM2")

    Returns:
        Path to the produced file (MKV if muxed, HEVC if mux is deferred).

    Raises:
        ValueError: If parameters are invalid
        FileNotFoundError: If required binaries not found
    """
    log = logging.getLogger(__name__)

    # Validate and calculate usable range
    sampling = SamplingParams(
        interval_frames=interval_frames,
        region_frames=region_frames,
        guard_start_frames=guard_start_frames,
        guard_end_frames=guard_end_frames,
        total_frames=total_frames,
    )
    sampling.validate()
    usable_range = calculate_usable_range(sampling)

    # Resolve and validate encoder paths
    paths = EncoderPaths.from_cwd(cwd)
    paths.validate()

    # Determine video format and build x265 params (lossless)
    is_hdr = is_hdr_video(video_info.color_trc)
    video_format = VideoFormat.HDR if is_hdr else VideoFormat.SDR
    x265_params = profile.to_x265_params(
        crf=0.0,
        video_format=video_format,
        video_info=video_info,
        is_lossless=True,
    )

    # Create temp files
    vpy_path, hevc_path = create_temp_encode_paths(
        temp_dir=temp_dir, name="concatenated_reference"
    )

    # Build and write VapourSynth script
    cache_file = source_path.parent / f"{source_path.stem}.ffindex"
    effective_crop = crop_values if enable_cropdetect else None
    vpy_content = build_sampling_vpy_script(
        source_path=source_path,
        cache_file=cache_file,
        usable_range=usable_range,
        interval_frames=interval_frames,
        region_frames=region_frames,
        fps=fps,
        cwd=cwd,
        crop_values=effective_crop,
    )

    write_vpy_script(vpy_path, vpy_content)

    # Log encoding info
    num_samples, total_sampled_frames = calculate_sample_count(
        usable_range.frame_count, interval_frames, region_frames
    )
    label_prefix = f"{metric_label} " if metric_label else ""
    log.info(
        "Encoding %slossless reference: %d samples, %d total frames (interval=%d, region=%d)",
        label_prefix,
        num_samples,
        total_sampled_frames,
        interval_frames,
        region_frames,
    )
    log.debug("VapourSynth script:\n%s", vpy_content)

    # Build and run x265 command
    x265_args = build_x265_command(
        paths=paths,
        vpy_path=vpy_path,
        hevc_path=hevc_path,
        x265_params=x265_params,
        preset=profile.preset,
        cwd=cwd,
    )

    vs_env = paths.vs_env.build_env()
    _ = run_x265_encode(x265_args, vs_env, cwd, line_handler, run_capture)

    # Mux and cleanup
    final_output = mux_and_cleanup(
        hevc_path=hevc_path,
        output_path=output_path,
        vpy_path=vpy_path,
        perform_mux=perform_mux,
        mux_fn=mux_hevc_to_mkv,
        mkvmerge_bin=mkvmerge_bin,
        cwd=cwd,
        mux_handler=mux_handler,
    )

    if perform_mux:
        log.info("%slossless reference created: %s", label_prefix, output_path.name)
    else:
        log.info(
            "%slossless reference encoded (mux deferred): %s",
            label_prefix,
            hevc_path.name,
        )

    return final_output


def encode_x265_concatenated_distorted(
    source_path: Path,
    output_path: Path,
    interval_frames: int,
    region_frames: int,
    guard_start_frames: int,
    guard_end_frames: int,
    total_frames: int,
    fps: float,
    profile: Profile,
    crf: float,
    video_info: VideoInfo,
    *,
    mkvmerge_bin: str = "mkvmerge",
    cwd: Path | None = None,
    temp_dir: Path | None = None,
    line_handler: Callable[[str], bool] | None = None,
    mux_handler: Callable[[str], bool] | None = None,
    perform_mux: bool = True,
    enable_cropdetect: bool = False,
    crop_values: CropValues | None = None,
    metric_label: str | None = None,
) -> Path:
    """
    Encode concatenated distorted clip using periodic sampling with CRF.

    Uses VapourSynth's SelectEvery to efficiently sample frames at regular
    intervals, then encodes with specified CRF value.

    Args:
        source_path: Input source video path
        output_path: Output MKV path
        interval_frames: Sample every N frames
        region_frames: Number of consecutive frames per sample
        guard_start_frames: Frames to skip at start (intros/credits)
        guard_end_frames: Frames to skip at end (credits)
        total_frames: Total frames in source video
        fps: Video framerate
        profile: x265 encoding profile
        crf: CRF value for encoding
        video_info: MediaInfo from ffprobe
        mkvmerge_bin: Path to mkvmerge binary
        cwd: Working directory
        temp_dir: Directory for temporary files
        line_handler: Optional callback for x265 progress
        mux_handler: Optional callback for mkvmerge progress
        enable_cropdetect: Whether to apply cropdetect
        crop_values: Pre-calculated crop values
        metric_label: Optional label for log messages (e.g., "VMAF", "SSIM2")

    Returns:
        Path to the produced file (MKV if muxed, HEVC if mux is deferred).

    Raises:
        ValueError: If parameters are invalid
        FileNotFoundError: If required binaries not found
    """
    log = logging.getLogger(__name__)

    # Validate and calculate usable range
    sampling = SamplingParams(
        interval_frames=interval_frames,
        region_frames=region_frames,
        guard_start_frames=guard_start_frames,
        guard_end_frames=guard_end_frames,
        total_frames=total_frames,
    )
    sampling.validate()
    usable_range = calculate_usable_range(sampling)

    # Resolve and validate encoder paths
    paths = EncoderPaths.from_cwd(cwd)
    paths.validate()

    # Determine video format and build x265 params (CRF encoding)
    is_hdr = is_hdr_video(video_info.color_trc)
    video_format = VideoFormat.HDR if is_hdr else VideoFormat.SDR
    x265_params = profile.to_x265_params(
        crf=crf,
        video_format=video_format,
        video_info=video_info,
        is_lossless=False,
    )

    # Create temp files
    vpy_path, hevc_path = create_temp_encode_paths(
        temp_dir=temp_dir, name=f"concatenated_distorted_crf{crf}"
    )

    # Build and write VapourSynth script
    cache_file = source_path.parent / f"{source_path.stem}.ffindex"
    effective_crop = crop_values if enable_cropdetect else None
    vpy_content = build_sampling_vpy_script(
        source_path=source_path,
        cache_file=cache_file,
        usable_range=usable_range,
        interval_frames=interval_frames,
        region_frames=region_frames,
        fps=fps,
        cwd=cwd,
        crop_values=effective_crop,
    )

    write_vpy_script(vpy_path, vpy_content)

    # Log encoding info
    num_samples, total_sampled_frames = calculate_sample_count(
        usable_range.frame_count, interval_frames, region_frames
    )
    label_prefix = f"{metric_label} " if metric_label else ""
    log.info(
        "Encoding %sdistorted clip (CRF %.1f): %d samples, %d total frames (interval=%d, region=%d)",
        label_prefix,
        crf,
        num_samples,
        total_sampled_frames,
        interval_frames,
        region_frames,
    )
    log.debug("VapourSynth script:\n%s", vpy_content)

    # Build and run x265 command
    x265_args = build_x265_command(
        paths=paths,
        vpy_path=vpy_path,
        hevc_path=hevc_path,
        x265_params=x265_params,
        preset=profile.preset,
        cwd=cwd,
    )

    vs_env = paths.vs_env.build_env()
    _ = run_x265_encode(x265_args, vs_env, cwd, line_handler, run_capture)

    # Mux and cleanup
    final_output = mux_and_cleanup(
        hevc_path=hevc_path,
        output_path=output_path,
        vpy_path=vpy_path,
        perform_mux=perform_mux,
        mux_fn=mux_hevc_to_mkv,
        mkvmerge_bin=mkvmerge_bin,
        cwd=cwd,
        mux_handler=mux_handler,
    )

    if perform_mux:
        log.info("%sdistorted clip created: %s", label_prefix, output_path.name)
    else:
        log.info(
            "%sdistorted HEVC encoded (mux deferred): %s",
            label_prefix,
            hevc_path.name,
        )

    return final_output


def encode_x265_concatenated_bitrate(
    source_path: Path,
    output_path: Path,
    interval_frames: int,
    region_frames: int,
    guard_start_frames: int,
    guard_end_frames: int,
    total_frames: int,
    fps: float,
    profile: "Profile",
    video_info: "VideoInfo",
    *,
    mkvmerge_bin: str = "mkvmerge",
    cwd: Path | None = None,
    temp_dir: Path | None = None,
    stats_file: Path | None = None,
    analysis_file: Path | None = None,
    line_handler: Callable[[str], bool] | None = None,
    mux_handler: Callable[[str], bool] | None = None,
    perform_mux: bool = True,
    enable_cropdetect: bool = False,
    crop_values: CropValues | None = None,
    metric_label: str | None = None,
) -> Path:
    """
    Encode concatenated clip using bitrate mode with optional multi-pass support.

    Similar to encode_x265_concatenated_distorted but uses bitrate mode instead of CRF.
    Supports single-pass (pass=1 or no pass) and multi-pass (pass=2/3) encoding.

    Args:
        source_path: Input source video path
        output_path: Output MKV path
        interval_frames: Sample every N frames
        region_frames: Number of consecutive frames per sample
        guard_start_frames: Frames to skip at start
        guard_end_frames: Frames to skip at end
        total_frames: Total frames in source video
        fps: Video framerate
        profile: x265 encoding profile (must have bitrate set)
        video_info: MediaInfo from ffprobe
        mkvmerge_bin: Path to mkvmerge binary
        cwd: Working directory
        temp_dir: Directory for temporary files
        stats_file: Path to stats file (for pass 2/3, or output location for pass 1)
        analysis_file: Path to analysis file for multi-pass optimization (optional)
        line_handler: Optional callback for x265 progress
        mux_handler: Optional callback for mkvmerge progress
        perform_mux: Whether to mux HEVC to MKV
        enable_cropdetect: Whether to apply cropdetect
        crop_values: Pre-calculated crop values
        metric_label: Optional label for log messages (e.g., "VMAF", "SSIM2")

    Returns:
        Path to the produced file (MKV if muxed, HEVC if mux is deferred)

    Raises:
        ValueError: If profile is not in bitrate mode or parameters are invalid
        FileNotFoundError: If required binaries not found
    """
    log = logging.getLogger(__name__)

    # Validate profile is in bitrate mode
    if not profile.is_bitrate_mode:
        raise ValueError(
            f"Profile '{profile.name}' is not in bitrate mode. "
            + "Use encode_x265_concatenated_distorted for CRF encoding."
        )

    pass_num = profile.pass_number or 1

    # Validate stats file requirements
    if pass_num in (2, 3) and stats_file is None:
        raise ValueError(f"Pass {pass_num} requires stats_file parameter")

    # Validate and calculate usable range
    sampling = SamplingParams(
        interval_frames=interval_frames,
        region_frames=region_frames,
        guard_start_frames=guard_start_frames,
        guard_end_frames=guard_end_frames,
        total_frames=total_frames,
    )
    sampling.validate()
    usable_range = calculate_usable_range(sampling)

    # Resolve and validate encoder paths
    paths = EncoderPaths.from_cwd(cwd)
    paths.validate()

    # Determine video format
    is_hdr = is_hdr_video(video_info.color_trc)
    video_format = VideoFormat.HDR if is_hdr else VideoFormat.SDR

    log_separator(log)
    log.info("Profile configuration:")
    log.info("  Name: %s", profile.name)
    log.info("  Preset: %s", profile.preset)
    log.info("  Bitrate mode: %s", profile.is_bitrate_mode)
    log.info("  Bitrate: %s kbps", profile.bitrate)
    log.info("  Pass number: %s", profile.pass_number)
    log.info("  Settings: %s", profile.settings)
    log_separator(log)

    # Build x265 params (bitrate encoding with stats file and optional analysis file)
    x265_params = profile.to_x265_params(
        crf=0.0,  # Ignored in bitrate mode
        video_format=video_format,
        video_info=video_info,
        is_lossless=False,
        stats_file=stats_file,
        analysis_file=analysis_file,
    )

    bitrate_kbps = profile.bitrate or 0

    log_separator(log)
    log.info("Generated x265 parameters:")
    log.info("  %s", " ".join(x265_params))
    log_separator(log)

    # Create temp files
    vpy_path, hevc_path = create_temp_encode_paths(
        temp_dir=temp_dir, name=f"concatenated_bitrate{bitrate_kbps}_pass{pass_num}"
    )

    # Build and write VapourSynth script
    cache_file = source_path.parent / f"{source_path.stem}.ffindex"
    effective_crop = crop_values if enable_cropdetect else None
    vpy_content = build_sampling_vpy_script(
        source_path=source_path,
        cache_file=cache_file,
        usable_range=usable_range,
        interval_frames=interval_frames,
        region_frames=region_frames,
        fps=fps,
        cwd=cwd,
        crop_values=effective_crop,
    )

    write_vpy_script(vpy_path, vpy_content)

    # Log encoding info
    num_samples, total_sampled_frames = calculate_sample_count(
        usable_range.frame_count, interval_frames, region_frames
    )
    label_prefix = f"{metric_label} " if metric_label else ""
    log.info(
        "Encoding %sbitrate clip (pass %d, %d kbps): %d samples, %d total frames",
        label_prefix,
        pass_num,
        bitrate_kbps,
        num_samples,
        total_sampled_frames,
    )

    log_separator(log)
    log.info("VapourSynth script content:")
    log_separator(log)
    log.info("%s", vpy_content)
    log_separator(log)

    # Build x265 command
    x265_args = build_x265_command(
        paths=paths,
        vpy_path=vpy_path,
        hevc_path=hevc_path,
        x265_params=x265_params,
        preset=profile.preset,
        cwd=cwd,
    )

    # Detailed execution logging (bitrate-specific verbose output)
    abs_vpy_path = resolve_absolute_path(vpy_path, cwd)
    abs_hevc_path = resolve_absolute_path(hevc_path, cwd)

    log_separator(log)
    log.info("x265 Execution Details:")
    log.info("  x265 binary: %s (exists: %s)", paths.x265_bin, paths.x265_bin.exists())
    log.info(
        "  VapourSynth DLL: %s (exists: %s)",
        paths.vs_env.vsscript_dll,
        paths.vs_env.vsscript_dll.exists(),
    )
    log.info(
        "  FFMS2 DLL: %s (exists: %s)",
        paths.vs_env.ffms2_dll,
        paths.vs_env.ffms2_dll.exists(),
    )
    log.info("  VPY script: %s (exists: %s)", abs_vpy_path, abs_vpy_path.exists())
    log.info("  Output HEVC: %s", abs_hevc_path)
    if stats_file:
        log.info("  Stats file: %s", stats_file)
    log.info("  Working directory: %s", cwd)
    log_separator(log)
    log.info("Full x265 command:")
    log.info("  %s", " ".join(shlex.quote(str(c)) for c in x265_args))
    log_separator(log)

    # Setup VapourSynth environment
    vs_env = paths.vs_env.build_env()

    log.info("Starting x265 encoding...")

    try:
        # Run x265 and capture all output for debugging
        try:
            output = run_capture(
                x265_args,
                cwd=cwd,
                env=vs_env,
                line_callback=line_handler,
            )
            log_separator(log)
            log.info("x265 output (last 50 lines):")
            log_separator(log)
            output_lines = output.strip().split("\n")
            for line in output_lines[-50:]:
                log.info("  %s", line)
            log_separator(log)
        except Exception as e:
            log_separator(log, logging.ERROR)
            log.error("x265 encoding FAILED!")
            log_separator(log, logging.ERROR)
            log.error("Error: %s", str(e))
            log_separator(log, logging.ERROR)
            raise

        final_output: Path
        if perform_mux:
            log.info("Muxing concatenated HEVC -> MKV")
            mux_hevc_to_mkv(
                hevc_path=hevc_path,
                output_path=output_path,
                mkvmerge_bin=mkvmerge_bin,
                cwd=cwd,
                line_handler=mux_handler,
            )
            final_output = output_path
            log.info("%sbitrate clip created: %s", label_prefix, output_path.name)
        else:
            final_output = hevc_path
            log.info(
                "%sbitrate HEVC encoded (mux deferred): %s",
                label_prefix,
                hevc_path.name,
            )

    finally:
        if perform_mux and hevc_path.exists():
            hevc_path.unlink()
        if vpy_path.exists():
            vpy_path.unlink()

    return final_output


def encode_x265_multipass_bitrate(
    source_path: Path,
    output_path: Path,
    interval_frames: int,
    region_frames: int,
    guard_start_frames: int,
    guard_end_frames: int,
    total_frames: int,
    fps: float,
    profile: "Profile",
    video_info: "VideoInfo",
    *,
    mkvmerge_bin: str = "mkvmerge",
    cwd: Path | None = None,
    temp_dir: Path | None = None,
    stats_file: Path,
    line_handler: Callable[[str], bool] | None = None,
    mux_handler: Callable[[str], bool] | None = None,
    enable_cropdetect: bool = False,
    crop_values: CropValues | None = None,
) -> Path:
    """
    Encode concatenated clip using multi-pass bitrate encoding.

    Automatically handles 2-pass or 3-pass encoding based on profile settings:
    - 2-pass mode (pass=2): Pass 1 → Pass 2
    - 3-pass mode (pass=3): Pass 1 → Pass 3 → Pass 2 (three sequential passes)

    Pass sequence:
    - Pass 1: Analyzes video and creates initial stats file
    - Pass 3: Refines stats file using initial analysis (intermediate pass, 3-pass only)
    - Pass 2: Creates final encode using refined stats file (final pass)

    Args:
        source_path: Input source video path
        output_path: Output MKV path
        interval_frames: Sample every N frames
        region_frames: Number of consecutive frames per sample
        guard_start_frames: Frames to skip at start
        guard_end_frames: Frames to skip at end
        total_frames: Total frames in source video
        fps: Video framerate
        profile: x265 encoding profile (must have pass=2 or pass=3)
        video_info: MediaInfo from ffprobe
        mkvmerge_bin: Path to mkvmerge binary
        cwd: Working directory
        temp_dir: Directory for temporary files
        stats_file: Path for stats file (shared across all passes)
        line_handler: Optional callback for x265 progress
        mux_handler: Optional callback for mkvmerge progress
        enable_cropdetect: Whether to apply cropdetect
        crop_values: Pre-calculated crop values

    Returns:
        Path to final output file

    Raises:
        ValueError: If profile doesn't use multi-pass (pass=2 or pass=3)
    """
    log = logging.getLogger(__name__)

    pass_num = profile.pass_number
    if pass_num not in (2, 3):
        raise ValueError(
            f"Profile '{profile.name}' must specify pass=2 or pass=3 for multi-pass encoding"
        )

    # Create a modified profile for pass 1
    pass1_profile = create_multipass_profile(profile, 1)

    # Determine pass sequence based on final pass number
    if pass_num == 3:
        log.info("Starting 3-pass bitrate encoding (Pass 1 -> Pass 3 -> Pass 2)")
    else:
        log.info("Starting 2-pass bitrate encoding (Pass 1 -> Pass 2)")

    # Pass 1: Create stats file
    log.info("=== Pass 1: Analyzing video and creating stats file ===")
    if temp_dir:
        pass1_output = temp_dir / f"pass1_{output_path.name}"
    else:
        pass1_output = output_path.parent / f"pass1_{output_path.name}"

    _ = encode_x265_concatenated_bitrate(
        source_path=source_path,
        output_path=pass1_output,
        interval_frames=interval_frames,
        region_frames=region_frames,
        guard_start_frames=guard_start_frames,
        guard_end_frames=guard_end_frames,
        total_frames=total_frames,
        fps=fps,
        profile=pass1_profile,
        video_info=video_info,
        mkvmerge_bin=mkvmerge_bin,
        cwd=cwd,
        temp_dir=temp_dir,
        stats_file=stats_file,
        line_handler=line_handler,
        mux_handler=None,  # Don't mux pass 1 output
        perform_mux=False,
        enable_cropdetect=enable_cropdetect,
        crop_values=crop_values,
    )

    # Clean up pass 1 output (we only need stats file)
    if pass1_output.exists():
        try:
            pass1_output.unlink()
            log.debug("Cleaned up pass 1 output: %s", pass1_output)
        except Exception as e:
            log.warning("Failed to clean up pass 1 output: %s", e)

    # For 3-pass: Run Pass 3 (intermediate) before Pass 2 (final)
    if pass_num == 3:
        log.info("=== Pass 3: Refining stats file (intermediate pass) ===")

        # Create a modified profile for pass 3
        pass3_profile = create_multipass_profile(profile, 3)

        if temp_dir:
            pass3_output = temp_dir / f"pass3_{output_path.name}"
        else:
            pass3_output = output_path.parent / f"pass3_{output_path.name}"

        _ = encode_x265_concatenated_bitrate(
            source_path=source_path,
            output_path=pass3_output,
            interval_frames=interval_frames,
            region_frames=region_frames,
            guard_start_frames=guard_start_frames,
            guard_end_frames=guard_end_frames,
            total_frames=total_frames,
            fps=fps,
            profile=pass3_profile,
            video_info=video_info,
            mkvmerge_bin=mkvmerge_bin,
            cwd=cwd,
            temp_dir=temp_dir,
            stats_file=stats_file,
            line_handler=line_handler,
            mux_handler=None,  # Don't mux pass 3 output
            perform_mux=False,
            enable_cropdetect=enable_cropdetect,
            crop_values=crop_values,
        )

        # Clean up pass 3 output (we only need stats file)
        if pass3_output.exists():
            try:
                pass3_output.unlink()
                log.debug("Cleaned up pass 3 output: %s", pass3_output)
            except Exception as e:
                log.warning("Failed to clean up pass 3 output: %s", e)

    # Final pass: Pass 2 (always the final encoding pass)
    # For 2-pass mode: this runs after Pass 1
    # For 3-pass mode: this runs after Pass 1 → Pass 3
    log.info("=== Pass 2: Final encode using refined stats file ===")

    # Create a modified profile for final pass 2
    pass2_profile = create_multipass_profile(profile, 2)

    final_output = encode_x265_concatenated_bitrate(
        source_path=source_path,
        output_path=output_path,
        interval_frames=interval_frames,
        region_frames=region_frames,
        guard_start_frames=guard_start_frames,
        guard_end_frames=guard_end_frames,
        total_frames=total_frames,
        fps=fps,
        profile=pass2_profile,  # Always use pass 2 for final encode
        video_info=video_info,
        mkvmerge_bin=mkvmerge_bin,
        cwd=cwd,
        temp_dir=temp_dir,
        stats_file=stats_file,
        line_handler=line_handler,
        mux_handler=mux_handler,
        perform_mux=True,
        enable_cropdetect=enable_cropdetect,
        crop_values=crop_values,
    )

    log.info("Multi-pass bitrate encoding complete")
    return final_output
