from __future__ import annotations

import json
import logging
import math
import os
import re
import shlex
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, cast, TYPE_CHECKING

from .media import (
    VideoInfo,
    parse_video_info,
    get_bit_depth_from_pix_fmt,
    get_assessment_frame_count,
)
from .tool_parsers import get_float
from .utils import run, make_relative_path
from .constants import (
    VMAF_THREAD_CPU_FRACTION,
    VMAF_MAX_THREADS,
    RESOLUTION_CONFIDENCE_THRESHOLD,
    RESOLUTION_RELATIVE_TOLERANCE,
    MIN_CROP_FRACTION,
)

if TYPE_CHECKING:
    from .profiles import Profile
    from .progress import PipelineDisplay


def get_cpu_count() -> int:
    """Get number of CPU cores, with fallback to 8 if detection fails."""
    try:
        return os.cpu_count() or 8
    except Exception:
        return 8


@dataclass(frozen=True)
class VMAFResult:
    mean: float
    harmonic_mean: float
    minimum: float
    p1_low: float
    frame_scores: Sequence[float] = field(default_factory=tuple)


def needs_tonemap(info: VideoInfo) -> bool:
    """Check if video needs tonemapping (BT.2020 color space only)."""
    return info.color_primaries == "BT.2020"


@dataclass(frozen=True)
class ResolutionResult:
    """
    Result of commercial resolution classification.

    Attributes:
        width: Actual video width in pixels
        height: Actual video height in pixels
        base_label: Resolution tier without scan type (e.g., "1080", "720")
        base_height: Standard height for this tier (e.g., 1080 for 1080p)
        confidence: Classification confidence from 0.0 to 1.0
        notes: Human-readable description of the match
        extras: Additional metadata (aspect ratios, error metrics, etc.)
    """

    width: int
    height: int
    base_label: str
    base_height: int
    confidence: float
    notes: str
    extras: dict[str, object] = field(default_factory=dict)


class CommercialResolutionInfer:
    """
    Classify arbitrary pixel dimensions into standard commercial resolution tiers:
        480p, 576p, 720p, 1080p, 1440p, 2160p, 4320p, 8640p

    Handles multiple aspect ratios and orientations:
    - 16:9 widescreen (1920x1080, 1280x720, etc.)
    - 4:3 classic TV/Academy ratio (960x720, 640x480, etc.)
    - 2.39:1 / 2.35:1 cinemascope (letterboxed)
    - 21:9 ultra-wide
    - Portrait/vertical video

    Returns the base resolution tier and confidence score (0.0-1.0).
    """

    # Standard base tiers: (label, base_height, canonical width)
    # Includes both 16:9 (widescreen) and 4:3 (classic/academy) variants
    BASES: list[tuple[str, int, int]] = [
        ("480", 480, 854),  # 16:9 SD
        ("480", 480, 640),  # 4:3 SD
        ("576", 576, 1024),  # 16:9 PAL
        ("576", 576, 768),  # 4:3 PAL
        ("720", 720, 1280),  # 16:9 HD
        ("720", 720, 960),  # 4:3 HD (pillarboxed from 1080p)
        ("1080", 1080, 1920),  # 16:9 Full HD
        ("1080", 1080, 1440),  # 4:3 Full HD (pillarboxed)
        ("1440", 1440, 2560),  # 16:9 QHD
        ("1440", 1440, 1920),  # 4:3 QHD (pillarboxed)
        ("2160", 2160, 3840),  # 16:9 4K/UHD
        ("2160", 2160, 2880),  # 4:3 4K (pillarboxed)
        ("4320", 4320, 7680),  # 16:9 8K
        ("4320", 4320, 5760),  # 4:3 8K (pillarboxed)
        ("8640", 8640, 15360),  # 16:9 16K
        ("8640", 8640, 11520),  # 4:3 16K (pillarboxed)
    ]

    # Absolute pixel tolerance for rounding errors
    ABS_TOL: int = 8

    @classmethod
    def infer(cls, width: int, height: int) -> ResolutionResult:
        """
        Infer commercial resolution from pixel dimensions.

        Tries both (w,h) and swapped (h,w) to handle portrait videos,
        returning whichever has higher confidence.

        Args:
            width: Video width in pixels
            height: Video height in pixels

        Returns:
            ResolutionResult with classification and confidence
        """
        cand1 = cls._infer_one(width, height)
        cand2 = cls._infer_one(height, width)
        return cand1 if cand1.confidence >= cand2.confidence else cand2

    @classmethod
    def _infer_one(cls, w: int, h: int) -> ResolutionResult:
        """Internal: classify a single orientation (w,h)."""
        if h == 0:
            # Prevent division by zero
            return ResolutionResult(
                width=w,
                height=h,
                base_label="480",
                base_height=480,
                confidence=0.0,
                notes="Invalid dimensions (height=0)",
                extras={},
            )

        best = None
        best_err = float("inf")

        for label, bh, bw in cls.BASES:
            base_ar = bw / bh
            obs_ar = w / h

            # tolerance helpers
            def tol(b: float) -> float:
                return max(cls.ABS_TOL, RESOLUTION_RELATIVE_TOLERANCE * b)

            # measure relative differences
            dw = abs(w - bw) / bw if bw > 0 else float("inf")
            dh = abs(h - bh) / bh if bh > 0 else float("inf")
            ar_err = abs(obs_ar - base_ar) / base_ar if base_ar > 0 else float("inf")

            # Three hypotheses: full-frame, cropped height (letterbox), cropped width (pillarbox)
            err_full = (dw + dh) / 2 + 0.25 * ar_err

            err_crop_h = None
            if w >= MIN_CROP_FRACTION * bw and h <= bh + tol(bh):
                deficit = max(0.0, (bh - h) / bh) if bh > 0 else 0.0
                err_crop_h = (
                    (abs(w - bw) / bw if bw > 0 else 0.0)
                    + 0.5 * deficit
                    + 0.25 * ar_err
                )

            err_crop_w = None
            if h >= MIN_CROP_FRACTION * bh and w <= bw + tol(bw):
                deficit = max(0.0, (bw - w) / bw) if bw > 0 else 0.0
                err_crop_w = (
                    (abs(h - bh) / bh if bh > 0 else 0.0)
                    + 0.5 * deficit
                    + 0.25 * ar_err
                )

            for err in (err_full, err_crop_h, err_crop_w):
                if err is None:
                    continue
                if err < best_err:
                    best_err = err
                    best = ResolutionResult(
                        width=w,
                        height=h,
                        base_label=label,
                        base_height=bh,
                        confidence=cls._err_to_conf(err),
                        notes=f"Matched to {label}p (base {bw}x{bh})",
                        extras={
                            "observed_ar": round(obs_ar, 5),
                            "base_ar": round(base_ar, 5),
                            "norm_error": round(err, 5),
                        },
                    )

                    # Early termination for excellent matches
                    if err < 0.01:
                        return best

        # Fallback if no match found (shouldn't happen)
        if best is None:
            label, bh, bw = min(cls.BASES, key=lambda b: abs(h - b[1]))
            best = ResolutionResult(
                width=w,
                height=h,
                base_label=label,
                base_height=bh,
                confidence=0.3,
                notes="Fallback by nearest base height",
                extras={"observed_ar": round(w / h, 5), "base_ar": round(bw / bh, 5)},
            )

        return best

    @staticmethod
    def _err_to_conf(err: float) -> float:
        """Map normalized error to confidence 0..1."""
        err = max(0.0, err)
        conf = 1.0 / (1.0 + 10.0 * err)
        return round(max(0.0, min(1.0, conf)), 3)


def validate_comparison(ref_info: VideoInfo, dis_info: VideoInfo) -> None:
    """Validate that the comparison is supported.

    Raises:
        ValueError: If validation fails
    """
    # Check for upscaling (distorted > reference)
    if ref_info.width and dis_info.width and dis_info.width > ref_info.width:
        raise ValueError(
            f"Upscaling not supported: distorted width ({dis_info.width}) > reference width ({ref_info.width})"
        )
    if ref_info.height and dis_info.height and dis_info.height > ref_info.height:
        raise ValueError(
            f"Upscaling not supported: distorted height ({dis_info.height}) > reference height ({ref_info.height})"
        )

    # Check for color space upgrade (reference is BT.709 but distorted is BT.2020)
    ref_is_2020 = needs_tonemap(ref_info)
    dis_is_2020 = needs_tonemap(dis_info)
    if not ref_is_2020 and dis_is_2020:
        raise ValueError(
            "Color space upgrade not supported: reference is BT.709 but distorted is BT.2020"
        )

    # Check for bit depth upgrade (distorted > reference)
    ref_depth = get_bit_depth_from_pix_fmt(ref_info.pix_fmt)
    dis_depth = get_bit_depth_from_pix_fmt(dis_info.pix_fmt)
    if dis_depth > ref_depth:
        raise ValueError(
            f"Bit depth upgrade not supported: distorted bit depth ({dis_depth}-bit) > reference bit depth ({ref_depth}-bit)"
        )


def select_vmaf_model(width: int, height: int) -> str:
    """Select appropriate VMAF model based on resolution tier classification.

    Uses smart resolution detection to handle letterboxed, pillarboxed, ultra-wide,
    and portrait content correctly.

    Args:
        width: Video width in pixels
        height: Video height in pixels

    Returns:
        Model version string (e.g., "vmaf_4k_v0.6.1" or "vmaf_v0.6.1")
    """
    log = logging.getLogger(__name__)

    # Classify resolution using smart detection
    result = CommercialResolutionInfer.infer(width, height)

    # Log classification details
    log.debug(
        f"Resolution classification: {width}x{height} → {result.base_label}p (confidence: {result.confidence}, AR: {result.extras.get('observed_ar', 'N/A')})"
    )

    # Warn on low confidence matches
    if result.confidence < RESOLUTION_CONFIDENCE_THRESHOLD:
        log.warning(
            f"Low confidence resolution classification for {width}x{height}: {result.notes}"
        )

    # Use 4K model for 2160p tier and above
    if result.base_height >= 2160:
        return "vmaf_4k_v0.6.1"
    return "vmaf_v0.6.1"


def build_vmaf_filter(
    ref_needs_tonemap: bool,
    _dis_needs_tonemap: bool,
    dis_bit_depth: int = 8,
    width: int = 3840,
    height: int = 2160,
    tonemap_policy: str = "auto",  # auto|force|off
) -> str:
    apply_tonemap = False
    if tonemap_policy == "force":
        apply_tonemap = True
    elif tonemap_policy == "auto":
        # Only tonemap if reference is BT.2020
        # (validation ensures distorted can't be BT.2020 if reference is BT.709)
        apply_tonemap = ref_needs_tonemap

    # Use pixel format based on distorted bit depth
    pix_fmt = "yuv420p10le" if dis_bit_depth == 10 else "yuv420p"
    final = f"format={pix_fmt},setpts=PTS-STARTPTS,settb=AVTB"

    if apply_tonemap:
        lp_sdr = (
            "libplacebo="
            f"w={width}:h={height}:"
            "colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=limited:"
            "tonemapping=bt.2390"
        )
        ref_chain = f"[0:v]{lp_sdr},{final}[ref]"
        dis_chain = f"[1:v]{lp_sdr},{final}[dis]"
    else:
        pre = f"scale={width}:{height}:flags=bicubic"
        ref_chain = f"[0:v]{pre},{final}[ref]"
        dis_chain = f"[1:v]{pre},{final}[dis]"

    return f"{ref_chain};{dis_chain};[dis][ref]"


def run_vmaf(
    ffmpeg_bin: str,
    ref_path: Path,
    dis_path: Path,
    ref_info: VideoInfo,
    dis_info: VideoInfo,
    model_spec: str | None = None,
    log_path: Path | None = None,
    tonemap_policy: str = "auto",
    cwd: Path | None = None,
    line_handler: Callable[[str], bool] | None = None,
) -> VMAFResult:
    log = logging.getLogger(__name__)

    # Validate comparison first
    validate_comparison(ref_info, dis_info)

    if log_path is None:
        raise ValueError("log_path must be provided for VMAF run")

    def _escape_filter_path(p: Path) -> str:
        # Convert to forward slashes and escape the drive letter colon for filter syntax
        s = str(p).replace("\\", "/")
        s = re.sub(r"^([A-Za-z]):/", r"\1\\:/", s)
        return s

    # Auto-select model based on distorted resolution if not specified
    if model_spec is None:
        if dis_info.width and dis_info.height:
            model_spec = select_vmaf_model(dis_info.width, dis_info.height)
        else:
            model_spec = "vmaf_v0.6.1"  # Default fallback

    # Build model option: prefer explicit path if it exists; otherwise accept
    # direct 'name=' or 'version=' input; fallback to 'version=<spec>'.
    if Path(model_spec).exists():
        model_opt = f"model=path={_escape_filter_path(Path(make_relative_path(Path(model_spec), cwd)))}"
    else:
        lower = model_spec.lower()
        if (
            lower.startswith("name=")
            or lower.startswith("version=")
            or lower.startswith("path=")
        ):
            model_opt = f"model={model_spec}"
        else:
            model_opt = f"model=version={model_spec}"

    # Use distorted file's properties for scaling and pixel format
    dis_bit_depth = get_bit_depth_from_pix_fmt(dis_info.pix_fmt)
    target_width = dis_info.width or 3840
    target_height = dis_info.height or 2160

    filter_prefix = build_vmaf_filter(
        ref_needs_tonemap=needs_tonemap(ref_info),
        _dis_needs_tonemap=needs_tonemap(dis_info),
        dis_bit_depth=dis_bit_depth,
        width=target_width,
        height=target_height,
        tonemap_policy=tonemap_policy,
    )
    # Use a relative path for log_path within the filter graph (avoid drive letters)
    log_path_escaped = _escape_filter_path(Path(make_relative_path(log_path, cwd)))

    # Calculate thread count based on CPU cores
    cpu_count = get_cpu_count()
    n_threads = min(max(1, int(cpu_count * VMAF_THREAD_CPU_FRACTION)), VMAF_MAX_THREADS)

    # VMAF options with sync parameters and explicit threading
    vmaf_opts = (
        f"libvmaf=shortest=true:ts_sync_mode=nearest:{model_opt}:"
        f"n_threads={n_threads}:log_fmt=json:log_path={log_path_escaped}"
    )
    filter_graph = f"{filter_prefix}{vmaf_opts}"

    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-y",
        "-stats",
        "-an",  # No audio
        "-sn",  # No subtitles
        "-dn",  # No data streams
        "-i",
        make_relative_path(ref_path, cwd),
        "-i",
        make_relative_path(dis_path, cwd),
        "-filter_complex",
        filter_graph,
        "-f",
        "null",
        "-",
    ]
    log.info("VMAF: %s", " ".join(shlex.quote(c) for c in cmd))
    run(cmd, live=True, cwd=cwd, line_callback=line_handler)

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            data = cast(object, json.load(f))
        frame_scores: list[float] = []
        mean: float | None = None
        harmonic_mean: float | None = None
        pooled_minimum: float | None = None
        # Prefer pooled mean if present
        if isinstance(data, dict):
            data_dict = cast(dict[str, object], data)
            if "pooled_metrics" in data_dict and isinstance(
                data_dict["pooled_metrics"], dict
            ):
                pooled_metrics = cast(dict[str, object], data_dict["pooled_metrics"])
                if "vmaf" in pooled_metrics and isinstance(
                    pooled_metrics["vmaf"], dict
                ):
                    vmaf_dict = cast(dict[str, object], pooled_metrics["vmaf"])
                    mean = get_float(vmaf_dict, "mean")
                    harmonic_mean = get_float(vmaf_dict, "harmonic_mean")
                    pooled_minimum = get_float(vmaf_dict, "min")
            elif "aggregate" in data_dict and isinstance(data_dict["aggregate"], dict):
                aggregate = cast(dict[str, object], data_dict["aggregate"])
                mean = get_float(aggregate, "vmaf")
            # Collect per-frame values for min and 1% low (and fallback mean)
            if "frames" in data_dict and isinstance(data_dict["frames"], list):
                frames = cast(list[object], data_dict["frames"])
                for fr in frames:
                    try:
                        if isinstance(fr, dict):
                            fr_dict = cast(dict[str, object], fr)
                            if "metrics" in fr_dict and isinstance(
                                fr_dict["metrics"], dict
                            ):
                                metrics = cast(dict[str, object], fr_dict["metrics"])
                                score = get_float(metrics, "vmaf")
                                if score is not None and not math.isnan(score):
                                    frame_scores.append(score)
                    except Exception:
                        continue
        minimum = float("nan")
        p1_low = float("nan")
        if frame_scores:
            if pooled_minimum is None:
                pooled_minimum = min(frame_scores)
            minimum = float(pooled_minimum)
            k = max(1, math.ceil(0.01 * len(frame_scores)))
            low = sorted(frame_scores)[:k]
            p1_low = sum(low) / len(low)
            if mean is None:
                mean = sum(frame_scores) / len(frame_scores)
            if harmonic_mean is None:
                denom = sum((1.0 / v) for v in frame_scores if v != 0)
                harmonic_mean = (len(frame_scores) / denom) if denom else float("nan")
        if harmonic_mean is None:
            harmonic_mean = float("nan")
        if pooled_minimum is None and not math.isnan(minimum):
            pooled_minimum = minimum
        if mean is None:
            log.warning("Could not locate VMAF metrics in %s", log_path)
            return VMAFResult(
                mean=float("nan"),
                harmonic_mean=float(harmonic_mean),
                minimum=float(pooled_minimum)
                if pooled_minimum is not None
                else float("nan"),
                p1_low=p1_low,
                frame_scores=tuple(frame_scores),
            )
        return VMAFResult(
            mean=float(mean),
            harmonic_mean=float(harmonic_mean),
            minimum=float(pooled_minimum)
            if pooled_minimum is not None
            else float("nan"),
            p1_low=float(p1_low),
            frame_scores=tuple(frame_scores),
        )
    except FileNotFoundError as e:
        log.error(
            "Failed to parse VMAF JSON (%s): File not found. Error: %s", log_path, e
        )
        return VMAFResult(
            mean=float("nan"),
            harmonic_mean=float("nan"),
            minimum=float("nan"),
            p1_low=float("nan"),
            frame_scores=tuple(),
        )
    except Exception as e:
        log.warning("Failed to parse VMAF JSON (%s): %s", log_path, e)
        return VMAFResult(
            mean=float("nan"),
            harmonic_mean=float("nan"),
            minimum=float("nan"),
            p1_low=float("nan"),
            frame_scores=tuple(),
        )


def assess_with_vmaf(
    *,
    ffmpeg_bin: str,
    ffprobe_bin: str,
    ref_path: Path,
    dis_path: Path,
    model_spec: str | None = None,
    log_path: Path,
    tonemap_policy: str = "auto",
    cwd: Path | None = None,
    line_handler: Callable[[str], bool] | None = None,
) -> VMAFResult:
    """High-level helper that probes both inputs and runs VMAF with optional tonemapping.

    Centralizes all VMAF-related orchestration here (probing, filtering, parsing).
    If model_spec is None, automatically selects the appropriate model based on
    the distorted video's resolution.
    """
    ref_info = parse_video_info(
        ref_path, ffprobe_bin=ffprobe_bin, log_hdr_metadata=False
    )
    dis_info = parse_video_info(
        dis_path, ffprobe_bin=ffprobe_bin, log_hdr_metadata=False
    )
    return run_vmaf(
        ffmpeg_bin=ffmpeg_bin,
        ref_path=ref_path,
        dis_path=dis_path,
        ref_info=ref_info,
        dis_info=dis_info,
        model_spec=model_spec,
        log_path=log_path,
        tonemap_policy=tonemap_policy,
        cwd=cwd,
        line_handler=line_handler,
    )


def assess_vmaf_concatenated(
    reference_path: Path,
    distorted_path: Path,
    workdir: Path,
    repo_root: Path,
    profile: Profile,
    ffmpeg_bin: str,
    ffprobe_bin: str,
    vmaf_model: str | None,
    tonemap_policy: str,
    display: PipelineDisplay,
    log: logging.Logger,
    iteration: int,
) -> list[VMAFResult]:
    """Run VMAF assessment on concatenated reference and distorted files.

    Args:
        reference_path: Path to concatenated lossless reference
        distorted_path: Path to concatenated distorted encode
        workdir: Working directory for output files
        repo_root: Repository root directory
        profile: Encoding profile (used for output directory organization)
        ffmpeg_bin: Path to ffmpeg binary
        ffprobe_bin: Path to ffprobe binary
        vmaf_model: VMAF model to use (None for auto-selection)
        tonemap_policy: HDR tonemapping policy
        display: Progress display manager
        log: Logger instance
        iteration: Current iteration number

    Returns:
        List containing single VMAFResult for the concatenated comparison
    """
    from .pipeline_types import get_vmaf_dir

    vmaf_log_path = (
        get_vmaf_dir(workdir, profile) / f"vmaf_concatenated_iter{iteration}.json"
    )

    # Get frame count for progress tracking
    total_frames = get_assessment_frame_count(reference_path, ffprobe_bin=ffprobe_bin)

    with display.stage(
        "Running VMAF assessment",
        total=total_frames,
        show_eta=True,
        transient=True,
        show_done=True,
    ) as stage:
        result = assess_with_vmaf(
            ffmpeg_bin=ffmpeg_bin,
            ffprobe_bin=ffprobe_bin,
            ref_path=reference_path,
            dis_path=distorted_path,
            model_spec=vmaf_model,
            log_path=vmaf_log_path,
            tonemap_policy=tonemap_policy,
            cwd=repo_root,
            line_handler=stage.make_ffmpeg_handler(total_frames=total_frames),
        )

    log.info(
        "VMAF: mean=%.2f, harmonic_mean=%.2f, 1%%=%.2f, min=%.2f",
        result.mean,
        result.harmonic_mean,
        result.p1_low,
        result.minimum,
    )
    return [result]
