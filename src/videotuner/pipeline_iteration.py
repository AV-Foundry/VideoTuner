"""Iteration execution for VideoTuner pipeline.

This module provides the core iteration functions for encoding and assessing
samples at specific CRF or bitrate settings. It consolidates shared logic
between CRF and bitrate modes to eliminate duplication.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from .media import parse_video_info, get_encode_stats

if TYPE_CHECKING:
    from .pipeline_types import IterationContext
    from .profiles import Profile
    from .ssimulacra2_assessment import SSIM2Result
    from .vmaf_assessment import VMAFResult

# Type alias for iteration return values
IterationResult = tuple[
    dict[str, float | None],  # scores
    list["VMAFResult"],  # vmaf_results
    list["SSIM2Result"],  # ssim2_results
    float,  # predicted_bitrate_kbps
    Path | None,  # vmaf_distorted_path
    Path | None,  # ssim2_distorted_path
]


@dataclass(frozen=True)
class MetricSampleParams:
    """Calculated sampling parameters for a metric."""

    metric_type: Literal["vmaf", "ssim2"]
    num_samples: int
    total_frames: int
    interval_frames: int
    region_frames: int


def calculate_metric_params(
    ctx: "IterationContext",
    metric_type: Literal["vmaf", "ssim2"],
) -> MetricSampleParams:
    """Calculate sample counts and frame totals for a metric.

    Args:
        ctx: Pipeline iteration context
        metric_type: Either "vmaf" or "ssim2"

    Returns:
        MetricSampleParams with calculated values
    """
    if metric_type == "vmaf":
        interval_frames = ctx.args.vmaf_interval_frames
        region_frames = ctx.args.vmaf_region_frames
    else:
        interval_frames = ctx.args.ssim2_interval_frames
        region_frames = ctx.args.ssim2_region_frames

    num_samples = (
        ctx.usable_frames + interval_frames - region_frames
    ) // interval_frames
    total_frames = num_samples * region_frames

    return MetricSampleParams(
        metric_type=metric_type,
        num_samples=num_samples,
        total_frames=total_frames,
        interval_frames=interval_frames,
        region_frames=region_frames,
    )


def run_vmaf_assessment(
    ctx: "IterationContext",
    distorted_path: Path,
    iteration: int,
) -> list["VMAFResult"]:
    """Run VMAF assessment with standard parameters.

    Args:
        ctx: Pipeline iteration context
        distorted_path: Path to distorted encode
        iteration: Iteration number for logging

    Returns:
        List of VMAFResult (typically single element for concatenated file)
    """
    from .vmaf_assessment import assess_vmaf_concatenated

    if ctx.vmaf_ref_path is None:
        ctx.log.warning("VMAF reference path is None, skipping VMAF assessment")
        return []

    try:
        return assess_vmaf_concatenated(
            reference_path=ctx.vmaf_ref_path,
            distorted_path=distorted_path,
            workdir=ctx.workdir,
            repo_root=ctx.repo_root,
            profile=ctx.selected_profile,
            ffmpeg_bin=ctx.args.ffmpeg_bin,
            ffprobe_bin=ctx.args.ffprobe_bin,
            vmaf_model=ctx.args.vmaf_model,
            tonemap_policy=ctx.args.tonemap,
            display=ctx.display,
            log=ctx.log,
            iteration=iteration,
        )
    except Exception as e:
        ctx.log.error("VMAF assessment failed: %s", e)
        return []


def run_ssim2_assessment(
    ctx: "IterationContext",
    distorted_path: Path,
    iteration: int,
) -> list["SSIM2Result"]:
    """Run SSIMULACRA2 assessment with VapourSynth config.

    Args:
        ctx: Pipeline iteration context
        distorted_path: Path to distorted encode
        iteration: Iteration number for logging

    Returns:
        List of SSIM2Result (typically single element for concatenated file)
    """
    from .encoding_utils import VapourSynthEnv, resolve_ssim2_bin
    from .ssimulacra2_assessment import assess_ssim2_concatenated

    if ctx.ssim2_ref_path is None:
        ctx.log.warning("SSIM2 reference path is None, skipping SSIM2 assessment")
        return []

    try:
        ssim2_bin = resolve_ssim2_bin(ctx.args.ssim2_bin)

        vs_env = VapourSynthEnv.from_args(
            ctx.args.vs_dir, ctx.args.vs_plugin_dir, ctx.repo_root
        )
        env = vs_env.build_env()

        return assess_ssim2_concatenated(
            reference_path=ctx.ssim2_ref_path,
            distorted_path=distorted_path,
            workdir=ctx.workdir,
            temp_dir=ctx.temp_dir,
            profile=ctx.selected_profile,
            ssim2_bin=ssim2_bin,
            vs_env=env,
            vs_plugin_dir=vs_env.vs_plugin_dir,
            display=ctx.display,
            log=ctx.log,
            iteration=iteration,
        )
    except Exception as e:
        ctx.log.error("SSIM2 assessment failed: %s", e)
        return []


def extract_scores(
    vmaf_results: list["VMAFResult"],
    ssim2_results: list["SSIM2Result"],
) -> dict[str, float | None]:
    """Build scores dictionary from assessment results.

    Args:
        vmaf_results: VMAF assessment results (may be empty)
        ssim2_results: SSIM2 assessment results (may be empty)

    Returns:
        Dictionary mapping metric names to scores
    """
    scores: dict[str, float | None] = {}

    if vmaf_results:
        result = vmaf_results[0]
        scores["vmaf_mean"] = result.mean
        scores["vmaf_hmean"] = result.harmonic_mean
        scores["vmaf_1pct"] = result.p1_low
        scores["vmaf_min"] = result.minimum

    if ssim2_results:
        result = ssim2_results[0]
        scores["ssim2_mean"] = result.mean
        scores["ssim2_median"] = result.median
        scores["ssim2_95pct"] = result.p95_high
        scores["ssim2_5pct"] = result.p5_low

    return scores


def calculate_predicted_bitrate(
    vmaf_distorted_path: Path | None,
    ssim2_distorted_path: Path | None,
    ffprobe_bin: str,
    log: logging.Logger,
) -> float:
    """Calculate predicted bitrate from concatenated distorted files.

    With periodic sampling, the concatenated file's bitrate directly represents
    the predicted bitrate for the full video. Duration is read from the file itself.

    Args:
        vmaf_distorted_path: Path to VMAF concatenated distorted file
        ssim2_distorted_path: Path to SSIM2 concatenated distorted file
        ffprobe_bin: Path to ffprobe binary
        log: Logger instance

    Returns:
        Predicted bitrate in kbps (duration-weighted average if both metrics used)
    """
    bitrates: list[tuple[float, float]] = []  # (bitrate_kbps, duration_seconds)

    if vmaf_distorted_path and vmaf_distorted_path.exists():
        stats = get_encode_stats(vmaf_distorted_path, ffprobe_bin=ffprobe_bin)
        if stats:
            # Get duration from the file itself
            info = parse_video_info(vmaf_distorted_path, ffprobe_bin=ffprobe_bin, log_hdr_metadata=False)
            if info and info.duration:
                bitrates.append((stats.bitrate_kbps, info.duration))
                log.debug(
                    "VMAF concatenated bitrate: %.0f kbps (%.1fs)",
                    stats.bitrate_kbps,
                    info.duration,
                )

    if ssim2_distorted_path and ssim2_distorted_path.exists():
        stats = get_encode_stats(ssim2_distorted_path, ffprobe_bin=ffprobe_bin)
        if stats:
            # Get duration from the file itself
            info = parse_video_info(ssim2_distorted_path, ffprobe_bin=ffprobe_bin, log_hdr_metadata=False)
            if info and info.duration:
                bitrates.append((stats.bitrate_kbps, info.duration))
                log.debug(
                    "SSIM2 concatenated bitrate: %.0f kbps (%.1fs)",
                    stats.bitrate_kbps,
                    info.duration,
                )

    if not bitrates:
        log.warning("No bitrate data available from concatenated files")
        return 0.0

    # Duration-weighted average
    total_weighted = sum(br * dur for br, dur in bitrates)
    total_duration = sum(dur for _, dur in bitrates)
    predicted = total_weighted / total_duration if total_duration > 0 else 0.0

    log.info("Predicted bitrate (duration-weighted): %.0f kbps", predicted)
    return predicted


def run_single_crf_iteration(
    ctx: "IterationContext",
    crf: float,
    iteration: int = 1,
) -> IterationResult:
    """Run a single CRF encoding and assessment iteration.

    Uses periodic sampling approach with concatenated files.

    Args:
        ctx: Pipeline iteration context
        crf: CRF value to encode at
        iteration: Iteration number for naming/logging

    Returns:
        Tuple of (scores_dict, vmaf_results, ssim2_results, predicted_bitrate_kbps,
                  vmaf_distorted_path, ssim2_distorted_path)
    """
    from .create_encodes import encode_x265_concatenated_distorted, mux_hevc_to_mkv
    from .pipeline_types import get_distorted_dir
    from .pipeline_validation import validate_assessment_results
    from .utils import log_section

    log_section(ctx.log, f"Encoding (CRF {crf:.1f})")

    # Encode VMAF concatenated distorted file
    vmaf_distorted_path: Path | None = None
    if ctx.args.vmaf and ctx.vmaf_ref_path:
        vmaf_params = calculate_metric_params(ctx, "vmaf")
        vmaf_distorted_path = (
            get_distorted_dir(ctx.workdir, ctx.selected_profile)
            / f"vmaf_distorted_crf{crf:.1f}_iter{iteration}.mkv"
        )

        vmaf_hevc_path: Path | None = None
        with ctx.display.stage(
            "Encoding VMAF samples",
            total=vmaf_params.total_frames,
            unit="frames",
            transient=True,
            show_done=True,
        ) as enc_stage:
            enc_handler = enc_stage.make_x265_handler(
                total_frames=vmaf_params.total_frames
            )

            try:
                vmaf_hevc_path = encode_x265_concatenated_distorted(
                    source_path=ctx.input_path,
                    output_path=vmaf_distorted_path,
                    interval_frames=vmaf_params.interval_frames,
                    region_frames=vmaf_params.region_frames,
                    guard_start_frames=ctx.guard_start_frames,
                    guard_end_frames=ctx.guard_end_frames,
                    total_frames=ctx.total_frames,
                    fps=ctx.info.fps,
                    profile=ctx.selected_profile,
                    video_info=ctx.info,
                    crf=crf,
                    mkvmerge_bin=ctx.args.mkvmerge_bin,
                    cwd=ctx.repo_root,
                    temp_dir=ctx.temp_dir,
                    line_handler=enc_handler,
                    mux_handler=None,
                    perform_mux=False,
                    enable_autocrop=ctx.args.auto_crop,
                    crop_values=ctx.crop_values,
                    metric_label="VMAF",
                )
                if vmaf_hevc_path:
                    ctx.log.info("VMAF distorted HEVC created: %s", vmaf_hevc_path.name)
            except Exception as e:
                ctx.log.error("Failed to create VMAF distorted encode: %s", e)
                ctx.args.vmaf = False
                vmaf_distorted_path = None
                if vmaf_hevc_path and vmaf_hevc_path.exists():
                    try:
                        vmaf_hevc_path.unlink()
                    except Exception:
                        pass
                vmaf_hevc_path = None

        if vmaf_hevc_path and ctx.args.vmaf:
            with ctx.display.stage(
                "Muxing VMAF samples",
                total=100,
                unit="%",
                transient=True,
                show_done=True,
            ) as mux_stage:
                mux_handler = mux_stage.make_percent_handler()

                try:
                    assert vmaf_distorted_path is not None
                    mux_hevc_to_mkv(
                        hevc_path=vmaf_hevc_path,
                        output_path=vmaf_distorted_path,
                        mkvmerge_bin=ctx.args.mkvmerge_bin,
                        cwd=ctx.repo_root,
                        line_handler=mux_handler,
                    )
                    if vmaf_distorted_path:
                        ctx.log.info(
                            "VMAF distorted encode created: %s",
                            vmaf_distorted_path.name,
                        )
                except Exception as e:
                    ctx.log.error("Failed to mux VMAF distorted encode: %s", e)
                    ctx.args.vmaf = False
                    vmaf_distorted_path = None
                finally:
                    try:
                        if vmaf_hevc_path.exists():
                            vmaf_hevc_path.unlink()
                    except Exception:
                        pass

    # Encode SSIM2 concatenated distorted file
    ssim2_distorted_path: Path | None = None
    if ctx.args.ssim2 and ctx.ssim2_ref_path:
        ssim2_params = calculate_metric_params(ctx, "ssim2")
        ssim2_distorted_path = (
            get_distorted_dir(ctx.workdir, ctx.selected_profile)
            / f"ssim2_distorted_crf{crf:.1f}_iter{iteration}.mkv"
        )

        ssim2_hevc_path: Path | None = None
        with ctx.display.stage(
            "Encoding SSIM2 samples",
            total=ssim2_params.total_frames,
            unit="frames",
            transient=True,
            show_done=True,
        ) as enc_stage:
            enc_handler = enc_stage.make_x265_handler(
                total_frames=ssim2_params.total_frames
            )

            try:
                ssim2_hevc_path = encode_x265_concatenated_distorted(
                    source_path=ctx.input_path,
                    output_path=ssim2_distorted_path,
                    interval_frames=ssim2_params.interval_frames,
                    region_frames=ssim2_params.region_frames,
                    guard_start_frames=ctx.guard_start_frames,
                    guard_end_frames=ctx.guard_end_frames,
                    total_frames=ctx.total_frames,
                    fps=ctx.info.fps,
                    profile=ctx.selected_profile,
                    video_info=ctx.info,
                    crf=crf,
                    mkvmerge_bin=ctx.args.mkvmerge_bin,
                    cwd=ctx.repo_root,
                    temp_dir=ctx.temp_dir,
                    line_handler=enc_handler,
                    mux_handler=None,
                    perform_mux=False,
                    enable_autocrop=ctx.args.auto_crop,
                    crop_values=ctx.crop_values,
                    metric_label="SSIM2",
                )
                if ssim2_hevc_path:
                    ctx.log.info(
                        "SSIM2 distorted HEVC created: %s", ssim2_hevc_path.name
                    )
            except Exception as e:
                ctx.log.error("Failed to create SSIM2 distorted encode: %s", e)
                ctx.args.ssim2 = False
                ssim2_distorted_path = None
                if ssim2_hevc_path and ssim2_hevc_path.exists():
                    try:
                        ssim2_hevc_path.unlink()
                    except Exception:
                        pass
                ssim2_hevc_path = None

        if ssim2_hevc_path and ctx.args.ssim2:
            with ctx.display.stage(
                "Muxing SSIM2 samples",
                total=100,
                unit="%",
                transient=True,
                show_done=True,
            ) as mux_stage:
                mux_handler = mux_stage.make_percent_handler()

                try:
                    assert ssim2_distorted_path is not None
                    mux_hevc_to_mkv(
                        hevc_path=ssim2_hevc_path,
                        output_path=ssim2_distorted_path,
                        mkvmerge_bin=ctx.args.mkvmerge_bin,
                        cwd=ctx.repo_root,
                        line_handler=mux_handler,
                    )
                    if ssim2_distorted_path:
                        ctx.log.info(
                            "SSIM2 distorted encode created: %s",
                            ssim2_distorted_path.name,
                        )
                except Exception as e:
                    ctx.log.error("Failed to mux SSIM2 distorted encode: %s", e)
                    ctx.args.ssim2 = False
                    ssim2_distorted_path = None
                finally:
                    try:
                        if ssim2_hevc_path.exists():
                            ssim2_hevc_path.unlink()
                    except Exception:
                        pass

    log_section(ctx.log, f"Assessment (CRF {crf:.1f})")

    # Run assessments using shared helpers
    vmaf_results: list["VMAFResult"] = []
    if ctx.args.vmaf and vmaf_distorted_path:
        vmaf_results = run_vmaf_assessment(ctx, vmaf_distorted_path, iteration)

    ssim2_results: list["SSIM2Result"] = []
    if ctx.args.ssim2 and ssim2_distorted_path:
        ssim2_results = run_ssim2_assessment(ctx, ssim2_distorted_path, iteration)

    # Validate assessment results
    validate_assessment_results(
        vmaf_results if vmaf_results else None,
        ssim2_results if ssim2_results else None,
        f"CRF {crf:.1f} iteration {iteration}",
        ctx.log,
    )

    # Extract scores and calculate predicted bitrate
    scores = extract_scores(vmaf_results, ssim2_results)
    predicted_bitrate = calculate_predicted_bitrate(
        vmaf_distorted_path=vmaf_distorted_path,
        ssim2_distorted_path=ssim2_distorted_path,
        ffprobe_bin=ctx.args.ffprobe_bin,
        log=ctx.log,
    )

    return (
        scores,
        vmaf_results,
        ssim2_results,
        predicted_bitrate,
        vmaf_distorted_path,
        ssim2_distorted_path,
    )


def run_single_bitrate_iteration(
    ctx: "IterationContext",
    iteration: int = 1,
) -> IterationResult:
    """Run a single bitrate encoding and assessment iteration.

    Similar to run_single_crf_iteration but uses bitrate mode instead of CRF.
    Handles multi-pass encoding automatically based on profile settings.

    Args:
        ctx: Pipeline context with all configuration and paths
        iteration: Iteration number for logging/naming (not used for optimization)

    Returns:
        Tuple of (scores_dict, vmaf_results, ssim2_results, predicted_bitrate_kbps,
                  vmaf_distorted_path, ssim2_distorted_path)
    """
    from .pipeline_types import get_distorted_dir
    from .profiles import ProfileError
    from .utils import log_section

    profile = ctx.selected_profile
    pass_num = profile.pass_number or 1
    bitrate_kbps = profile.bitrate or 0

    ctx.log.debug("Bitrate iteration: profile=%s, bitrate=%s kbps, pass=%s, iteration=%s",
                   profile.name, bitrate_kbps, pass_num, iteration)

    log_section(ctx.log, f"Encoding (Bitrate {bitrate_kbps} kbps, Pass {pass_num})")

    # Determine if multi-pass encoding
    is_multipass = pass_num in (2, 3)

    # Log pass information once at the start
    if is_multipass:
        if pass_num == 3:
            ctx.log.info(
                "3-pass encoding: Pass 1 (analysis) -> Pass 3 (refine) -> Pass 2 (final)"
            )
        else:
            ctx.log.info("2-pass encoding: Pass 1 (analysis) -> Pass 2 (final)")

    # Create separate stats files for VMAF and SSIM2 (they encode different frame patterns)
    vmaf_stats_file: Path | None = None
    ssim2_stats_file: Path | None = None
    if is_multipass:
        vmaf_stats_file = ctx.workdir / f"{ctx.input_path.stem}_bitrate_stats_vmaf"
        ssim2_stats_file = ctx.workdir / f"{ctx.input_path.stem}_bitrate_stats_ssim2"

    # Create analysis files if multi-pass optimization is enabled
    vmaf_analysis_file: Path | None = None
    ssim2_analysis_file: Path | None = None
    has_multipass_opt = profile.settings.get(
        "multi-pass-opt-analysis", False
    ) or profile.settings.get("multi-pass-opt-distortion", False)
    if is_multipass and has_multipass_opt:
        vmaf_analysis_file = (
            ctx.workdir / f"{ctx.input_path.stem}_bitrate_analysis_vmaf.dat"
        )
        ssim2_analysis_file = (
            ctx.workdir / f"{ctx.input_path.stem}_bitrate_analysis_ssim2.dat"
        )

    # Encode VMAF concatenated distorted file
    vmaf_distorted_path: Path | None = None
    if ctx.args.vmaf and ctx.vmaf_ref_path:
        vmaf_params = calculate_metric_params(ctx, "vmaf")
        vmaf_distorted_path = (
            get_distorted_dir(ctx.workdir, profile)
            / f"vmaf_distorted_bitrate{bitrate_kbps}_iter{iteration}.mkv"
        )

        ctx.log.debug("VMAF encode: bitrate=%d kbps, multipass=%s, pass=%d, frames=%d",
                      bitrate_kbps, is_multipass, pass_num, vmaf_params.total_frames)

        try:
            vmaf_distorted_path = _encode_bitrate_metric(
                ctx=ctx,
                metric_params=vmaf_params,
                output_path=vmaf_distorted_path,
                profile=profile,
                pass_num=pass_num,
                is_multipass=is_multipass,
                stats_file=vmaf_stats_file,
                analysis_file=vmaf_analysis_file,
                metric_label="VMAF",
            )
            if vmaf_distorted_path:
                ctx.log.info(
                    "VMAF distorted bitrate encode created: %s",
                    vmaf_distorted_path.name,
                )
        except ProfileError:
            raise
        except Exception as e:
            import traceback
            ctx.log.error("VMAF encoding failed: %s", e)
            ctx.log.debug("Traceback:\n%s", traceback.format_exc())
            ctx.args.vmaf = False
            vmaf_distorted_path = None

    # Encode SSIM2 concatenated distorted file
    ssim2_distorted_path: Path | None = None
    if ctx.args.ssim2 and ctx.ssim2_ref_path:
        ssim2_params = calculate_metric_params(ctx, "ssim2")
        ssim2_distorted_path = (
            get_distorted_dir(ctx.workdir, profile)
            / f"ssim2_distorted_bitrate{bitrate_kbps}_iter{iteration}.mkv"
        )

        ctx.log.debug("SSIM2 encode: bitrate=%d kbps, multipass=%s, pass=%d, frames=%d",
                      bitrate_kbps, is_multipass, pass_num, ssim2_params.total_frames)

        try:
            ssim2_distorted_path = _encode_bitrate_metric(
                ctx=ctx,
                metric_params=ssim2_params,
                output_path=ssim2_distorted_path,
                profile=profile,
                pass_num=pass_num,
                is_multipass=is_multipass,
                stats_file=ssim2_stats_file,
                analysis_file=ssim2_analysis_file,
                metric_label="SSIM2",
            )
            if ssim2_distorted_path:
                ctx.log.info(
                    "SSIM2 distorted bitrate encode created: %s",
                    ssim2_distorted_path.name,
                )
        except ProfileError:
            raise
        except Exception as e:
            import traceback
            ctx.log.error("SSIM2 encoding failed: %s", e)
            ctx.log.debug("Traceback:\n%s", traceback.format_exc())
            ctx.args.ssim2 = False
            ssim2_distorted_path = None

    # Run assessments using shared helpers
    log_section(ctx.log, "Quality Assessment")

    vmaf_results: list["VMAFResult"] = []
    if ctx.args.vmaf and vmaf_distorted_path:
        vmaf_results = run_vmaf_assessment(ctx, vmaf_distorted_path, iteration)
        if vmaf_results:
            ctx.log.info("VMAF assessment complete")

    ssim2_results: list["SSIM2Result"] = []
    if ctx.args.ssim2 and ssim2_distorted_path:
        ssim2_results = run_ssim2_assessment(ctx, ssim2_distorted_path, iteration)
        if ssim2_results:
            ctx.log.info("SSIM2 assessment complete")

    # Extract scores and calculate predicted bitrate
    scores = extract_scores(vmaf_results, ssim2_results)
    predicted_bitrate = calculate_predicted_bitrate(
        vmaf_distorted_path=vmaf_distorted_path,
        ssim2_distorted_path=ssim2_distorted_path,
        ffprobe_bin=ctx.args.ffprobe_bin,
        log=ctx.log,
    )

    return (
        scores,
        vmaf_results,
        ssim2_results,
        predicted_bitrate,
        vmaf_distorted_path,
        ssim2_distorted_path,
    )


def _encode_bitrate_metric(
    ctx: "IterationContext",
    metric_params: MetricSampleParams,
    output_path: Path,
    profile: "Profile",
    pass_num: int,
    is_multipass: bool,
    stats_file: Path | None,
    analysis_file: Path | None,
    metric_label: str,
) -> Path | None:
    """Encode samples for a single metric in bitrate mode.

    Handles single-pass and multi-pass (2-pass, 3-pass) encoding.

    Args:
        ctx: Pipeline iteration context
        metric_params: Calculated sampling parameters
        output_path: Final output path for encoded file
        profile: Encoding profile
        pass_num: Pass number (1, 2, or 3)
        is_multipass: Whether multi-pass encoding is enabled
        stats_file: Path to stats file for multi-pass
        analysis_file: Path to analysis file for multi-pass optimization
        metric_label: Label for display ("VMAF" or "SSIM2")

    Returns:
        Path to the encoded file, or None if encoding failed
    """
    from .create_encodes import encode_x265_concatenated_bitrate
    from .profiles import create_multipass_profile

    if is_multipass:
        # Multi-pass: Run pass 1, then pass 2/3 with separate progress bars
        # Pass 1
        pass1_profile = create_multipass_profile(profile, 1)

        with ctx.display.stage(
            f"{metric_label} Pass 1: Analyzing",
            total=metric_params.total_frames,
            unit="frames",
            transient=True,
            show_done=False,
        ) as enc_stage:
            enc_handler = enc_stage.make_x265_handler(
                total_frames=metric_params.total_frames
            )

            if ctx.temp_dir:
                pass1_output = ctx.temp_dir / f"pass1_{output_path.name}"
            else:
                pass1_output = output_path.parent / f"pass1_{output_path.name}"

            _ = encode_x265_concatenated_bitrate(
                source_path=ctx.input_path,
                output_path=pass1_output,
                interval_frames=metric_params.interval_frames,
                region_frames=metric_params.region_frames,
                guard_start_frames=ctx.guard_start_frames,
                guard_end_frames=ctx.guard_end_frames,
                total_frames=ctx.total_frames,
                fps=ctx.info.fps,
                profile=pass1_profile,
                video_info=ctx.info,
                mkvmerge_bin=ctx.args.mkvmerge_bin,
                cwd=ctx.repo_root,
                temp_dir=ctx.temp_dir,
                stats_file=stats_file,
                analysis_file=analysis_file,
                line_handler=enc_handler,
                mux_handler=None,
                perform_mux=False,
                enable_autocrop=ctx.args.auto_crop,
                crop_values=ctx.crop_values,
                metric_label=metric_label,
            )

        # Clean up pass 1 output
        if pass1_output.exists():
            try:
                pass1_output.unlink()
                ctx.log.debug("Cleaned up pass 1 output: %s", pass1_output)
            except Exception as e:
                ctx.log.warning("Failed to clean up pass 1 output: %s", e)

        # For 3-pass: Run Pass 3 before Pass 2
        if pass_num == 3:
            pass3_profile = create_multipass_profile(profile, 3)

            with ctx.display.stage(
                f"{metric_label} Pass 3: Refining",
                total=metric_params.total_frames,
                unit="frames",
                transient=True,
                show_done=False,
            ) as enc_stage:
                enc_handler = enc_stage.make_x265_handler(
                    total_frames=metric_params.total_frames
                )

                if ctx.temp_dir:
                    pass3_output = ctx.temp_dir / f"pass3_{output_path.name}"
                else:
                    pass3_output = output_path.parent / f"pass3_{output_path.name}"

                _ = encode_x265_concatenated_bitrate(
                    source_path=ctx.input_path,
                    output_path=pass3_output,
                    interval_frames=metric_params.interval_frames,
                    region_frames=metric_params.region_frames,
                    guard_start_frames=ctx.guard_start_frames,
                    guard_end_frames=ctx.guard_end_frames,
                    total_frames=ctx.total_frames,
                    fps=ctx.info.fps,
                    profile=pass3_profile,
                    video_info=ctx.info,
                    mkvmerge_bin=ctx.args.mkvmerge_bin,
                    cwd=ctx.repo_root,
                    temp_dir=ctx.temp_dir,
                    stats_file=stats_file,
                    analysis_file=analysis_file,
                    line_handler=enc_handler,
                    mux_handler=None,
                    perform_mux=False,
                    enable_autocrop=ctx.args.auto_crop,
                    crop_values=ctx.crop_values,
                    metric_label=metric_label,
                )

            # Clean up pass 3 output
            if pass3_output.exists():
                try:
                    pass3_output.unlink()
                    ctx.log.debug("Cleaned up pass 3 output: %s", pass3_output)
                except Exception as e:
                    ctx.log.warning("Failed to clean up pass 3 output: %s", e)

        # Final pass: Always Pass 2
        pass2_profile = create_multipass_profile(profile, 2)
        ctx.log.debug("%s Pass 2 settings: %s", metric_label, pass2_profile.settings)
        ctx.log.debug("%s Pass 2 analysis_file: %s", metric_label, analysis_file)
        ctx.log.debug("%s Pass 2 stats_file: %s", metric_label, stats_file)

        with ctx.display.stage(
            f"{metric_label} Pass 2: Encoding",
            total=metric_params.total_frames,
            unit="frames",
            transient=True,
            show_done=True,
        ) as enc_stage:
            enc_handler = enc_stage.make_x265_handler(
                total_frames=metric_params.total_frames
            )

            _ = encode_x265_concatenated_bitrate(
                source_path=ctx.input_path,
                output_path=output_path,
                interval_frames=metric_params.interval_frames,
                region_frames=metric_params.region_frames,
                guard_start_frames=ctx.guard_start_frames,
                guard_end_frames=ctx.guard_end_frames,
                total_frames=ctx.total_frames,
                fps=ctx.info.fps,
                profile=pass2_profile,
                video_info=ctx.info,
                mkvmerge_bin=ctx.args.mkvmerge_bin,
                cwd=ctx.repo_root,
                temp_dir=ctx.temp_dir,
                stats_file=stats_file,
                analysis_file=analysis_file,
                line_handler=enc_handler,
                enable_autocrop=ctx.args.auto_crop,
                crop_values=ctx.crop_values,
                metric_label=metric_label,
            )
    else:
        # Single-pass encoding
        with ctx.display.stage(
            f"Encoding {metric_label} samples",
            total=metric_params.total_frames,
            unit="frames",
            transient=True,
            show_done=True,
        ) as enc_stage:
            enc_handler = enc_stage.make_x265_handler(
                total_frames=metric_params.total_frames
            )

            _ = encode_x265_concatenated_bitrate(
                source_path=ctx.input_path,
                output_path=output_path,
                interval_frames=metric_params.interval_frames,
                region_frames=metric_params.region_frames,
                guard_start_frames=ctx.guard_start_frames,
                guard_end_frames=ctx.guard_end_frames,
                total_frames=ctx.total_frames,
                fps=ctx.info.fps,
                profile=profile,
                video_info=ctx.info,
                mkvmerge_bin=ctx.args.mkvmerge_bin,
                cwd=ctx.repo_root,
                temp_dir=ctx.temp_dir,
                stats_file=None,
                analysis_file=None,
                line_handler=enc_handler,
                enable_autocrop=ctx.args.auto_crop,
                crop_values=ctx.crop_values,
                metric_label=metric_label,
            )

    return output_path
