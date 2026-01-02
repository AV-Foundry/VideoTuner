"""Validation utilities for the encoding pipeline."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline_cli import PipelineArgs
    from .ssimulacra2_assessment import SSIM2Result
    from .vmaf_assessment import VMAFResult

from .crf_search import QualityTarget


class AssessmentError(Exception):
    """Raised when assessment results are invalid or unavailable."""


@dataclass(frozen=True)
class SamplingValidation:
    """Result of validating sampling parameters for a metric.

    Attributes:
        is_valid: Whether the sampling configuration is valid
        num_samples: Number of samples that would be extracted
        total_frames: Total frames that would be assessed
        coverage_percent: Percentage of total video frames covered
        reason: Explanation if invalid (None if valid)
    """

    is_valid: bool
    num_samples: int
    total_frames: int
    coverage_percent: float
    reason: str | None = None


def validate_assessment_results(
    vmaf_results: list["VMAFResult"] | None,
    ssim2_results: list["SSIM2Result"] | None,
    context: str,
    log: logging.Logger,
) -> None:
    """Validate that assessment results contain valid scores.

    Args:
        vmaf_results: List of VMAF results (may be None)
        ssim2_results: List of SSIM2 results (may be None)
        context: Description of where validation is happening (for error messages)
        log: Logger for error messages

    Raises:
        AssessmentError: If assessment results are invalid or all scores are NaN
    """
    has_vmaf = vmaf_results is not None and len(vmaf_results) > 0
    has_ssim2 = ssim2_results is not None and len(ssim2_results) > 0

    if not has_vmaf and not has_ssim2:
        log.error("No assessment results available for %s", context)
        raise AssessmentError(
            f"Assessment failed: No VMAF or SSIMULACRA2 results available for {context}. Check the log for details."
        )

    # Check if VMAF results are all NaN
    if has_vmaf and vmaf_results is not None:
        all_nan = all(math.isnan(r.mean) for r in vmaf_results)
        if all_nan:
            log.error("All VMAF scores are NaN for %s", context)
            raise AssessmentError(
                f"Assessment failed: All VMAF scores are unparseable for {context}. Check the log for details."
            )

    # Check if SSIM2 results are all NaN
    if has_ssim2 and ssim2_results is not None:
        all_nan = all(math.isnan(r.mean) for r in ssim2_results)
        if all_nan:
            log.error("All SSIMULACRA2 scores are NaN for %s", context)
            raise AssessmentError(
                f"Assessment failed: All SSIMULACRA2 scores are unparseable for {context}. Check the log for details."
            )


def has_targets(args: "PipelineArgs") -> bool:
    """Check if any quality targets are specified.

    Args:
        args: Pipeline arguments from CLI

    Returns:
        True if at least one quality target is specified
    """
    return any(
        (
            args.vmaf_target is not None,
            args.vmaf_hmean_target is not None,
            args.vmaf_1pct_target is not None,
            args.vmaf_min_target is not None,
            args.ssim2_mean_target is not None,
            args.ssim2_median_target is not None,
            args.ssim2_95pct_target is not None,
            args.ssim2_5pct_target is not None,
        )
    )


def build_targets(args: "PipelineArgs") -> list[QualityTarget]:
    """Build list of quality targets from command-line arguments.

    Args:
        args: Pipeline arguments from CLI

    Returns:
        List of QualityTarget objects for all specified targets
    """
    decimals = args.metric_decimals
    targets: list[QualityTarget] = []
    if args.vmaf_target is not None:
        targets.append(
            QualityTarget("vmaf_mean", args.vmaf_target, metric_decimals=decimals)
        )
    if args.vmaf_hmean_target is not None:
        targets.append(
            QualityTarget(
                "vmaf_hmean", args.vmaf_hmean_target, metric_decimals=decimals
            )
        )
    if args.vmaf_1pct_target is not None:
        targets.append(
            QualityTarget("vmaf_1pct", args.vmaf_1pct_target, metric_decimals=decimals)
        )
    if args.vmaf_min_target is not None:
        targets.append(
            QualityTarget("vmaf_min", args.vmaf_min_target, metric_decimals=decimals)
        )
    if args.ssim2_mean_target is not None:
        targets.append(
            QualityTarget(
                "ssim2_mean", args.ssim2_mean_target, metric_decimals=decimals
            )
        )
    if args.ssim2_median_target is not None:
        targets.append(
            QualityTarget(
                "ssim2_median", args.ssim2_median_target, metric_decimals=decimals
            )
        )
    if args.ssim2_95pct_target is not None:
        targets.append(
            QualityTarget(
                "ssim2_95pct", args.ssim2_95pct_target, metric_decimals=decimals
            )
        )
    if args.ssim2_5pct_target is not None:
        targets.append(
            QualityTarget(
                "ssim2_5pct", args.ssim2_5pct_target, metric_decimals=decimals
            )
        )
    return targets


def check_scores_meet_targets(
    scores: dict[str, float | None], targets: list[QualityTarget]
) -> bool:
    """Check if given scores meet all quality targets.

    Uses each target's metric_decimals for consistent rounding.

    Args:
        scores: Dictionary of metric scores (may contain None values)
        targets: List of quality targets to check

    Returns:
        True if all targets are met, False otherwise
    """
    for target in targets:
        if target.metric_name not in scores:
            return False
        score = scores[target.metric_name]
        if score is None:
            return False
        if round(score, target.metric_decimals) < round(
            target.target_value, target.metric_decimals
        ):
            return False
    return True


def validate_metric_sampling(
    usable_frames: int,
    total_frames: int,
    interval_frames: int,
    region_frames: int,
    metric_name: str,
    log: logging.Logger,
) -> SamplingValidation:
    """Validate sampling parameters for a single metric.

    Calculates whether the given sampling configuration is valid and computes
    the resulting sample count and coverage.

    Args:
        usable_frames: Frames available after guard bands
        total_frames: Total frames in the video
        interval_frames: Frames between sample starts
        region_frames: Frames per sample region
        metric_name: Name of the metric (for logging)
        log: Logger instance

    Returns:
        SamplingValidation with validity status and computed metrics
    """
    # Check if region fits within usable frames
    if usable_frames < region_frames:
        log.warning(
            "Usable frames (%d) less than %s region size (%d) - disabling %s",
            usable_frames,
            metric_name.upper(),
            region_frames,
            metric_name.upper(),
        )
        return SamplingValidation(
            is_valid=False,
            num_samples=0,
            total_frames=0,
            coverage_percent=0.0,
            reason=f"Usable frames ({usable_frames}) less than region size ({region_frames})",
        )

    # Calculate number of samples and total frames
    num_samples = (usable_frames + interval_frames - region_frames) // interval_frames
    total_metric_frames = num_samples * region_frames

    if num_samples < 1:
        log.warning(
            "No %s samples possible with current parameters - disabling %s",
            metric_name.upper(),
            metric_name.upper(),
        )
        return SamplingValidation(
            is_valid=False,
            num_samples=0,
            total_frames=0,
            coverage_percent=0.0,
            reason="No samples possible with current parameters",
        )

    coverage_percent = (total_metric_frames / total_frames) * 100
    log.info(
        "%s periodic sampling: %d samples × %d frames = %d total frames (%.1f%% coverage)",
        metric_name.upper(),
        num_samples,
        region_frames,
        total_metric_frames,
        coverage_percent,
    )

    return SamplingValidation(
        is_valid=True,
        num_samples=num_samples,
        total_frames=total_metric_frames,
        coverage_percent=coverage_percent,
    )


def validate_sampling_parameters(
    args: "PipelineArgs",
    total_frames: int,
    guard_start_frames: int,
    guard_end_frames: int,
    log: logging.Logger,
) -> tuple[bool, bool]:
    """Validate sampling parameters for both VMAF and SSIM2 metrics.

    Checks if the sampling configuration is valid for each enabled metric
    and logs appropriate warnings if not.

    Args:
        args: Pipeline arguments containing metric flags and sampling params
        total_frames: Total frames in the video
        guard_start_frames: Frames excluded at start
        guard_end_frames: Frames excluded at end
        log: Logger instance

    Returns:
        Tuple of (vmaf_valid, ssim2_valid) indicating which metrics are valid
    """
    usable_frames = total_frames - guard_start_frames - guard_end_frames

    vmaf_valid = args.vmaf
    ssim2_valid = args.ssim2

    # Validate VMAF sampling
    if args.vmaf:
        validation = validate_metric_sampling(
            usable_frames=usable_frames,
            total_frames=total_frames,
            interval_frames=args.vmaf_interval_frames,
            region_frames=args.vmaf_region_frames,
            metric_name="vmaf",
            log=log,
        )
        vmaf_valid = validation.is_valid

    # Validate SSIM2 sampling
    if args.ssim2:
        validation = validate_metric_sampling(
            usable_frames=usable_frames,
            total_frames=total_frames,
            interval_frames=args.ssim2_interval_frames,
            region_frames=args.ssim2_region_frames,
            metric_name="ssim2",
            log=log,
        )
        ssim2_valid = validation.is_valid

    return (vmaf_valid, ssim2_valid)
