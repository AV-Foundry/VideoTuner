"""Display utilities for VideoTuner pipeline.

This module provides Rich console display functions for pipeline settings,
assessment summaries, and multi-profile comparison results.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

from rich.console import Console
from rich.table import Table

from .constants import METRIC_DECIMALS
from .crf_search import QualityTarget
from .pipeline_cli import DEFAULT_CRF_START_VALUE, DEFAULT_CRF_INTERVAL, get_default

if TYPE_CHECKING:
    from .pipeline_types import MultiProfileResult


class PipelineArgsProtocol(Protocol):
    """Protocol for pipeline arguments used in display functions."""

    assessment_only: bool
    multi_profile_search: list[str] | None
    profile: str | None
    preset: str | None
    auto_crop: bool
    vmaf: bool
    ssim2: bool
    vmaf_interval_frames: int
    vmaf_region_frames: int
    ssim2_interval_frames: int
    ssim2_region_frames: int
    vmaf_model: str | None
    tonemap: str
    guard_start_percent: float
    guard_end_percent: float
    guard_seconds: float
    predicted_bitrate_warning_percent: float | None
    vmaf_target: float | None
    vmaf_hmean_target: float | None
    vmaf_1pct_target: float | None
    vmaf_min_target: float | None
    ssim2_mean_target: float | None
    ssim2_median_target: float | None
    ssim2_95pct_target: float | None
    ssim2_5pct_target: float | None


def display_ignored_args_warnings(
    console: Console,
    log: logging.Logger,
    *,
    bitrate_profile_names: list[str],
    crf_start_value: float,
    crf_interval: float,
    has_targets: bool,
) -> None:
    """Display warnings for arguments that will be ignored when using bitrate profiles.

    Args:
        console: Rich console for output
        log: Logger for file logging
        bitrate_profile_names: Names of profiles that use bitrate mode (empty if none)
        crf_start_value: The CRF start value from args
        crf_interval: The CRF interval from args
        has_targets: True if any quality targets are specified
    """
    if not bitrate_profile_names:
        return

    warnings: list[str] = []
    profiles_str = ", ".join(bitrate_profile_names)

    # Check for explicitly-provided CRF arguments (differ from defaults)
    if crf_start_value != DEFAULT_CRF_START_VALUE:
        warnings.append(
            f"--crf-start-value {crf_start_value} will be ignored for bitrate profiles: {profiles_str}"
        )
    if crf_interval != DEFAULT_CRF_INTERVAL:
        warnings.append(
            f"--crf-interval {crf_interval} will be ignored for bitrate profiles: {profiles_str}"
        )

    # Check for targets (these are ignored for bitrate profiles)
    if has_targets:
        warnings.append(
            f"Quality targets will be ignored for bitrate profiles: {profiles_str}"
        )

    # Display warnings
    for warning in warnings:
        console.print(
            f"[bold yellow]⚠ Warning:[/bold yellow] [yellow]{warning}[/yellow]"
        )
        log.warning(warning)

    if warnings:
        console.print()


def display_settings_summary(
    console: Console,
    args: PipelineArgsProtocol,
    multi_profile_display: str = "",
    source_name: str = "",
) -> None:
    """Display active settings summary to console.

    Args:
        console: Rich console for output
        args: Pipeline arguments
        multi_profile_display: Pre-formatted string showing profiles/groups for multi-profile mode
        source_name: Name of the source video file
    """
    console.print()
    console.print("[bold]Settings[/bold]")

    if args.assessment_only:
        console.print("  Mode: Assessment Only")
        console.print(f"  Profile/Preset: {args.profile or f'preset-{args.preset}'}")
    elif args.multi_profile_search:
        console.print("  Mode: Multi-Profile Search")
        console.print(f"  Profiles: {multi_profile_display}")
    else:
        console.print("  Mode: CRF Search")
        console.print(f"  Profile/Preset: {args.profile or f'preset-{args.preset}'}")

    # Build assessments string
    assessments: list[str] = []
    if args.vmaf:
        assessments.append("VMAF")
    if args.ssim2:
        assessments.append("SSIMULACRA2")
    console.print(f"  Assessments: {', '.join(assessments)}")

    console.print(f"  AutoCrop: {'Enabled' if args.auto_crop else 'Disabled'}")

    # Log non-default sampling parameters
    if args.vmaf_interval_frames != get_default("vmaf_interval_frames"):
        console.print(f"  VMAF Interval: {args.vmaf_interval_frames} frames")
    if args.vmaf_region_frames != get_default("vmaf_region_frames"):
        console.print(f"  VMAF Region: {args.vmaf_region_frames} frames")
    if args.ssim2_interval_frames != get_default("ssim2_interval_frames"):
        console.print(f"  SSIM2 Interval: {args.ssim2_interval_frames} frames")
    if args.ssim2_region_frames != get_default("ssim2_region_frames"):
        console.print(f"  SSIM2 Region: {args.ssim2_region_frames} frames")

    # Log non-default analysis options
    if args.vmaf_model is not None:
        console.print(f"  VMAF Model: {args.vmaf_model}")
    if args.tonemap != get_default("tonemap"):
        console.print(f"  Tonemap: {args.tonemap}")

    # Log non-default guard bands
    if args.guard_start_percent > 0:
        console.print(f"  Guard Start: {args.guard_start_percent * 100:.1f}%")
    if args.guard_end_percent > 0:
        console.print(f"  Guard End: {args.guard_end_percent * 100:.1f}%")
    if args.guard_seconds > 0:
        console.print(f"  Guard Seconds: {args.guard_seconds:.1f}s")

    # Log bitrate warning if specified
    if args.predicted_bitrate_warning_percent is not None:
        console.print(
            f"  Bitrate Warning: {args.predicted_bitrate_warning_percent:.0f}%"
        )

    # Log quality targets if specified
    target_parts: list[str] = []
    if args.vmaf_target is not None:
        target_parts.append(f"VMAF Mean ≥ {args.vmaf_target}")
    if args.vmaf_hmean_target is not None:
        target_parts.append(f"VMAF HMean ≥ {args.vmaf_hmean_target}")
    if args.vmaf_1pct_target is not None:
        target_parts.append(f"VMAF 1% ≥ {args.vmaf_1pct_target}")
    if args.vmaf_min_target is not None:
        target_parts.append(f"VMAF Min ≥ {args.vmaf_min_target}")
    if args.ssim2_mean_target is not None:
        target_parts.append(f"SSIM2 Mean ≥ {args.ssim2_mean_target}")
    if args.ssim2_median_target is not None:
        target_parts.append(f"SSIM2 Median ≥ {args.ssim2_median_target}")
    if args.ssim2_95pct_target is not None:
        target_parts.append(f"SSIM2 95% ≥ {args.ssim2_95pct_target}")
    if args.ssim2_5pct_target is not None:
        target_parts.append(f"SSIM2 5% ≥ {args.ssim2_5pct_target}")

    if target_parts:
        console.print()
        console.print("[bold]Targets[/bold]")
        for target in target_parts:
            console.print(f"  {target}")

    if source_name:
        console.print()
        console.print(f"[bold]Source:[/bold] {source_name}")
        console.print()


def display_assessment_summary(
    console: Console,
    scores: dict[str, float | None],
    targets: list[QualityTarget] | None = None,
    iteration: int | None = None,
    targets_only: bool = False,
    custom_title: str | None = None,
    metric_decimals: int = METRIC_DECIMALS,
) -> None:
    """Display assessment summary with optional target annotations.

    Args:
        console: Rich console for output
        scores: Dictionary of metric scores
        targets: Optional list of quality targets (None for exploration mode)
        iteration: Optional iteration number (None for final display)
        targets_only: If True, only show metrics that are targets (for iteration displays)
        custom_title: Optional custom title (overrides iteration-based title)
        metric_decimals: Decimal places for metric display (default: METRIC_DECIMALS)
    """
    # Build target lookup
    target_map = {}
    if targets:
        target_map: dict[str, QualityTarget] = {t.metric_name: t for t in targets}

    # Metric display configuration: (metric_key, full_display_name)
    metric_display = [
        # VMAF metrics
        ("vmaf_mean", "VMAF Mean"),
        ("vmaf_hmean", "VMAF Harmonic Mean"),
        ("vmaf_1pct", "VMAF 1% Low"),
        ("vmaf_min", "VMAF Minimum"),
        # SSIMULACRA2 metrics
        ("ssim2_mean", "SSIMULACRA2 Mean"),
        ("ssim2_median", "SSIMULACRA2 Median"),
        ("ssim2_95pct", "SSIMULACRA2 95% High"),
        ("ssim2_5pct", "SSIMULACRA2 5% Low"),
    ]

    # Determine title
    console.print()
    if custom_title is not None:
        title = custom_title
    elif iteration is not None:
        if targets_only:
            title = f"Targets: Iteration {iteration}"
        else:
            title = f"Assessment Summary: Iteration {iteration}"
    else:
        title = "Assessment Summary"

    # Create single table
    table = Table(
        title=f"[bold cyan]{title}[/bold cyan]",
        show_header=True,
        header_style="bold cyan",
        title_justify="left",
    )

    # Determine if we have targets to show
    has_targets = targets is not None and len(targets) > 0

    table.add_column("Metric", style="white")
    table.add_column("Value", justify="right")
    if has_targets:
        table.add_column("Target Met?", justify="center")
        table.add_column("Delta", justify="right")

    # Add rows for all metrics
    has_rows = False
    for metric_key, display_name in metric_display:
        if metric_key not in scores or scores[metric_key] is None:
            continue

        target = target_map.get(metric_key)

        # Skip non-targets if targets_only mode
        if targets_only and target is None:
            continue

        value = scores[metric_key]
        has_rows = True

        if target is not None and has_targets:
            # This is a target metric - use target's metric_decimals
            decimals = target.metric_decimals
            delta = target.delta()
            delta_str = f"{delta:+.{decimals}f}" if delta is not None else "N/A"

            if target.is_met():
                # Target met: green
                table.add_row(
                    display_name,
                    f"[bold green]{value:.{decimals}f}[/bold green]",
                    "[green]✓[/green]",
                    delta_str,
                )
            else:
                # Target not met: red
                table.add_row(
                    display_name,
                    f"[bold red]{value:.{decimals}f}[/bold red]",
                    "[red]✗[/red]",
                    delta_str,
                )
        else:
            # Not a target: default color (only metric and value when no targets)
            table.add_row(display_name, f"[cyan]{value:.{metric_decimals}f}[/cyan]")

    # Only print table if it has rows
    if has_rows:
        console.print(table)

    console.print()


def format_bitrate_percentage(
    predicted_bitrate_kbps: float,
    input_bitrate_kbps: float | None,
) -> str:
    """Format predicted bitrate with percentage of input if available.

    Args:
        predicted_bitrate_kbps: Predicted output bitrate
        input_bitrate_kbps: Input video bitrate (None if unavailable)

    Returns:
        Formatted string with bitrate and percentage (if available)
    """
    base_str = f"{predicted_bitrate_kbps:,.0f} kbps"

    if (
        input_bitrate_kbps is not None
        and input_bitrate_kbps > 0
        and predicted_bitrate_kbps > 0
    ):
        bitrate_percent = (predicted_bitrate_kbps / input_bitrate_kbps) * 100.0
        return f"{base_str} ({bitrate_percent:.1f}% of input)"

    return base_str


def check_and_display_bitrate_warning(
    console: Console,
    log: logging.Logger,
    predicted_bitrate_kbps: float,
    input_bitrate_kbps: float | None,
    threshold_percent: float | None,
    profile_name: str | None = None,
) -> None:
    """Check if predicted bitrate exceeds threshold and display warning.

    Args:
        console: Rich console for output
        log: Logger for file logging
        predicted_bitrate_kbps: Predicted output bitrate
        input_bitrate_kbps: Input video bitrate (None if unavailable)
        threshold_percent: Warning threshold as percentage (1-100, None if disabled)
        profile_name: Optional profile name for warning message
    """
    if threshold_percent is None:
        return  # Feature disabled

    if input_bitrate_kbps is None or input_bitrate_kbps <= 0:
        log.debug("Predicted bitrate warning skipped: input bitrate unavailable")
        return

    if predicted_bitrate_kbps <= 0:
        log.debug("Predicted bitrate warning skipped: predicted bitrate unavailable")
        return

    # Calculate percentage
    bitrate_percent = (predicted_bitrate_kbps / input_bitrate_kbps) * 100.0

    if bitrate_percent > threshold_percent:
        profile_str = f" ({profile_name})" if profile_name else ""
        console.print()
        console.print(
            f"[bold yellow]⚠ Warning: Predicted bitrate{profile_str} "
            + f"({predicted_bitrate_kbps:,.0f} kbps) exceeds {threshold_percent:.0f}% "
            + f"of input bitrate ({input_bitrate_kbps:,.0f} kbps)[/bold yellow]"
        )
        console.print(f"[yellow]  Predicted: {bitrate_percent:.1f}% of input[/yellow]")
        log.warning(
            "Predicted bitrate%s (%.0f kbps) exceeds threshold: %.1f%% of input (%.0f kbps)",
            profile_str,
            predicted_bitrate_kbps,
            bitrate_percent,
            input_bitrate_kbps,
        )


def display_multi_profile_results(
    console: Console,
    results: list[MultiProfileResult],
    targets: list[QualityTarget],
    metric_decimals: int = METRIC_DECIMALS,
) -> None:
    """Display ranked multi-profile search results (transposed: metrics as rows, profiles as columns).

    Args:
        console: Rich console for output
        results: Sorted list of profile results (best first)
        targets: Quality targets that were set
        metric_decimals: Decimal places for metric display (default: METRIC_DECIMALS)
    """
    if not results:
        return

    # Determine which metrics to display based on available data
    metric_keys: list[str] = []
    metric_names: list[str] = []

    # Check which metrics are present in results
    if results and results[0].scores:
        sample_scores = results[0].scores

        # VMAF metrics
        if "vmaf_mean" in sample_scores:
            metric_keys.append("vmaf_mean")
            metric_names.append("VMAF Mean")
        if "vmaf_hmean" in sample_scores:
            metric_keys.append("vmaf_hmean")
            metric_names.append("VMAF Harmonic")
        if "vmaf_1pct" in sample_scores:
            metric_keys.append("vmaf_1pct")
            metric_names.append("VMAF 1% Low")
        if "vmaf_min" in sample_scores:
            metric_keys.append("vmaf_min")
            metric_names.append("VMAF Min")

        # SSIMULACRA2 metrics
        if "ssim2_mean" in sample_scores:
            metric_keys.append("ssim2_mean")
            metric_names.append("SSIMULACRA2 Mean")
        if "ssim2_median" in sample_scores:
            metric_keys.append("ssim2_median")
            metric_names.append("SSIM2 Median")
        if "ssim2_95pct" in sample_scores:
            metric_keys.append("ssim2_95pct")
            metric_names.append("SSIM2 95% High")
        if "ssim2_5pct" in sample_scores:
            metric_keys.append("ssim2_5pct")
            metric_names.append("SSIM2 5% Low")

    # Create table (transposed: metrics as rows, profiles as columns)
    table = Table(
        title="[bold cyan]Multi-Profile Comparison Results[/bold cyan]",
        show_header=True,
        header_style="bold cyan",
        title_justify="left",
    )

    # Check if we have any CRF profiles (needed for winner highlighting)
    has_crf_profiles = any(result.optimal_crf is not None for result in results)

    # Add columns: Metric name + one column per profile
    table.add_column("Metric", style="white", min_width=18)
    for rank, result in enumerate(results, 1):
        is_winner = (
            rank == 1 and has_crf_profiles
        )  # Only highlight winner for CRF profiles
        profile_header = f"#{rank} {result.profile_name}"
        if is_winner:
            table.add_column(
                profile_header, justify="right", style="bold green", min_width=12
            )
        else:
            table.add_column(profile_header, justify="right", min_width=12)

    # Add rows: one row per metric
    has_checkmarks = False
    for _metric_idx, (metric_key, metric_name) in enumerate(
        zip(metric_keys, metric_names)
    ):
        row_values = [metric_name]

        # Check if this metric is a target
        is_target = any(t.metric_name == metric_key for t in targets)

        # Get decimals from target if this is a target metric, else use parameter
        target_for_metric = next(
            (t for t in targets if t.metric_name == metric_key), None
        )
        decimals = (
            target_for_metric.metric_decimals if target_for_metric else metric_decimals
        )

        for rank, result in enumerate(results, 1):
            value = result.scores.get(metric_key)

            if value is not None:
                formatted_value = f"{value:.{decimals}f}"
                # Add checkmark if target is met (CRF profiles only)
                if is_target and not result.is_bitrate_mode:
                    target = next(
                        (t for t in targets if t.metric_name == metric_key), None
                    )
                    if target and value >= target.target_value:
                        formatted_value = f"✓ {formatted_value}"
                        has_checkmarks = True
                row_values.append(formatted_value)
            else:
                row_values.append("-")

        table.add_row(*row_values)

    # Add separator row (empty)
    if metric_keys:
        table.add_row(*[""] * (len(results) + 1))

    # Add Avg Bitrate row
    row_values: list[str] = ["Predicted Bitrate (kbps)"]
    for result in results:
        row_values.append(f"{result.predicted_bitrate_kbps:,.0f}")
    table.add_row(*row_values)

    # Add Mode row (CRF or Bitrate)
    row_values = ["Mode"]
    for result in results:
        if result.optimal_crf is not None:
            row_values.append("CRF")
        else:
            row_values.append("Bitrate")
    table.add_row(*row_values)

    # Add Optimal CRF row (only if at least one profile is CRF mode)
    if has_crf_profiles:
        row_values = ["Optimal CRF"]
        for result in results:
            if result.optimal_crf is not None:
                row_values.append(f"{result.optimal_crf:.1f}")
            else:
                row_values.append("-")
        table.add_row(*row_values)

    # Add Targets Met row (only if there are targets and at least one CRF profile)
    if targets and has_crf_profiles:
        row_values = ["All Targets Met"]
        for result in results:
            if result.is_bitrate_mode:
                # Bitrate profile - targets don't apply
                row_values.append("-")
            elif result.meets_all_targets is True:
                row_values.append("✓")
            else:
                row_values.append("✗")
        table.add_row(*row_values)

    console.print()
    console.print(table)

    # Show appropriate ranking explanation based on profile types
    if has_crf_profiles and targets:
        console.print(
            "[dim]Ranked by target achievement, then lowest predicted bitrate (winner highlighted)[/dim]"
        )
    elif has_crf_profiles:
        console.print(
            "[dim]Ranked by lowest predicted bitrate (winner highlighted)[/dim]"
        )
    else:
        # Bitrate-only profiles
        console.print("[dim]Sorted by predicted bitrate[/dim]")

    # Show key legend only when checkmarks are displayed
    if has_checkmarks:
        console.print("[dim]Key: ✓ = Target met[/dim]")
