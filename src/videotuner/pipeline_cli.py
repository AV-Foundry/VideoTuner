"""CLI argument parsing and validation for VideoTuner pipeline.

This module provides command-line argument parsing and validation functionality
for the VideoTuner pipeline, including the PipelineArgs dataclass, parser
construction, and comprehensive argument validation.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass, fields, MISSING
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .profiles import Profile


# Default values exposed as constants for use in validation/warnings
DEFAULT_CRF_START_VALUE: float = 28.0
DEFAULT_CRF_INTERVAL: float = 0.5


@dataclass
class PipelineArgs:
    input: Path
    output: Path | None

    # Periodic sampling parameters
    vmaf_interval_frames: int = 1600
    vmaf_region_frames: int = 20
    ssim2_interval_frames: int = 1600
    ssim2_region_frames: int = 20

    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    mkvmerge_bin: str = "mkvmerge"

    profile: str | None = None
    preset: str | None = None  # None means use default "slow"
    crf_start_value: float = DEFAULT_CRF_START_VALUE
    crf_interval: float = DEFAULT_CRF_INTERVAL

    # Mode flags
    assessment_only: bool = False
    multi_profile_search: list[str] | None = None

    vmaf_target: float | None = None
    vmaf_hmean_target: float | None = None
    vmaf_1pct_target: float | None = None
    vmaf_min_target: float | None = None

    ssim2_mean_target: float | None = None
    ssim2_median_target: float | None = None
    ssim2_95pct_target: float | None = None
    ssim2_5pct_target: float | None = None

    workdir: Path | None = None

    vmaf: bool = True
    vmaf_model: str | None = None
    vmaf_log: Path | None = None
    tonemap: str = "auto"

    guard_start_percent: float = 0.0
    guard_end_percent: float = 0.0
    guard_seconds: float = 0.0

    ssim2: bool = True
    ssim2_log: Path | None = None
    vs_dir: Path | None = None
    vs_plugin_dir: Path | None = None
    auto_crop: bool = True
    predicted_bitrate_warning_percent: float | None = None

    log_file: str | Path | None = None
    quiet: bool = False
    verbose: bool = False


def get_default(field_name: str) -> object:
    """Get default value from PipelineArgs dataclass field."""
    for field in fields(PipelineArgs):
        if field.name == field_name:
            # Field defaults can be Any type by design
            return field.default if field.default is not MISSING else None  # pyright: ignore[reportAny]
    raise ValueError(f"Field {field_name} not found in PipelineArgs")


# Internal alias for use within this module
_get_default = get_default


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CRF optimization and encoder benchmarking using VMAF and SSIMULACRA2 quality metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Positional arguments
    _ = p.add_argument("input", type=Path, help="Input video path")
    _ = p.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="Output directory (default: jobs/<name>_<timestamp>)",
    )

    # -------------------------------------------------------------------------
    # Mode Selection
    # -------------------------------------------------------------------------
    mode_group = p.add_argument_group("Mode Selection")
    _ = mode_group.add_argument(
        "--assessment-only",
        action="store_true",
        help="Single assessment at starting CRF without CRF search",
    )
    _ = mode_group.add_argument(
        "--multi-profile-search",
        type=str,
        metavar="PROFILES",
        default=_get_default("multi_profile_search"),
        help="Compare multiple profiles/groups (comma-separated)",
    )

    # -------------------------------------------------------------------------
    # Encoding Options
    # -------------------------------------------------------------------------
    encoding_group = p.add_argument_group("Encoding Options")
    _ = encoding_group.add_argument(
        "--preset",
        type=str,
        default=_get_default("preset"),
        choices=[
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
        ],
        help="x265 preset (mutually exclusive with --profile)",
    )
    _ = encoding_group.add_argument(
        "--profile",
        type=str,
        metavar="NAME",
        default=_get_default("profile"),
        help="Profile name from x265_profiles.yaml (mutually exclusive with --preset)",
    )
    _ = encoding_group.add_argument(
        "--crf-start-value",
        type=float,
        metavar="CRF",
        default=_get_default("crf_start_value"),
        help=f"Starting CRF for search (default: {_get_default('crf_start_value')})",
    )
    _ = encoding_group.add_argument(
        "--crf-interval",
        type=float,
        metavar="STEP",
        default=_get_default("crf_interval"),
        help=f"Minimum CRF step size (default: {_get_default('crf_interval')})",
    )
    _ = encoding_group.add_argument(
        "--no-autocrop",
        action="store_false",
        dest="auto_crop",
        help="Disable automatic crop detection",
    )

    # -------------------------------------------------------------------------
    # VMAF Targets
    # -------------------------------------------------------------------------
    vmaf_target_group = p.add_argument_group("VMAF Targets")
    _ = vmaf_target_group.add_argument(
        "--vmaf-target",
        type=float,
        metavar="SCORE",
        default=_get_default("vmaf_target"),
        help="Target VMAF mean score",
    )
    _ = vmaf_target_group.add_argument(
        "--vmaf-hmean-target",
        type=float,
        metavar="SCORE",
        default=_get_default("vmaf_hmean_target"),
        help="Target VMAF harmonic mean score",
    )
    _ = vmaf_target_group.add_argument(
        "--vmaf-1pct-target",
        type=float,
        metavar="SCORE",
        default=_get_default("vmaf_1pct_target"),
        help="Target VMAF 1%% low score",
    )
    _ = vmaf_target_group.add_argument(
        "--vmaf-min-target",
        type=float,
        metavar="SCORE",
        default=_get_default("vmaf_min_target"),
        help="Target VMAF minimum score",
    )

    # -------------------------------------------------------------------------
    # SSIMULACRA2 Targets
    # -------------------------------------------------------------------------
    ssim2_target_group = p.add_argument_group("SSIMULACRA2 Targets")
    _ = ssim2_target_group.add_argument(
        "--ssim2-mean-target",
        type=float,
        metavar="SCORE",
        default=_get_default("ssim2_mean_target"),
        help="Target SSIMULACRA2 mean score",
    )
    _ = ssim2_target_group.add_argument(
        "--ssim2-median-target",
        type=float,
        metavar="SCORE",
        default=_get_default("ssim2_median_target"),
        help="Target SSIMULACRA2 median score",
    )
    _ = ssim2_target_group.add_argument(
        "--ssim2-95pct-target",
        type=float,
        metavar="SCORE",
        default=_get_default("ssim2_95pct_target"),
        help="Target SSIMULACRA2 95%% high score",
    )
    _ = ssim2_target_group.add_argument(
        "--ssim2-5pct-target",
        type=float,
        metavar="SCORE",
        default=_get_default("ssim2_5pct_target"),
        help="Target SSIMULACRA2 5%% low score",
    )

    # -------------------------------------------------------------------------
    # Sampling Parameters
    # -------------------------------------------------------------------------
    sampling_group = p.add_argument_group("Sampling Parameters")
    _ = sampling_group.add_argument(
        "--vmaf-interval-frames",
        type=int,
        metavar="N",
        default=_get_default("vmaf_interval_frames"),
        help=f"Sample every N frames for VMAF (default: {_get_default('vmaf_interval_frames')})",
    )
    _ = sampling_group.add_argument(
        "--vmaf-region-frames",
        type=int,
        metavar="N",
        default=_get_default("vmaf_region_frames"),
        help=f"Consecutive frames per VMAF sample (default: {_get_default('vmaf_region_frames')})",
    )
    _ = sampling_group.add_argument(
        "--ssim2-interval-frames",
        type=int,
        metavar="N",
        default=_get_default("ssim2_interval_frames"),
        help=f"Sample every N frames for SSIM2 (default: {_get_default('ssim2_interval_frames')})",
    )
    _ = sampling_group.add_argument(
        "--ssim2-region-frames",
        type=int,
        metavar="N",
        default=_get_default("ssim2_region_frames"),
        help=f"Consecutive frames per SSIM2 sample (default: {_get_default('ssim2_region_frames')})",
    )

    # -------------------------------------------------------------------------
    # Analysis Options
    # -------------------------------------------------------------------------
    analysis_group = p.add_argument_group("Analysis Options")
    _ = analysis_group.add_argument(
        "--no-vmaf", dest="vmaf", action="store_false", help="Disable VMAF assessment"
    )
    _ = analysis_group.add_argument(
        "--no-ssim2",
        dest="ssim2",
        action="store_false",
        help="Disable SSIMULACRA2 assessment",
    )
    _ = analysis_group.add_argument(
        "--vmaf-model",
        type=str,
        metavar="MODEL",
        default=_get_default("vmaf_model"),
        help="VMAF model name or path (default: auto-select by resolution)",
    )
    _ = analysis_group.add_argument(
        "--tonemap",
        choices=["auto", "force", "off"],
        default=_get_default("tonemap"),
        help="HDR tonemapping: auto|force|off (default: auto)",
    )
    p.set_defaults(vmaf=_get_default("vmaf"), ssim2=_get_default("ssim2"))

    # -------------------------------------------------------------------------
    # Guard Bands
    # -------------------------------------------------------------------------
    guard_group = p.add_argument_group("Guard Bands")
    _ = guard_group.add_argument(
        "--guard-start-percent",
        type=float,
        metavar="PCT",
        default=_get_default("guard_start_percent"),
        help=f"Exclude head percentage (default: {_get_default('guard_start_percent')})",
    )
    _ = guard_group.add_argument(
        "--guard-end-percent",
        type=float,
        metavar="PCT",
        default=_get_default("guard_end_percent"),
        help=f"Exclude tail percentage (default: {_get_default('guard_end_percent')})",
    )
    _ = guard_group.add_argument(
        "--guard-seconds",
        type=float,
        metavar="SEC",
        default=_get_default("guard_seconds"),
        help=f"Minimum guard in seconds per side (default: {_get_default('guard_seconds')})",
    )

    # -------------------------------------------------------------------------
    # Bitrate Warning
    # -------------------------------------------------------------------------
    bitrate_group = p.add_argument_group("Bitrate Warning")
    _ = bitrate_group.add_argument(
        "--predicted-bitrate-warning-percent",
        type=float,
        metavar="PCT",
        default=_get_default("predicted_bitrate_warning_percent"),
        help="Warn if output exceeds this %% of input bitrate (1-100)",
    )

    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    paths_group = p.add_argument_group("Paths")
    _ = paths_group.add_argument(
        "--workdir",
        type=Path,
        metavar="DIR",
        default=_get_default("workdir"),
        help="Working directory (default: jobs/<name>_<timestamp>)",
    )
    _ = paths_group.add_argument(
        "--ffmpeg",
        dest="ffmpeg_bin",
        metavar="PATH",
        default=_get_default("ffmpeg_bin"),
        help="FFmpeg binary path",
    )
    _ = paths_group.add_argument(
        "--ffprobe",
        dest="ffprobe_bin",
        metavar="PATH",
        default=_get_default("ffprobe_bin"),
        help="FFprobe binary path",
    )
    _ = paths_group.add_argument(
        "--mkvmerge",
        dest="mkvmerge_bin",
        metavar="PATH",
        default=_get_default("mkvmerge_bin"),
        help="MKVmerge binary path",
    )
    _ = paths_group.add_argument(
        "--vs-dir",
        type=Path,
        metavar="DIR",
        default=_get_default("vs_dir"),
        help="VapourSynth portable directory",
    )
    _ = paths_group.add_argument(
        "--vs-plugin-dir",
        type=Path,
        metavar="DIR",
        default=_get_default("vs_plugin_dir"),
        help="VapourSynth plugin directory",
    )
    _ = paths_group.add_argument(
        "--vmaf-log",
        type=Path,
        metavar="PATH",
        default=_get_default("vmaf_log"),
        help="VMAF JSON log path (default: under workdir)",
    )
    _ = paths_group.add_argument(
        "--ssim2-log",
        type=Path,
        metavar="PATH",
        default=_get_default("ssim2_log"),
        help="SSIMULACRA2 JSON log path (default: under workdir)",
    )

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------
    logging_group = p.add_argument_group("Logging")
    _ = logging_group.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    _ = logging_group.add_argument(
        "-q", "--quiet", action="store_true", help="Reduce logging (warnings only)"
    )
    _ = logging_group.add_argument(
        "--log-file",
        type=str,
        metavar="PATH",
        nargs="?",
        const="",
        default=_get_default("log_file"),
        help="Write summary log to file (default: <workdir>/<name>.log)",
    )

    return p


def parse_cli(argv: Iterable[str] | None = None) -> PipelineArgs:
    parser = build_arg_parser()
    argv_list: list[str] | None = list(argv) if argv is not None else None
    parsed = parser.parse_args(argv_list)

    # Convert comma-separated multi_profile_search to list
    mps = getattr(parsed, "multi_profile_search", None)
    if isinstance(mps, str):
        parsed.multi_profile_search = [g.strip() for g in mps.split(",") if g.strip()]

    return PipelineArgs(**vars(parsed))  # pyright: ignore[reportAny]


# =============================================================================
# Argument Validation
# =============================================================================


@dataclass
class ValidationResult:
    """Result of argument validation with resolved profiles."""

    selected_profile: "Profile | None"  # None when using multi-profile search
    multi_profile_list: list["Profile"]
    multi_profile_display: str
    has_quality_targets: bool


def validate_time_args(args: PipelineArgs, parser: argparse.ArgumentParser) -> None:
    """Validate time-based and guard band arguments."""
    if args.guard_seconds < 0:
        parser.error(f"--guard-seconds must be non-negative (got {args.guard_seconds})")
    if args.guard_start_percent < 0:
        parser.error(
            f"--guard-start-percent must be non-negative (got {args.guard_start_percent})"
        )
    if args.guard_end_percent < 0:
        parser.error(
            f"--guard-end-percent must be non-negative (got {args.guard_end_percent})"
        )


def validate_sampling_args(args: PipelineArgs, parser: argparse.ArgumentParser) -> None:
    """Validate periodic sampling parameters."""
    from .constants import MAX_COMBINED_GUARD_PERCENT

    periodic_sampling_args = [
        ("--vmaf-interval-frames", args.vmaf_interval_frames),
        ("--vmaf-region-frames", args.vmaf_region_frames),
        ("--ssim2-interval-frames", args.ssim2_interval_frames),
        ("--ssim2-region-frames", args.ssim2_region_frames),
    ]
    for arg_name, arg_value in periodic_sampling_args:
        if arg_value < 1:
            parser.error(f"{arg_name} must be at least 1 (got {arg_value})")

    if args.guard_start_percent + args.guard_end_percent >= MAX_COMBINED_GUARD_PERCENT:
        parser.error(
            f"--guard-start-percent + --guard-end-percent must be less than {MAX_COMBINED_GUARD_PERCENT} (got {args.guard_start_percent + args.guard_end_percent:.2f})"
        )


def validate_metric_args(args: PipelineArgs, parser: argparse.ArgumentParser) -> None:
    """Validate metric enable/disable flags."""
    if not args.vmaf and not args.ssim2:
        parser.error(
            "At least one metric must be enabled. Cannot use both --no-vmaf and --no-ssim2 together."
        )


def validate_crf_args(args: PipelineArgs, parser: argparse.ArgumentParser) -> None:
    """Validate CRF search parameters."""
    from .constants import CRF_MIN, CRF_MAX

    if args.crf_interval <= 0:
        parser.error(f"--crf-interval must be greater than 0 (got {args.crf_interval})")
    if args.crf_start_value < CRF_MIN or args.crf_start_value > CRF_MAX:
        parser.error(
            f"--crf-start-value must be between {CRF_MIN:.0f} and {CRF_MAX:.0f} (got {args.crf_start_value})"
        )


def validate_target_args(args: PipelineArgs, parser: argparse.ArgumentParser) -> None:
    """Validate quality targets vs enabled metrics."""
    vmaf_targets = [
        args.vmaf_target,
        args.vmaf_hmean_target,
        args.vmaf_1pct_target,
        args.vmaf_min_target,
    ]
    ssim2_targets = [
        args.ssim2_mean_target,
        args.ssim2_median_target,
        args.ssim2_5pct_target,
    ]

    if any(t is not None for t in vmaf_targets) and not args.vmaf:
        parser.error(
            "VMAF targets require VMAF to be enabled. Remove --no-vmaf or remove VMAF targets."
        )
    if any(t is not None for t in ssim2_targets) and not args.ssim2:
        parser.error(
            "SSIMULACRA2 targets require SSIMULACRA2 to be enabled. Remove --no-ssim2 or remove SSIM2 targets."
        )


def validate_mode_args(
    args: PipelineArgs, parser: argparse.ArgumentParser, has_quality_targets: bool
) -> None:
    """Validate mode-related argument combinations."""
    if args.assessment_only and has_quality_targets:
        parser.error(
            "--assessment-only cannot be used with quality targets. Remove targets or remove --assessment-only."
        )

    if args.assessment_only and args.multi_profile_search:
        parser.error(
            "--assessment-only and --multi-profile-search are mutually exclusive. Use --assessment-only for a single encode, or --multi-profile-search for profile comparison."
        )

    if args.profile is not None and args.preset is not None:
        parser.error(
            "--preset and --profile are mutually exclusive. Use one or the other."
        )

    if args.multi_profile_search and args.profile is not None:
        parser.error(
            "--profile and --multi-profile-search are mutually exclusive. Use --multi-profile-search to specify profiles to compare."
        )

    if args.multi_profile_search and args.preset is not None:
        parser.error(
            "--preset and --multi-profile-search are mutually exclusive. Use --multi-profile-search to specify profiles to compare."
        )


def validate_bitrate_warning_args(
    args: PipelineArgs, parser: argparse.ArgumentParser
) -> None:
    """Validate bitrate warning percentage range."""
    from .constants import BITRATE_WARNING_PERCENT_MIN, BITRATE_WARNING_PERCENT_MAX

    if args.predicted_bitrate_warning_percent is not None:
        if not (
            BITRATE_WARNING_PERCENT_MIN
            <= args.predicted_bitrate_warning_percent
            <= BITRATE_WARNING_PERCENT_MAX
        ):
            parser.error(
                f"--predicted-bitrate-warning-percent must be between {BITRATE_WARNING_PERCENT_MIN:.0f} and {BITRATE_WARNING_PERCENT_MAX:.0f} (got {args.predicted_bitrate_warning_percent})"
            )


def _resolve_multi_profile_search(
    args: PipelineArgs,
    parser: argparse.ArgumentParser,
    profiles: dict[str, "Profile"],
) -> tuple[list["Profile"], str]:
    """Resolve multi-profile-search references to profile objects.

    Returns:
        Tuple of (profile_list, display_string)
    """
    from .profiles import get_all_groups, get_profiles_by_groups

    multi_profile_list: list["Profile"] = []
    all_groups = get_all_groups(profiles)

    passed_groups: list[tuple[str, list[str]]] = []
    passed_profiles: list[str] = []

    for name in args.multi_profile_search or []:
        if name in profiles:
            multi_profile_list.append(profiles[name])
            passed_profiles.append(name)
        elif name in all_groups:
            group_profiles = get_profiles_by_groups(profiles, [name])
            multi_profile_list.extend(group_profiles)
            passed_groups.append((name, [p.name for p in group_profiles]))
        else:
            available_profiles = ", ".join(profiles.keys())
            available_groups = ", ".join(sorted(all_groups)) if all_groups else "(none)"
            error_msg = "\n".join(
                [
                    f"'{name}' is not a valid profile or group name.",
                    f"Available profiles: {available_profiles}",
                    f"Available groups: {available_groups}",
                ]
            )
            parser.error(error_msg)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_profiles: list["Profile"] = []
    for p in multi_profile_list:
        if p.name not in seen:
            seen.add(p.name)
            unique_profiles.append(p)
    multi_profile_list = unique_profiles

    # Build display string
    display_parts: list[str] = []
    for group_name, profile_names in passed_groups:
        display_parts.append(f"{group_name} ({', '.join(profile_names)})")
    if passed_profiles:
        display_parts.append(", ".join(passed_profiles))
    multi_profile_display = ", ".join(display_parts) if display_parts else ""

    if len(multi_profile_list) < 2:
        parser.error(
            f"--multi-profile-search requires at least 2 profiles, but only {len(multi_profile_list)} resolved. Specify more profiles or groups."
        )

    return multi_profile_list, multi_profile_display


def _validate_target_requirements(
    args: PipelineArgs,
    parser: argparse.ArgumentParser,
    has_quality_targets: bool,
    multi_profile_list: list["Profile"],
) -> None:
    """Validate that required targets are specified based on mode."""
    if args.assessment_only:
        return  # No targets required

    if has_quality_targets:
        return  # Targets are specified

    # Check if multi-profile search with all bitrate profiles
    if args.multi_profile_search and multi_profile_list:
        all_bitrate = all(p.is_bitrate_mode for p in multi_profile_list)
        if all_bitrate:
            return  # All bitrate profiles don't require targets

    parser.error(
        "At least one quality target is required (e.g., --vmaf-target, --ssim2-mean-target). Use --assessment-only for a single encode without targets."
    )


def _resolve_selected_profile(
    args: PipelineArgs,
    parser: argparse.ArgumentParser,
    profiles: dict[str, "Profile"] | None,
) -> "Profile | None":
    """Resolve the selected encoding profile.

    Returns None for multi-profile search mode (which uses multi_profile_list instead).
    """
    from .profiles import Profile, ProfileError, get_profile, list_profiles

    if args.multi_profile_search:
        # Multi-profile search doesn't use selected_profile
        return None

    if args.profile is not None:
        if profiles is None:
            parser.error("Profiles not loaded but --profile was specified")
        try:
            return get_profile(profiles, args.profile)
        except ProfileError as e:
            print(list_profiles(profiles))
            parser.error(str(e))

    # Use preset-based profile (default mode)
    return Profile(
        name=f"preset-{args.preset}",
        description=f"x265 {args.preset} preset",
        settings={"preset": args.preset},
        is_preset=True,
    )


def _has_targets(args: PipelineArgs) -> bool:
    """Check if any quality targets are specified.

    Local implementation to avoid circular import with pipeline_validation.
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


def validate_args(
    args: PipelineArgs, parser: argparse.ArgumentParser
) -> ValidationResult:
    """Validate all pipeline arguments and resolve profiles.

    This is the main validation entry point that calls all validation helpers.

    Args:
        args: Parsed pipeline arguments
        parser: Argument parser for error reporting

    Returns:
        ValidationResult with validated profile information

    Raises:
        SystemExit: Via parser.error() if validation fails
    """
    from .profiles import ProfileError, load_profiles

    # Run validation checks
    validate_time_args(args, parser)
    validate_sampling_args(args, parser)
    validate_metric_args(args, parser)
    validate_crf_args(args, parser)
    validate_target_args(args, parser)

    has_quality_targets = _has_targets(args)
    validate_mode_args(args, parser, has_quality_targets)
    validate_bitrate_warning_args(args, parser)

    # Validate that profile/preset/multi-profile-search is specified
    if args.profile is None and args.preset is None and not args.multi_profile_search:
        parser.error(
            "One of --profile, --preset, or --multi-profile-search is required."
        )

    # Load profiles if needed
    profiles: dict[str, "Profile"] | None = None
    if args.profile is not None or args.multi_profile_search:
        try:
            profiles = load_profiles()
        except ProfileError as e:
            parser.error(f"Failed to load profiles: {e}")

    # Resolve multi-profile search
    multi_profile_list: list["Profile"] = []
    multi_profile_display: str = ""
    if args.multi_profile_search and profiles is not None:
        multi_profile_list, multi_profile_display = _resolve_multi_profile_search(
            args, parser, profiles
        )

    # Validate target requirements
    _validate_target_requirements(args, parser, has_quality_targets, multi_profile_list)

    # Resolve selected profile
    selected_profile = _resolve_selected_profile(args, parser, profiles)

    # Validate bitrate profiles are not used in CRF search mode
    if not args.assessment_only and not args.multi_profile_search:
        # selected_profile is guaranteed to be set when not in multi-profile search mode
        assert selected_profile is not None
        if selected_profile.is_bitrate_mode:
            parser.error(
                f"Profile '{selected_profile.name}' uses bitrate mode and cannot be used with CRF search. Use --assessment-only for a single bitrate encode, or --multi-profile-search for profile comparison."
            )

    return ValidationResult(
        selected_profile=selected_profile,
        multi_profile_list=multi_profile_list,
        multi_profile_display=multi_profile_display,
        has_quality_targets=has_quality_targets,
    )
