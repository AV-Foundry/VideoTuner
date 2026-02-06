from __future__ import annotations

import logging
import os
import sys
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from .constants import METRIC_DECIMALS
from .ssimulacra2_assessment import SSIM2Result
from .vmaf_assessment import VMAFResult
from .create_encodes import (
    build_ffms2_index,
    calculate_autocrop_values,
)
from .encoding_utils import CropValues, VapourSynthEnv
from .pipeline_iteration import run_single_crf_iteration, run_single_bitrate_iteration
from .pipeline_cli import (
    PipelineArgs,
    build_arg_parser,
    parse_cli,
    validate_args,
    get_default,
)
from .pipeline_types import get_reference_dir
from .pipeline_display import (
    display_settings_summary,
    display_assessment_summary,
    display_ignored_args_warnings,
    format_bitrate_percentage,
    check_and_display_bitrate_warning,
    display_multi_profile_results,
)
from .pipeline_types import IterationContext
from .progress import PipelineDisplay
from .media import parse_video_info, InvalidVideoFileError, get_frame_count
from .profiles import Profile
from .utils import ensure_dir
from .crf_search import (
    CRFSearchState,
    QualityTarget,
)
from .pipeline_validation import (
    has_targets,
    build_targets,
    validate_sampling_parameters,
)
from .pipeline_multi_profile import (
    MultiProfileSearchParams,
    run_multi_profile_search,
    rank_profile_results,
)
from .pipeline_reference import (
    MetricSamplingParams,
    are_sampling_params_equal,
    generate_metric_reference,
)
from .utils import get_app_root, log_section
from .constants import (
    CRF_SEARCH_MAX_ITERATIONS,
)


def run_pipeline(args: PipelineArgs) -> int:
    parser = build_arg_parser()

    # Validate arguments and resolve profiles
    validation = validate_args(args, parser)
    selected_profile = validation.selected_profile
    multi_profile_list = validation.multi_profile_list
    multi_profile_display = validation.multi_profile_display
    has_quality_targets = validation.has_quality_targets

    auto_crop: bool = bool(args.auto_crop)

    display = PipelineDisplay(show_title=True)

    # Configure logging (file-only; console output is driven by the Rich UI)
    verbose: bool = bool(args.verbose)
    quiet: bool = bool(args.quiet)
    level = logging.INFO
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[logging.NullHandler()],
        force=True,
    )
    log = logging.getLogger(__name__)

    input_path: Path = Path(args.input)
    if not input_path.exists():
        parser.error(f"Input not found: {input_path}")

    log.info("Probing input with ffprobe ...")
    ffprobe_bin: str = args.ffprobe_bin
    try:
        info = parse_video_info(input_path, ffprobe_bin=ffprobe_bin)
    except InvalidVideoFileError as e:
        display.console.print(f"\n[bold red]Error:[/bold red] {e}\n")
        log.error("Invalid video file: %s", e)
        return 1
    res = f"{info.width}x{info.height}" if (info.width and info.height) else "unknown"
    log.info(
        "Detected fps=%.3f, duration=%.2fs, pix_fmt=%s, res=%s",
        info.fps,
        info.duration,
        info.pix_fmt or "unknown",
        res,
    )

    # Determine app root early for tool detection
    repo_root: Path = get_app_root()

    # Display active settings summary
    display_settings_summary(
        display.console, args, multi_profile_display, input_path.name
    )

    # Warn about ignored arguments when using bitrate profiles
    bitrate_profile_names = [p.name for p in multi_profile_list if p.is_bitrate_mode]
    display_ignored_args_warnings(
        display.console,
        log,
        bitrate_profile_names=bitrate_profile_names,
        crf_start_value=args.crf_start_value,
        crf_interval=args.crf_interval,
        has_targets=has_quality_targets,
    )

    # Validate window sizes fit within video duration (after guard bands)
    guard_start_percent: float = max(0.0, args.guard_start_percent)
    guard_end_percent: float = max(0.0, args.guard_end_percent)
    guard_seconds: float = max(0.0, args.guard_seconds)
    start_guard_time = max(info.duration * guard_start_percent, guard_seconds)
    end_guard_time = max(info.duration * guard_end_percent, guard_seconds)
    usable_duration = info.duration - (start_guard_time + end_guard_time)

    if usable_duration <= 0:
        msg1 = (
            f"ERROR: Guard bands ({guard_start_percent * 100:.1f}% start, "
            f"{guard_end_percent * 100:.1f}% end, seconds min {guard_seconds:.1f}) "
            f"leave no usable duration in {info.duration:.1f}s video"
        )
        msg2 = "ERROR: Reduce --guard-start-percent, --guard-end-percent, or --guard-seconds"
        print(msg1, file=sys.stderr)
        print(msg2, file=sys.stderr)
        log.error(
            "Guard bands (%.1f%% = %.1fs) leave no usable duration in %.1fs video",
            (guard_start_percent + guard_end_percent) * 100,
            start_guard_time + end_guard_time,
            info.duration,
        )
        log.error("Reduce guard settings")
        return 1

    # Check if autocrop plugin is available when autocrop is enabled
    if auto_crop:
        vs_env = VapourSynthEnv.from_args(args.vs_dir, args.vs_plugin_dir, repo_root)
        if not vs_env.has_autocrop_plugin():
            display.console.print(
                f"[bold yellow]⚠ Warning:[/bold yellow] [yellow]autocrop.dll not found in {vs_env.vs_plugin_dir} - disabling autocrop[/yellow]"
            )
            display.console.print(
                "[dim]Download from: https://github.com/Irrational-Encoding-Wizardry/vapoursynth-autocrop[/dim]"
            )
            display.console.print()
            auto_crop = False

    # Default workdir is <repo_root>/jobs/<name>_<timestamp> unless overridden
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.workdir:
        workdir: Path = args.workdir
    else:
        jobs_root = repo_root / "jobs"
        workdir = jobs_root / f"{input_path.stem}_{timestamp}"
    _ = ensure_dir(workdir)

    # Create temp subdirectory for temporary files
    temp_dir = ensure_dir(workdir / "temp")

    # Add file logger (default in job folder)
    if not args.log_file:
        log_file: Path = workdir / f"{input_path.stem}_{timestamp}.log"
    else:
        log_file = Path(args.log_file)
        try:
            _ = ensure_dir(log_file.parent)
        except OSError as e:
            log.warning("Could not create log directory %s: %s", log_file.parent, e)

    def _rel(p: Path) -> str:
        try:
            return os.path.relpath(p, repo_root)
        except Exception:
            return str(p)

    try:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        logging.getLogger().addHandler(fh)

    except Exception as e:
        log.warning("Could not attach file logger at %s: %s", log_file, e)

    log_section(log, "Initialization")

    try:
        log.info("Log file: %s", _rel(log_file))
        log.info("Job folder: %s", _rel(workdir))
    except OSError as e:
        log.warning("Failed to log file/job folder paths due to OSError: %s", e)

    # Log job settings
    mode_str = "CRF Search"
    if args.assessment_only:
        mode_str = "Assessment Only"
    elif args.multi_profile_search:
        mode_str = "Multi-Profile Search"

    profile_str = selected_profile.name if selected_profile else "None"
    if args.multi_profile_search:
        profile_str = multi_profile_display

    log.info("")
    log.info("Settings")
    log.info("  Mode: %s", mode_str)
    log.info("  Profile: %s", profile_str)

    # Build assessments string
    assessments: list[str] = []
    if args.vmaf:
        assessments.append("VMAF")
    if args.ssim2:
        assessments.append("SSIMULACRA2")
    log.info("  Assessments: %s", ", ".join(assessments))

    log.info("  AutoCrop: %s", "Enabled" if auto_crop else "Disabled")

    # Log non-default sampling parameters
    if args.vmaf_interval_frames != get_default("vmaf_interval_frames"):
        log.info("  VMAF Interval: %d frames", args.vmaf_interval_frames)
    if args.vmaf_region_frames != get_default("vmaf_region_frames"):
        log.info("  VMAF Region: %d frames", args.vmaf_region_frames)
    if args.ssim2_interval_frames != get_default("ssim2_interval_frames"):
        log.info("  SSIM2 Interval: %d frames", args.ssim2_interval_frames)
    if args.ssim2_region_frames != get_default("ssim2_region_frames"):
        log.info("  SSIM2 Region: %d frames", args.ssim2_region_frames)

    # Log non-default analysis options
    if args.vmaf_model is not None:
        log.info("  VMAF Model: %s", args.vmaf_model)
    if args.tonemap != get_default("tonemap"):
        log.info("  Tonemap: %s", args.tonemap)

    # Log non-default guard bands
    if args.guard_start_percent > 0:
        log.info("  Guard Start: %.1f%%", args.guard_start_percent * 100)
    if args.guard_end_percent > 0:
        log.info("  Guard End: %.1f%%", args.guard_end_percent * 100)
    if args.guard_seconds > 0:
        log.info("  Guard Seconds: %.1fs", args.guard_seconds)

    # Log bitrate warning if specified
    if args.predicted_bitrate_warning_percent is not None:
        log.info("  Bitrate Warning: %.0f%%", args.predicted_bitrate_warning_percent)

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
        log.info("")
        log.info("Targets")
        for target in target_parts:
            log.info("  %s", target)

    log.info("")
    log.info("Source: %s", input_path.name)

    # Build FFMS2 index for source (shared across autocrop and all encodes)
    # This is done BEFORE autocrop to avoid delays during autocrop detection
    # Index is stored alongside the source file for persistence across job runs
    if auto_crop or args.vmaf or args.ssim2:
        # Store index alongside source file (e.g., sources/video.mkv -> sources/video.ffindex)
        cache_file = input_path.parent / f"{input_path.stem}.ffindex"

        # Only build if index doesn't exist
        if not cache_file.exists():
            with (
                display.stage(
                    "Building FFMS2 Index",
                    total=1,  # Placeholder - will be updated when ffmsindex reports frame count
                    show_eta=True,
                    transient=True,  # Transient to avoid duplicate header line
                    show_done=True,
                ) as index_stage
            ):
                _ = build_ffms2_index(
                    source_path=input_path,
                    cache_file=cache_file,
                    cwd=repo_root,
                    line_handler=index_stage.make_ffmsindex_handler(),
                )
        else:
            display.console.print("[cyan]Using Existing FFMS2 Index[/cyan]")

    log_section(log, "Analysis")

    # Calculate autocrop values once (shared across ALL encodes)
    crop_values: CropValues | None = None

    if auto_crop:
        # Analyze the entire video to get representative crop values
        total_frames = get_frame_count(input_path, info)
        log.info(
            "Calculating shared autocrop values from entire video (%d frames)...",
            total_frames,
        )
        with display.stage(
            "Calculating AutoCrop Values",
            total=1,  # Will be updated by handler when total samples is known
            show_eta=True,
            transient=True,
            show_done=True,
        ) as autocrop_stage:
            crop_values = calculate_autocrop_values(
                source_path=input_path,
                start_frame=0,
                num_frames=total_frames,
                fps=info.fps,
                cwd=repo_root,
                temp_dir=workdir / "temp",
                line_handler=autocrop_stage.make_autocrop_handler(),
            )
            # Calculate final dimensions after crop
            final_width = (info.width or 0) - crop_values.left - crop_values.right
            final_height = (info.height or 0) - crop_values.top - crop_values.bottom
            autocrop_stage.done_suffix = (
                f"[white]({final_width}x{final_height})[/white]"
            )
        log.info(
            "Autocrop values calculated: left=%d, top=%d, right=%d, bottom=%d (final: %dx%d)",
            crop_values.left,
            crop_values.top,
            crop_values.right,
            crop_values.bottom,
            final_width,
            final_height,
        )
        log.info("Autocrop will be applied to ALL encoding operations")
    elif not auto_crop:
        log.info("Autocrop disabled - analyzing full frame including any letterboxing")

    # ===================================================================
    # PERIODIC SAMPLING APPROACH
    # ===================================================================

    log_section(log, "Sample Selection")

    # Calculate total frames and guard frames for periodic sampling
    total_frames = get_frame_count(input_path, info)
    # Note: guard percentages are already in decimal form (0.0-1.0), not 0-100
    guard_start_ratio = args.guard_start_percent
    guard_end_ratio = args.guard_end_percent
    guard_start_frames = int(guard_start_ratio * total_frames)
    guard_end_frames = int(guard_end_ratio * total_frames)

    log.info(
        "Total frames: %d (%.3f fps × %.1fs)", total_frames, info.fps, info.duration
    )

    # Log HDR metadata if present (logged here after file logger is set up)
    if info.mastering_display_color_primaries or info.maximum_content_light_level:
        hdr_parts: list[str] = []
        if info.mastering_display_color_primaries:
            hdr_parts.append(f"colorspace={info.mastering_display_color_primaries}")
        if info.maximum_content_light_level:
            hdr_parts.append(f"MaxCLL={info.maximum_content_light_level}")
        if info.maximum_frameaverage_light_level:
            hdr_parts.append(f"MaxFALL={info.maximum_frameaverage_light_level}")
        log.info("HDR metadata: %s", ", ".join(hdr_parts))

    log.info(
        "Guard bands: start=%d frames (%.1f%%), end=%d frames (%.1f%%)",
        guard_start_frames,
        guard_start_ratio * 100,
        guard_end_frames,
        guard_end_ratio * 100,
    )

    # Validate periodic sampling parameters for both metrics
    vmaf_valid, ssim2_valid = validate_sampling_parameters(
        args=args,
        total_frames=total_frames,
        guard_start_frames=guard_start_frames,
        guard_end_frames=guard_end_frames,
        log=log,
    )
    args.vmaf = vmaf_valid
    args.ssim2 = ssim2_valid

    # Validate that at least one metric is enabled
    if not args.vmaf and not args.ssim2:
        log.error("No metrics have valid samples - cannot proceed")
        return 1

    log_section(log, "Reference Generation")

    # Create concatenated reference files for each metric
    lossless_profile = Profile(
        name="lossless",
        description="Lossless reference extraction",
        settings={"preset": "ultrafast"},
    )

    vmaf_ref_path: Path | None = None
    ssim2_ref_path: Path | None = None
    reference_dir = get_reference_dir(workdir)

    # Check if we can share samples between metrics (identical sampling parameters)
    sharing_samples = are_sampling_params_equal(args)

    if sharing_samples:
        # Generate ONE shared reference file for both metrics
        log.info(
            "Using shared samples for VMAF and SSIM2 (identical sampling parameters)"
        )
        display.console.print("[cyan]Using shared samples for VMAF and SSIM2[/cyan]")

        shared_sampling = MetricSamplingParams(
            interval_frames=args.vmaf_interval_frames,  # Same as ssim2
            region_frames=args.vmaf_region_frames,  # Same as ssim2
            guard_start_frames=guard_start_frames,
            guard_end_frames=guard_end_frames,
            total_frames=total_frames,
        )
        shared_ref_path = generate_metric_reference(
            metric_type="shared",
            source_path=input_path,
            output_dir=reference_dir,
            sampling_params=shared_sampling,
            fps=info.fps,
            lossless_profile=lossless_profile,
            video_info=info,
            mkvmerge_bin=args.mkvmerge_bin,
            repo_root=repo_root,
            temp_dir=temp_dir,
            display=display,
            log=log,
            auto_crop=auto_crop,
            crop_values=crop_values,
        )
        if shared_ref_path is None:
            args.vmaf = False
            args.ssim2 = False
        else:
            # Point both metrics to the same reference file
            vmaf_ref_path = shared_ref_path
            ssim2_ref_path = shared_ref_path
    else:
        # Generate separate references for each metric (different sampling parameters)
        if args.vmaf:
            vmaf_sampling = MetricSamplingParams(
                interval_frames=args.vmaf_interval_frames,
                region_frames=args.vmaf_region_frames,
                guard_start_frames=guard_start_frames,
                guard_end_frames=guard_end_frames,
                total_frames=total_frames,
            )
            vmaf_ref_path = generate_metric_reference(
                metric_type="vmaf",
                source_path=input_path,
                output_dir=reference_dir,
                sampling_params=vmaf_sampling,
                fps=info.fps,
                lossless_profile=lossless_profile,
                video_info=info,
                mkvmerge_bin=args.mkvmerge_bin,
                repo_root=repo_root,
                temp_dir=temp_dir,
                display=display,
                log=log,
                auto_crop=auto_crop,
                crop_values=crop_values,
            )
            if vmaf_ref_path is None:
                args.vmaf = False

        if args.ssim2:
            ssim2_sampling = MetricSamplingParams(
                interval_frames=args.ssim2_interval_frames,
                region_frames=args.ssim2_region_frames,
                guard_start_frames=guard_start_frames,
                guard_end_frames=guard_end_frames,
                total_frames=total_frames,
            )
            ssim2_ref_path = generate_metric_reference(
                metric_type="ssim2",
                source_path=input_path,
                output_dir=reference_dir,
                sampling_params=ssim2_sampling,
                fps=info.fps,
                lossless_profile=lossless_profile,
                video_info=info,
                mkvmerge_bin=args.mkvmerge_bin,
                repo_root=repo_root,
                temp_dir=temp_dir,
                display=display,
                log=log,
                auto_crop=auto_crop,
                crop_values=crop_values,
            )
            if ssim2_ref_path is None:
                args.ssim2 = False

    # Build targets (needed for any search module)
    has_quality_targets = has_targets(args)
    targets: list[QualityTarget] = build_targets(args) if has_quality_targets else []
    targets_for_display: list[QualityTarget] | None = (
        targets if has_quality_targets else None
    )

    # Initialize result variables
    vmaf_results: list[VMAFResult] = []
    ssim2_results: list[SSIM2Result] = []
    scores: dict[str, float | None] = {}
    # Optimal scores from CRF search (if converged) - used for final display
    # Note: CRF search filters out None values, so optimal scores never contain None
    optimal_search_scores: dict[str, float] | None = None
    optimal_crf: float | None = None
    _predicted_bitrate: float = 0.0
    optimal_predicted_bitrate: float = 0.0

    # ========== ASSESSMENT ONLY MODE ==========
    if args.assessment_only:
        # selected_profile is guaranteed when not in multi-profile search mode
        assert selected_profile is not None
        log.info("\n=== Running Assessment Only ===")
        log.info(
            "%s: %s", selected_profile.display_label, selected_profile.display_name
        )

        # Create iteration context
        ctx = IterationContext(
            input_path=input_path,
            workdir=workdir,
            temp_dir=temp_dir,
            repo_root=repo_root,
            info=info,
            selected_profile=selected_profile,
            total_frames=total_frames,
            guard_start_frames=guard_start_frames,
            guard_end_frames=guard_end_frames,
            vmaf_ref_path=vmaf_ref_path,
            ssim2_ref_path=ssim2_ref_path,
            args=args,
            display=display,
            log=log,
            crop_values=crop_values,
            sharing_samples=sharing_samples,
        )

        # Detect bitrate vs CRF mode
        if selected_profile.is_bitrate_mode:
            # Bitrate mode: determine stats file location if multi-pass
            bitrate_kbps = selected_profile.bitrate or 0
            pass_num = selected_profile.pass_number or 1
            log.info("Bitrate: %d kbps", bitrate_kbps)
            log.info("Pass: %d\n", pass_num)

            # Display console header for assessment-only bitrate mode
            pass_mode = selected_profile.pass_mode_description
            display.console.print()
            display.console.print(
                f"[bold cyan]Encoding at {bitrate_kbps} kbps ({pass_mode})[/bold cyan]"
            )

            # Run bitrate iteration
            (
                scores,
                vmaf_results,
                ssim2_results,
                _predicted_bitrate,
                _vmaf_distorted_path,
                _ssim2_distorted_path,
            ) = run_single_bitrate_iteration(ctx, iteration=1)
        else:
            # CRF mode (existing behavior)
            log.info("CRF: %.1f\n", args.crf_start_value)

            # Display console header for assessment-only CRF mode
            display.console.print()
            display.console.print(
                f"[bold cyan]Encoding at CRF {args.crf_start_value:.1f}[/bold cyan]"
            )

            # Run single iteration
            (
                scores,
                vmaf_results,
                ssim2_results,
                _predicted_bitrate,
                _vmaf_distorted_path,
                _ssim2_distorted_path,
            ) = run_single_crf_iteration(ctx, args.crf_start_value, iteration=1)

        # Note: predicted_bitrate is already calculated inside iteration function

    # ========== CRF SEARCH MODE (default) ==========
    elif not args.multi_profile_search:
        # selected_profile is guaranteed when not in multi-profile search mode
        assert selected_profile is not None
        crf_search_state = CRFSearchState(targets, args.crf_interval)

        # Create iteration context with selected profile (may have been updated by profile search)
        ctx = IterationContext(
            input_path=input_path,
            workdir=workdir,
            temp_dir=temp_dir,
            repo_root=repo_root,
            info=info,
            selected_profile=selected_profile,
            total_frames=total_frames,
            guard_start_frames=guard_start_frames,
            guard_end_frames=guard_end_frames,
            vmaf_ref_path=vmaf_ref_path,
            ssim2_ref_path=ssim2_ref_path,
            args=args,
            display=display,
            log=log,
            crop_values=crop_values,
            sharing_samples=sharing_samples,
        )

        iteration = 0
        current_crf: float = args.crf_start_value
        max_iterations = CRF_SEARCH_MAX_ITERATIONS
        scores = {}
        crf_to_predicted_bitrate_single: dict[
            float, float
        ] = {}  # Track CRF -> predicted bitrate

        log.info("\n=== Starting CRF Search ===")
        log.info(
            "%s: %s", selected_profile.display_label, selected_profile.display_name
        )
        log.info("Targets:")
        for target in targets:
            log.info("  %s >= %.2f", target.metric_name, target.target_value)
        log.info("Starting CRF: %.1f", current_crf)
        log.info("CRF Interval: %.1f\n", args.crf_interval)

        display.console.print()
        display.console.print(
            f"[bold]CRF Search: {selected_profile.display_label} '{selected_profile.display_name}'[/bold]"
        )

        while iteration < max_iterations:
            iteration += 1

            # Display iteration header
            display.console.print()
            display.console.print(
                f"[bold cyan]CRF Iteration {iteration}: CRF {current_crf:.1f}[/bold cyan]"
            )

            log.info("\n=== CRF Iteration %d: CRF %.1f ===", iteration, current_crf)

            # Run encoding and assessment for this CRF
            (
                scores,
                vmaf_results,
                ssim2_results,
                _predicted_bitrate,
                _vmaf_distorted_path,
                _ssim2_distorted_path,
            ) = run_single_crf_iteration(ctx, current_crf, iteration)

            # Store predicted bitrate for this CRF
            crf_to_predicted_bitrate_single[current_crf] = _predicted_bitrate

            # Add result to search state
            search_scores = {k: v for k, v in scores.items() if v is not None}
            crf_search_state.add_result(current_crf, search_scores)

            # Display iteration targets summary
            display_assessment_summary(
                display.console,
                scores,
                targets=targets,
                iteration=iteration,
                targets_only=True,
                metric_decimals=args.metric_decimals,
            )

            # Display search status summary
            if crf_search_state.all_targets_met():
                display.console.print("[green]✓ All targets met[/green]")
            else:
                unmet_count = sum(1 for t in targets if not t.is_met())
                display.console.print(f"[red]✗ {unmet_count} target(s) not met[/red]")

            # Log iteration summary
            log.info("Iteration %d results:", iteration)
            for key, value in scores.items():
                if value is not None:
                    log.info("  %s: %.3f", key, value)
            log.info("Targets:")
            for target in targets:
                status = "MET" if target.is_met() else "NOT MET"
                delta_str = (
                    f"(Δ={target.delta():.{METRIC_DECIMALS}f})"
                    if target.delta() is not None
                    else ""
                )
                log.info(
                    f"  %s >= %.{METRIC_DECIMALS}f: %s %s",
                    target.metric_name,
                    target.target_value,
                    status,
                    delta_str,
                )

            # Check for convergence
            if crf_search_state.is_converged():
                optimal_crf = crf_search_state.get_optimal_crf()
                optimal_search_scores = crf_search_state.get_optimal_scores()

                # Get predicted bitrate for the optimal CRF
                if (
                    optimal_crf is not None
                    and optimal_crf in crf_to_predicted_bitrate_single
                ):
                    optimal_predicted_bitrate = crf_to_predicted_bitrate_single[
                        optimal_crf
                    ]
                else:
                    optimal_predicted_bitrate = 0.0

                display.console.print(
                    f"[bold green]CRF Search complete: Optimal CRF = {optimal_crf:.1f}[/bold green]"
                )
                log.info("\nCRF Search complete after %d iterations", iteration)
                log.info("Optimal CRF: %.1f", optimal_crf)
                break

            # Calculate next CRF
            next_crf = crf_search_state.calculate_next_crf(current_crf)

            if next_crf is None:
                if crf_search_state.all_targets_met():
                    optimal_crf = crf_search_state.get_optimal_crf()
                    optimal_search_scores = crf_search_state.get_optimal_scores()

                    # Get predicted bitrate for the optimal CRF
                    if (
                        optimal_crf is not None
                        and optimal_crf in crf_to_predicted_bitrate_single
                    ):
                        optimal_predicted_bitrate = crf_to_predicted_bitrate_single[
                            optimal_crf
                        ]
                    else:
                        optimal_predicted_bitrate = 0.0

                    display.console.print(
                        f"[bold green]CRF Search complete: Optimal CRF = {optimal_crf:.1f}[/bold green]"
                    )
                    log.info("\nCRF Search complete after %d iterations", iteration)
                    log.info("Optimal CRF: %.1f", optimal_crf)
                else:
                    display.console.print(
                        "[bold yellow]CRF Search exhausted - targets may be unreachable[/bold yellow]"
                    )
                    log.warning(
                        "\nCRF Search exhausted after %d iterations - targets may be unreachable",
                        iteration,
                    )
                    if crf_search_state.get_optimal_crf() is not None:
                        log.info(
                            "Best CRF found: %.1f", crf_search_state.get_optimal_crf()
                        )
                break

            # Display next step
            direction = "down" if next_crf < current_crf else "up"
            crf_delta = next_crf - current_crf
            delta_str = f"{crf_delta:+.1f}"
            if (
                crf_search_state.passing_crf is not None
                and crf_search_state.failing_crf is not None
            ):
                display.console.print(
                    f"[cyan]→ Next: CRF {next_crf:.1f} (refining, {delta_str})[/cyan]"
                )
                log.info("Next CRF: %.1f (refining, delta=%s)", next_crf, delta_str)
            else:
                display.console.print(
                    f"[cyan]→ Next: CRF {next_crf:.1f} (searching {direction}, {delta_str})[/cyan]"
                )
                log.info(
                    "Next CRF: %.1f (searching %s, delta=%s)",
                    next_crf,
                    direction,
                    delta_str,
                )

            current_crf = next_crf

        if iteration >= max_iterations:
            display.console.print()
            display.console.print("[bold red]Maximum CRF iterations reached[/bold red]")
            log.warning("\nMaximum CRF iterations (%d) reached", max_iterations)

    # ========== MULTI-PROFILE SEARCH MODE ==========
    else:
        # Create context factory for multi-profile search
        def ctx_factory(profile: Profile) -> IterationContext:
            return IterationContext(
                input_path=input_path,
                workdir=workdir,
                temp_dir=temp_dir,
                repo_root=repo_root,
                info=info,
                selected_profile=profile,
                total_frames=total_frames,
                guard_start_frames=guard_start_frames,
                guard_end_frames=guard_end_frames,
                vmaf_ref_path=vmaf_ref_path,
                ssim2_ref_path=ssim2_ref_path,
                args=args,
                display=display,
                log=log,
                crop_values=crop_values,
                sharing_samples=sharing_samples,
            )

        # Run multi-profile search
        search_params = MultiProfileSearchParams(
            profiles=multi_profile_list,
            targets=targets,
            crf_start_value=args.crf_start_value,
            crf_interval=args.crf_interval,
            max_iterations=CRF_SEARCH_MAX_ITERATIONS,
            args=args,
            display=display,
            log=log,
        )
        profile_results = run_multi_profile_search(search_params, ctx_factory)

        # Rank results
        ranked_results = rank_profile_results(profile_results)

        if not ranked_results:
            display.console.print(
                "[bold red]No profiles successfully met targets[/bold red]"
            )
            log.error("Multi-profile search failed - no valid results")
            return 1

        winner = ranked_results[0]
        has_crf_profiles = any(r.optimal_crf is not None for r in ranked_results)

        # Display ranked comparison table
        display_multi_profile_results(
            display.console, ranked_results, targets, args.metric_decimals
        )

        # Display winner (only for CRF profiles - bitrate profiles aren't comparable)
        display.console.print()
        if has_crf_profiles:
            if winner.optimal_crf is not None:
                display.console.print(
                    f"[bold green]Winner: {winner.profile_name} at CRF {winner.optimal_crf:.1f}[/bold green]"
                )
            else:
                display.console.print(
                    f"[bold green]Winner: {winner.profile_name} (Bitrate mode)[/bold green]"
                )
            bitrate_display = format_bitrate_percentage(
                winner.predicted_bitrate_kbps, info.video_bitrate_kbps
            )
            display.console.print(f"[cyan]Predicted Bitrate: {bitrate_display}[/cyan]")

        # Show warning if applicable
        check_and_display_bitrate_warning(
            display.console,
            log,
            winner.predicted_bitrate_kbps,
            info.video_bitrate_kbps,
            args.predicted_bitrate_warning_percent,
            profile_name=winner.profile_name,
        )

        if has_crf_profiles:
            if winner.optimal_crf is not None:
                log.info(
                    "Multi-profile search complete - Winner: %s at CRF %.1f (%.0f kbps)",
                    winner.profile_name,
                    winner.optimal_crf,
                    winner.predicted_bitrate_kbps,
                )
            else:
                log.info(
                    "Multi-profile search complete - Winner: %s (Bitrate mode, %.0f kbps)",
                    winner.profile_name,
                    winner.predicted_bitrate_kbps,
                )
        else:
            log.info("Multi-profile search complete (bitrate profiles)")

        # Set winner's scores for final results display
        optimal_search_scores = {
            k: v for k, v in winner.scores.items() if v is not None
        }

    # ========== DISPLAY FINAL RESULTS ==========
    # Build final scores dict for display
    # Prefer optimal scores from CRF search (if converged) over last iteration's results
    final_scores: dict[str, float | None] = {}
    if optimal_search_scores:
        # Use scores from the optimal CRF (the one that actually met targets)
        final_scores = {k: v for k, v in optimal_search_scores.items()}
    elif vmaf_results:
        # Fallback to last iteration's results (exploration mode or no convergence)
        result = vmaf_results[0]
        final_scores["vmaf_mean"] = result.mean
        final_scores["vmaf_hmean"] = result.harmonic_mean
        final_scores["vmaf_1pct"] = result.p1_low
        final_scores["vmaf_min"] = result.minimum

    # Add SSIM2 scores if available (not tracked in CRF search scores yet)
    if ssim2_results and "ssim2_mean" not in final_scores:
        result = ssim2_results[0]
        final_scores["ssim2_mean"] = result.mean
        final_scores["ssim2_median"] = result.median
        final_scores["ssim2_95pct"] = result.p95_high
        final_scores["ssim2_5pct"] = result.p5_low

    # Display Assessment Summary using shared helper
    # Skip in multi-profile mode since the comparison table already shows everything
    if final_scores and not multi_profile_list:
        # selected_profile is guaranteed when not in multi-profile mode
        assert selected_profile is not None
        # Update targets with final scores (for targeting mode)
        if targets_for_display:
            for target in targets_for_display:
                if target.metric_name in final_scores:
                    target.current_value = final_scores[target.metric_name]

        # targets is None in exploration mode, populated in targeting mode
        display_assessment_summary(
            display.console,
            final_scores,
            targets=targets_for_display,
            iteration=None,
            metric_decimals=args.metric_decimals,
        )

        # Display optimal CRF for CRF search mode
        if not args.assessment_only and optimal_crf is not None:
            display.console.print(
                f"[bold cyan]Optimal CRF: {optimal_crf:.1f}[/bold cyan]"
            )

        # Display predicted bitrate
        # Use the appropriate bitrate value based on mode
        predicted_bitrate_for_display: float
        if args.assessment_only:
            predicted_bitrate_for_display = _predicted_bitrate
        else:
            # CRF search mode
            predicted_bitrate_for_display = optimal_predicted_bitrate

        if predicted_bitrate_for_display > 0:
            bitrate_display = format_bitrate_percentage(
                predicted_bitrate_for_display, info.video_bitrate_kbps
            )
            display.console.print(f"[cyan]Predicted Bitrate: {bitrate_display}[/cyan]")

        # Show warning if applicable
        check_and_display_bitrate_warning(
            display.console,
            log,
            predicted_bitrate_for_display,
            info.video_bitrate_kbps,
            args.predicted_bitrate_warning_percent,
            profile_name=selected_profile.name,
        )

    log_section(log, "Results")

    # Log VMAF scores
    if vmaf_results:
        result = vmaf_results[0]
        log.info("\n=== VMAF Scores ===")
        log.info("VMAF (mean):            %.2f", result.mean)
        log.info("VMAF (harmonic mean):   %.2f", result.harmonic_mean)
        log.info("VMAF (1%% low):          %.2f", result.p1_low)
        log.info("VMAF (minimum):         %.2f", result.minimum)

    # Log SSIMULACRA2 scores
    if ssim2_results:
        result = ssim2_results[0]
        log.info("\n=== SSIMULACRA2 Scores ===")
        log.info("SSIMULACRA2 (mean):    %.2f", result.mean)
        log.info("SSIMULACRA2 (median):  %.2f", result.median)
        log.info("SSIMULACRA2 (95%%):     %.2f", result.p95_high)
        log.info("SSIMULACRA2 (5%% low):  %.2f", result.p5_low)
        log.info("SSIMULACRA2 (stddev):  %.2f", result.std_dev)

    return 0


def main(argv: Iterable[str] | None = None) -> int:
    """CLI wrapper for entry points."""
    return run_pipeline(parse_cli(argv))


if __name__ == "__main__":
    raise SystemExit(run_pipeline(parse_cli()))
