"""Multi-profile search functionality for the encoding pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, cast

if TYPE_CHECKING:
    from .crf_search import QualityTarget
    from .progress import PipelineDisplay
    from .pipeline_cli import PipelineArgs
    from .pipeline_types import IterationContext, MultiProfileResult
    from .profiles import Profile


@dataclass
class MultiProfileSearchParams:
    """Parameters for multi-profile search.

    Attributes:
        profiles: List of profiles to compare
        targets: Quality targets to meet
        crf_start_value: Default starting CRF value
        crf_interval: CRF interval for search
        max_iterations: Maximum CRF search iterations per profile
        args: Pipeline arguments
        display: Pipeline display instance
        log: Logger instance
    """

    profiles: list["Profile"]
    targets: list["QualityTarget"]
    crf_start_value: float
    crf_interval: float
    max_iterations: int
    args: "PipelineArgs"
    display: "PipelineDisplay"
    log: logging.Logger


def run_multi_profile_search(
    params: MultiProfileSearchParams,
    ctx_factory: Callable[["Profile"], "IterationContext"],
) -> list["MultiProfileResult"]:
    """Run CRF search across multiple profiles.

    For each profile, either runs a bitrate iteration (for bitrate-mode profiles)
    or a full CRF search (for CRF-mode profiles). Uses the previous profile's
    optimal CRF as the starting point for the next profile to speed up convergence.

    Args:
        params: Search parameters including profiles, targets, and display settings
        ctx_factory: Factory function to create IterationContext for each profile

    Returns:
        List of MultiProfileResult for each profile tested
    """
    profiles = params.profiles
    targets = params.targets
    display = params.display
    log = params.log

    log.info("\n=== Multi-Profile Search Mode ===")
    log.info("Profiles to compare: %s", ", ".join(p.name for p in profiles))

    profile_results: list[MultiProfileResult] = []
    previous_optimal_crf: float | None = None

    for profile_idx, profile in enumerate(profiles, 1):
        display.console.print()
        display.console.print(
            f"[bold cyan]Testing Profile {profile_idx}/{len(profiles)}: {profile.name}[/bold cyan]"
        )
        log.info("\n--- Testing profile: %s ---", profile.name)

        ctx = ctx_factory(profile)

        if profile.is_bitrate_mode:
            result = _run_bitrate_profile(ctx, profile, targets, display, log)
            profile_results.append(result)
            continue

        # CRF mode: run full CRF search
        result, optimal_crf = _run_crf_profile_search(
            ctx=ctx,
            profile=profile,
            targets=targets,
            crf_start_value=params.crf_start_value,
            crf_interval=params.crf_interval,
            max_iterations=params.max_iterations,
            previous_optimal_crf=previous_optimal_crf,
            display=display,
            log=log,
        )
        profile_results.append(result)

        # Update previous optimal CRF for next profile
        if result.is_valid() and optimal_crf is not None:
            previous_optimal_crf = optimal_crf

    return profile_results


def _run_bitrate_profile(
    ctx: "IterationContext",
    profile: "Profile",
    targets: list["QualityTarget"],
    display: "PipelineDisplay",
    log: logging.Logger,
) -> "MultiProfileResult":
    """Run single bitrate iteration for a bitrate-mode profile.

    Args:
        ctx: Iteration context
        profile: Bitrate-mode profile to encode
        targets: Quality targets to evaluate (empty list if none specified)
        display: Pipeline display instance
        log: Logger instance

    Returns:
        MultiProfileResult for this profile
    """
    from .pipeline_iteration import run_single_bitrate_iteration
    from .pipeline_types import MultiProfileResult
    from .pipeline_validation import check_scores_meet_targets

    bitrate_kbps = profile.bitrate or 0
    pass_num = profile.pass_number or 1
    log.info("Bitrate mode: %d kbps, pass %d", bitrate_kbps, pass_num)

    pass_mode = profile.pass_mode_description
    display.console.print(f"[bold]Encoding at {bitrate_kbps} kbps ({pass_mode})[/bold]")

    try:
        (
            scores,
            _vmaf_results,
            _ssim2_results,
            predicted_bitrate,
            _vmaf_distorted_path,
            _ssim2_distorted_path,
        ) = run_single_bitrate_iteration(ctx, iteration=1)

        # Evaluate targets when specified (pass/fail only, no CRF iteration)
        meets_all: bool | None = None
        if targets:
            meets_all = check_scores_meet_targets(scores, targets)

        result = MultiProfileResult(
            profile_name=profile.name,
            optimal_crf=None,
            scores=scores,
            predicted_bitrate_kbps=predicted_bitrate,
            converged=True,
            meets_all_targets=meets_all,
        )

        if meets_all is True:
            display.console.print(
                f"[green]✓ Profile {profile.name}: Encoded successfully ({predicted_bitrate:.0f} kbps) - All targets met[/green]"
            )
        elif meets_all is False:
            display.console.print(
                f"[yellow]⚠ Profile {profile.name}: Encoded successfully ({predicted_bitrate:.0f} kbps) - Not all targets met[/yellow]"
            )
        else:
            display.console.print(
                f"[green]✓ Profile {profile.name}: Encoded successfully ({predicted_bitrate:.0f} kbps)[/green]"
            )
        log.info(
            "Profile %s: Encoded successfully at %d kbps",
            profile.name,
            bitrate_kbps,
        )

    except Exception as e:
        log.error("Profile %s bitrate encoding failed: %s", profile.name, e)
        display.console.print(f"[red]✗ Profile {profile.name}: Encoding failed[/red]")

        result = MultiProfileResult(
            profile_name=profile.name,
            optimal_crf=None,
            scores={},
            predicted_bitrate_kbps=0.0,
            converged=False,
            meets_all_targets=None,
        )

    return result


def _run_crf_profile_search(
    ctx: "IterationContext",
    profile: "Profile",
    targets: list["QualityTarget"],
    crf_start_value: float,
    crf_interval: float,
    max_iterations: int,
    previous_optimal_crf: float | None,
    display: "PipelineDisplay",
    log: logging.Logger,
) -> tuple["MultiProfileResult", float | None]:
    """Run CRF search for a single profile.

    Args:
        ctx: Iteration context
        profile: CRF-mode profile to search
        targets: Quality targets to meet
        crf_start_value: Default starting CRF
        crf_interval: CRF interval for search
        max_iterations: Maximum iterations
        previous_optimal_crf: Optimal CRF from previous profile (or None)
        display: Pipeline display instance
        log: Logger instance

    Returns:
        Tuple of (MultiProfileResult, optimal_crf or None)
    """
    from .crf_search import CRFFloorError, CRFSearchState
    from .pipeline_display import display_assessment_summary
    from .pipeline_iteration import run_single_crf_iteration
    from .pipeline_types import MultiProfileResult
    from .pipeline_validation import check_scores_meet_targets
    from .constants import CRF_FLOOR_VALUE, CRF_FLOOR_TOLERANCE

    crf_search_state = CRFSearchState(targets, crf_interval)
    iteration = 0

    # Use previous profile's optimal CRF as starting point (if available)
    current_crf = (
        previous_optimal_crf if previous_optimal_crf is not None else crf_start_value
    )

    if previous_optimal_crf is not None:
        log.info("Starting CRF search at %.1f (from previous profile)", current_crf)
    else:
        log.info("Starting CRF search at %.1f (default)", current_crf)

    crf_to_predicted_bitrate: dict[float, float] = {}
    iteration_final_scores: dict[str, float | None] = {}

    while iteration < max_iterations:
        iteration += 1
        display.console.print(
            f"[bold]CRF Search Iteration {iteration}: CRF {current_crf:.1f}[/bold]"
        )

        (
            scores,
            _vmaf_results,
            _ssim2_results,
            predicted_bitrate,
            _vmaf_distorted_path,
            _ssim2_distorted_path,
        ) = run_single_crf_iteration(ctx, current_crf, iteration=iteration)

        crf_to_predicted_bitrate[current_crf] = predicted_bitrate
        iteration_final_scores = scores

        search_scores = {k: v for k, v in scores.items() if v is not None}
        crf_search_state.add_result(current_crf, search_scores)

        # Display iteration summary
        # Get metric_decimals from first target (all targets share the same value)
        metric_decimals = targets[0].metric_decimals if targets else 2
        display_assessment_summary(
            display.console,
            scores,
            targets=targets,
            iteration=iteration,
            targets_only=True,
            metric_decimals=metric_decimals,
        )

        # Check convergence
        if crf_search_state.all_targets_met():
            display.console.print("[green]✓ All targets met[/green]")
        else:
            unmet_count = sum(1 for t in targets if not t.is_met())
            display.console.print(f"[red]✗ {unmet_count} target(s) not met[/red]")

        # Check floor
        try:
            crf_search_state.check_floor_reached(current_crf)
        except CRFFloorError as e:
            display.console.print(f"[bold red]CRF floor reached: {e}[/bold red]")
            log.warning("Profile %s hit CRF floor: %s", profile.name, e)
            break

        if crf_search_state.is_converged():
            display.console.print("[bold green]CRF Search converged[/bold green]")
            break

        next_crf = crf_search_state.calculate_next_crf(current_crf)
        if next_crf is None or crf_search_state.has_been_tested(next_crf):
            break

        # Display next step
        direction = "down" if next_crf < current_crf else "up"
        crf_delta = next_crf - current_crf
        display.console.print(
            f"[cyan]→ Next: CRF {next_crf:.1f} ({direction}, {crf_delta:+.1f})[/cyan]"
        )

        current_crf = next_crf

    # Collect results
    optimal_crf = crf_search_state.get_optimal_crf()
    converged = crf_search_state.is_converged()

    # Get predicted bitrate for the optimal CRF
    if optimal_crf is not None and optimal_crf in crf_to_predicted_bitrate:
        predicted_bitrate = crf_to_predicted_bitrate[optimal_crf]
    else:
        predicted_bitrate = 0.0

    # Use optimal scores rather than last iteration's scores
    optimal_scores = crf_search_state.get_optimal_scores()
    result_scores: dict[str, float | None] = cast(
        dict[str, float | None],
        optimal_scores if optimal_scores is not None else iteration_final_scores,
    )

    meets_all_targets_crf = converged and check_scores_meet_targets(
        result_scores, targets
    )

    result = MultiProfileResult(
        profile_name=profile.name,
        optimal_crf=optimal_crf,
        scores=result_scores,
        predicted_bitrate_kbps=predicted_bitrate,
        converged=converged,
        meets_all_targets=meets_all_targets_crf,
    )

    if not result.is_valid():
        display.console.print(
            f"[yellow]⚠ Profile {profile.name} failed to meet targets[/yellow]"
        )
        log.warning("Profile %s failed to meet targets", profile.name)
    else:
        crf_note = ""
        if (
            optimal_crf is not None
            and abs(optimal_crf - CRF_FLOOR_VALUE) < CRF_FLOOR_TOLERANCE
        ):
            crf_note = " (at ceiling)"
        display.console.print(
            f"[green]✓ Profile {profile.name}: Found optimal CRF {optimal_crf:.1f}{crf_note}[/green]"
        )
        log.info(
            "Profile %s: Found optimal CRF %.1f%s",
            profile.name,
            optimal_crf,
            crf_note,
        )

    return result, optimal_crf


def get_effective_metric_priority(
    targets: list["QualityTarget"],
) -> tuple[str, ...]:
    """Compute effective metric priority with target promotion.

    Metrics that have user-specified targets are promoted to the top of the
    priority list, preserving their relative order from METRIC_PRIORITY.
    Non-targeted metrics follow in their default order.

    Args:
        targets: Quality targets specified by the user

    Returns:
        Tuple of metric names in effective priority order
    """
    from .constants import METRIC_PRIORITY

    target_names = {t.metric_name for t in targets}
    promoted = tuple(m for m in METRIC_PRIORITY if m in target_names)
    remaining = tuple(m for m in METRIC_PRIORITY if m not in target_names)
    return promoted + remaining


def _metric_priority_sort_key(
    result: "MultiProfileResult",
    priority: tuple[str, ...],
) -> tuple[float, ...]:
    """Create a sort key from scores based on metric priority.

    Returns a tuple of negated scores in priority order (negated because
    Python sorts ascending, but higher scores are better). Missing scores
    are treated as negative infinity (worst possible).

    Args:
        result: Profile result to create key for
        priority: Metric names in priority order

    Returns:
        Tuple of negated scores for use as sort key (lower = better)
    """
    key: list[float] = []
    for metric_name in priority:
        score = result.scores.get(metric_name)
        if score is not None:
            key.append(-score)  # Negate: higher score = lower sort value = better
        else:
            key.append(float("inf"))  # Missing score = worst
    return tuple(key)


def rank_profile_results(
    results: list["MultiProfileResult"],
    targets: list["QualityTarget"] | None = None,
) -> list["MultiProfileResult"]:
    """Rank profile results using tiered ranking with metric priority.

    Ranking depends on whether the group contains any CRF profiles:

    **All-ABR groups:**
    - Tier 1: profiles that met all targets (if targets specified)
    - Tier 2: profiles that didn't meet targets / no targets specified
    - Within each tier: rank by metric priority (highest score wins)
    - Bitrate is NOT used for ranking in all-ABR groups

    **Mixed groups (any ABR + any CRF):**
    - Tier 1: profiles that met all targets (CRF and ABR treated equally)
    - Tier 2: profiles that didn't meet targets
    - Within each tier: rank by lowest predicted bitrate, with metric
      priority as tiebreaker

    Args:
        results: List of MultiProfileResult to rank
        targets: Quality targets (used for metric priority promotion)

    Returns:
        Sorted list with best results first
    """
    # Filter to valid results only
    valid_results = [r for r in results if r.is_valid()]

    if not valid_results:
        return []

    # Compute effective metric priority with target promotion
    effective_targets = targets or []
    priority = get_effective_metric_priority(effective_targets)

    # Determine if this is an all-ABR group
    has_crf = any(not r.is_bitrate_mode for r in valid_results)

    # Split into tiers
    tier1: list[MultiProfileResult] = []  # Met all targets
    tier2: list[MultiProfileResult] = []  # Failed / no targets

    for r in valid_results:
        if r.meets_all_targets is True:
            tier1.append(r)
        else:
            tier2.append(r)

    if has_crf:
        # Mixed group: sort by bitrate, metric priority as tiebreaker
        tier1.sort(
            key=lambda r: (
                r.predicted_bitrate_kbps,
                _metric_priority_sort_key(r, priority),
            )
        )
        tier2.sort(
            key=lambda r: (
                r.predicted_bitrate_kbps,
                _metric_priority_sort_key(r, priority),
            )
        )
    else:
        # All-ABR group: sort by metric priority only (no bitrate ranking)
        tier1.sort(key=lambda r: _metric_priority_sort_key(r, priority))
        tier2.sort(key=lambda r: _metric_priority_sort_key(r, priority))

    return tier1 + tier2
