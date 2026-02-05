"""CRF search algorithm for iterative quality targeting.

Implements interpolated binary search to find the optimal CRF value
that meets all user-specified quality targets.
"""

from __future__ import annotations

from dataclasses import dataclass

from .constants import METRIC_DECIMALS
from .pipeline_cli import DEFAULT_CRF_START_VALUE


class CRFFloorError(Exception):
    """Raised when CRF reaches the minimum floor without meeting targets."""

    crf_floor: float
    unmet_targets: list[tuple[str, float, float]]

    def __init__(self, crf_floor: float, unmet_targets: list[tuple[str, float, float]]):
        """
        Args:
            crf_floor: The CRF floor that was reached
            unmet_targets: List of (metric_name, target_value, current_value) tuples
        """
        self.crf_floor = crf_floor
        self.unmet_targets = unmet_targets

        target_details = ", ".join(
            f"{name}: {current:.{METRIC_DECIMALS}f} < {target:.{METRIC_DECIMALS}f}"
            for name, target, current in unmet_targets
        )
        super().__init__(
            f"CRF search reached minimum CRF {crf_floor} without meeting all targets. "
            + f"Unmet targets: {target_details}. "
            + "Try lowering your target values or using different encoding settings."
        )


@dataclass
class QualityTarget:
    """Represents a single quality metric target."""

    metric_name: str  # e.g., "vmaf_mean", "ssim2_5pct"
    target_value: float
    current_value: float | None = None
    metric_decimals: int = METRIC_DECIMALS

    def is_met(self) -> bool:
        """Check if target is met (current >= target, rounded to metric_decimals)."""
        if self.current_value is None:
            return False
        return round(self.current_value, self.metric_decimals) >= round(
            self.target_value, self.metric_decimals
        )

    def delta(self) -> float | None:
        """Calculate how far we are from target (positive = met, negative = not met)."""
        if self.current_value is None:
            return None
        return self.current_value - self.target_value


@dataclass(frozen=True)
class CRFResult:
    """Represents a tested CRF value and its resulting scores."""

    crf: float
    scores: dict[str, float]  # metric_name -> score


# Minimum CRF value - if we reach this without meeting targets, raise an error
CRF_FLOOR = 1.0
CRF_CEILING = 51.0


class CRFSearchState:
    """Manages the state of iterative CRF search using interpolated binary search."""

    def __init__(self, targets: list[QualityTarget], crf_interval: float):
        self.targets: list[QualityTarget] = targets
        self.crf_interval: float = crf_interval
        self.history: list[CRFResult] = []

        # Track bounds for interpolation
        # passing_crf = highest CRF where ALL targets are met (what we want!)
        # failing_crf = lowest CRF where targets are NOT met (need better quality)
        self.passing_crf: CRFResult | None = None
        self.failing_crf: CRFResult | None = None

    def add_result(self, crf: float, scores: dict[str, float]) -> None:
        """Add a test result and update bounds."""
        result = CRFResult(crf, scores)
        self.history.append(result)

        # Update targets with current values
        for target in self.targets:
            if target.metric_name in scores:
                target.current_value = scores[target.metric_name]

        # Check if all targets met with this CRF
        targets_met = self._check_targets_met(scores)

        if targets_met:
            # This CRF meets all targets - candidate for passing_crf
            # We want the HIGHEST CRF that passes (smallest file size)
            if self.passing_crf is None or crf > self.passing_crf.crf:
                self.passing_crf = result
        else:
            # This CRF doesn't meet targets - candidate for failing_crf
            # We track the LOWEST CRF that fails (closest to passing)
            if self.failing_crf is None or crf < self.failing_crf.crf:
                self.failing_crf = result

    def _check_targets_met(self, scores: dict[str, float]) -> bool:
        """Check if all targets are met with given scores (rounded to metric_decimals)."""
        for target in self.targets:
            if target.metric_name not in scores:
                return False
            if round(scores[target.metric_name], target.metric_decimals) < round(
                target.target_value, target.metric_decimals
            ):
                return False
        return True

    def all_targets_met(self) -> bool:
        """Check if all targets are currently met (with last tested CRF)."""
        return all(t.is_met() for t in self.targets)

    def has_been_tested(self, crf: float) -> bool:
        """Check if we've already tested this CRF value."""
        return any(abs(h.crf - crf) < 0.01 for h in self.history)

    def calculate_next_crf(self, current_crf: float) -> float | None:
        """Calculate the next CRF value to test.

        Returns:
            Next CRF to test, or None if converged/error
        """

        # Case 1: All targets met - try higher CRF to find optimal (smallest file)
        if self.all_targets_met():
            # Do we have a failing CRF above us?
            if self.failing_crf is not None and self.failing_crf.crf > current_crf:
                # We're bracketed: current CRF passes, failing_crf doesn't
                # Try midpoint
                next_crf = (current_crf + self.failing_crf.crf) / 2.0
                next_crf = self._round_to_interval(next_crf)

                # Ensure it's at least one interval higher and not already tested
                if next_crf <= current_crf or self.has_been_tested(next_crf):
                    # Bracket too narrow or already tested - converged
                    return None
                return min(next_crf, CRF_CEILING)
            else:
                # No failing CRF above us - make exploratory jump up
                # Be aggressive - if we overshoot, binary search will refine
                closest_target = self._find_closest_to_target()
                if closest_target is not None:
                    closest_delta = closest_target.delta()
                else:
                    closest_delta = None
                if closest_target is not None and closest_delta is not None:
                    delta = closest_delta
                    # Scale jump with headroom - larger gaps allow bigger jumps
                    if delta >= 1.0:
                        # Scale proportionally: 1 CRF per VMAF point, min 3.0, max 15.0
                        jump = max(3.0, min(delta * 1.0, 15.0))
                    elif delta >= 0.3:
                        jump = 2.0  # Moderate headroom
                    elif delta >= 0.1:
                        jump = 1.0  # Small headroom
                    else:
                        jump = self.crf_interval  # Very close - refining
                else:
                    jump = 2.0  # No data - moderate aggressive

                next_crf = current_crf + jump
                next_crf = self._round_to_interval(next_crf)
                next_crf = min(next_crf, CRF_CEILING)

                # Ensure we haven't already tested this value
                if self.has_been_tested(next_crf):
                    # Already tested - if we've tested ceiling and it passes, we've converged at ceiling
                    # Otherwise, we're converged within the interval
                    return None

                return next_crf

        # Case 2: Targets NOT met - need better quality (lower CRF)

        # Find the worst-failing target to guide interpolation
        worst_target = self._find_worst_target()

        # If we have both passing and failing CRFs, try interpolation
        if (
            self.passing_crf is not None
            and self.failing_crf is not None
            and worst_target is not None
        ):
            interpolated = self._interpolate(worst_target)
            if interpolated is not None:
                # Check if interpolation is within bounds
                lower_bound = min(self.passing_crf.crf, self.failing_crf.crf)
                upper_bound = max(self.passing_crf.crf, self.failing_crf.crf)

                if lower_bound <= interpolated <= upper_bound:
                    # Interpolation is within tested range
                    rounded = self._round_to_interval(interpolated)

                    # Check if we've already tested this value
                    if not self.has_been_tested(rounded):
                        return rounded

                    # Interpolation rounded to already-tested value
                    # Try the midpoint between bounds instead
                    midpoint = (lower_bound + upper_bound) / 2.0
                    rounded_mid = self._round_to_interval(midpoint)

                    if (
                        not self.has_been_tested(rounded_mid)
                        and lower_bound < rounded_mid < upper_bound
                    ):
                        return rounded_mid

                    # Bracket too narrow to continue
                    return None

        # Try interpolation with just passing CRF if available
        if self.passing_crf is not None and worst_target is not None:
            # We're above a passing CRF - interpolate down
            interpolated = self._interpolate_from_passing(worst_target, current_crf)
            if interpolated is not None and interpolated < current_crf:
                rounded = self._round_to_interval(max(interpolated, CRF_FLOOR))
                if not self.has_been_tested(rounded):
                    return rounded

        # Fallback: Make exploratory jump down
        # Scale jump with deficit size - larger gaps need bigger jumps
        worst_delta = worst_target.delta() if worst_target is not None else None
        if worst_delta is not None:
            abs_delta = abs(worst_delta)
            if abs_delta >= 1.0:
                # Scale proportionally: 1 CRF per VMAF point, min 3.0, max 15.0
                jump = max(3.0, min(abs_delta * 1.0, 15.0))
            elif abs_delta >= 0.3:
                jump = 2.0  # Moderate deficit
            elif abs_delta >= 0.1:
                jump = 1.0  # Small deficit
            else:
                jump = self.crf_interval  # Very close - refining
        else:
            jump = 3.0  # No data - aggressive jump

        next_crf = current_crf - jump
        next_crf = self._round_to_interval(max(next_crf, CRF_FLOOR))

        # Check if we've hit the CRF floor
        if next_crf <= CRF_FLOOR and self.has_been_tested(CRF_FLOOR):
            # We've already tested CRF floor and still don't meet targets
            self._raise_floor_error()

        # Ensure we haven't already tested this value
        if self.has_been_tested(next_crf):
            # Try smaller jump
            next_crf = current_crf - self.crf_interval
            next_crf = self._round_to_interval(max(next_crf, CRF_FLOOR))

            # Check floor again
            if next_crf <= CRF_FLOOR and self.has_been_tested(CRF_FLOOR):
                self._raise_floor_error()

            if self.has_been_tested(next_crf):
                # Can't make progress
                return None

        return next_crf

    def _find_worst_target(self) -> QualityTarget | None:
        """Find the target that's farthest from being met (most negative delta)."""
        targets_with_values: list[tuple[QualityTarget, float]] = []
        for t in self.targets:
            delta = t.delta()
            if delta is None:
                continue
            targets_with_values.append((t, delta))
        if not targets_with_values:
            return None
        return min(targets_with_values, key=lambda item: item[1])[0]

    def _find_closest_to_target(self) -> QualityTarget | None:
        """Find the target that's closest to its target value (smallest positive delta)."""
        targets_with_values: list[tuple[QualityTarget, float]] = []
        for t in self.targets:
            delta = t.delta()
            if delta is None or not t.is_met():
                continue
            targets_with_values.append((t, delta))
        if not targets_with_values:
            return None
        return min(targets_with_values, key=lambda item: item[1])[0]

    def _has_exact_match_when_all_met(self) -> bool:
        """Check if all targets are met AND at least one is an exact match.

        When a metric exactly equals its target (at display precision), going to
        a higher CRF will almost certainly cause it to fail. So if all targets
        are met and one is exact, we've found the optimal CRF.

        Returns:
            True if all targets met and at least one score exactly equals its target
        """
        if self.passing_crf is None:
            return False

        # Verify all targets are met with the passing CRF
        if not self._check_targets_met(self.passing_crf.scores):
            return False

        # Check if any target is an exact match
        for target in self.targets:
            if target.metric_name in self.passing_crf.scores:
                score = self.passing_crf.scores[target.metric_name]
                rounded_score = round(score, target.metric_decimals)
                rounded_target = round(target.target_value, target.metric_decimals)
                if rounded_score == rounded_target:
                    return True

        return False

    def _interpolate(self, target: QualityTarget) -> float | None:
        """Interpolate CRF between passing and failing bounds for given target."""
        if self.passing_crf is None or self.failing_crf is None:
            return None

        metric_name = target.metric_name
        if (
            metric_name not in self.passing_crf.scores
            or metric_name not in self.failing_crf.scores
        ):
            return None

        # Get scores at bounds
        pass_score = self.passing_crf.scores[metric_name]
        fail_score = self.failing_crf.scores[metric_name]
        target_score = target.target_value

        # Passing CRF has lower quality score (higher CRF)
        # Failing CRF has higher quality score (lower CRF) but still doesn't meet ALL targets
        # Note: This might seem counterintuitive, but failing_crf fails because OTHER targets aren't met

        # For interpolation, we want to find where target_score would be
        score_range = pass_score - fail_score
        if abs(score_range) < 0.01:
            return None  # Scores too similar

        # Linear interpolation
        ratio = (target_score - fail_score) / score_range
        crf_range = self.passing_crf.crf - self.failing_crf.crf
        interpolated_crf = self.failing_crf.crf + (ratio * crf_range)

        return interpolated_crf

    def _interpolate_from_passing(
        self, target: QualityTarget, current_crf: float
    ) -> float | None:
        """Interpolate between current position and passing CRF."""
        if self.passing_crf is None:
            return None

        metric_name = target.metric_name
        if metric_name not in self.passing_crf.scores or target.current_value is None:
            return None

        pass_score = self.passing_crf.scores[metric_name]
        current_score = target.current_value
        target_score = target.target_value

        score_range = pass_score - current_score
        if abs(score_range) < 0.01:
            return None

        ratio = (target_score - current_score) / score_range
        crf_range = self.passing_crf.crf - current_crf
        interpolated_crf = current_crf + (ratio * crf_range)

        return interpolated_crf

    def _round_to_interval(self, crf: float) -> float:
        """Round CRF to nearest interval."""
        return round(crf / self.crf_interval) * self.crf_interval

    def is_converged(self) -> bool:
        """Check if we've found and confirmed the optimal (highest) CRF that meets all targets.

        This is true when:
        1. We have a CRF that meets all targets (passing_crf), AND
        2. Either:
           - We're at the CRF ceiling, OR
           - All targets are met and at least one score exactly equals its target
             (going higher would almost certainly fail), OR
           - We've tested a higher CRF that doesn't meet targets, and the gap is too
             small to test anything in between (considering the CRF interval)
        """
        if self.passing_crf is None:
            return False

        # Special case: If passing CRF is at ceiling, we've found the optimal CRF
        if abs(self.passing_crf.crf - CRF_CEILING) < 0.01:
            return True

        # Special case: All targets met with at least one exact match
        # Going to a higher CRF would almost certainly cause the exact match to fail
        if self._has_exact_match_when_all_met():
            return True

        # Check if we've tested a higher CRF that failed to meet targets
        if self.failing_crf is not None and self.failing_crf.crf > self.passing_crf.crf:
            # We have confirmed bounds - check if the gap is narrow enough
            gap = abs(self.failing_crf.crf - self.passing_crf.crf)

            # If gap <= crf_interval, we can't test anything in between
            # (the midpoint would round to one of the already-tested values)
            if gap <= self.crf_interval:
                return True

            # Gap is still wide enough to refine - not converged yet
            return False

        # No failing CRF above us and not at ceiling - haven't confirmed optimal yet
        return False

    def get_optimal_crf(self) -> float | None:
        """Get the optimal CRF value (highest CRF that meets all targets)."""
        if self.passing_crf is not None:
            return self.passing_crf.crf
        return None

    def get_optimal_scores(self) -> dict[str, float] | None:
        """Get the scores from the optimal CRF (highest CRF that meets all targets)."""
        if self.passing_crf is not None:
            return self.passing_crf.scores
        return None

    def _raise_floor_error(self) -> None:
        """Raise CRFFloorError with details about unmet targets."""
        unmet: list[tuple[str, float, float]] = []
        for target in self.targets:
            if not target.is_met() and target.current_value is not None:
                unmet.append(
                    (target.metric_name, target.target_value, target.current_value)
                )
        raise CRFFloorError(CRF_FLOOR, unmet)

    def check_floor_reached(self, current_crf: float) -> None:
        """Check if CRF floor has been reached without meeting targets.

        Should be called after add_result() when targets are not met.

        Args:
            current_crf: The current CRF value that was just tested

        Raises:
            CRFFloorError: If we're at or below CRF floor and targets not met
        """
        if current_crf <= CRF_FLOOR and not self.all_targets_met():
            self._raise_floor_error()


def estimate_starting_crf_from_results(
    previous_results: list[CRFResult],
    targets: list[QualityTarget],
    default_crf: float = DEFAULT_CRF_START_VALUE,
) -> float:
    """Estimate a good starting CRF based on previous search results.

    This is used when switching from preset-based search to profile-based search.
    It analyzes how close the previous results were to targets and estimates
    a starting point that should be close to meeting targets.

    Args:
        previous_results: Results from previous CRF search (e.g., preset-based)
        targets: Quality targets to meet
        default_crf: Default CRF if no useful data available

    Returns:
        Estimated starting CRF for the new search
    """
    if not previous_results:
        return default_crf

    # Find the result closest to meeting all targets
    # We want the highest CRF where we're close to meeting targets
    best_result: CRFResult | None = None
    best_score = float("-inf")

    for result in previous_results:
        # Calculate aggregate "closeness" to targets
        # Positive score = meets targets, negative = below targets
        min_delta = float("inf")
        all_metrics_present = True

        for target in targets:
            if target.metric_name not in result.scores:
                all_metrics_present = False
                break
            delta = result.scores[target.metric_name] - target.target_value
            min_delta = min(min_delta, delta)

        if not all_metrics_present:
            continue

        # Score: prefer results that meet targets (positive min_delta)
        # Among those that meet, prefer higher CRF
        # Among those that don't meet, prefer ones closer to meeting
        if min_delta >= 0:
            # Meets all targets - score by CRF (higher is better)
            score = result.crf + 100  # Offset to prioritize meeting targets
        else:
            # Doesn't meet all - score by how close we are
            score = min_delta

        if score > best_score:
            best_score = score
            best_result = result

    if best_result is None:
        return default_crf

    # If best result meets targets, start slightly higher (try for better compression)
    if best_score >= 100:  # Means it met targets (score = crf + 100)
        return min(best_result.crf + 2.0, CRF_CEILING)

    # If best result doesn't quite meet targets, start at that CRF
    # (the new profile might perform better)
    return best_result.crf
