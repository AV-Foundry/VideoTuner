"""Tests for CRF search algorithm."""

from videotuner.crf_search import CRFSearchState, QualityTarget, CRF_CEILING


class TestCRFSearchConvergence:
    """Tests for CRF search convergence logic."""

    def test_convergence_with_tight_bracket(self):
        """Test convergence when passing and failing CRFs are within one interval."""
        targets = [QualityTarget("vmaf_mean", 95.0)]
        state = CRFSearchState(targets, crf_interval=0.5)

        # CRF 16.0 meets targets
        state.add_result(16.0, {"vmaf_mean": 96.0})
        assert not state.is_converged()  # No failing CRF yet

        # CRF 16.5 also meets targets (higher is better)
        state.add_result(16.5, {"vmaf_mean": 95.5})
        assert not state.is_converged()  # Still no failing CRF

        # CRF 17.0 fails to meet targets
        state.add_result(17.0, {"vmaf_mean": 94.5})
        # Now we have passing_crf=16.5, failing_crf=17.0, gap=0.5 (equals interval)
        assert state.is_converged()
        assert state.get_optimal_crf() == 16.5

    def test_no_convergence_with_wide_bracket(self):
        """Test that search doesn't converge when bracket is wider than interval."""
        targets = [QualityTarget("vmaf_mean", 95.0)]
        state = CRFSearchState(targets, crf_interval=0.5)

        # CRF 16.0 meets targets
        state.add_result(16.0, {"vmaf_mean": 96.0})

        # CRF 18.0 fails to meet targets (gap=2.0, wider than interval)
        state.add_result(18.0, {"vmaf_mean": 94.0})

        assert not state.is_converged()  # Bracket too wide, can refine further
        assert state.get_optimal_crf() == 16.0

    def test_convergence_at_ceiling(self):
        """Test convergence when optimal CRF is at the ceiling."""
        targets = [QualityTarget("vmaf_mean", 95.0)]
        state = CRFSearchState(targets, crf_interval=0.5)

        # CRF at ceiling meets targets
        state.add_result(CRF_CEILING, {"vmaf_mean": 96.0})

        assert state.is_converged()  # At ceiling, can't go higher
        assert state.get_optimal_crf() == CRF_CEILING

    def test_no_convergence_without_failing_crf(self):
        """Test that search doesn't converge without testing a failing CRF above."""
        targets = [QualityTarget("vmaf_mean", 95.0)]
        state = CRFSearchState(targets, crf_interval=0.5)

        # Only tested one CRF that meets targets
        state.add_result(16.0, {"vmaf_mean": 96.0})

        assert not state.is_converged()  # Don't know if we can go higher
        assert state.get_optimal_crf() == 16.0

    def test_convergence_when_last_iteration_failed(self):
        """Test convergence works even when the most recent iteration didn't meet targets."""
        targets = [
            QualityTarget("vmaf_mean", 99.0),
            QualityTarget("ssim2_mean", 85.0),
        ]
        state = CRFSearchState(targets, crf_interval=0.5)

        # Iteration 1: CRF 16.0 meets all targets
        state.add_result(16.0, {"vmaf_mean": 99.4, "ssim2_mean": 85.9})
        assert not state.is_converged()

        # Iteration 2: CRF 16.5 also meets all targets (new optimal)
        state.add_result(16.5, {"vmaf_mean": 99.3, "ssim2_mean": 85.2})
        assert not state.is_converged()

        # Iteration 3: CRF 17.0 fails to meet targets (SSIM2 too low)
        # This is the LAST iteration, but we still have optimal CRF at 16.5
        state.add_result(17.0, {"vmaf_mean": 99.2, "ssim2_mean": 84.4})

        # Should converge! We have passing_crf=16.5, failing_crf=17.0, gap=0.5
        assert state.is_converged()
        assert state.get_optimal_crf() == 16.5

        # Verify we don't require all_targets_met() for the current iteration
        assert not state.all_targets_met()  # Current iteration failed

    def test_multiple_targets_all_must_pass(self):
        """Test that all targets must be met for a CRF to be considered passing."""
        targets = [
            QualityTarget("vmaf_mean", 99.0),
            QualityTarget("ssim2_mean", 85.0),
        ]
        state = CRFSearchState(targets, crf_interval=0.5)

        # CRF 16.0: VMAF passes but SSIM2 fails
        state.add_result(16.0, {"vmaf_mean": 99.5, "ssim2_mean": 84.0})
        assert state.get_optimal_crf() is None  # No passing CRF yet

        # CRF 15.0: Both pass
        state.add_result(15.0, {"vmaf_mean": 99.8, "ssim2_mean": 86.0})
        assert state.get_optimal_crf() == 15.0

    def test_optimal_scores_match_optimal_crf(self):
        """Test that get_optimal_scores returns scores from the optimal CRF."""
        targets = [QualityTarget("vmaf_mean", 95.0)]
        state = CRFSearchState(targets, crf_interval=0.5)

        state.add_result(16.0, {"vmaf_mean": 96.0, "ssim2_mean": 84.0})
        state.add_result(16.5, {"vmaf_mean": 95.5, "ssim2_mean": 83.0})
        state.add_result(17.0, {"vmaf_mean": 94.5, "ssim2_mean": 82.0})

        # Optimal is 16.5 (highest passing CRF)
        assert state.get_optimal_crf() == 16.5
        scores = state.get_optimal_scores()
        assert scores is not None
        assert scores["vmaf_mean"] == 95.5
        assert scores["ssim2_mean"] == 83.0

    def test_no_optimal_when_no_passing_crf(self):
        """Test that optimal CRF is None when no CRF meets targets."""
        targets = [QualityTarget("vmaf_mean", 99.0)]
        state = CRFSearchState(targets, crf_interval=0.5)

        # Both fail to meet target
        state.add_result(16.0, {"vmaf_mean": 98.0})
        state.add_result(15.0, {"vmaf_mean": 98.5})

        assert state.get_optimal_crf() is None
        assert state.get_optimal_scores() is None


class TestCRFSearchNextCalculation:
    """Tests for calculating the next CRF to test."""

    def test_increase_crf_when_targets_met(self):
        """Test that CRF increases when targets are met."""
        targets = [QualityTarget("vmaf_mean", 95.0)]
        state = CRFSearchState(targets, crf_interval=0.5)

        state.add_result(16.0, {"vmaf_mean": 96.0})
        next_crf = state.calculate_next_crf(16.0)

        assert next_crf is not None
        assert next_crf > 16.0  # Should try higher CRF

    def test_decrease_crf_when_targets_not_met(self):
        """Test that CRF decreases when targets are not met."""
        targets = [QualityTarget("vmaf_mean", 95.0)]
        state = CRFSearchState(targets, crf_interval=0.5)

        state.add_result(16.0, {"vmaf_mean": 94.0})
        next_crf = state.calculate_next_crf(16.0)

        assert next_crf is not None
        assert next_crf < 16.0  # Should try lower CRF

    def test_no_next_when_converged(self):
        """Test that calculate_next_crf returns None when converged."""
        targets = [QualityTarget("vmaf_mean", 95.0)]
        state = CRFSearchState(targets, crf_interval=0.5)

        state.add_result(16.0, {"vmaf_mean": 96.0})
        state.add_result(16.5, {"vmaf_mean": 95.5})
        state.add_result(17.0, {"vmaf_mean": 94.5})

        # Converged: passing=16.5, failing=17.0, gap=0.5
        assert state.is_converged()
        next_crf = state.calculate_next_crf(17.0)
        assert next_crf is None

    def test_binary_search_between_bounds(self):
        """Test that next CRF is between passing and failing bounds when both exist."""
        targets = [QualityTarget("vmaf_mean", 95.0)]
        state = CRFSearchState(targets, crf_interval=0.5)

        # Create wide bracket
        state.add_result(15.0, {"vmaf_mean": 97.0})  # Passing
        state.add_result(20.0, {"vmaf_mean": 92.0})  # Failing

        # Next CRF should be between 15.0 and 20.0
        next_crf = state.calculate_next_crf(20.0)
        assert next_crf is not None
        assert 15.0 < next_crf < 20.0


class TestQualityTarget:
    """Tests for QualityTarget class."""

    def test_target_met_when_current_exceeds_target(self):
        """Test that is_met returns True when current >= target."""
        target = QualityTarget("vmaf_mean", 95.0, current_value=96.0)
        assert target.is_met()

    def test_target_met_when_current_equals_target(self):
        """Test that is_met returns True when current equals target."""
        target = QualityTarget("vmaf_mean", 95.0, current_value=95.0)
        assert target.is_met()

    def test_target_not_met_when_current_below_target(self):
        """Test that is_met returns False when current < target."""
        target = QualityTarget("vmaf_mean", 95.0, current_value=94.5)
        assert not target.is_met()

    def test_target_not_met_when_no_current_value(self):
        """Test that is_met returns False when current_value is None."""
        target = QualityTarget("vmaf_mean", 95.0)
        assert not target.is_met()

    def test_delta_calculation(self):
        """Test that delta correctly calculates distance from target."""
        target = QualityTarget("vmaf_mean", 95.0, current_value=96.5)
        assert target.delta() == 1.5  # 96.5 - 95.0

        target2 = QualityTarget("vmaf_mean", 95.0, current_value=94.0)
        assert target2.delta() == -1.0  # 94.0 - 95.0

    def test_delta_none_when_no_current_value(self):
        """Test that delta returns None when current_value is None."""
        target = QualityTarget("vmaf_mean", 95.0)
        assert target.delta() is None

    def test_target_met_with_rounded_value(self):
        """Test that values rounded to 2 decimal places meet targets correctly.

        This ensures consistency between display (always 2 decimals) and
        comparison logic. If it displays as 99.00, it should pass a 99.0 target.
        """
        # Value exactly at target threshold
        target = QualityTarget("vmaf_mean", 99.0, current_value=99.0)
        assert target.is_met()
        assert target.delta() == 0.0
