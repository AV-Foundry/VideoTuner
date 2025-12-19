"""Tests for pipeline multi-profile search module."""

from unittest.mock import MagicMock

from videotuner.crf_search import QualityTarget
from videotuner.pipeline_multi_profile import (
    MultiProfileSearchParams,
    rank_profile_results,
)
from videotuner.pipeline_types import MultiProfileResult
from videotuner.profiles import Profile


class TestMultiProfileSearchParams:
    """Tests for MultiProfileSearchParams dataclass."""

    def test_creation_with_all_fields(self):
        """Test that MultiProfileSearchParams can be created with all fields."""
        profiles = [Profile(name="test", description="Test profile", settings={})]
        targets = [QualityTarget(metric_name="vmaf_mean", target_value=90.0)]
        args = MagicMock()
        display = MagicMock()
        log = MagicMock()

        params = MultiProfileSearchParams(
            profiles=profiles,
            targets=targets,
            crf_start_value=18.0,
            crf_interval=0.5,
            max_iterations=10,
            args=args,
            display=display,
            log=log,
        )

        assert params.profiles == profiles
        assert params.targets == targets
        assert params.crf_start_value == 18.0
        assert params.crf_interval == 0.5
        assert params.max_iterations == 10


class TestRankProfileResults:
    """Tests for rank_profile_results function."""

    def test_ranks_meeting_targets_first(self):
        """Test that profiles meeting targets are ranked before those that don't."""
        result_meets = MultiProfileResult(
            profile_name="profile_a",
            optimal_crf=18.0,
            scores={"vmaf_mean": 95.0},
            predicted_bitrate_kbps=5000.0,
            converged=True,
            meets_all_targets=True,
        )
        result_fails = MultiProfileResult(
            profile_name="profile_b",
            optimal_crf=16.0,
            scores={"vmaf_mean": 89.0},
            predicted_bitrate_kbps=3000.0,  # Lower bitrate, but fails targets
            converged=True,
            meets_all_targets=False,
        )

        ranked = rank_profile_results([result_fails, result_meets])

        assert len(ranked) == 2
        assert ranked[0].profile_name == "profile_a"  # Meets targets
        assert ranked[1].profile_name == "profile_b"  # Fails targets

    def test_sorts_by_bitrate_within_tier(self):
        """Test that profiles are sorted by bitrate within the same tier."""
        result_a = MultiProfileResult(
            profile_name="profile_a",
            optimal_crf=18.0,
            scores={"vmaf_mean": 95.0},
            predicted_bitrate_kbps=5000.0,
            converged=True,
            meets_all_targets=True,
        )
        result_b = MultiProfileResult(
            profile_name="profile_b",
            optimal_crf=17.0,
            scores={"vmaf_mean": 96.0},
            predicted_bitrate_kbps=3000.0,  # Lower bitrate, same tier
            converged=True,
            meets_all_targets=True,
        )
        result_c = MultiProfileResult(
            profile_name="profile_c",
            optimal_crf=19.0,
            scores={"vmaf_mean": 94.0},
            predicted_bitrate_kbps=4000.0,
            converged=True,
            meets_all_targets=True,
        )

        ranked = rank_profile_results([result_a, result_b, result_c])

        assert len(ranked) == 3
        assert ranked[0].profile_name == "profile_b"  # 3000 kbps
        assert ranked[1].profile_name == "profile_c"  # 4000 kbps
        assert ranked[2].profile_name == "profile_a"  # 5000 kbps

    def test_bitrate_profiles_rank_with_meeting_targets(self):
        """Test that bitrate profiles rank alongside CRF profiles meeting targets."""
        result_crf = MultiProfileResult(
            profile_name="crf_profile",
            optimal_crf=18.0,
            scores={"vmaf_mean": 95.0},
            predicted_bitrate_kbps=5000.0,
            converged=True,
            meets_all_targets=True,
        )
        result_bitrate = MultiProfileResult(
            profile_name="bitrate_profile",
            optimal_crf=None,  # Bitrate mode
            scores={"vmaf_mean": 92.0},
            predicted_bitrate_kbps=3000.0,
            converged=True,
            meets_all_targets=None,  # N/A for bitrate mode
        )

        ranked = rank_profile_results([result_crf, result_bitrate])

        # Bitrate profile should rank first (lower bitrate)
        assert len(ranked) == 2
        assert ranked[0].profile_name == "bitrate_profile"
        assert ranked[1].profile_name == "crf_profile"

    def test_filters_invalid_results(self):
        """Test that invalid results are filtered out."""
        result_valid = MultiProfileResult(
            profile_name="valid_profile",
            optimal_crf=18.0,
            scores={"vmaf_mean": 95.0},
            predicted_bitrate_kbps=5000.0,
            converged=True,
            meets_all_targets=True,
        )
        result_invalid = MultiProfileResult(
            profile_name="invalid_profile",
            optimal_crf=None,
            scores={},
            predicted_bitrate_kbps=0.0,
            converged=False,
            meets_all_targets=False,
        )

        ranked = rank_profile_results([result_valid, result_invalid])

        assert len(ranked) == 1
        assert ranked[0].profile_name == "valid_profile"

    def test_returns_empty_list_for_no_valid_results(self):
        """Test that empty list is returned when no valid results exist."""
        result_invalid = MultiProfileResult(
            profile_name="invalid_profile",
            optimal_crf=None,
            scores={},
            predicted_bitrate_kbps=0.0,
            converged=False,
            meets_all_targets=False,
        )

        ranked = rank_profile_results([result_invalid])

        assert ranked == []

    def test_handles_empty_input(self):
        """Test that empty input returns empty list."""
        ranked = rank_profile_results([])
        assert ranked == []

    def test_crf_profiles_failing_targets_rank_last(self):
        """Test that CRF profiles failing targets rank after bitrate profiles."""
        result_bitrate = MultiProfileResult(
            profile_name="bitrate_profile",
            optimal_crf=None,
            scores={"vmaf_mean": 88.0},
            predicted_bitrate_kbps=6000.0,  # Higher bitrate
            converged=True,
            meets_all_targets=None,  # N/A
        )
        result_crf_fail = MultiProfileResult(
            profile_name="crf_fail",
            optimal_crf=16.0,
            scores={"vmaf_mean": 89.0},
            predicted_bitrate_kbps=3000.0,  # Lower bitrate but fails
            converged=True,
            meets_all_targets=False,
        )

        ranked = rank_profile_results([result_crf_fail, result_bitrate])

        # Bitrate profile (targets N/A) ranks before CRF profile failing targets
        assert len(ranked) == 2
        assert ranked[0].profile_name == "bitrate_profile"
        assert ranked[1].profile_name == "crf_fail"

    def test_complex_ranking_scenario(self):
        """Test ranking with mix of meeting, failing, and bitrate profiles."""
        result_meets_high_br = MultiProfileResult(
            profile_name="meets_high",
            optimal_crf=18.0,
            scores={"vmaf_mean": 96.0},
            predicted_bitrate_kbps=6000.0,
            converged=True,
            meets_all_targets=True,
        )
        result_meets_low_br = MultiProfileResult(
            profile_name="meets_low",
            optimal_crf=19.0,
            scores={"vmaf_mean": 95.0},
            predicted_bitrate_kbps=4000.0,
            converged=True,
            meets_all_targets=True,
        )
        result_bitrate = MultiProfileResult(
            profile_name="bitrate",
            optimal_crf=None,
            scores={"vmaf_mean": 92.0},
            predicted_bitrate_kbps=5000.0,
            converged=True,
            meets_all_targets=None,
        )
        result_fails = MultiProfileResult(
            profile_name="fails",
            optimal_crf=16.0,
            scores={"vmaf_mean": 88.0},
            predicted_bitrate_kbps=3000.0,  # Lowest bitrate but fails
            converged=True,
            meets_all_targets=False,
        )

        ranked = rank_profile_results(
            [
                result_meets_high_br,
                result_fails,
                result_bitrate,
                result_meets_low_br,
            ]
        )

        # Expected order:
        # 1. meets_low (4000 kbps, meets targets)
        # 2. bitrate (5000 kbps, targets N/A)
        # 3. meets_high (6000 kbps, meets targets)
        # 4. fails (3000 kbps, but in failing tier)
        assert len(ranked) == 4
        assert ranked[0].profile_name == "meets_low"
        assert ranked[1].profile_name == "bitrate"
        assert ranked[2].profile_name == "meets_high"
        assert ranked[3].profile_name == "fails"
