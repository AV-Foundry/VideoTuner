"""Tests for pipeline multi-profile search module."""

from unittest.mock import MagicMock

from videotuner.crf_search import QualityTarget
from videotuner.constants import METRIC_PRIORITY
from videotuner.pipeline_multi_profile import (
    MultiProfileSearchParams,
    rank_profile_results,
    get_effective_metric_priority,
    metric_priority_sort_key,
)
from videotuner.pipeline_types import MultiProfileResult
from videotuner.encoder_type import EncoderType
from videotuner.profiles import Profile


class TestMultiProfileSearchParams:
    """Tests for MultiProfileSearchParams dataclass."""

    def test_creation_with_all_fields(self):
        """Test that MultiProfileSearchParams can be created with all fields."""
        profiles = [
            Profile(
                name="test",
                description="Test profile",
                settings={},
                encoder=EncoderType.X265,
            )
        ]
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


class TestMetricPriority:
    """Tests for metric priority computation and sort keys."""

    def test_default_priority_order(self):
        """Test default priority when no targets specified."""
        priority = get_effective_metric_priority([])
        assert priority == METRIC_PRIORITY

    def test_target_promotion(self):
        """Test that targeted metrics are promoted to top of priority."""
        targets = [
            QualityTarget("vmaf_1pct", 90.0),
            QualityTarget("ssim2_95pct", 70.0),
        ]
        priority = get_effective_metric_priority(targets)
        # vmaf_1pct and ssim2_95pct promoted, rest follow in default order
        assert priority[0] == "vmaf_1pct"
        assert priority[1] == "ssim2_95pct"
        assert priority[2] == "vmaf_mean"  # First non-targeted in default order

    def test_promotion_preserves_default_relative_order(self):
        """Test that promoted metrics maintain their default relative order."""
        # Specify targets in reverse of default order
        targets = [
            QualityTarget("ssim2_95pct", 70.0),
            QualityTarget("vmaf_mean", 90.0),
        ]
        priority = get_effective_metric_priority(targets)
        # Should be in default order, not target specification order
        assert priority[0] == "vmaf_mean"
        assert priority[1] == "ssim2_95pct"

    def test_all_metrics_targeted(self):
        """Test that targeting all metrics produces the default order."""
        targets = [QualityTarget(m, 50.0) for m in METRIC_PRIORITY]
        priority = get_effective_metric_priority(targets)
        assert priority == METRIC_PRIORITY

    def test_sort_key_higher_score_ranks_first(self):
        """Test that higher scores produce lower sort keys (better rank)."""
        result_high = MultiProfileResult(
            profile_name="high",
            optimal_crf=None,
            scores={"vmaf_mean": 96.0, "vmaf_hmean": 95.0},
            predicted_bitrate_kbps=5000.0,
            converged=True,
        )
        result_low = MultiProfileResult(
            profile_name="low",
            optimal_crf=None,
            scores={"vmaf_mean": 92.0, "vmaf_hmean": 91.0},
            predicted_bitrate_kbps=5000.0,
            converged=True,
        )
        key_high = metric_priority_sort_key(result_high, METRIC_PRIORITY)
        key_low = metric_priority_sort_key(result_low, METRIC_PRIORITY)
        assert key_high < key_low  # Lower key = better

    def test_sort_key_missing_scores_rank_last(self):
        """Test that missing scores produce worst sort keys."""
        result_full = MultiProfileResult(
            profile_name="full",
            optimal_crf=None,
            scores={"vmaf_mean": 50.0},
            predicted_bitrate_kbps=5000.0,
            converged=True,
        )
        result_empty = MultiProfileResult(
            profile_name="empty",
            optimal_crf=None,
            scores={},
            predicted_bitrate_kbps=5000.0,
            converged=True,
        )
        key_full = metric_priority_sort_key(result_full, METRIC_PRIORITY)
        key_empty = metric_priority_sort_key(result_empty, METRIC_PRIORITY)
        assert key_full < key_empty

    def test_sort_key_tiebreaker_on_secondary_metric(self):
        """Test that tied primary metric falls through to secondary."""
        result_a = MultiProfileResult(
            profile_name="a",
            optimal_crf=None,
            scores={"vmaf_mean": 95.0, "vmaf_hmean": 94.0},
            predicted_bitrate_kbps=5000.0,
            converged=True,
        )
        result_b = MultiProfileResult(
            profile_name="b",
            optimal_crf=None,
            scores={"vmaf_mean": 95.0, "vmaf_hmean": 93.0},
            predicted_bitrate_kbps=5000.0,
            converged=True,
        )
        key_a = metric_priority_sort_key(result_a, METRIC_PRIORITY)
        key_b = metric_priority_sort_key(result_b, METRIC_PRIORITY)
        assert key_a < key_b  # Higher vmaf_hmean = better


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
        """Test that CRF profiles are sorted by bitrate within the same tier."""
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

    def test_bitrate_profiles_with_met_targets_rank_in_tier1(self):
        """Test that bitrate profiles meeting targets rank in tier 1 alongside CRF."""
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
            optimal_crf=None,
            scores={"vmaf_mean": 92.0},
            predicted_bitrate_kbps=3000.0,
            converged=True,
            meets_all_targets=True,  # Met targets
        )
        targets = [QualityTarget("vmaf_mean", 90.0)]

        ranked = rank_profile_results([result_crf, result_bitrate], targets)

        # Both in tier 1, bitrate profile wins on lower bitrate (mixed group)
        assert len(ranked) == 2
        assert ranked[0].profile_name == "bitrate_profile"
        assert ranked[1].profile_name == "crf_profile"

    def test_bitrate_profiles_without_targets_rank_in_tier2(self):
        """Test that bitrate profiles with meets_all_targets=None rank in tier 2."""
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
            optimal_crf=None,
            scores={"vmaf_mean": 92.0},
            predicted_bitrate_kbps=3000.0,
            converged=True,
            meets_all_targets=None,  # No targets evaluated
        )

        ranked = rank_profile_results([result_crf, result_bitrate])

        # CRF in tier 1, bitrate in tier 2
        assert len(ranked) == 2
        assert ranked[0].profile_name == "crf_profile"
        assert ranked[1].profile_name == "bitrate_profile"

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
        """Test that CRF profiles failing targets rank in tier 2."""
        result_crf_meets = MultiProfileResult(
            profile_name="crf_meets",
            optimal_crf=18.0,
            scores={"vmaf_mean": 95.0},
            predicted_bitrate_kbps=5000.0,
            converged=True,
            meets_all_targets=True,
        )
        result_crf_fail = MultiProfileResult(
            profile_name="crf_fail",
            optimal_crf=16.0,
            scores={"vmaf_mean": 89.0},
            predicted_bitrate_kbps=3000.0,  # Lower bitrate but fails
            converged=True,
            meets_all_targets=False,
        )

        ranked = rank_profile_results([result_crf_fail, result_crf_meets])

        assert len(ranked) == 2
        assert ranked[0].profile_name == "crf_meets"
        assert ranked[1].profile_name == "crf_fail"

    def test_complex_mixed_ranking_scenario(self):
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
        result_bitrate_meets = MultiProfileResult(
            profile_name="bitrate",
            optimal_crf=None,
            scores={"vmaf_mean": 92.0},
            predicted_bitrate_kbps=5000.0,
            converged=True,
            meets_all_targets=True,  # Met targets
        )
        result_fails = MultiProfileResult(
            profile_name="fails",
            optimal_crf=16.0,
            scores={"vmaf_mean": 88.0},
            predicted_bitrate_kbps=3000.0,  # Lowest bitrate but fails
            converged=True,
            meets_all_targets=False,
        )
        targets = [QualityTarget("vmaf_mean", 90.0)]

        ranked = rank_profile_results(
            [
                result_meets_high_br,
                result_fails,
                result_bitrate_meets,
                result_meets_low_br,
            ],
            targets,
        )

        # Tier 1 (met targets): sorted by bitrate
        # 1. meets_low (4000 kbps)
        # 2. bitrate (5000 kbps)
        # 3. meets_high (6000 kbps)
        # Tier 2 (failed): fails (3000 kbps)
        assert len(ranked) == 4
        assert ranked[0].profile_name == "meets_low"
        assert ranked[1].profile_name == "bitrate"
        assert ranked[2].profile_name == "meets_high"
        assert ranked[3].profile_name == "fails"


class TestAllABRRanking:
    """Tests for all-ABR group ranking (quality-based, not bitrate-based)."""

    def test_all_abr_ranked_by_quality_not_bitrate(self):
        """Test all-ABR profiles rank by quality scores, ignoring bitrate."""
        high_quality_high_br = MultiProfileResult(
            profile_name="high_q",
            optimal_crf=None,
            scores={"vmaf_mean": 96.0},
            predicted_bitrate_kbps=8000.0,
            converged=True,
            meets_all_targets=None,
        )
        low_quality_low_br = MultiProfileResult(
            profile_name="low_q",
            optimal_crf=None,
            scores={"vmaf_mean": 90.0},
            predicted_bitrate_kbps=4000.0,
            converged=True,
            meets_all_targets=None,
        )

        ranked = rank_profile_results([low_quality_low_br, high_quality_high_br])

        # Higher quality wins despite higher bitrate
        assert ranked[0].profile_name == "high_q"
        assert ranked[1].profile_name == "low_q"

    def test_all_abr_with_targets_tiers(self):
        """Test all-ABR with targets: met-targets tier ranks first."""
        meets = MultiProfileResult(
            profile_name="meets",
            optimal_crf=None,
            scores={"vmaf_mean": 91.0},
            predicted_bitrate_kbps=8000.0,
            converged=True,
            meets_all_targets=True,
        )
        fails = MultiProfileResult(
            profile_name="fails",
            optimal_crf=None,
            scores={"vmaf_mean": 96.0},  # Higher quality but fails targets
            predicted_bitrate_kbps=4000.0,
            converged=True,
            meets_all_targets=False,
        )
        targets = [QualityTarget("vmaf_mean", 90.0)]

        ranked = rank_profile_results([fails, meets], targets)

        assert ranked[0].profile_name == "meets"
        assert ranked[1].profile_name == "fails"

    def test_all_abr_tiebreaker_uses_secondary_metric(self):
        """Test that tied primary metric falls through to secondary."""
        result_a = MultiProfileResult(
            profile_name="a",
            optimal_crf=None,
            scores={"vmaf_mean": 95.0, "vmaf_hmean": 94.0},
            predicted_bitrate_kbps=5000.0,
            converged=True,
            meets_all_targets=None,
        )
        result_b = MultiProfileResult(
            profile_name="b",
            optimal_crf=None,
            scores={"vmaf_mean": 95.0, "vmaf_hmean": 93.0},
            predicted_bitrate_kbps=5000.0,
            converged=True,
            meets_all_targets=None,
        )

        ranked = rank_profile_results([result_b, result_a])

        assert ranked[0].profile_name == "a"  # Higher vmaf_hmean wins

    def test_all_abr_target_promotion_affects_ranking(self):
        """Test that target promotion changes which metric is ranked first."""
        # Profile a has higher vmaf_mean, profile b has higher ssim2_5pct
        result_a = MultiProfileResult(
            profile_name="a",
            optimal_crf=None,
            scores={"vmaf_mean": 96.0, "ssim2_5pct": 70.0},
            predicted_bitrate_kbps=5000.0,
            converged=True,
            meets_all_targets=True,
        )
        result_b = MultiProfileResult(
            profile_name="b",
            optimal_crf=None,
            scores={"vmaf_mean": 94.0, "ssim2_5pct": 80.0},
            predicted_bitrate_kbps=5000.0,
            converged=True,
            meets_all_targets=True,
        )

        # Without target promotion: vmaf_mean is primary, so "a" wins
        ranked_default = rank_profile_results([result_b, result_a])
        assert ranked_default[0].profile_name == "a"

        # With ssim2_5pct as target: promoted to top, so "b" wins
        targets = [QualityTarget("ssim2_5pct", 60.0)]
        ranked_promoted = rank_profile_results([result_b, result_a], targets)
        assert ranked_promoted[0].profile_name == "b"


class TestMixedGroupRanking:
    """Tests for mixed CRF+ABR group ranking."""

    def test_mixed_abr_met_targets_in_tier1(self):
        """Test ABR profiles with met targets rank in tier 1 alongside CRF."""
        crf_meets = MultiProfileResult(
            profile_name="crf",
            optimal_crf=18.0,
            scores={"vmaf_mean": 95.0},
            predicted_bitrate_kbps=5000.0,
            converged=True,
            meets_all_targets=True,
        )
        abr_meets = MultiProfileResult(
            profile_name="abr",
            optimal_crf=None,
            scores={"vmaf_mean": 93.0},
            predicted_bitrate_kbps=3000.0,
            converged=True,
            meets_all_targets=True,
        )
        targets = [QualityTarget("vmaf_mean", 90.0)]

        ranked = rank_profile_results([crf_meets, abr_meets], targets)

        # ABR wins in tier 1 due to lower bitrate
        assert ranked[0].profile_name == "abr"
        assert ranked[1].profile_name == "crf"

    def test_mixed_abr_no_targets_in_tier2(self):
        """Test ABR profiles with no target evaluation go to tier 2."""
        crf_meets = MultiProfileResult(
            profile_name="crf",
            optimal_crf=18.0,
            scores={"vmaf_mean": 95.0},
            predicted_bitrate_kbps=5000.0,
            converged=True,
            meets_all_targets=True,
        )
        abr_no_eval = MultiProfileResult(
            profile_name="abr",
            optimal_crf=None,
            scores={"vmaf_mean": 98.0},
            predicted_bitrate_kbps=3000.0,
            converged=True,
            meets_all_targets=None,  # Not evaluated
        )

        ranked = rank_profile_results([abr_no_eval, crf_meets])

        # CRF in tier 1, ABR in tier 2
        assert ranked[0].profile_name == "crf"
        assert ranked[1].profile_name == "abr"

    def test_mixed_bitrate_tiebreaker(self):
        """Test mixed group uses metric priority as bitrate tiebreaker."""
        crf_a = MultiProfileResult(
            profile_name="a",
            optimal_crf=18.0,
            scores={"vmaf_mean": 96.0},
            predicted_bitrate_kbps=5000.0,
            converged=True,
            meets_all_targets=True,
        )
        crf_b = MultiProfileResult(
            profile_name="b",
            optimal_crf=19.0,
            scores={"vmaf_mean": 94.0},
            predicted_bitrate_kbps=5000.0,  # Same bitrate
            converged=True,
            meets_all_targets=True,
        )
        abr = MultiProfileResult(
            profile_name="abr",
            optimal_crf=None,
            scores={"vmaf_mean": 95.0},
            predicted_bitrate_kbps=5000.0,  # Same bitrate
            converged=True,
            meets_all_targets=True,
        )
        targets = [QualityTarget("vmaf_mean", 90.0)]

        ranked = rank_profile_results([crf_b, abr, crf_a], targets)

        # All same bitrate, so metric priority breaks tie
        assert ranked[0].profile_name == "a"  # vmaf 96
        assert ranked[1].profile_name == "abr"  # vmaf 95
        assert ranked[2].profile_name == "b"  # vmaf 94
