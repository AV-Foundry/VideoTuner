"""Tests for pipeline validation utilities."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

if TYPE_CHECKING:
    from videotuner.profiles import Profile

import pytest

from videotuner.pipeline_validation import (
    AssessmentError,
    validate_assessment_results,
    has_targets,
    build_targets,
    check_scores_meet_targets,
    validate_metric_sampling,
    validate_sampling_parameters,
)
from videotuner.pipeline_cli import PipelineArgs, build_arg_parser, validate_args
from videotuner.crf_search import QualityTarget


class TestValidateAssessmentResults:
    """Tests for validate_assessment_results function."""

    def test_raises_when_no_results(self):
        """Test that AssessmentError is raised when no results available."""
        log = logging.getLogger("test")
        with pytest.raises(AssessmentError) as exc_info:
            validate_assessment_results(None, None, "test context", log)
        assert "No VMAF or SSIMULACRA2 results available" in str(exc_info.value)

    def test_raises_when_empty_results(self):
        """Test that AssessmentError is raised when results are empty lists."""
        log = logging.getLogger("test")
        with pytest.raises(AssessmentError) as exc_info:
            validate_assessment_results([], [], "test context", log)
        assert "No VMAF or SSIMULACRA2 results available" in str(exc_info.value)

    def test_raises_when_vmaf_all_nan(self):
        """Test that AssessmentError is raised when all VMAF scores are NaN."""
        log = logging.getLogger("test")
        vmaf_result = MagicMock()
        vmaf_result.mean = float("nan")
        with pytest.raises(AssessmentError) as exc_info:
            validate_assessment_results([vmaf_result], None, "test context", log)
        assert "All VMAF scores are unparseable" in str(exc_info.value)

    def test_raises_when_ssim2_all_nan(self):
        """Test that AssessmentError is raised when all SSIM2 scores are NaN."""
        log = logging.getLogger("test")
        ssim2_result = MagicMock()
        ssim2_result.mean = float("nan")
        with pytest.raises(AssessmentError) as exc_info:
            validate_assessment_results(None, [ssim2_result], "test context", log)
        assert "All SSIMULACRA2 scores are unparseable" in str(exc_info.value)

    def test_passes_with_valid_vmaf(self):
        """Test that validation passes with valid VMAF results."""
        log = logging.getLogger("test")
        vmaf_result = MagicMock()
        vmaf_result.mean = 95.5
        # Should not raise
        validate_assessment_results([vmaf_result], None, "test context", log)

    def test_passes_with_valid_ssim2(self):
        """Test that validation passes with valid SSIM2 results."""
        log = logging.getLogger("test")
        ssim2_result = MagicMock()
        ssim2_result.mean = 85.0
        # Should not raise
        validate_assessment_results(None, [ssim2_result], "test context", log)

    def test_passes_with_both_valid(self):
        """Test that validation passes with both valid results."""
        log = logging.getLogger("test")
        vmaf_result = MagicMock()
        vmaf_result.mean = 95.5
        ssim2_result = MagicMock()
        ssim2_result.mean = 85.0
        # Should not raise
        validate_assessment_results([vmaf_result], [ssim2_result], "test context", log)

    def test_fails_when_ssim2_nan_even_if_vmaf_valid(self):
        """Test that validation fails when SSIM2 is all NaN even if VMAF is valid."""
        log = logging.getLogger("test")
        vmaf_result = MagicMock()
        vmaf_result.mean = 95.5
        ssim2_result = MagicMock()
        ssim2_result.mean = float("nan")
        # Should fail because SSIM2 results are all NaN
        with pytest.raises(AssessmentError) as exc_info:
            validate_assessment_results(
                [vmaf_result], [ssim2_result], "test context", log
            )
        assert "All SSIMULACRA2 scores are unparseable" in str(exc_info.value)

    def test_passes_with_valid_vmaf_and_no_ssim2(self):
        """Test that validation passes with valid VMAF and no SSIM2 results."""
        log = logging.getLogger("test")
        vmaf_result = MagicMock()
        vmaf_result.mean = 95.5
        # Should pass because VMAF is valid and SSIM2 is None (not NaN)
        validate_assessment_results([vmaf_result], None, "test context", log)


class TestHasTargets:
    """Tests for has_targets function."""

    def test_returns_false_when_no_targets(self):
        """Test that has_targets returns False when no targets set."""
        args = PipelineArgs(input=Path("test.mkv"), output=Path("output.mkv"))
        assert has_targets(args) is False

    def test_returns_true_with_vmaf_target(self):
        """Test that has_targets returns True with vmaf_target."""
        args = PipelineArgs(
            input=Path("test.mkv"), output=Path("output.mkv"), vmaf_target=95.0
        )
        assert has_targets(args) is True

    def test_returns_true_with_vmaf_hmean_target(self):
        """Test that has_targets returns True with vmaf_hmean_target."""
        args = PipelineArgs(
            input=Path("test.mkv"), output=Path("output.mkv"), vmaf_hmean_target=93.0
        )
        assert has_targets(args) is True

    def test_returns_true_with_vmaf_1pct_target(self):
        """Test that has_targets returns True with vmaf_1pct_target."""
        args = PipelineArgs(
            input=Path("test.mkv"), output=Path("output.mkv"), vmaf_1pct_target=90.0
        )
        assert has_targets(args) is True

    def test_returns_true_with_vmaf_min_target(self):
        """Test that has_targets returns True with vmaf_min_target."""
        args = PipelineArgs(
            input=Path("test.mkv"), output=Path("output.mkv"), vmaf_min_target=85.0
        )
        assert has_targets(args) is True

    def test_returns_true_with_ssim2_mean_target(self):
        """Test that has_targets returns True with ssim2_mean_target."""
        args = PipelineArgs(
            input=Path("test.mkv"), output=Path("output.mkv"), ssim2_mean_target=80.0
        )
        assert has_targets(args) is True

    def test_returns_true_with_ssim2_median_target(self):
        """Test that has_targets returns True with ssim2_median_target."""
        args = PipelineArgs(
            input=Path("test.mkv"), output=Path("output.mkv"), ssim2_median_target=80.0
        )
        assert has_targets(args) is True

    def test_returns_true_with_ssim2_95pct_target(self):
        """Test that has_targets returns True with ssim2_95pct_target."""
        args = PipelineArgs(
            input=Path("test.mkv"), output=Path("output.mkv"), ssim2_95pct_target=75.0
        )
        assert has_targets(args) is True

    def test_returns_true_with_ssim2_5pct_target(self):
        """Test that has_targets returns True with ssim2_5pct_target."""
        args = PipelineArgs(
            input=Path("test.mkv"), output=Path("output.mkv"), ssim2_5pct_target=70.0
        )
        assert has_targets(args) is True

    def test_returns_true_with_multiple_targets(self):
        """Test that has_targets returns True with multiple targets."""
        args = PipelineArgs(
            input=Path("test.mkv"),
            output=Path("output.mkv"),
            vmaf_target=95.0,
            ssim2_mean_target=80.0,
        )
        assert has_targets(args) is True


class TestBuildTargets:
    """Tests for build_targets function."""

    def test_returns_empty_list_when_no_targets(self):
        """Test that build_targets returns empty list when no targets set."""
        args = PipelineArgs(input=Path("test.mkv"), output=Path("output.mkv"))
        targets = build_targets(args)
        assert targets == []

    def test_builds_vmaf_target(self):
        """Test that build_targets creates QualityTarget for vmaf_target."""
        args = PipelineArgs(
            input=Path("test.mkv"), output=Path("output.mkv"), vmaf_target=95.0
        )
        targets = build_targets(args)
        assert len(targets) == 1
        assert targets[0].metric_name == "vmaf_mean"
        assert targets[0].target_value == 95.0

    def test_builds_ssim2_target(self):
        """Test that build_targets creates QualityTarget for ssim2_mean_target."""
        args = PipelineArgs(
            input=Path("test.mkv"), output=Path("output.mkv"), ssim2_mean_target=80.0
        )
        targets = build_targets(args)
        assert len(targets) == 1
        assert targets[0].metric_name == "ssim2_mean"
        assert targets[0].target_value == 80.0

    def test_builds_multiple_targets(self):
        """Test that build_targets creates multiple QualityTargets."""
        args = PipelineArgs(
            input=Path("test.mkv"),
            output=Path("output.mkv"),
            vmaf_target=95.0,
            vmaf_1pct_target=90.0,
            ssim2_mean_target=80.0,
        )
        targets = build_targets(args)
        assert len(targets) == 3
        metric_names = [t.metric_name for t in targets]
        assert "vmaf_mean" in metric_names
        assert "vmaf_1pct" in metric_names
        assert "ssim2_mean" in metric_names

    def test_builds_all_target_types(self):
        """Test that build_targets handles all target types correctly."""
        args = PipelineArgs(
            input=Path("test.mkv"),
            output=Path("output.mkv"),
            vmaf_target=95.0,
            vmaf_hmean_target=93.0,
            vmaf_1pct_target=90.0,
            vmaf_min_target=85.0,
            ssim2_mean_target=80.0,
            ssim2_median_target=79.0,
            ssim2_95pct_target=75.0,
            ssim2_5pct_target=70.0,
        )
        targets = build_targets(args)
        assert len(targets) == 8


class TestCheckScoresMeetTargets:
    """Tests for check_scores_meet_targets function."""

    def test_returns_true_when_no_targets(self):
        """Test that check returns True when no targets specified."""
        scores: dict[str, float | None] = {"vmaf_mean": 95.0}
        targets: list[QualityTarget] = []
        assert check_scores_meet_targets(scores, targets) is True

    def test_returns_true_when_target_met(self):
        """Test that check returns True when target is met."""
        scores: dict[str, float | None] = {"vmaf_mean": 95.0}
        targets = [QualityTarget("vmaf_mean", 90.0)]
        assert check_scores_meet_targets(scores, targets) is True

    def test_returns_true_when_target_exactly_met(self):
        """Test that check returns True when target is exactly met."""
        scores: dict[str, float | None] = {"vmaf_mean": 90.0}
        targets = [QualityTarget("vmaf_mean", 90.0)]
        assert check_scores_meet_targets(scores, targets) is True

    def test_returns_false_when_target_not_met(self):
        """Test that check returns False when target is not met."""
        scores: dict[str, float | None] = {"vmaf_mean": 89.0}
        targets = [QualityTarget("vmaf_mean", 90.0)]
        assert check_scores_meet_targets(scores, targets) is False

    def test_returns_false_when_metric_missing(self):
        """Test that check returns False when metric is missing from scores."""
        scores: dict[str, float | None] = {"ssim2_mean": 80.0}
        targets = [QualityTarget("vmaf_mean", 90.0)]
        assert check_scores_meet_targets(scores, targets) is False

    def test_returns_false_when_score_is_none(self):
        """Test that check returns False when score value is None."""
        scores: dict[str, float | None] = {"vmaf_mean": None}
        targets = [QualityTarget("vmaf_mean", 90.0)]
        assert check_scores_meet_targets(scores, targets) is False

    def test_returns_true_when_all_targets_met(self):
        """Test that check returns True when all targets are met."""
        scores: dict[str, float | None] = {"vmaf_mean": 95.0, "ssim2_mean": 85.0}
        targets = [
            QualityTarget("vmaf_mean", 90.0),
            QualityTarget("ssim2_mean", 80.0),
        ]
        assert check_scores_meet_targets(scores, targets) is True

    def test_returns_false_when_any_target_not_met(self):
        """Test that check returns False when any target is not met."""
        scores: dict[str, float | None] = {"vmaf_mean": 95.0, "ssim2_mean": 75.0}
        targets = [
            QualityTarget("vmaf_mean", 90.0),
            QualityTarget("ssim2_mean", 80.0),
        ]
        assert check_scores_meet_targets(scores, targets) is False


class TestValidateMetricSampling:
    """Tests for validate_metric_sampling function."""

    def test_returns_valid_when_sufficient_frames(self):
        """Test that validation returns valid when frames are sufficient."""
        log = logging.getLogger("test")
        result = validate_metric_sampling(
            usable_frames=10000,
            total_frames=12000,
            interval_frames=1000,
            region_frames=500,
            metric_name="vmaf",
            log=log,
        )
        assert result.is_valid is True
        assert result.num_samples > 0
        assert result.total_frames > 0
        assert result.coverage_percent > 0
        assert result.reason is None

    def test_returns_invalid_when_region_exceeds_usable(self):
        """Test that validation returns invalid when region frames exceed usable frames."""
        log = logging.getLogger("test")
        result = validate_metric_sampling(
            usable_frames=100,
            total_frames=1000,
            interval_frames=1000,
            region_frames=500,  # More than usable_frames
            metric_name="vmaf",
            log=log,
        )
        assert result.is_valid is False
        assert result.num_samples == 0
        assert result.total_frames == 0
        assert result.reason is not None
        assert "region size" in result.reason

    def test_returns_valid_with_large_interval(self):
        """Test that at least one sample is possible when usable_frames >= region_frames."""
        log = logging.getLogger("test")
        result = validate_metric_sampling(
            usable_frames=500,
            total_frames=1000,
            interval_frames=10000,  # Large interval still yields 1 sample
            region_frames=100,
            metric_name="ssim2",
            log=log,
        )
        # Formula: (usable + interval - region) // interval = (500 + 10000 - 100) // 10000 = 1
        assert result.is_valid is True
        assert result.num_samples == 1

    def test_calculates_correct_coverage(self):
        """Test that coverage percentage is calculated correctly."""
        log = logging.getLogger("test")
        # With 10000 total frames, 5 samples of 500 frames = 2500 frames = 25%
        result = validate_metric_sampling(
            usable_frames=8000,  # 8000 usable out of 10000
            total_frames=10000,
            interval_frames=1600,  # (8000 + 1600 - 500) // 1600 = 9100 // 1600 = 5 samples
            region_frames=500,
            metric_name="vmaf",
            log=log,
        )
        assert result.is_valid is True
        # 5 samples * 500 frames = 2500 frames
        # 2500 / 10000 = 25%
        assert result.num_samples == 5
        assert result.total_frames == 2500
        assert abs(result.coverage_percent - 25.0) < 0.1


class TestValidateSamplingParameters:
    """Tests for validate_sampling_parameters function."""

    def test_validates_both_metrics_when_enabled(self):
        """Test that both metrics are validated when both are enabled."""
        log = logging.getLogger("test")
        args = PipelineArgs(
            input=Path("test.mkv"),
            output=Path("output.mkv"),
            vmaf=True,
            ssim2=True,
            vmaf_interval_frames=1000,
            vmaf_region_frames=500,
            ssim2_interval_frames=1000,
            ssim2_region_frames=500,
        )
        vmaf_valid, ssim2_valid = validate_sampling_parameters(
            args=args,
            total_frames=10000,
            guard_start_frames=500,
            guard_end_frames=500,
            log=log,
        )
        assert vmaf_valid is True
        assert ssim2_valid is True

    def test_disables_vmaf_when_invalid(self):
        """Test that VMAF is marked invalid when region exceeds usable frames."""
        log = logging.getLogger("test")
        args = PipelineArgs(
            input=Path("test.mkv"),
            output=Path("output.mkv"),
            vmaf=True,
            ssim2=True,
            vmaf_interval_frames=1000,
            vmaf_region_frames=10000,  # Way too large
            ssim2_interval_frames=1000,
            ssim2_region_frames=500,
        )
        vmaf_valid, ssim2_valid = validate_sampling_parameters(
            args=args,
            total_frames=10000,
            guard_start_frames=500,
            guard_end_frames=500,
            log=log,
        )
        assert vmaf_valid is False
        assert ssim2_valid is True

    def test_disables_ssim2_when_invalid(self):
        """Test that SSIM2 is marked invalid when region exceeds usable frames."""
        log = logging.getLogger("test")
        args = PipelineArgs(
            input=Path("test.mkv"),
            output=Path("output.mkv"),
            vmaf=True,
            ssim2=True,
            vmaf_interval_frames=1000,
            vmaf_region_frames=500,
            ssim2_interval_frames=1000,
            ssim2_region_frames=10000,  # Way too large
        )
        vmaf_valid, ssim2_valid = validate_sampling_parameters(
            args=args,
            total_frames=10000,
            guard_start_frames=500,
            guard_end_frames=500,
            log=log,
        )
        assert vmaf_valid is True
        assert ssim2_valid is False

    def test_respects_disabled_metrics(self):
        """Test that already disabled metrics stay disabled."""
        log = logging.getLogger("test")
        args = PipelineArgs(
            input=Path("test.mkv"),
            output=Path("output.mkv"),
            vmaf=False,
            ssim2=True,
            vmaf_interval_frames=1000,
            vmaf_region_frames=500,
            ssim2_interval_frames=1000,
            ssim2_region_frames=500,
        )
        vmaf_valid, ssim2_valid = validate_sampling_parameters(
            args=args,
            total_frames=10000,
            guard_start_frames=500,
            guard_end_frames=500,
            log=log,
        )
        assert vmaf_valid is False  # Was already disabled
        assert ssim2_valid is True

    def test_handles_large_guard_bands(self):
        """Test that large guard bands reduce usable frames correctly."""
        log = logging.getLogger("test")
        args = PipelineArgs(
            input=Path("test.mkv"),
            output=Path("output.mkv"),
            vmaf=True,
            ssim2=True,
            vmaf_interval_frames=1000,
            vmaf_region_frames=500,
            ssim2_interval_frames=1000,
            ssim2_region_frames=500,
        )
        # 10000 total - 4000 start guard - 4000 end guard = 2000 usable
        vmaf_valid, ssim2_valid = validate_sampling_parameters(
            args=args,
            total_frames=10000,
            guard_start_frames=4000,
            guard_end_frames=4000,
            log=log,
        )
        # 2000 usable frames can still fit 500-frame regions
        assert vmaf_valid is True
        assert ssim2_valid is True


class TestValidateArgsCLI:
    """Tests for CLI argument validation in validate_args."""

    @pytest.fixture
    def parser(self) -> argparse.ArgumentParser:
        """Create argument parser for validation tests."""
        return build_arg_parser()

    def test_errors_when_no_profile_preset_or_multi_profile(
        self, parser: argparse.ArgumentParser
    ) -> None:
        """Verify at least one encoding option is required."""
        args = PipelineArgs(
            input=Path("test.mkv"),
            output=Path("output"),
            vmaf_target=95.0,
        )
        with pytest.raises(SystemExit):
            _ = validate_args(args, parser)

    def test_errors_when_preset_with_profile(
        self, parser: argparse.ArgumentParser
    ) -> None:
        """Verify --preset and --profile are mutually exclusive."""
        args = PipelineArgs(
            input=Path("test.mkv"),
            output=Path("output"),
            preset="slow",
            profile="some-profile",
            vmaf_target=95.0,
        )
        with pytest.raises(SystemExit):
            _ = validate_args(args, parser)

    def test_errors_when_preset_with_multi_profile_search(
        self, parser: argparse.ArgumentParser
    ) -> None:
        """Verify --preset and --multi-profile-search are mutually exclusive."""
        args = PipelineArgs(
            input=Path("test.mkv"),
            output=Path("output"),
            preset="slow",
            multi_profile_search=["group1"],
            vmaf_target=95.0,
        )
        with pytest.raises(SystemExit):
            _ = validate_args(args, parser)

    def test_errors_when_profile_with_multi_profile_search(
        self, parser: argparse.ArgumentParser
    ) -> None:
        """Verify --profile and --multi-profile-search are mutually exclusive."""
        args = PipelineArgs(
            input=Path("test.mkv"),
            output=Path("output"),
            profile="some-profile",
            multi_profile_search=["group1"],
            vmaf_target=95.0,
        )
        with pytest.raises(SystemExit):
            _ = validate_args(args, parser)

    def test_accepts_preset_alone(self, parser: argparse.ArgumentParser) -> None:
        """Verify --preset with --encoder is valid (with required targets)."""
        args = PipelineArgs(
            input=Path("test.mkv"),
            output=Path("output"),
            preset="slow",
            encoder="x265",
            vmaf_target=95.0,
        )
        result = validate_args(args, parser)
        assert result.selected_profile is not None
        assert result.selected_profile.name == "preset-slow"

    def test_accepts_preset_with_assessment_only(
        self, parser: argparse.ArgumentParser
    ) -> None:
        """Verify --preset with --assessment-only is valid (no targets required)."""
        args = PipelineArgs(
            input=Path("test.mkv"),
            output=Path("output"),
            preset="medium",
            encoder="x265",
            assessment_only=True,
        )
        result = validate_args(args, parser)
        assert result.selected_profile is not None
        assert result.selected_profile.name == "preset-medium"

    def test_errors_when_preset_without_encoder(
        self, parser: argparse.ArgumentParser
    ) -> None:
        """Verify --preset without --encoder errors."""
        args = PipelineArgs(
            input=Path("test.mkv"),
            output=Path("output"),
            preset="slow",
            vmaf_target=95.0,
        )
        with pytest.raises(SystemExit):
            _ = validate_args(args, parser)

    def test_accepts_multi_profile_search_alone(
        self, parser: argparse.ArgumentParser, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify --multi-profile-search alone is valid (selected_profile is None)."""
        # Use sample profiles file so test doesn't depend on user's profiles.yaml
        from videotuner.profiles import load_profiles as _real_load

        sample_file = Path(__file__).resolve().parent.parent / "profiles.yaml.sample"
        sample_profiles = _real_load(sample_file)

        def _load_sample(_profile_file: Path | None = None) -> dict[str, Profile]:
            return sample_profiles

        monkeypatch.setattr("videotuner.profiles.load_profiles", _load_sample)
        args = PipelineArgs(
            input=Path("test.mkv"),
            output=Path("output"),
            multi_profile_search=["Film (x265)"],
            vmaf_target=93.0,
        )
        result = validate_args(args, parser)
        # In multi-profile search mode, selected_profile is None
        assert result.selected_profile is None
        # But multi_profile_list should be populated
        assert len(result.multi_profile_list) > 0
