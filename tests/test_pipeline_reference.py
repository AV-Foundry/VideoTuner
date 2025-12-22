"""Tests for pipeline reference module."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from videotuner.pipeline_cli import PipelineArgs
from videotuner.pipeline_reference import (
    MetricSamplingParams,
    are_sampling_params_equal,
)


class TestAreSamplingParamsEqual:
    """Tests for are_sampling_params_equal function."""

    def test_returns_true_when_both_enabled_and_params_match(self):
        """Test returns True when both metrics enabled with matching params."""
        args = PipelineArgs(
            input=Path("test.mkv"),
            output=Path("output.mkv"),
            vmaf=True,
            ssim2=True,
            vmaf_interval_frames=1600,
            vmaf_region_frames=20,
            ssim2_interval_frames=1600,
            ssim2_region_frames=20,
        )
        assert are_sampling_params_equal(args) is True

    def test_returns_true_with_default_params(self):
        """Test returns True with default sampling parameters (1600/20)."""
        args = PipelineArgs(
            input=Path("test.mkv"),
            output=Path("output.mkv"),
            vmaf=True,
            ssim2=True,
            # Using defaults: interval=1600, region=20 for both
        )
        assert are_sampling_params_equal(args) is True

    def test_returns_false_when_vmaf_disabled(self):
        """Test returns False when VMAF is disabled."""
        args = PipelineArgs(
            input=Path("test.mkv"),
            output=Path("output.mkv"),
            vmaf=False,
            ssim2=True,
            vmaf_interval_frames=1600,
            vmaf_region_frames=20,
            ssim2_interval_frames=1600,
            ssim2_region_frames=20,
        )
        assert are_sampling_params_equal(args) is False

    def test_returns_false_when_ssim2_disabled(self):
        """Test returns False when SSIM2 is disabled."""
        args = PipelineArgs(
            input=Path("test.mkv"),
            output=Path("output.mkv"),
            vmaf=True,
            ssim2=False,
            vmaf_interval_frames=1600,
            vmaf_region_frames=20,
            ssim2_interval_frames=1600,
            ssim2_region_frames=20,
        )
        assert are_sampling_params_equal(args) is False

    def test_returns_false_when_both_disabled(self):
        """Test returns False when both metrics are disabled."""
        args = PipelineArgs(
            input=Path("test.mkv"),
            output=Path("output.mkv"),
            vmaf=False,
            ssim2=False,
        )
        assert are_sampling_params_equal(args) is False

    def test_returns_false_when_interval_frames_differ(self):
        """Test returns False when interval_frames differ between metrics."""
        args = PipelineArgs(
            input=Path("test.mkv"),
            output=Path("output.mkv"),
            vmaf=True,
            ssim2=True,
            vmaf_interval_frames=1600,
            vmaf_region_frames=20,
            ssim2_interval_frames=800,  # Different
            ssim2_region_frames=20,
        )
        assert are_sampling_params_equal(args) is False

    def test_returns_false_when_region_frames_differ(self):
        """Test returns False when region_frames differ between metrics."""
        args = PipelineArgs(
            input=Path("test.mkv"),
            output=Path("output.mkv"),
            vmaf=True,
            ssim2=True,
            vmaf_interval_frames=1600,
            vmaf_region_frames=20,
            ssim2_interval_frames=1600,
            ssim2_region_frames=40,  # Different
        )
        assert are_sampling_params_equal(args) is False

    def test_returns_false_when_both_params_differ(self):
        """Test returns False when both interval and region differ."""
        args = PipelineArgs(
            input=Path("test.mkv"),
            output=Path("output.mkv"),
            vmaf=True,
            ssim2=True,
            vmaf_interval_frames=1600,
            vmaf_region_frames=20,
            ssim2_interval_frames=3200,  # Different
            ssim2_region_frames=40,  # Different
        )
        assert are_sampling_params_equal(args) is False

    def test_returns_true_with_non_default_matching_params(self):
        """Test returns True with custom matching params (not defaults)."""
        args = PipelineArgs(
            input=Path("test.mkv"),
            output=Path("output.mkv"),
            vmaf=True,
            ssim2=True,
            vmaf_interval_frames=3200,
            vmaf_region_frames=50,
            ssim2_interval_frames=3200,
            ssim2_region_frames=50,
        )
        assert are_sampling_params_equal(args) is True


class TestMetricSamplingParams:
    """Tests for MetricSamplingParams dataclass."""

    def test_creation_with_all_fields(self):
        """Test that MetricSamplingParams can be created with all fields."""
        params = MetricSamplingParams(
            interval_frames=1600,
            region_frames=20,
            guard_start_frames=100,
            guard_end_frames=100,
            total_frames=10000,
        )
        assert params.interval_frames == 1600
        assert params.region_frames == 20
        assert params.guard_start_frames == 100
        assert params.guard_end_frames == 100
        assert params.total_frames == 10000

    def test_is_frozen(self):
        """Test that MetricSamplingParams is immutable."""
        params = MetricSamplingParams(
            interval_frames=1600,
            region_frames=20,
            guard_start_frames=100,
            guard_end_frames=100,
            total_frames=10000,
        )
        with pytest.raises(AttributeError):
            setattr(params, "interval_frames", 3200)

    def test_usable_frames_property(self):
        """Test usable_frames excludes guard bands."""
        params = MetricSamplingParams(
            interval_frames=1600,
            region_frames=20,
            guard_start_frames=100,
            guard_end_frames=200,
            total_frames=10000,
        )
        # usable = 10000 - 100 - 200 = 9700
        assert params.usable_frames == 9700

    def test_usable_frames_with_no_guards(self):
        """Test usable_frames equals total when no guard bands."""
        params = MetricSamplingParams(
            interval_frames=1600,
            region_frames=20,
            guard_start_frames=0,
            guard_end_frames=0,
            total_frames=10000,
        )
        assert params.usable_frames == 10000

    def test_num_samples_property(self):
        """Test num_samples calculation."""
        params = MetricSamplingParams(
            interval_frames=1600,
            region_frames=20,
            guard_start_frames=100,
            guard_end_frames=100,
            total_frames=10000,
        )
        # usable = 10000 - 100 - 100 = 9800
        # num_samples = (9800 + 1600 - 20) // 1600 = 11380 // 1600 = 7
        assert params.num_samples == 7

    def test_num_samples_with_small_video(self):
        """Test num_samples with a small video returns at least 1."""
        params = MetricSamplingParams(
            interval_frames=1600,
            region_frames=20,
            guard_start_frames=100,
            guard_end_frames=100,
            total_frames=500,
        )
        # usable = 500 - 100 - 100 = 300
        # num_samples = (300 + 1600 - 20) // 1600 = 1880 // 1600 = 1
        assert params.num_samples == 1

    def test_total_sample_frames_property(self):
        """Test total_sample_frames calculation."""
        params = MetricSamplingParams(
            interval_frames=1600,
            region_frames=20,
            guard_start_frames=100,
            guard_end_frames=100,
            total_frames=10000,
        )
        # num_samples = 7 (from previous test)
        # total_sample_frames = 7 * 20 = 140
        assert params.total_sample_frames == 140

    def test_coverage_percent_property(self):
        """Test coverage_percent calculation."""
        params = MetricSamplingParams(
            interval_frames=1600,
            region_frames=20,
            guard_start_frames=100,
            guard_end_frames=100,
            total_frames=10000,
        )
        # total_sample_frames = 140
        # coverage = (140 / 10000) * 100 = 1.4%
        assert math.isclose(params.coverage_percent, 1.4)

    def test_coverage_percent_with_zero_total_frames(self):
        """Test coverage_percent returns 0 when total_frames is 0."""
        params = MetricSamplingParams(
            interval_frames=1600,
            region_frames=20,
            guard_start_frames=0,
            guard_end_frames=0,
            total_frames=0,
        )
        assert params.coverage_percent == 0.0

    def test_coverage_percent_with_high_coverage(self):
        """Test coverage_percent with high sampling rate."""
        params = MetricSamplingParams(
            interval_frames=100,  # Sample every 100 frames
            region_frames=50,  # 50 frame regions
            guard_start_frames=0,
            guard_end_frames=0,
            total_frames=1000,
        )
        # usable = 1000
        # num_samples = (1000 + 100 - 50) // 100 = 1050 // 100 = 10
        # total_sample_frames = 10 * 50 = 500
        # coverage = (500 / 1000) * 100 = 50%
        assert params.num_samples == 10
        assert params.total_sample_frames == 500
        assert math.isclose(params.coverage_percent, 50.0)
