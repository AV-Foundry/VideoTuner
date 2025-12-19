"""Tests for constants module."""

from __future__ import annotations

from videotuner.constants import (
    # CRF constants
    CRF_MIN,
    CRF_MAX,
    CRF_SEARCH_MAX_ITERATIONS,
    CRF_FLOOR_VALUE,
    CRF_FLOOR_TOLERANCE,
    # Guard band constants
    MAX_COMBINED_GUARD_PERCENT,
    # Bitrate warning constants
    BITRATE_WARNING_PERCENT_MIN,
    BITRATE_WARNING_PERCENT_MAX,
    # VMAF constants
    VMAF_THREAD_CPU_FRACTION,
    VMAF_MAX_THREADS,
    RESOLUTION_RELATIVE_TOLERANCE,
    MIN_CROP_FRACTION,
    RESOLUTION_CONFIDENCE_THRESHOLD,
    # Display constants
    PROGRESS_BAR_WIDTH,
    LOG_SEPARATOR_WIDTH,
    LOG_SEPARATOR_CHAR,
    # Frame rate constants
    FPS_DENOMINATOR,
    # Percentile constants
    PERCENTILE_1PCT,
    PERCENTILE_5PCT,
    PERCENTILE_95PCT,
)


class TestCRFConstants:
    """Tests for CRF-related constants."""

    def test_crf_min_is_zero(self):
        """Test CRF minimum is 0 (lossless)."""
        assert CRF_MIN == 0.0

    def test_crf_max_is_51(self):
        """Test CRF maximum is 51 (worst quality)."""
        assert CRF_MAX == 51.0

    def test_crf_min_less_than_max(self):
        """Test CRF min is less than max."""
        assert CRF_MIN < CRF_MAX

    def test_crf_search_max_iterations_positive(self):
        """Test max iterations is a positive integer."""
        assert CRF_SEARCH_MAX_ITERATIONS > 0
        assert isinstance(CRF_SEARCH_MAX_ITERATIONS, int)

    def test_crf_floor_value_equals_max(self):
        """Test CRF floor value equals CRF max."""
        assert CRF_FLOOR_VALUE == CRF_MAX

    def test_crf_floor_tolerance_small(self):
        """Test CRF floor tolerance is small."""
        assert 0 < CRF_FLOOR_TOLERANCE < 1


class TestGuardBandConstants:
    """Tests for guard band constants."""

    def test_max_combined_guard_less_than_one(self):
        """Test max combined guard percent is less than 100%."""
        assert MAX_COMBINED_GUARD_PERCENT < 1.0

    def test_max_combined_guard_positive(self):
        """Test max combined guard percent is positive."""
        assert MAX_COMBINED_GUARD_PERCENT > 0


class TestBitrateWarningConstants:
    """Tests for bitrate warning constants."""

    def test_min_less_than_max(self):
        """Test min percent is less than max."""
        assert BITRATE_WARNING_PERCENT_MIN < BITRATE_WARNING_PERCENT_MAX

    def test_min_is_positive(self):
        """Test min percent is positive."""
        assert BITRATE_WARNING_PERCENT_MIN > 0

    def test_max_is_100(self):
        """Test max percent is 100."""
        assert BITRATE_WARNING_PERCENT_MAX == 100.0


class TestVMAFConstants:
    """Tests for VMAF-related constants."""

    def test_thread_fraction_valid(self):
        """Test thread CPU fraction is between 0 and 1."""
        assert 0 < VMAF_THREAD_CPU_FRACTION <= 1.0

    def test_max_threads_reasonable(self):
        """Test max threads is reasonable."""
        assert 1 <= VMAF_MAX_THREADS <= 64

    def test_resolution_tolerance_small(self):
        """Test resolution tolerance is small percentage."""
        assert 0 < RESOLUTION_RELATIVE_TOLERANCE < 0.1

    def test_min_crop_fraction_valid(self):
        """Test min crop fraction is between 0 and 1."""
        assert 0 < MIN_CROP_FRACTION < 1.0

    def test_confidence_threshold_valid(self):
        """Test confidence threshold is between 0 and 1."""
        assert 0 < RESOLUTION_CONFIDENCE_THRESHOLD < 1.0


class TestDisplayConstants:
    """Tests for display-related constants."""

    def test_progress_bar_width_positive(self):
        """Test progress bar width is positive."""
        assert PROGRESS_BAR_WIDTH > 0
        assert isinstance(PROGRESS_BAR_WIDTH, int)

    def test_log_separator_width_positive(self):
        """Test log separator width is positive."""
        assert LOG_SEPARATOR_WIDTH > 0
        assert isinstance(LOG_SEPARATOR_WIDTH, int)

    def test_log_separator_char_single(self):
        """Test log separator char is a single character."""
        assert len(LOG_SEPARATOR_CHAR) == 1
        assert isinstance(LOG_SEPARATOR_CHAR, str)


class TestFrameRateConstants:
    """Tests for frame rate constants."""

    def test_fps_denominator_positive(self):
        """Test FPS denominator is positive."""
        assert FPS_DENOMINATOR > 0
        assert isinstance(FPS_DENOMINATOR, int)


class TestPercentileConstants:
    """Tests for percentile constants."""

    def test_percentiles_ordered(self):
        """Test percentiles are in ascending order."""
        assert PERCENTILE_1PCT < PERCENTILE_5PCT < PERCENTILE_95PCT

    def test_percentiles_valid_range(self):
        """Test percentiles are between 0 and 1."""
        assert 0 < PERCENTILE_1PCT < 1
        assert 0 < PERCENTILE_5PCT < 1
        assert 0 < PERCENTILE_95PCT < 1

    def test_1pct_value(self):
        """Test 1st percentile value."""
        assert PERCENTILE_1PCT == 0.01

    def test_5pct_value(self):
        """Test 5th percentile value."""
        assert PERCENTILE_5PCT == 0.05

    def test_95pct_value(self):
        """Test 95th percentile value."""
        assert PERCENTILE_95PCT == 0.95
