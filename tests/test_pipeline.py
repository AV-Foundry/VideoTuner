"""Tests for pipeline functionality including predicted bitrate features."""

import logging
import re
from io import StringIO
from pathlib import Path

from rich.console import Console

from videotuner.pipeline_types import MultiProfileResult
from videotuner.pipeline_cli import PipelineArgs
from videotuner.pipeline_display import (
    check_and_display_bitrate_warning,
    display_ignored_args_warnings,
    format_bitrate_percentage,
)
from videotuner.pipeline_cli import DEFAULT_CRF_START_VALUE, DEFAULT_CRF_INTERVAL


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    return ansi_escape.sub("", text)


def _make_console() -> tuple[Console, StringIO]:
    """Create a console that writes to an in-memory buffer."""
    buffer = StringIO()
    return Console(file=buffer, force_terminal=True, width=200), buffer


class TestPipelineArgs:
    """Tests for PipelineArgs dataclass and validation."""

    def test_predicted_bitrate_warning_percent_default(self):
        """Test that predicted_bitrate_warning_percent defaults to None."""
        args = PipelineArgs(input=Path("test.mkv"), output=Path("output.mkv"))
        assert args.predicted_bitrate_warning_percent is None

    def test_predicted_bitrate_warning_percent_valid_values(self):
        """Test that predicted_bitrate_warning_percent accepts valid values."""
        # Test minimum valid value
        args = PipelineArgs(
            input=Path("test.mkv"),
            output=Path("output.mkv"),
            predicted_bitrate_warning_percent=1.0,
        )
        assert args.predicted_bitrate_warning_percent == 1.0

        # Test maximum valid value
        args = PipelineArgs(
            input=Path("test.mkv"),
            output=Path("output.mkv"),
            predicted_bitrate_warning_percent=100.0,
        )
        assert args.predicted_bitrate_warning_percent == 100.0

        # Test middle value
        args = PipelineArgs(
            input=Path("test.mkv"),
            output=Path("output.mkv"),
            predicted_bitrate_warning_percent=50.0,
        )
        assert args.predicted_bitrate_warning_percent == 50.0


class TestMultiProfileResult:
    """Tests for MultiProfileResult dataclass."""

    def test_predicted_bitrate_field_exists(self):
        """Test that MultiProfileResult has predicted_bitrate_kbps field."""
        result = MultiProfileResult(
            profile_name="TestProfile",
            optimal_crf=28.0,
            scores={"vmaf_mean": 95.0},
            predicted_bitrate_kbps=5000.0,
            converged=True,
            meets_all_targets=True,
        )
        assert result.predicted_bitrate_kbps == 5000.0

    def test_is_valid_with_converged_result(self):
        """Test is_valid() returns True for converged results."""
        result = MultiProfileResult(
            profile_name="TestProfile",
            optimal_crf=28.0,
            scores={"vmaf_mean": 95.0},
            predicted_bitrate_kbps=5000.0,
            converged=True,
            meets_all_targets=True,
        )
        assert result.is_valid() is True

    def test_is_valid_with_non_converged_result(self):
        """Test is_valid() returns False for non-converged results."""
        result = MultiProfileResult(
            profile_name="TestProfile",
            optimal_crf=None,
            scores={"vmaf_mean": 95.0},
            predicted_bitrate_kbps=5000.0,
            converged=False,
            meets_all_targets=False,
        )
        assert result.is_valid() is False

    def test_is_bitrate_mode_for_crf_profile(self):
        """Test is_bitrate_mode returns False for CRF profiles."""
        result = MultiProfileResult(
            profile_name="CRFProfile",
            optimal_crf=28.0,
            scores={"vmaf_mean": 95.0},
            predicted_bitrate_kbps=5000.0,
            converged=True,
            meets_all_targets=True,
        )
        assert result.is_bitrate_mode is False

    def test_is_bitrate_mode_for_bitrate_profile(self):
        """Test is_bitrate_mode returns True for bitrate profiles."""
        result = MultiProfileResult(
            profile_name="BitrateProfile",
            optimal_crf=None,
            scores={"vmaf_mean": 95.0},
            predicted_bitrate_kbps=8000.0,
            converged=True,
            meets_all_targets=None,
        )
        assert result.is_bitrate_mode is True

    def test_meets_all_targets_true_for_crf_meeting_targets(self):
        """Test meets_all_targets is True for CRF profiles meeting targets."""
        result = MultiProfileResult(
            profile_name="CRFProfile",
            optimal_crf=28.0,
            scores={"vmaf_mean": 95.0},
            predicted_bitrate_kbps=5000.0,
            converged=True,
            meets_all_targets=True,
        )
        assert result.meets_all_targets is True

    def test_meets_all_targets_false_for_crf_failing_targets(self):
        """Test meets_all_targets is False for CRF profiles failing targets."""
        result = MultiProfileResult(
            profile_name="CRFProfile",
            optimal_crf=28.0,
            scores={"vmaf_mean": 90.0},
            predicted_bitrate_kbps=5000.0,
            converged=True,
            meets_all_targets=False,
        )
        assert result.meets_all_targets is False

    def test_meets_all_targets_none_for_bitrate_profile_without_targets(self):
        """Test meets_all_targets is None for bitrate profiles when no targets specified."""
        result = MultiProfileResult(
            profile_name="BitrateProfile",
            optimal_crf=None,
            scores={"vmaf_mean": 95.0},
            predicted_bitrate_kbps=8000.0,
            converged=True,
            meets_all_targets=None,
        )
        assert result.meets_all_targets is None

    def test_meets_all_targets_bool_for_bitrate_profile_with_targets(self):
        """Test meets_all_targets can be True/False for bitrate profiles with targets."""
        result_meets = MultiProfileResult(
            profile_name="BitrateProfile",
            optimal_crf=None,
            scores={"vmaf_mean": 95.0},
            predicted_bitrate_kbps=8000.0,
            converged=True,
            meets_all_targets=True,
        )
        assert result_meets.meets_all_targets is True
        assert result_meets.is_bitrate_mode is True

        result_fails = MultiProfileResult(
            profile_name="BitrateProfile",
            optimal_crf=None,
            scores={"vmaf_mean": 85.0},
            predicted_bitrate_kbps=8000.0,
            converged=True,
            meets_all_targets=False,
        )
        assert result_fails.meets_all_targets is False
        assert result_fails.is_bitrate_mode is True


class TestMultiProfileRanking:
    """Tests for multi-profile ranking logic using rank_profile_results."""

    def test_crf_meeting_targets_ranks_before_crf_failing_targets(self):
        """Test CRF profiles meeting targets rank before those failing."""
        from videotuner.pipeline_multi_profile import rank_profile_results

        meets = MultiProfileResult(
            profile_name="MeetsTargets",
            optimal_crf=28.0,
            scores={"vmaf_mean": 95.0},
            predicted_bitrate_kbps=6000.0,  # Higher bitrate
            converged=True,
            meets_all_targets=True,
        )
        fails = MultiProfileResult(
            profile_name="FailsTargets",
            optimal_crf=32.0,
            scores={"vmaf_mean": 90.0},
            predicted_bitrate_kbps=4000.0,  # Lower bitrate
            converged=True,
            meets_all_targets=False,
        )

        ranked = rank_profile_results([meets, fails])

        # MeetsTargets should win despite higher bitrate
        assert ranked[0].profile_name == "MeetsTargets"
        assert ranked[1].profile_name == "FailsTargets"

    def test_bitrate_only_profiles_sorted_by_quality(self):
        """Test bitrate-only profiles are sorted by quality scores, not bitrate."""
        from videotuner.pipeline_multi_profile import rank_profile_results

        bitrate_high_q = MultiProfileResult(
            profile_name="BitrateHighQ",
            optimal_crf=None,
            scores={"vmaf_mean": 96.0},
            predicted_bitrate_kbps=8000.0,
            converged=True,
            meets_all_targets=None,
        )
        bitrate_low_q = MultiProfileResult(
            profile_name="BitrateLowQ",
            optimal_crf=None,
            scores={"vmaf_mean": 92.0},
            predicted_bitrate_kbps=4000.0,
            converged=True,
            meets_all_targets=None,
        )

        ranked = rank_profile_results([bitrate_low_q, bitrate_high_q])

        # Higher quality wins (all-ABR ranks by quality, not bitrate)
        assert ranked[0].profile_name == "BitrateHighQ"
        assert ranked[1].profile_name == "BitrateLowQ"

    def test_within_tier_sorted_by_lowest_bitrate(self):
        """Test CRF profiles within same tier are sorted by lowest bitrate."""
        from videotuner.pipeline_multi_profile import rank_profile_results

        crf_a = MultiProfileResult(
            profile_name="CRF_A",
            optimal_crf=26.0,
            scores={"vmaf_mean": 96.0},
            predicted_bitrate_kbps=7000.0,
            converged=True,
            meets_all_targets=True,
        )
        crf_b = MultiProfileResult(
            profile_name="CRF_B",
            optimal_crf=28.0,
            scores={"vmaf_mean": 95.0},
            predicted_bitrate_kbps=5000.0,
            converged=True,
            meets_all_targets=True,
        )

        ranked = rank_profile_results([crf_a, crf_b])

        # CRF_B wins with lower bitrate
        assert ranked[0].profile_name == "CRF_B"
        assert ranked[1].profile_name == "CRF_A"


class TestBitrateWarning:
    """Tests for predicted bitrate warning functionality."""

    def test_warning_disabled_when_threshold_none(self):
        """Test that no warning is displayed when threshold is None."""
        console, buffer = _make_console()
        log = logging.getLogger("test")

        check_and_display_bitrate_warning(
            console=console,
            log=log,
            predicted_bitrate_kbps=10000.0,
            input_bitrate_kbps=5000.0,
            threshold_percent=None,
            profile_name="TestProfile",
        )

        output = buffer.getvalue()
        assert "Warning" not in output

    def test_warning_skipped_when_input_bitrate_none(self):
        """Test that warning is skipped when input bitrate is unavailable."""
        console, buffer = _make_console()
        log = logging.getLogger("test")

        check_and_display_bitrate_warning(
            console=console,
            log=log,
            predicted_bitrate_kbps=10000.0,
            input_bitrate_kbps=None,
            threshold_percent=80.0,
            profile_name="TestProfile",
        )

        output = buffer.getvalue()
        assert "Warning" not in output

    def test_warning_skipped_when_input_bitrate_zero(self):
        """Test that warning is skipped when input bitrate is zero."""
        console, buffer = _make_console()
        log = logging.getLogger("test")

        check_and_display_bitrate_warning(
            console=console,
            log=log,
            predicted_bitrate_kbps=10000.0,
            input_bitrate_kbps=0.0,
            threshold_percent=80.0,
            profile_name="TestProfile",
        )

        output = buffer.getvalue()
        assert "Warning" not in output

    def test_warning_skipped_when_predicted_bitrate_zero(self):
        """Test that warning is skipped when predicted bitrate is zero."""
        console, buffer = _make_console()
        log = logging.getLogger("test")

        check_and_display_bitrate_warning(
            console=console,
            log=log,
            predicted_bitrate_kbps=0.0,
            input_bitrate_kbps=5000.0,
            threshold_percent=80.0,
            profile_name="TestProfile",
        )

        output = buffer.getvalue()
        assert "Warning" not in output

    def test_warning_displayed_when_threshold_exceeded(self):
        """Test that warning is displayed when predicted bitrate exceeds threshold."""
        console, buffer = _make_console()
        log = logging.getLogger("test")

        # Predicted is 10,000 kbps, input is 5,000 kbps = 200% of input
        # Threshold is 80%, so warning should be displayed
        check_and_display_bitrate_warning(
            console=console,
            log=log,
            predicted_bitrate_kbps=10000.0,
            input_bitrate_kbps=5000.0,
            threshold_percent=80.0,
            profile_name="TestProfile",
        )

        output = strip_ansi(buffer.getvalue())
        assert "Warning" in output
        assert "10,000 kbps" in output  # Check comma formatting
        assert "5,000 kbps" in output  # Check comma formatting
        assert "80" in output  # Check threshold percentage
        assert "200.0%" in output  # Check calculated percentage

    def test_no_warning_when_threshold_not_exceeded(self):
        """Test that no warning is displayed when predicted bitrate is below threshold."""
        console, buffer = _make_console()
        log = logging.getLogger("test")

        # Predicted is 3,000 kbps, input is 5,000 kbps = 60% of input
        # Threshold is 80%, so no warning should be displayed
        check_and_display_bitrate_warning(
            console=console,
            log=log,
            predicted_bitrate_kbps=3000.0,
            input_bitrate_kbps=5000.0,
            threshold_percent=80.0,
            profile_name="TestProfile",
        )

        output = buffer.getvalue()
        assert "Warning" not in output

    def test_warning_at_exact_threshold(self):
        """Test behavior when predicted bitrate equals threshold exactly."""
        console, buffer = _make_console()
        log = logging.getLogger("test")

        # Predicted is 4,000 kbps, input is 5,000 kbps = 80% of input
        # Threshold is 80%, so no warning (not exceeding)
        check_and_display_bitrate_warning(
            console=console,
            log=log,
            predicted_bitrate_kbps=4000.0,
            input_bitrate_kbps=5000.0,
            threshold_percent=80.0,
            profile_name="TestProfile",
        )

        output = buffer.getvalue()
        assert "Warning" not in output

    def test_warning_with_profile_name(self):
        """Test that warning includes profile name when provided."""
        console, buffer = _make_console()
        log = logging.getLogger("test")

        check_and_display_bitrate_warning(
            console=console,
            log=log,
            predicted_bitrate_kbps=10000.0,
            input_bitrate_kbps=5000.0,
            threshold_percent=80.0,
            profile_name="Film Clean",
        )

        output = buffer.getvalue()
        assert "Film Clean" in output

    def test_warning_without_profile_name(self):
        """Test that warning works when profile name is None."""
        console, buffer = _make_console()
        log = logging.getLogger("test")

        check_and_display_bitrate_warning(
            console=console,
            log=log,
            predicted_bitrate_kbps=10000.0,
            input_bitrate_kbps=5000.0,
            threshold_percent=80.0,
            profile_name=None,
        )

        output = buffer.getvalue()
        assert "Warning" in output
        # Should not crash and should display warning without profile name


class TestBitrateFormatting:
    """Tests for bitrate number formatting with comma separators."""

    def test_comma_separator_in_thousands(self):
        """Test that bitrates display with comma separators."""
        console, buffer = _make_console()
        log = logging.getLogger("test")

        check_and_display_bitrate_warning(
            console=console,
            log=log,
            predicted_bitrate_kbps=31144.0,
            input_bitrate_kbps=15000.0,
            threshold_percent=50.0,
            profile_name="TestProfile",
        )

        output = strip_ansi(buffer.getvalue())
        # Check that bitrates are formatted with commas
        assert "31,144" in output
        assert "15,000" in output
        # Check that unformatted numbers don't appear
        assert "31144" not in output.replace("31,144", "")
        assert "15000" not in output.replace("15,000", "")


class TestBitratePercentageFormatting:
    """Tests for _format_bitrate_percentage helper function."""

    def test_format_with_valid_input_bitrate(self):
        """Test that percentage is included when input bitrate is available."""
        result = format_bitrate_percentage(5000.0, 10000.0)
        assert result == "5,000 kbps (50.0% of input)"

    def test_format_with_none_input_bitrate(self):
        """Test that only kbps is shown when input bitrate is None."""
        result = format_bitrate_percentage(5000.0, None)
        assert result == "5,000 kbps"
        assert "%" not in result

    def test_format_with_zero_input_bitrate(self):
        """Test that only kbps is shown when input bitrate is zero."""
        result = format_bitrate_percentage(5000.0, 0.0)
        assert result == "5,000 kbps"
        assert "%" not in result

    def test_format_with_zero_predicted_bitrate(self):
        """Test that only kbps is shown when predicted bitrate is zero."""
        result = format_bitrate_percentage(0.0, 10000.0)
        assert result == "0 kbps"
        assert "%" not in result

    def test_format_with_high_percentage(self):
        """Test formatting when predicted exceeds input (>100%)."""
        result = format_bitrate_percentage(15000.0, 10000.0)
        assert result == "15,000 kbps (150.0% of input)"

    def test_format_with_low_percentage(self):
        """Test formatting with low percentage."""
        result = format_bitrate_percentage(1000.0, 10000.0)
        assert result == "1,000 kbps (10.0% of input)"

    def test_format_comma_separator(self):
        """Test that large numbers are formatted with comma separators."""
        result = format_bitrate_percentage(31144.0, 50000.0)
        assert "31,144" in result
        assert "62.3% of input" in result


class TestIgnoredArgsWarning:
    """Tests for ignored arguments warnings with bitrate profiles."""

    def test_no_warning_when_no_bitrate_profiles(self):
        """Test that no warning is displayed when no profiles are bitrate mode."""
        console, buffer = _make_console()
        log = logging.getLogger("test")

        display_ignored_args_warnings(
            console,
            log,
            bitrate_profile_names=[],
            crf_start_value=15.0,  # Non-default
            crf_interval=0.25,  # Non-default
        )

        output = buffer.getvalue()
        assert "Warning" not in output

    def test_warning_for_non_default_crf_start_value(self):
        """Test that warning is displayed when CRF start value differs from default."""
        console, buffer = _make_console()
        log = logging.getLogger("test")

        display_ignored_args_warnings(
            console,
            log,
            bitrate_profile_names=["Streaming 1080p"],
            crf_start_value=15.0,  # Non-default
            crf_interval=DEFAULT_CRF_INTERVAL,
        )

        output = strip_ansi(buffer.getvalue())
        assert "Warning" in output
        assert "--crf-start-value 15.0" in output
        assert "ignored" in output
        assert "Streaming 1080p" in output

    def test_warning_for_non_default_crf_interval(self):
        """Test that warning is displayed when CRF interval differs from default."""
        console, buffer = _make_console()
        log = logging.getLogger("test")

        display_ignored_args_warnings(
            console,
            log,
            bitrate_profile_names=["Streaming 1080p"],
            crf_start_value=DEFAULT_CRF_START_VALUE,
            crf_interval=0.25,  # Non-default
        )

        output = strip_ansi(buffer.getvalue())
        assert "Warning" in output
        assert "--crf-interval 0.25" in output
        assert "ignored" in output
        assert "Streaming 1080p" in output

    def test_no_warning_with_default_values(self):
        """Test that no warning is displayed when using default CRF values."""
        console, buffer = _make_console()
        log = logging.getLogger("test")

        display_ignored_args_warnings(
            console,
            log,
            bitrate_profile_names=["Streaming 1080p"],
            crf_start_value=DEFAULT_CRF_START_VALUE,
            crf_interval=DEFAULT_CRF_INTERVAL,
        )

        output = buffer.getvalue()
        assert "Warning" not in output

    def test_multiple_warnings_combined(self):
        """Test that multiple warnings are displayed together."""
        console, buffer = _make_console()
        log = logging.getLogger("test")

        display_ignored_args_warnings(
            console,
            log,
            bitrate_profile_names=["Streaming 1080p"],
            crf_start_value=15.0,  # Non-default
            crf_interval=0.25,  # Non-default
        )

        output = strip_ansi(buffer.getvalue())
        assert output.count("Warning") == 2
        assert "--crf-start-value 15.0" in output
        assert "--crf-interval 0.25" in output

    def test_warning_lists_multiple_bitrate_profiles(self):
        """Test that warning lists all bitrate profile names."""
        console, buffer = _make_console()
        log = logging.getLogger("test")

        display_ignored_args_warnings(
            console,
            log,
            bitrate_profile_names=[
                "Streaming 1080p",
                "Streaming 720p",
                "Streaming 480p",
            ],
            crf_start_value=15.0,  # Non-default
            crf_interval=DEFAULT_CRF_INTERVAL,
        )

        output = strip_ansi(buffer.getvalue())
        assert "Warning" in output
        assert "Streaming 1080p" in output
        assert "Streaming 720p" in output
        assert "Streaming 480p" in output
