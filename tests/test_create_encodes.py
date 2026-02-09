"""Tests for create_encodes module."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest

from videotuner.create_encodes import calculate_cropdetect_values
from videotuner.encoding_utils import CropValues
from videotuner.tool_parsers import CROPDETECT_RE

_TEST_PATH = Path("test.mkv")

# Number of samples for num_frames=10000, fps=24.0, interval=30 (defaults):
# skip=1000, safe_start=1000, safe_end=9000, step=720 → 12 samples
_SAMPLES_10K = len(range(1000, 9000, 720))


def _passthrough_resolve(p: Path, _c: Path | None) -> Path:
    """Typed side_effect for resolve_absolute_path mock."""
    return p


def _crop_result(crop_line: str) -> subprocess.CompletedProcess[str]:
    """Create a CompletedProcess with cropdetect output on stderr."""
    return subprocess.CompletedProcess(
        args=[], returncode=0, stdout="", stderr=crop_line
    )


class TestCropValues:
    """Tests for CropValues dataclass."""

    def test_create_crop_values(self) -> None:
        """Test creating CropValues with all fields."""
        crop = CropValues(left=10, right=20, top=5, bottom=15)
        assert crop.left == 10
        assert crop.right == 20
        assert crop.top == 5
        assert crop.bottom == 15

    def test_frozen_dataclass(self) -> None:
        """Test CropValues is frozen (immutable)."""
        crop = CropValues(left=10, right=20, top=5, bottom=15)
        with pytest.raises(AttributeError):
            setattr(crop, "left", 100)


class TestCropdetectParsing:
    """Tests for CROPDETECT_RE regex."""

    def test_parse_crop_line(self) -> None:
        line = "[Parsed_cropdetect_0 @ 0x...] x1:0 x2:3839 y1:276 y2:1863 w:3840 h:1584 x:0 y:278 pts:1001 t:1.001000 limit:0.094118 crop=3840:1584:0:278"
        matches: list[tuple[str, str, str, str]] = CROPDETECT_RE.findall(line)
        assert len(matches) == 1
        w, h, x, y = matches[0]
        assert (w, h, x, y) == ("3840", "1584", "0", "278")

    def test_parse_multiple_lines(self) -> None:
        output = "crop=3840:1584:0:278\ncrop=3840:1580:0:280\ncrop=3840:1584:0:278\n"
        matches: list[tuple[str, str, str, str]] = CROPDETECT_RE.findall(output)
        assert len(matches) == 3

    def test_no_match(self) -> None:
        output = "frame= 100 fps=50.0 q=28.0 size=N/A time=00:00:04.00"
        matches: list[tuple[str, str, str, str]] = CROPDETECT_RE.findall(output)
        assert len(matches) == 0


class TestCalculateCropdetectValues:
    """Tests for calculate_cropdetect_values with mocked FFmpeg."""

    def test_basic_letterbox(self) -> None:
        """Test detection of consistent letterboxing (278px bars)."""
        with (
            patch(
                "videotuner.create_encodes.subprocess.run",
                return_value=_crop_result("crop=3840:1584:0:278"),
            ),
            patch(
                "videotuner.create_encodes.resolve_absolute_path",
                side_effect=_passthrough_resolve,
            ),
        ):
            result = calculate_cropdetect_values(
                source_path=_TEST_PATH,
                start_frame=0,
                num_frames=10000,
                fps=24.0,
                source_width=3840,
                source_height=2160,
            )
        assert result.left == 0
        assert result.right == 0
        assert result.top == 278
        assert result.bottom == 298  # 2160 - 1584 - 278 = 298

    def test_minimum_across_frames(self) -> None:
        """Test that the minimum crop is taken across all frames."""
        # Most samples return letterbox, one returns full frame (no crop)
        results = [_crop_result("crop=3840:1584:0:278")] * _SAMPLES_10K
        results[2] = _crop_result("crop=3840:2160:0:0")
        with (
            patch(
                "videotuner.create_encodes.subprocess.run",
                side_effect=results,
            ),
            patch(
                "videotuner.create_encodes.resolve_absolute_path",
                side_effect=_passthrough_resolve,
            ),
        ):
            result = calculate_cropdetect_values(
                source_path=_TEST_PATH,
                start_frame=0,
                num_frames=10000,
                fps=24.0,
                source_width=3840,
                source_height=2160,
            )
        # One frame had no crop, so minimum should be 0
        assert result.top == 0
        assert result.bottom == 0

    def test_no_cropdetect_output(self) -> None:
        """Test graceful handling when cropdetect produces no output."""
        with (
            patch(
                "videotuner.create_encodes.subprocess.run",
                return_value=_crop_result("no crop lines here"),
            ),
            patch(
                "videotuner.create_encodes.resolve_absolute_path",
                side_effect=_passthrough_resolve,
            ),
        ):
            result = calculate_cropdetect_values(
                source_path=_TEST_PATH,
                start_frame=0,
                num_frames=10000,
                fps=24.0,
                source_width=3840,
                source_height=2160,
            )
        assert result == CropValues(left=0, right=0, top=0, bottom=0)

    def test_hdr_inserts_tonemap(self) -> None:
        """Test that HDR mode inserts tonemapping in the filter chain."""
        with (
            patch(
                "videotuner.create_encodes.subprocess.run",
                return_value=_crop_result("crop=3840:2160:0:0"),
            ) as mock_run,
            patch(
                "videotuner.create_encodes.resolve_absolute_path",
                side_effect=_passthrough_resolve,
            ),
            patch("videotuner.create_encodes.has_vulkan_support", return_value=True),
        ):
            _ = calculate_cropdetect_values(
                source_path=_TEST_PATH,
                start_frame=0,
                num_frames=10000,
                fps=24.0,
                is_hdr=True,
                source_width=3840,
                source_height=2160,
            )
        # Check the first sample's -vf filter contains libplacebo
        first_call = mock_run.call_args_list[0]
        cmd = cast(list[str], first_call[0][0])
        vf_idx = cmd.index("-vf")
        vf_arg = cmd[vf_idx + 1]
        assert "libplacebo=" in vf_arg
        assert "cropdetect=" in vf_arg

    def test_format_conversion_before_cropdetect(self) -> None:
        """Test that format=yuv420p is inserted before cropdetect."""
        with (
            patch(
                "videotuner.create_encodes.subprocess.run",
                return_value=_crop_result("crop=3840:2160:0:0"),
            ) as mock_run,
            patch(
                "videotuner.create_encodes.resolve_absolute_path",
                side_effect=_passthrough_resolve,
            ),
        ):
            _ = calculate_cropdetect_values(
                source_path=_TEST_PATH,
                start_frame=0,
                num_frames=10000,
                fps=24.0,
                source_width=3840,
                source_height=2160,
            )
        first_call = mock_run.call_args_list[0]
        cmd = cast(list[str], first_call[0][0])
        vf_idx = cmd.index("-vf")
        vf_arg = cmd[vf_idx + 1]
        # format=yuv420p must appear before cropdetect
        assert "format=yuv420p" in vf_arg
        assert vf_arg.index("format=yuv420p") < vf_arg.index("cropdetect=")
        # Uses FFmpeg default limit (24), no custom limit parameter
        assert "limit=" not in vf_arg
        # skip=0 ensures cropdetect evaluates the very first frame
        assert "skip=0" in vf_arg

    def test_uses_ss_seeking(self) -> None:
        """Test that per-sample seeking uses -ss before -i."""
        with (
            patch(
                "videotuner.create_encodes.subprocess.run",
                return_value=_crop_result("crop=3840:2160:0:0"),
            ) as mock_run,
            patch(
                "videotuner.create_encodes.resolve_absolute_path",
                side_effect=_passthrough_resolve,
            ),
        ):
            _ = calculate_cropdetect_values(
                source_path=_TEST_PATH,
                start_frame=0,
                num_frames=10000,
                fps=24.0,
                source_width=3840,
                source_height=2160,
            )
        # Should make one subprocess call per sample
        assert mock_run.call_count == _SAMPLES_10K
        # Check first call uses -ss before -i
        first_call = mock_run.call_args_list[0]
        cmd = cast(list[str], first_call[0][0])
        ss_idx = cmd.index("-ss")
        i_idx = cmd.index("-i")
        assert ss_idx < i_idx
        assert "-frames:v" in cmd

    def test_custom_limit(self) -> None:
        """Test that a custom limit is included in the filter string."""
        with (
            patch(
                "videotuner.create_encodes.subprocess.run",
                return_value=_crop_result("crop=3840:2160:0:0"),
            ) as mock_run,
            patch(
                "videotuner.create_encodes.resolve_absolute_path",
                side_effect=_passthrough_resolve,
            ),
        ):
            _ = calculate_cropdetect_values(
                source_path=_TEST_PATH,
                start_frame=0,
                num_frames=10000,
                fps=24.0,
                source_width=3840,
                source_height=2160,
                cropdetect_limit=30,
            )
        first_call = mock_run.call_args_list[0]
        cmd = cast(list[str], first_call[0][0])
        vf_idx = cmd.index("-vf")
        vf_arg = cmd[vf_idx + 1]
        assert "limit=30" in vf_arg

    def test_custom_round(self) -> None:
        """Test that a custom round value is included in the filter string."""
        with (
            patch(
                "videotuner.create_encodes.subprocess.run",
                return_value=_crop_result("crop=3840:2160:0:0"),
            ) as mock_run,
            patch(
                "videotuner.create_encodes.resolve_absolute_path",
                side_effect=_passthrough_resolve,
            ),
        ):
            _ = calculate_cropdetect_values(
                source_path=_TEST_PATH,
                start_frame=0,
                num_frames=10000,
                fps=24.0,
                source_width=3840,
                source_height=2160,
                cropdetect_round=16,
            )
        first_call = mock_run.call_args_list[0]
        cmd = cast(list[str], first_call[0][0])
        vf_idx = cmd.index("-vf")
        vf_arg = cmd[vf_idx + 1]
        assert "round=16" in vf_arg

    def test_mvedges_mode(self) -> None:
        """Test that mvedges mode and its params are included in the filter string."""
        with (
            patch(
                "videotuner.create_encodes.subprocess.run",
                return_value=_crop_result("crop=3840:2160:0:0"),
            ) as mock_run,
            patch(
                "videotuner.create_encodes.resolve_absolute_path",
                side_effect=_passthrough_resolve,
            ),
        ):
            _ = calculate_cropdetect_values(
                source_path=_TEST_PATH,
                start_frame=0,
                num_frames=10000,
                fps=24.0,
                source_width=3840,
                source_height=2160,
                cropdetect_mode="mvedges",
                cropdetect_mv_threshold=10,
                cropdetect_low=0.05,
                cropdetect_high=0.10,
            )
        first_call = mock_run.call_args_list[0]
        cmd = cast(list[str], first_call[0][0])
        vf_idx = cmd.index("-vf")
        vf_arg = cmd[vf_idx + 1]
        assert "mode=mvedges" in vf_arg
        assert "mv_threshold=10" in vf_arg
        assert "low=0.05" in vf_arg
        assert "high=0.1" in vf_arg

    def test_default_mode_not_in_filter(self) -> None:
        """Test that default black mode is not explicitly in the filter string."""
        with (
            patch(
                "videotuner.create_encodes.subprocess.run",
                return_value=_crop_result("crop=3840:2160:0:0"),
            ) as mock_run,
            patch(
                "videotuner.create_encodes.resolve_absolute_path",
                side_effect=_passthrough_resolve,
            ),
        ):
            _ = calculate_cropdetect_values(
                source_path=_TEST_PATH,
                start_frame=0,
                num_frames=10000,
                fps=24.0,
                source_width=3840,
                source_height=2160,
            )
        first_call = mock_run.call_args_list[0]
        cmd = cast(list[str], first_call[0][0])
        vf_idx = cmd.index("-vf")
        vf_arg = cmd[vf_idx + 1]
        # Default mode (black) should not be explicitly passed
        assert "mode=" not in vf_arg
