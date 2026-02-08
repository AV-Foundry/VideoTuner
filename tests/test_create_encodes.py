"""Tests for create_encodes module."""

from __future__ import annotations

from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest

from videotuner.create_encodes import CropConfig, mod2, calculate_autocrop_values
from videotuner.encoding_utils import CropValues
from videotuner.tool_parsers import CROPDETECT_RE

_TEST_PATH = Path("test.mkv")


def _passthrough_resolve(p: Path, _c: Path | None) -> Path:
    """Typed side_effect for resolve_absolute_path mock."""
    return p


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


class TestCropConfig:
    """Tests for CropConfig dataclass."""

    def test_default_values(self) -> None:
        """Test CropConfig default values."""
        config = CropConfig()
        assert config.enabled is False
        assert config.values is None

    def test_disabled_factory(self) -> None:
        """Test CropConfig.disabled() factory method."""
        config = CropConfig.disabled()
        assert config.enabled is False
        assert config.values is None

    def test_with_values_factory(self) -> None:
        """Test CropConfig.with_values() factory method."""
        crop = CropValues(left=10, right=20, top=5, bottom=15)
        config = CropConfig.with_values(crop)
        assert config.enabled is True
        assert config.values is crop
        assert config.values is not None and config.values.left == 10

    def test_auto_factory(self) -> None:
        """Test CropConfig.auto() factory method."""
        config = CropConfig.auto()
        assert config.enabled is True
        assert config.values is None

    def test_frozen_dataclass(self) -> None:
        """Test CropConfig is frozen (immutable)."""
        config = CropConfig()
        with pytest.raises(AttributeError):
            setattr(config, "enabled", True)

    def test_explicit_enabled_with_values(self) -> None:
        """Test creating CropConfig with explicit enabled and values."""
        crop = CropValues(left=0, right=0, top=100, bottom=100)
        config = CropConfig(enabled=True, values=crop)
        assert config.enabled is True
        assert config.values is not None
        assert config.values.top == 100
        assert config.values.bottom == 100

    def test_disabled_with_values_ignored(self) -> None:
        """Test that values can be set even when disabled (not enforced)."""
        crop = CropValues(left=10, right=10, top=10, bottom=10)
        # This is technically valid even though semantically odd
        config = CropConfig(enabled=False, values=crop)
        assert config.enabled is False
        assert config.values is crop


class TestMod2:
    """Tests for mod2 helper."""

    def test_even_value_unchanged(self) -> None:
        assert mod2(10, "increase") == 10
        assert mod2(10, "decrease") == 10

    def test_odd_increase(self) -> None:
        assert mod2(11, "increase") == 12

    def test_odd_decrease(self) -> None:
        assert mod2(11, "decrease") == 10

    def test_zero(self) -> None:
        assert mod2(0, "increase") == 0
        assert mod2(0, "decrease") == 0

    def test_one_increase(self) -> None:
        assert mod2(1, "increase") == 2

    def test_one_decrease(self) -> None:
        assert mod2(1, "decrease") == 0


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


class TestCalculateAutocropValues:
    """Tests for calculate_autocrop_values with mocked FFmpeg."""

    def _mock_cropdetect_output(self, *crop_lines: str) -> str:
        """Build mock FFmpeg output from crop=W:H:X:Y lines."""
        return "\n".join(crop_lines)

    def test_basic_letterbox(self) -> None:
        """Test detection of consistent letterboxing (278px bars)."""
        output = self._mock_cropdetect_output(
            "crop=3840:1584:0:278",
            "crop=3840:1584:0:278",
            "crop=3840:1584:0:278",
        )
        with (
            patch("videotuner.create_encodes.run_capture", return_value=output),
            patch(
                "videotuner.create_encodes.resolve_absolute_path",
                side_effect=_passthrough_resolve,
            ),
        ):
            result = calculate_autocrop_values(
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
        output = self._mock_cropdetect_output(
            "crop=3840:1584:0:278",
            "crop=3840:2160:0:0",  # Full frame — no crop
            "crop=3840:1584:0:278",
        )
        with (
            patch("videotuner.create_encodes.run_capture", return_value=output),
            patch(
                "videotuner.create_encodes.resolve_absolute_path",
                side_effect=_passthrough_resolve,
            ),
        ):
            result = calculate_autocrop_values(
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
                "videotuner.create_encodes.run_capture", return_value="no crop lines"
            ),
            patch(
                "videotuner.create_encodes.resolve_absolute_path",
                side_effect=_passthrough_resolve,
            ),
        ):
            result = calculate_autocrop_values(
                source_path=_TEST_PATH,
                start_frame=0,
                num_frames=10000,
                fps=24.0,
                source_width=3840,
                source_height=2160,
            )
        assert result == CropValues(left=0, right=0, top=0, bottom=0)

    def test_mod2_increase(self) -> None:
        """Test mod-2 alignment with increase direction."""
        output = self._mock_cropdetect_output("crop=3838:1583:1:279")
        with (
            patch("videotuner.create_encodes.run_capture", return_value=output),
            patch(
                "videotuner.create_encodes.resolve_absolute_path",
                side_effect=_passthrough_resolve,
            ),
        ):
            result = calculate_autocrop_values(
                source_path=_TEST_PATH,
                start_frame=0,
                num_frames=10000,
                fps=24.0,
                mod_direction="increase",
                source_width=3840,
                source_height=2160,
            )
        # left=1→2, top=279→280, right=3840-3838-1=1→2, bottom=2160-1583-279=298→298
        assert result.left == 2
        assert result.top == 280
        assert result.right == 2
        assert result.bottom == 298

    def test_mod2_decrease(self) -> None:
        """Test mod-2 alignment with decrease direction."""
        output = self._mock_cropdetect_output("crop=3838:1583:1:279")
        with (
            patch("videotuner.create_encodes.run_capture", return_value=output),
            patch(
                "videotuner.create_encodes.resolve_absolute_path",
                side_effect=_passthrough_resolve,
            ),
        ):
            result = calculate_autocrop_values(
                source_path=_TEST_PATH,
                start_frame=0,
                num_frames=10000,
                fps=24.0,
                mod_direction="decrease",
                source_width=3840,
                source_height=2160,
            )
        # left=1→0, top=279→278, right=1→0, bottom=298→298
        assert result.left == 0
        assert result.top == 278
        assert result.right == 0
        assert result.bottom == 298

    def test_hdr_inserts_tonemap(self) -> None:
        """Test that HDR mode inserts tonemapping in the filter chain."""
        output = self._mock_cropdetect_output("crop=3840:2160:0:0")
        with (
            patch(
                "videotuner.create_encodes.run_capture", return_value=output
            ) as mock_run,
            patch(
                "videotuner.create_encodes.resolve_absolute_path",
                side_effect=_passthrough_resolve,
            ),
            patch("videotuner.create_encodes.has_vulkan_support", return_value=True),
        ):
            _ = calculate_autocrop_values(
                source_path=_TEST_PATH,
                start_frame=0,
                num_frames=10000,
                fps=24.0,
                is_hdr=True,
                source_width=3840,
                source_height=2160,
            )
        # Check that the -vf filter contains libplacebo
        cmd = cast(list[str], mock_run.call_args[0][0])
        vf_idx = cmd.index("-vf")
        vf_arg = cmd[vf_idx + 1]
        assert "libplacebo=" in vf_arg
        assert "cropdetect=" in vf_arg

    def test_threshold_conversion(self) -> None:
        """Test threshold is converted to cropdetect limit format."""
        output = self._mock_cropdetect_output("crop=3840:2160:0:0")
        with (
            patch(
                "videotuner.create_encodes.run_capture", return_value=output
            ) as mock_run,
            patch(
                "videotuner.create_encodes.resolve_absolute_path",
                side_effect=_passthrough_resolve,
            ),
        ):
            _ = calculate_autocrop_values(
                source_path=_TEST_PATH,
                start_frame=0,
                num_frames=10000,
                fps=24.0,
                threshold=10.0,
                source_width=3840,
                source_height=2160,
            )
        cmd = cast(list[str], mock_run.call_args[0][0])
        vf_idx = cmd.index("-vf")
        vf_arg = cmd[vf_idx + 1]
        # limit = (16 + 10/100 * 219) / 255 ≈ 0.148627
        assert "cropdetect=limit=0.14" in vf_arg
