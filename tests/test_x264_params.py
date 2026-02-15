"""Tests for x264 parameter building."""

from __future__ import annotations

from dataclasses import replace

from videotuner.media import VideoInfo
from videotuner.x264_params import GLOBAL_X264_PARAMS, build_global_x264_params

# Base SDR BT.709 VideoInfo used across tests.  Override fields via
# ``dataclasses.replace`` where a test needs different values.
_SDR_INFO = VideoInfo(
    fps=23.976,
    duration=120.0,
    pix_fmt="yuv420p",
    width=1920,
    height=1080,
    color_primaries="BT.709",
    color_trc=None,
    color_space="BT.709",
    color_range="tv",
)


class TestGlobalX264Params:
    """Tests for GLOBAL_X264_PARAMS constant."""

    def test_contains_expected_params(self):
        """Test GLOBAL_X264_PARAMS contains core auto-detected parameters."""
        assert "colorprim" in GLOBAL_X264_PARAMS
        assert "transfer" in GLOBAL_X264_PARAMS
        assert "colormatrix" in GLOBAL_X264_PARAMS
        assert "chromaloc" in GLOBAL_X264_PARAMS
        assert "output-depth" in GLOBAL_X264_PARAMS
        assert "range" in GLOBAL_X264_PARAMS

    def test_does_not_contain_hdr_params(self):
        """Test GLOBAL_X264_PARAMS excludes HDR-specific parameters."""
        assert "hdr10" not in GLOBAL_X264_PARAMS
        assert "hdr10-opt" not in GLOBAL_X264_PARAMS
        assert "master-display" not in GLOBAL_X264_PARAMS
        assert "max-cll" not in GLOBAL_X264_PARAMS
        assert "repeat-headers" not in GLOBAL_X264_PARAMS


class TestBuildGlobalX264Params:
    """Tests for build_global_x264_params function."""

    def test_sdr_bt709_basic(self):
        """Test basic SDR BT.709 content produces expected params."""
        params = build_global_x264_params(_SDR_INFO)

        assert "--output-depth" in params
        assert "--colorprim" in params
        assert "--colormatrix" in params
        assert "--range" in params

    def test_lossless_uses_qp_0(self):
        """Test lossless mode uses --qp 0 instead of --lossless."""
        params = build_global_x264_params(_SDR_INFO, is_lossless=True)

        assert "--qp" in params
        idx = params.index("--qp")
        assert params[idx + 1] == "0"
        # Should NOT contain --lossless (that's x265 only)
        assert "--lossless" not in params

    def test_8bit_output_depth(self):
        """Test 8-bit source produces 8-bit output depth."""
        params = build_global_x264_params(_SDR_INFO)

        idx = params.index("--output-depth")
        assert params[idx + 1] == "8"

    def test_10bit_output_depth(self):
        """Test 10-bit source produces 10-bit output depth."""
        info = replace(_SDR_INFO, pix_fmt="yuv420p10le")
        params = build_global_x264_params(info)

        idx = params.index("--output-depth")
        assert params[idx + 1] == "10"

    def test_12bit_capped_to_10bit(self):
        """Test 12-bit source is capped to 10-bit for x264."""
        info = replace(_SDR_INFO, pix_fmt="yuv420p12le")
        params = build_global_x264_params(info)

        idx = params.index("--output-depth")
        assert params[idx + 1] == "10"

    def test_color_range_tv(self):
        """Test TV color range uses 'tv' naming."""
        params = build_global_x264_params(_SDR_INFO)

        idx = params.index("--range")
        assert params[idx + 1] == "tv"

    def test_color_range_pc(self):
        """Test full color range uses 'pc' naming."""
        info = replace(_SDR_INFO, color_range="pc")
        params = build_global_x264_params(info)

        idx = params.index("--range")
        assert params[idx + 1] == "pc"

    def test_chroma_location_passed(self):
        """Test chroma location is included when provided."""
        params = build_global_x264_params(_SDR_INFO, chroma_location=1)

        assert "--chromaloc" in params
        idx = params.index("--chromaloc")
        assert params[idx + 1] == "1"

    def test_chroma_location_omitted_when_none(self):
        """Test chroma location is omitted when not provided."""
        params = build_global_x264_params(_SDR_INFO, chroma_location=None)

        assert "--chromaloc" not in params

    def test_skip_params(self):
        """Test skip_params excludes specified parameters."""
        params = build_global_x264_params(_SDR_INFO, skip_params={"colorprim", "range"})

        assert "--colorprim" not in params
        assert "--range" not in params
        # Other params should still be present
        assert "--output-depth" in params
        assert "--colormatrix" in params

    def test_no_hdr_metadata_flags(self):
        """Test HDR source does not produce HDR metadata flags."""
        info = replace(
            _SDR_INFO,
            color_trc="PQ",
            color_primaries="BT.2020",
            color_space="BT.2020 non-constant",
            mastering_display_luminance="min: 0.0050 cd/m2, max: 1000.0000 cd/m2",
            maximum_content_light_level="1000",
        )
        params = build_global_x264_params(info)

        # x264 should NOT include HDR metadata flags
        assert "--hdr10" not in params
        assert "--hdr10-opt" not in params
        assert "--master-display" not in params
        assert "--max-cll" not in params
        assert "--repeat-headers" not in params

    def test_bt2020_color_primaries(self):
        """Test BT.2020 color primaries mapping."""
        info = replace(_SDR_INFO, color_primaries="BT.2020")
        params = build_global_x264_params(info)

        idx = params.index("--colorprim")
        assert params[idx + 1] == "bt2020"

    def test_bt2020nc_color_matrix(self):
        """Test BT.2020 non-constant color matrix mapping."""
        info = replace(_SDR_INFO, color_space="BT.2020 non-constant")
        params = build_global_x264_params(info)

        idx = params.index("--colormatrix")
        assert params[idx + 1] == "bt2020nc"
