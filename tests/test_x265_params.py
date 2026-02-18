"""Tests for x265_params module."""

from __future__ import annotations

from videotuner.x265_params import build_global_x265_params, VALID_PRESETS
from videotuner.media import VideoInfo


class TestBuildGlobalX265Params:
    """Tests for building global x265 parameters."""

    def test_sdr_video_basic_params(self):
        """Test basic SDR video parameter generation."""
        video_info = VideoInfo(
            fps=24.0,
            duration=100.0,
            pix_fmt="yuv420p",
            color_trc="BT.709",
            color_primaries="BT.709",
            color_space="BT.709",
            color_range="tv",
        )

        params = build_global_x265_params(video_info)

        # Should include basic params
        assert "--aud" in params
        assert "--hrd" in params
        assert "--output-depth" in params
        assert "8" in params  # 8-bit output
        assert "--no-repeat-headers" in params  # SDR should not repeat headers
        assert "--no-hdr10" in params  # SDR should disable HDR10

    def test_hdr10_video_with_pq_transfer(self):
        """Test HDR10 video with PQ transfer characteristic."""
        video_info = VideoInfo(
            fps=24.0,
            duration=100.0,
            pix_fmt="yuv420p10le",
            color_trc="PQ",
            color_primaries="BT.2020",
            color_space="BT.2020 non-constant",
            color_range="tv",
        )

        params = build_global_x265_params(video_info)

        # HDR-specific params
        assert "--hdr10" in params
        assert "--hdr10-opt" in params
        assert "--repeat-headers" in params
        assert "--output-depth" in params
        assert "10" in params  # 10-bit output

        # Color parameters
        assert "--colorprim" in params
        assert "bt2020" in params
        assert "--transfer" in params
        assert "smpte2084" in params
        assert "--colormatrix" in params
        assert "bt2020nc" in params

    def test_hlg_transfer_is_hdr(self):
        """Test that HLG transfer is detected as HDR."""
        video_info = VideoInfo(
            fps=24.0,
            duration=100.0,
            pix_fmt="yuv420p10le",
            color_trc="HLG",
            color_primaries="BT.2020",
        )

        params = build_global_x265_params(video_info)

        assert "--hdr10" in params
        assert "--hdr10-opt" in params
        assert "--transfer" in params
        assert "arib-std-b67" in params

    def test_lossless_mode_adds_flag(self):
        """Test that lossless mode adds --lossless flag."""
        video_info = VideoInfo(fps=24.0, duration=100.0, pix_fmt="yuv420p")

        params = build_global_x265_params(video_info, is_lossless=True)

        assert "--lossless" in params

    def test_lossless_mode_excludes_hrd(self):
        """Test that lossless mode excludes --hrd flag."""
        video_info = VideoInfo(fps=24.0, duration=100.0, pix_fmt="yuv420p")

        params = build_global_x265_params(video_info, is_lossless=True)

        assert "--hrd" not in params

    def test_chroma_location_preserved(self):
        """Test that chroma location is added to params."""
        video_info = VideoInfo(fps=24.0, duration=100.0, pix_fmt="yuv420p")

        params = build_global_x265_params(video_info, chroma_location=2)

        assert "--chromaloc" in params
        assert "2" in params

    def test_color_range_full(self):
        """Test that full color range (pc) is converted correctly."""
        video_info = VideoInfo(
            fps=24.0, duration=100.0, pix_fmt="yuv420p", color_range="pc"
        )

        params = build_global_x265_params(video_info)

        assert "--range" in params
        assert "full" in params

    def test_color_range_limited(self):
        """Test that limited color range (tv) is converted correctly."""
        video_info = VideoInfo(
            fps=24.0, duration=100.0, pix_fmt="yuv420p", color_range="tv"
        )

        params = build_global_x265_params(video_info)

        assert "--range" in params
        assert "limited" in params

    def test_master_display_metadata_added(self):
        """Test that master display metadata is added when present."""
        video_info = VideoInfo(
            fps=24.0,
            duration=100.0,
            pix_fmt="yuv420p10le",
            color_trc="PQ",
            mastering_display_color_primaries="Display P3",
            mastering_display_luminance="min: 0.0050 cd/m2, max: 1000 cd/m2",
        )

        params = build_global_x265_params(video_info)

        assert "--master-display" in params
        # Should contain the parsed master display string
        master_display_idx = params.index("--master-display")
        master_display_value = params[master_display_idx + 1]
        assert "G(" in master_display_value
        assert "L(" in master_display_value

    def test_max_cll_metadata_added(self):
        """Test that MaxCLL/MaxFALL metadata is added when present."""
        video_info = VideoInfo(
            fps=24.0,
            duration=100.0,
            pix_fmt="yuv420p10le",
            maximum_content_light_level="1000 cd/m2",
            maximum_frameaverage_light_level="400 cd/m2",
        )

        params = build_global_x265_params(video_info)

        assert "--max-cll" in params
        max_cll_idx = params.index("--max-cll")
        max_cll_value = params[max_cll_idx + 1]
        assert "1000,400" == max_cll_value

    def test_skip_params_honored(self):
        """Test that skip_params prevents certain parameters from being added."""
        video_info = VideoInfo(
            fps=24.0,
            duration=100.0,
            pix_fmt="yuv420p10le",
            color_trc="PQ",
            color_primaries="BT.2020",
        )

        skip = {"hdr10", "colorprim"}
        params = build_global_x265_params(video_info, skip_params=skip)

        # These should be skipped
        assert "--hdr10" not in params
        assert "--colorprim" not in params

        # These should still be present
        assert "--hdr10-opt" in params
        assert "--transfer" in params

    def test_10bit_output_depth(self):
        """Test that 10-bit pixel format sets output-depth to 10."""
        video_info = VideoInfo(fps=24.0, duration=100.0, pix_fmt="yuv420p10le")

        params = build_global_x265_params(video_info)

        assert "--output-depth" in params
        depth_idx = params.index("--output-depth")
        assert params[depth_idx + 1] == "10"

    def test_12bit_output_depth(self):
        """Test that 12-bit pixel format sets output-depth to 12."""
        video_info = VideoInfo(fps=24.0, duration=100.0, pix_fmt="yuv420p12le")

        params = build_global_x265_params(video_info)

        assert "--output-depth" in params
        depth_idx = params.index("--output-depth")
        assert params[depth_idx + 1] == "12"

    def test_bt601_color_primaries(self):
        """Test BT.601 NTSC color primaries mapping."""
        video_info = VideoInfo(
            fps=24.0,
            duration=100.0,
            pix_fmt="yuv420p",
            color_primaries="BT.601 NTSC",
            color_space="BT.601",
        )

        params = build_global_x265_params(video_info)

        assert "--colorprim" in params
        assert "smpte170m" in params
        assert "--colormatrix" in params
        assert "smpte170m" in params

    def test_colormatrix_fallback_from_primaries(self):
        """Test that color matrix is inferred from primaries when color_space is unknown."""
        video_info = VideoInfo(
            fps=24.0,
            duration=100.0,
            pix_fmt="yuv420p",
            color_primaries="BT.2020",
            color_space=None,  # Unknown matrix
        )

        params = build_global_x265_params(video_info)

        # Should infer bt2020nc from BT.2020 primaries
        assert "--colormatrix" in params
        assert "bt2020nc" in params


class TestValidPresets:
    """Tests for valid preset constants."""

    def test_all_valid_presets_defined(self):
        """Test that all x265 presets are defined."""
        expected = {
            "ultrafast",
            "superfast",
            "veryfast",
            "faster",
            "fast",
            "medium",
            "slow",
            "slower",
            "veryslow",
            "placebo",
        }
        assert set(VALID_PRESETS) == expected
