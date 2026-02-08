"""Tests for tonemapping module."""

from __future__ import annotations

import subprocess
from typing import cast
from unittest.mock import patch, MagicMock

from videotuner.tonemapping import has_vulkan_support, build_tonemap_chain


class TestHasVulkanSupport:
    """Tests for Vulkan GPU detection."""

    def setup_method(self) -> None:
        # Clear the lru_cache between tests
        has_vulkan_support.cache_clear()

    def test_vulkan_available(self) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("videotuner.tonemapping.subprocess.run", return_value=mock_result):
            assert has_vulkan_support("ffmpeg") is True

    def test_vulkan_not_available(self) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 1
        with patch("videotuner.tonemapping.subprocess.run", return_value=mock_result):
            assert has_vulkan_support("ffmpeg") is False

    def test_ffmpeg_not_found(self) -> None:
        with patch(
            "videotuner.tonemapping.subprocess.run", side_effect=FileNotFoundError
        ):
            assert has_vulkan_support("nonexistent") is False

    def test_timeout(self) -> None:
        with patch(
            "videotuner.tonemapping.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="ffmpeg", timeout=10),
        ):
            assert has_vulkan_support("ffmpeg") is False

    def test_caching(self) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch(
            "videotuner.tonemapping.subprocess.run", return_value=mock_result
        ) as mock_run:
            assert has_vulkan_support("ffmpeg") is True
            assert has_vulkan_support("ffmpeg") is True
            # Should only be called once due to lru_cache
            mock_run.assert_called_once()

    def test_probe_command_structure(self) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch(
            "videotuner.tonemapping.subprocess.run", return_value=mock_result
        ) as mock_run:
            _ = has_vulkan_support("ffmpeg")
            assert mock_run.call_args is not None
            cmd = cast(list[str], mock_run.call_args[0][0])
            kwargs = cast(dict[str, object], mock_run.call_args[1])
            assert cmd[0] == "ffmpeg"
            assert "-init_hw_device" in cmd
            assert "vulkan" in cmd
            assert kwargs["timeout"] == 10


class TestBuildTonemapChain:
    """Tests for tonemapping filter chain building."""

    def test_gpu_path_contains_libplacebo(self) -> None:
        chain = build_tonemap_chain(3840, 2160, use_gpu=True)
        assert "libplacebo=" in chain

    def test_gpu_path_contains_bt2390(self) -> None:
        chain = build_tonemap_chain(3840, 2160, use_gpu=True)
        assert "tonemapping=bt.2390" in chain

    def test_gpu_path_peak_detect_off(self) -> None:
        chain = build_tonemap_chain(3840, 2160, use_gpu=True)
        assert "peak_detect=0" in chain

    def test_gpu_path_bt709_output(self) -> None:
        chain = build_tonemap_chain(3840, 2160, use_gpu=True)
        assert "colorspace=bt709" in chain
        assert "color_primaries=bt709" in chain
        assert "color_trc=bt709" in chain

    def test_gpu_path_dimensions(self) -> None:
        chain = build_tonemap_chain(1920, 1080, use_gpu=True)
        assert "w=1920" in chain
        assert "h=1080" in chain

    def test_gpu_path_limited_range(self) -> None:
        chain = build_tonemap_chain(3840, 2160, use_gpu=True)
        assert "range=limited" in chain

    def test_cpu_path_contains_zscale(self) -> None:
        chain = build_tonemap_chain(3840, 2160, use_gpu=False)
        assert "zscale=" in chain

    def test_cpu_path_contains_hable(self) -> None:
        chain = build_tonemap_chain(3840, 2160, use_gpu=False)
        assert "tonemap=hable" in chain

    def test_cpu_path_dimensions(self) -> None:
        chain = build_tonemap_chain(1920, 1080, use_gpu=False)
        assert "scale=1920:1080" in chain

    def test_cpu_path_npl(self) -> None:
        chain = build_tonemap_chain(3840, 2160, use_gpu=False)
        assert "npl=100" in chain

    def test_cpu_path_desat(self) -> None:
        chain = build_tonemap_chain(3840, 2160, use_gpu=False)
        assert "desat=2" in chain

    def test_cpu_path_no_libplacebo(self) -> None:
        chain = build_tonemap_chain(3840, 2160, use_gpu=False)
        assert "libplacebo" not in chain

    def test_gpu_path_no_zscale(self) -> None:
        chain = build_tonemap_chain(3840, 2160, use_gpu=True)
        assert "zscale" not in chain
