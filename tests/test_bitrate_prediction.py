"""Tests for bitrate prediction module."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from videotuner.pipeline_iteration import calculate_predicted_bitrate


class TestCalculatePredictedBitrate:
    """Tests for predicted bitrate calculation."""

    def test_calculates_single_vmaf_file_bitrate(self):
        """Test calculates bitrate from single VMAF file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vmaf_path = Path(tmpdir) / "vmaf.mkv"
            vmaf_path.touch()  # Create empty file

            # Mock get_encode_stats and parse_video_info
            mock_stats = MagicMock()
            mock_stats.bitrate_kbps = 5000.0

            mock_info = MagicMock()
            mock_info.duration = 120.0  # 120 seconds

            with (
                patch(
                    "videotuner.pipeline_iteration.get_encode_stats",
                    return_value=mock_stats,
                ),
                patch(
                    "videotuner.pipeline_iteration.parse_video_info",
                    return_value=mock_info,
                ),
            ):
                logger = logging.getLogger("test")
                result = calculate_predicted_bitrate(
                    vmaf_distorted_path=vmaf_path,
                    ssim2_distorted_path=None,
                    ffprobe_bin="ffprobe",
                    log=logger,
                )

                assert result == 5000.0

    def test_calculates_single_ssim2_file_bitrate(self):
        """Test calculates bitrate from single SSIM2 file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ssim2_path = Path(tmpdir) / "ssim2.mkv"
            ssim2_path.touch()

            mock_stats = MagicMock()
            mock_stats.bitrate_kbps = 6000.0

            mock_info = MagicMock()
            mock_info.duration = 60.0

            with (
                patch(
                    "videotuner.pipeline_iteration.get_encode_stats",
                    return_value=mock_stats,
                ),
                patch(
                    "videotuner.pipeline_iteration.parse_video_info",
                    return_value=mock_info,
                ),
            ):
                logger = logging.getLogger("test")
                result = calculate_predicted_bitrate(
                    vmaf_distorted_path=None,
                    ssim2_distorted_path=ssim2_path,
                    ffprobe_bin="ffprobe",
                    log=logger,
                )

                assert result == 6000.0

    def test_calculates_duration_weighted_average_for_both_files(self):
        """Test calculates duration-weighted average when both files present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vmaf_path = Path(tmpdir) / "vmaf.mkv"
            ssim2_path = Path(tmpdir) / "ssim2.mkv"
            vmaf_path.touch()
            ssim2_path.touch()

            def mock_get_encode_stats(
                path: Path | str, *, ffprobe_bin: str = "ffprobe"
            ) -> MagicMock:
                _ = ffprobe_bin  # Intentionally unused in mock
                if "vmaf" in str(path):
                    stats = MagicMock()
                    stats.bitrate_kbps = 5000.0  # VMAF: 5000 kbps
                    return stats
                else:
                    stats = MagicMock()
                    stats.bitrate_kbps = 6000.0  # SSIM2: 6000 kbps
                    return stats

            def mock_parse_video_info(
                path: Path | str, *, ffprobe_bin: str = "ffprobe", log_hdr_metadata: bool = True
            ) -> MagicMock:
                _ = ffprobe_bin  # Intentionally unused in mock
                _ = log_hdr_metadata  # Intentionally unused in mock
                if "vmaf" in str(path):
                    info = MagicMock()
                    info.duration = 120.0  # VMAF: 120 seconds
                    return info
                else:
                    info = MagicMock()
                    info.duration = 60.0  # SSIM2: 60 seconds
                    return info

            with (
                patch(
                    "videotuner.pipeline_iteration.get_encode_stats",
                    side_effect=mock_get_encode_stats,
                ),
                patch(
                    "videotuner.pipeline_iteration.parse_video_info",
                    side_effect=mock_parse_video_info,
                ),
            ):
                logger = logging.getLogger("test")
                result = calculate_predicted_bitrate(
                    vmaf_distorted_path=vmaf_path,
                    ssim2_distorted_path=ssim2_path,
                    ffprobe_bin="ffprobe",
                    log=logger,
                )

                # Duration-weighted: (5000 * 120 + 6000 * 60) / (120 + 60)
                # = (600000 + 360000) / 180 = 960000 / 180 = 5333.33...
                expected = (5000.0 * 120.0 + 6000.0 * 60.0) / (120.0 + 60.0)
                assert abs(result - expected) < 0.01

    def test_returns_zero_when_no_files_provided(self):
        """Test returns 0.0 when no files are provided."""
        logger = logging.getLogger("test")
        result = calculate_predicted_bitrate(
            vmaf_distorted_path=None,
            ssim2_distorted_path=None,
            ffprobe_bin="ffprobe",
            log=logger,
        )

        assert result == 0.0

    def test_returns_zero_when_files_dont_exist(self):
        """Test returns 0.0 when provided files don't exist."""
        nonexistent_vmaf = Path("/nonexistent/vmaf.mkv")
        nonexistent_ssim2 = Path("/nonexistent/ssim2.mkv")

        logger = logging.getLogger("test")
        result = calculate_predicted_bitrate(
            vmaf_distorted_path=nonexistent_vmaf,
            ssim2_distorted_path=nonexistent_ssim2,
            ffprobe_bin="ffprobe",
            log=logger,
        )

        assert result == 0.0

    def test_handles_missing_stats(self):
        """Test handles case where get_encode_stats returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vmaf_path = Path(tmpdir) / "vmaf.mkv"
            vmaf_path.touch()

            with patch(
                "videotuner.pipeline_iteration.get_encode_stats", return_value=None
            ):
                logger = logging.getLogger("test")
                result = calculate_predicted_bitrate(
                    vmaf_distorted_path=vmaf_path,
                    ssim2_distorted_path=None,
                    ffprobe_bin="ffprobe",
                    log=logger,
                )

                assert result == 0.0

    def test_handles_missing_duration(self):
        """Test handles case where duration is None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vmaf_path = Path(tmpdir) / "vmaf.mkv"
            vmaf_path.touch()

            mock_stats = MagicMock()
            mock_stats.bitrate_kbps = 5000.0

            mock_info = MagicMock()
            mock_info.duration = None  # Missing duration

            with (
                patch(
                    "videotuner.pipeline_iteration.get_encode_stats",
                    return_value=mock_stats,
                ),
                patch(
                    "videotuner.pipeline_iteration.parse_video_info",
                    return_value=mock_info,
                ),
            ):
                logger = logging.getLogger("test")
                result = calculate_predicted_bitrate(
                    vmaf_distorted_path=vmaf_path,
                    ssim2_distorted_path=None,
                    ffprobe_bin="ffprobe",
                    log=logger,
                )

                # Should skip the file if duration is missing
                assert result == 0.0

    def test_handles_missing_video_info(self):
        """Test handles case where parse_video_info returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vmaf_path = Path(tmpdir) / "vmaf.mkv"
            vmaf_path.touch()

            mock_stats = MagicMock()
            mock_stats.bitrate_kbps = 5000.0

            with (
                patch(
                    "videotuner.pipeline_iteration.get_encode_stats",
                    return_value=mock_stats,
                ),
                patch("videotuner.pipeline_iteration.parse_video_info", return_value=None),
            ):
                logger = logging.getLogger("test")
                result = calculate_predicted_bitrate(
                    vmaf_distorted_path=vmaf_path,
                    ssim2_distorted_path=None,
                    ffprobe_bin="ffprobe",
                    log=logger,
                )

                assert result == 0.0

    def test_skips_file_with_valid_stats_but_no_duration(self):
        """Test skips file if stats exist but duration is unavailable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vmaf_path = Path(tmpdir) / "vmaf.mkv"
            ssim2_path = Path(tmpdir) / "ssim2.mkv"
            vmaf_path.touch()
            ssim2_path.touch()

            def mock_get_encode_stats(
                path: Path | str, *, ffprobe_bin: str = "ffprobe"
            ) -> MagicMock:
                _ = (path, ffprobe_bin)  # Intentionally unused in mock
                stats = MagicMock()
                stats.bitrate_kbps = 5000.0
                return stats

            def mock_parse_video_info(
                path: Path | str, *, ffprobe_bin: str = "ffprobe", log_hdr_metadata: bool = True
            ) -> MagicMock | None:
                _ = ffprobe_bin  # Intentionally unused in mock
                _ = log_hdr_metadata  # Intentionally unused in mock
                if "vmaf" in str(path):
                    # VMAF has no duration
                    return None
                else:
                    # SSIM2 has valid duration
                    info = MagicMock()
                    info.duration = 60.0
                    return info

            with (
                patch(
                    "videotuner.pipeline_iteration.get_encode_stats",
                    side_effect=mock_get_encode_stats,
                ),
                patch(
                    "videotuner.pipeline_iteration.parse_video_info",
                    side_effect=mock_parse_video_info,
                ),
            ):
                logger = logging.getLogger("test")
                result = calculate_predicted_bitrate(
                    vmaf_distorted_path=vmaf_path,
                    ssim2_distorted_path=ssim2_path,
                    ffprobe_bin="ffprobe",
                    log=logger,
                )

                # Should only use SSIM2 file
                assert result == 5000.0

    def test_handles_zero_total_duration(self):
        """Test handles edge case where total duration is 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vmaf_path = Path(tmpdir) / "vmaf.mkv"
            vmaf_path.touch()

            mock_stats = MagicMock()
            mock_stats.bitrate_kbps = 5000.0

            mock_info = MagicMock()
            mock_info.duration = 0.0  # Zero duration

            with (
                patch(
                    "videotuner.pipeline_iteration.get_encode_stats",
                    return_value=mock_stats,
                ),
                patch(
                    "videotuner.pipeline_iteration.parse_video_info",
                    return_value=mock_info,
                ),
            ):
                logger = logging.getLogger("test")
                result = calculate_predicted_bitrate(
                    vmaf_distorted_path=vmaf_path,
                    ssim2_distorted_path=None,
                    ffprobe_bin="ffprobe",
                    log=logger,
                )

                # Should return 0 if total duration is 0 (division by zero prevention)
                assert result == 0.0

    def test_uses_provided_ffprobe_bin_path(self):
        """Test uses the provided ffprobe binary path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vmaf_path = Path(tmpdir) / "vmaf.mkv"
            vmaf_path.touch()

            mock_stats = MagicMock()
            mock_stats.bitrate_kbps = 5000.0

            mock_info = MagicMock()
            mock_info.duration = 120.0

            with (
                patch(
                    "videotuner.pipeline_iteration.get_encode_stats",
                    return_value=mock_stats,
                ) as mock_get_stats,
                patch(
                    "videotuner.pipeline_iteration.parse_video_info",
                    return_value=mock_info,
                ) as mock_parse,
            ):
                logger = logging.getLogger("test")
                _ = calculate_predicted_bitrate(
                    vmaf_distorted_path=vmaf_path,
                    ssim2_distorted_path=None,
                    ffprobe_bin="/custom/path/to/ffprobe",
                    log=logger,
                )

                # Verify custom ffprobe path was passed to both functions
                mock_get_stats.assert_called_once_with(
                    vmaf_path, ffprobe_bin="/custom/path/to/ffprobe"
                )
                mock_parse.assert_called_once_with(
                    vmaf_path, ffprobe_bin="/custom/path/to/ffprobe", log_hdr_metadata=False
                )
