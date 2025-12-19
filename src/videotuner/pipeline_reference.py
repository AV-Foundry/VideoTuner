"""Reference generation utilities for the encoding pipeline.

This module contains helpers for generating lossless reference encodes
used for quality assessment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .progress import PipelineDisplay
    from .encoding_utils import CropValues
    from .media import VideoInfo
    from .profiles import Profile


@dataclass(frozen=True)
class MetricSamplingParams:
    """Parameters for periodic sampling of a metric."""

    interval_frames: int
    region_frames: int
    guard_start_frames: int
    guard_end_frames: int
    total_frames: int

    @property
    def usable_frames(self) -> int:
        """Number of frames available for sampling (excluding guard bands)."""
        return self.total_frames - self.guard_start_frames - self.guard_end_frames

    @property
    def num_samples(self) -> int:
        """Number of samples that will be generated."""
        return (
            self.usable_frames + self.interval_frames - self.region_frames
        ) // self.interval_frames

    @property
    def total_sample_frames(self) -> int:
        """Total number of frames across all samples."""
        return self.num_samples * self.region_frames

    @property
    def coverage_percent(self) -> float:
        """Percentage of video covered by samples."""
        if self.total_frames == 0:
            return 0.0
        return (self.total_sample_frames / self.total_frames) * 100


def generate_metric_reference(
    metric_type: Literal["vmaf", "ssim2"],
    source_path: Path,
    output_dir: Path,
    sampling_params: MetricSamplingParams,
    fps: float,
    lossless_profile: "Profile",
    video_info: "VideoInfo",
    mkvmerge_bin: str,
    repo_root: Path,
    temp_dir: Path,
    display: "PipelineDisplay",
    log: logging.Logger,
    auto_crop: bool = False,
    crop_values: "CropValues | None" = None,
) -> Path | None:
    """Generate a concatenated reference file for quality metric assessment.

    This function handles both VMAF and SSIM2 reference generation using the same
    logic, reducing code duplication.

    Args:
        metric_type: Type of metric ("vmaf" or "ssim2")
        source_path: Path to source video
        output_dir: Directory for reference output
        sampling_params: Parameters controlling periodic sampling
        fps: Video frame rate
        lossless_profile: Profile for lossless encoding
        video_info: Video metadata
        mkvmerge_bin: Path to mkvmerge binary
        repo_root: Repository root directory
        temp_dir: Temporary directory for intermediate files
        display: Pipeline display for progress UI
        log: Logger for status messages
        auto_crop: Whether to apply autocrop
        crop_values: Crop values if autocropping

    Returns:
        Path to the generated reference MKV, or None if generation failed
    """
    from .create_encodes import encode_x265_concatenated_reference, mux_hevc_to_mkv

    metric_upper = metric_type.upper()
    ref_path = output_dir / f"{metric_type}_reference_concatenated.mkv"
    total_frames = sampling_params.total_sample_frames

    hevc_path: Path | None = None

    # Encode reference HEVC
    with display.stage(
        f"Encoding {metric_upper} reference",
        total=total_frames,
        unit="frames",
        transient=True,
        show_done=True,
    ) as enc_stage:
        enc_handler = enc_stage.make_x265_handler(total_frames=total_frames)

        try:
            hevc_path = encode_x265_concatenated_reference(
                source_path=source_path,
                output_path=ref_path,
                interval_frames=sampling_params.interval_frames,
                region_frames=sampling_params.region_frames,
                guard_start_frames=sampling_params.guard_start_frames,
                guard_end_frames=sampling_params.guard_end_frames,
                total_frames=sampling_params.total_frames,
                fps=fps,
                profile=lossless_profile,
                video_info=video_info,
                mkvmerge_bin=mkvmerge_bin,
                cwd=repo_root,
                temp_dir=temp_dir,
                line_handler=enc_handler,
                mux_handler=None,
                perform_mux=False,
                enable_autocrop=auto_crop,
                crop_values=crop_values,
                metric_label=metric_upper,
            )
            log.info(
                "%s concatenated reference HEVC created: %s",
                metric_upper,
                hevc_path.name,
            )
        except Exception as e:
            log.error("Failed to create %s reference: %s", metric_upper, e)
            _cleanup_hevc(hevc_path)
            return None

    # hevc_path is guaranteed to be Path here (exception would have returned None)
    assert hevc_path is not None

    # Mux HEVC to MKV
    with display.stage(
        f"Muxing {metric_upper} reference",
        total=100,
        unit="%",
        transient=True,
        show_done=True,
    ) as mux_stage:
        mux_handler = mux_stage.make_percent_handler()

        try:
            mux_hevc_to_mkv(
                hevc_path=hevc_path,
                output_path=ref_path,
                mkvmerge_bin=mkvmerge_bin,
                cwd=repo_root,
                line_handler=mux_handler,
            )
            log.info(
                "%s concatenated reference created: %s",
                metric_upper,
                ref_path.name,
            )
        except Exception as e:
            log.error("Failed to mux %s reference: %s", metric_upper, e)
            return None
        finally:
            _cleanup_hevc(hevc_path)

    return ref_path


def _cleanup_hevc(hevc_path: Path | None) -> None:
    """Safely remove HEVC intermediate file."""
    if hevc_path is not None and hevc_path.exists():
        try:
            hevc_path.unlink()
        except Exception:
            pass
