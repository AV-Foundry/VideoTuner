"""Shared types and path utilities for the VideoTuner pipeline.

This module contains dataclasses and path management functions shared across
pipeline modules to avoid circular imports.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .progress import PipelineDisplay
from .encoding_utils import CropValues
from .media import VideoInfo
from .pipeline_cli import PipelineArgs
from .profiles import Profile
from .utils import ensure_dir


@dataclass
class IterationContext:
    """Shared context for pipeline iterations."""

    # Paths
    input_path: Path
    workdir: Path
    temp_dir: Path
    repo_root: Path

    # Video info
    info: VideoInfo

    # Profile and settings
    selected_profile: Profile

    # Periodic sampling parameters
    total_frames: int
    guard_start_frames: int
    guard_end_frames: int

    # Reference files (single concatenated file per metric)
    vmaf_ref_path: Path | None
    ssim2_ref_path: Path | None

    # Command-line args
    args: PipelineArgs

    # Display
    display: PipelineDisplay
    log: logging.Logger

    # CropDetect values (calculated once, shared across all encodes)
    crop_values: CropValues | None = None

    # Sample sharing: True when VMAF and SSIM2 use identical sampling parameters
    sharing_samples: bool = False

    @property
    def usable_frames(self) -> int:
        """Number of frames available for sampling (excluding guard bands)."""
        return self.total_frames - self.guard_start_frames - self.guard_end_frames


@dataclass(frozen=True)
class MultiProfileResult:
    """Results from a single profile in multi-profile mode (CRF search or bitrate).

    For CRF profiles: optimal_crf contains the searched value, meets_all_targets is bool
    For bitrate profiles: optimal_crf is None, meets_all_targets is None (targets N/A)
    """

    profile_name: str
    optimal_crf: float | None  # None if failed to converge or bitrate mode
    scores: dict[
        str, float | None
    ]  # Final metric scores (None for unavailable metrics)
    predicted_bitrate_kbps: float  # Predicted bitrate across all samples
    converged: bool  # True if search converged successfully
    meets_all_targets: bool | None = None  # True/False for CRF, None for bitrate (N/A)

    @property
    def is_bitrate_mode(self) -> bool:
        """Check if this result is from a bitrate profile (not CRF)."""
        return self.optimal_crf is None

    def is_valid(self) -> bool:
        """Check if this profile produced valid results.

        For CRF profiles: converged and has optimal_crf
        For bitrate profiles: converged (optimal_crf will be None)
        """
        return self.converged


# Path management utilities


def _profile_slug(profile: Profile) -> str:
    """Convert profile name to filesystem-safe slug."""
    return profile.name.replace(" ", "_").replace("/", "_").replace("\\", "_")


def get_reference_dir(workdir: Path) -> Path:
    """Get path to reference files directory, creating if needed.

    Args:
        workdir: Working directory for the job

    Returns:
        Path to reference directory
    """
    ref_dir = workdir / "reference"
    return ensure_dir(ref_dir)


def get_distorted_dir(workdir: Path, profile: Profile) -> Path:
    """Get path to distorted files directory for a profile, creating if needed.

    Args:
        workdir: Working directory for the job
        profile: Encoding profile (used for directory naming)

    Returns:
        Path to distorted files directory for this profile
    """
    dist_dir = workdir / "distorted" / f"profile_{_profile_slug(profile)}"
    return ensure_dir(dist_dir)


def get_vmaf_dir(workdir: Path, profile: Profile) -> Path:
    """Get path to VMAF output directory for a profile, creating if needed.

    Args:
        workdir: Working directory for the job
        profile: Encoding profile (used for directory naming)

    Returns:
        Path to VMAF assessment output directory
    """
    vmaf_dir = workdir / "vmaf" / f"{_profile_slug(profile)}_profile"
    return ensure_dir(vmaf_dir)


def get_ssim2_dir(workdir: Path, profile: Profile) -> Path:
    """Get path to SSIMULACRA2 output directory for a profile, creating if needed.

    Args:
        workdir: Working directory for the job
        profile: Encoding profile (used for directory naming)

    Returns:
        Path to SSIMULACRA2 assessment output directory
    """
    ssim2_dir = workdir / "ssimulacra2" / f"{_profile_slug(profile)}_profile"
    return ensure_dir(ssim2_dir)
