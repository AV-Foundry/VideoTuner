"""Shared encoding utilities for VideoTuner."""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# Directory name for bundled VapourSynth portable installation
VAPOURSYNTH_PORTABLE_DIR = "vapoursynth-portable"

# Relative path to bundled x265 encoder binary
X265_BIN_PATH = Path("tools") / "x265.exe"


@dataclass(frozen=True)
class CropValues:
    """Crop values for consistent cropping across all encodes."""

    left: int
    right: int
    top: int
    bottom: int


HDR_TRANSFER_CHARACTERISTICS: set[str] = {
    "pq",
    "smpte2084",
    "smpte 2084",
    "hlg",
    "arib-std-b67",
    "arib std-b67",
}


def is_hdr_video(color_trc: str | None) -> bool:
    """Check if video uses HDR transfer characteristics.

    Args:
        color_trc: Color transfer characteristic from video metadata

    Returns:
        True if video uses HDR transfer (PQ or HLG), False otherwise
    """
    if not color_trc:
        return False
    return color_trc.lower() in HDR_TRANSFER_CHARACTERISTICS


def get_x265_bin(cwd: Path | None = None) -> Path:
    """Get path to x265 binary.

    Args:
        cwd: Working directory (if None, uses relative path)

    Returns:
        Path to x265.exe binary
    """
    if cwd:
        return Path(cwd) / X265_BIN_PATH
    return X265_BIN_PATH


def get_vapoursynth_portable_dir(cwd: Path | None = None) -> Path:
    """Get path to VapourSynth portable directory.

    Args:
        cwd: Working directory (if None, uses relative path)

    Returns:
        Path to vapoursynth-portable directory
    """
    if cwd:
        return Path(cwd) / VAPOURSYNTH_PORTABLE_DIR
    return Path(VAPOURSYNTH_PORTABLE_DIR)


def resolve_absolute_path(path: Path, cwd: Path | None = None) -> Path:
    """Convert path to absolute, resolving relative to cwd if provided.

    Args:
        path: Path to resolve
        cwd: Working directory for relative path resolution

    Returns:
        Absolute path
    """
    if path.is_absolute():
        return path
    if cwd:
        return Path(cwd) / path
    return path.resolve()


def resolve_ssim2_bin(ssim2_bin_arg: str | None) -> str:
    """Resolve SSIMULACRA2 binary path from CLI argument or PATH.

    Args:
        ssim2_bin_arg: User-provided --ssim2-bin argument (None if not specified)

    Returns:
        Path to ssimulacra2_rs binary as string

    Raises:
        FileNotFoundError: If ssimulacra2_rs is not found on PATH when no arg provided
    """
    if ssim2_bin_arg:
        return str(ssim2_bin_arg)

    ssim2_bin = shutil.which("ssimulacra2_rs")
    if ssim2_bin is None:
        raise FileNotFoundError(
            "ssimulacra2_rs not found on PATH. Install with: cargo install ssimulacra2_rs\n"
            + "See: https://crates.io/crates/ssimulacra2_rs"
        )
    return ssim2_bin


def calculate_usable_frames(
    total_frames: int,
    guard_start_frames: int,
    guard_end_frames: int,
) -> int:
    """Calculate number of usable frames after excluding guard bands.

    Guard bands are regions at the start and end of the video that are
    excluded from sampling to avoid credits, intros, and outros.

    Args:
        total_frames: Total number of frames in the video.
        guard_start_frames: Frames to skip at the start.
        guard_end_frames: Frames to skip at the end.

    Returns:
        Number of frames available for sampling.

    Raises:
        ValueError: If parameters are invalid or result in no usable frames.
    """
    if total_frames < 1:
        raise ValueError(f"total_frames must be >= 1, got {total_frames}")
    if guard_start_frames < 0:
        raise ValueError(f"guard_start_frames must be >= 0, got {guard_start_frames}")
    if guard_end_frames < 0:
        raise ValueError(f"guard_end_frames must be >= 0, got {guard_end_frames}")

    usable = total_frames - guard_start_frames - guard_end_frames
    if usable <= 0:
        msg = (
            f"No usable frames after guards (total={total_frames}, "
            f"start={guard_start_frames}, end={guard_end_frames})"
        )
        raise ValueError(msg)
    return usable


def write_vpy_script(vpy_path: Path, content: str) -> None:
    """Write VapourSynth script content to file.

    Args:
        vpy_path: Path to .vpy file
        content: VapourSynth script content
    """
    _ = vpy_path.write_text(content, encoding="utf-8")


def create_temp_encode_paths(
    temp_dir: Path | None = None,
    name: str = "encode",
) -> tuple[Path, Path]:
    """Create temporary VPY and HEVC file paths.

    Args:
        temp_dir: Directory for temporary files (None for system temp)
        name: Base filename (without extension)

    Returns:
        Tuple of (vpy_path, hevc_path)
    """
    if temp_dir:
        from .utils import ensure_dir
        _ = ensure_dir(temp_dir)
        vpy_path = temp_dir / f"{name}.vpy"
        hevc_path = temp_dir / f"{name}.hevc"
    else:
        with tempfile.NamedTemporaryFile(
            suffix=".vpy", prefix=f"{name}_", delete=False, mode="w", encoding="utf-8"
        ) as tmp:
            vpy_path = Path(tmp.name)
        with tempfile.NamedTemporaryFile(
            suffix=".hevc", prefix=f"{name}_", delete=False
        ) as tmp:
            hevc_path = Path(tmp.name)

    return vpy_path, hevc_path


@dataclass(frozen=True)
class VapourSynthEnv:
    """Unified VapourSynth environment for encoding and assessment tools.

    VapourSynth is mandatory for both x265 encoding and ssimulacra2_rs.
    This class provides strict validation and comprehensive environment setup.

    Usage:
        vs_env = VapourSynthEnv.from_cwd(cwd)
        vs_env.validate()  # Raises FileNotFoundError if files missing
        env = vs_env.build_env()  # Get comprehensive environment dict
    """

    vs_dir: Path
    vs_plugin_dir: Path
    vsscript_dll: Path
    ffms2_dll: Path

    @classmethod
    def from_cwd(cls, cwd: Path | None) -> "VapourSynthEnv":
        """Resolve VapourSynth paths from working directory.

        Args:
            cwd: Working directory. If None, uses relative path from current dir.

        Returns:
            VapourSynthEnv with all paths resolved.
        """
        vs_dir = (
            Path(cwd) / VAPOURSYNTH_PORTABLE_DIR
            if cwd
            else Path(VAPOURSYNTH_PORTABLE_DIR)
        )
        vs_plugin_dir = vs_dir / "vs-plugins"
        return cls(
            vs_dir=vs_dir,
            vs_plugin_dir=vs_plugin_dir,
            vsscript_dll=vs_dir / "VSScript.dll",
            ffms2_dll=vs_plugin_dir / "ffms2.dll",
        )

    @classmethod
    def from_args(
        cls,
        vs_dir_arg: Path | None,
        vs_plugin_dir_arg: Path | None,
        repo_root: Path,
    ) -> "VapourSynthEnv":
        """Resolve VapourSynth paths from CLI args with repo root fallback.

        Args:
            vs_dir_arg: User-provided --vs-dir argument (None if not specified)
            vs_plugin_dir_arg: User-provided --vs-plugin-dir argument (None if not specified)
            repo_root: Repository root for default path resolution

        Returns:
            VapourSynthEnv with all paths resolved.
        """
        vs_dir = vs_dir_arg if vs_dir_arg is not None else repo_root / VAPOURSYNTH_PORTABLE_DIR
        vs_plugin_dir = vs_plugin_dir_arg if vs_plugin_dir_arg is not None else vs_dir / "vs-plugins"
        return cls(
            vs_dir=vs_dir,
            vs_plugin_dir=vs_plugin_dir,
            vsscript_dll=vs_dir / "VSScript.dll",
            ffms2_dll=vs_plugin_dir / "ffms2.dll",
        )

    def validate(self) -> None:
        """Validate that required VapourSynth files exist.

        Raises:
            FileNotFoundError: If VSScript.dll or ffms2.dll is missing.
        """
        if not self.vsscript_dll.exists():
            raise FileNotFoundError(
                f"VapourSynth VSScript.dll not found at: {self.vsscript_dll}"
            )
        if not self.ffms2_dll.exists():
            raise FileNotFoundError(f"FFMS2 plugin not found at: {self.ffms2_dll}")

    def build_env(self, base_env: dict[str, str] | None = None) -> dict[str, str]:
        """Build comprehensive environment dict with all VapourSynth paths.

        Configures VAPOURSYNTH_PORTABLE, PATH, VAPOURSYNTH_PLUGIN_PATH,
        VAPOURSYNTH_LIBRARY_PATH, and VSSCRIPT_LIBRARY_PATH.

        Args:
            base_env: Base environment to copy from. If None, uses os.environ.

        Returns:
            New environment dict with all VS paths configured.
        """
        env = dict(base_env) if base_env is not None else os.environ.copy()

        # Set VAPOURSYNTH_PORTABLE for tools that use it
        env["VAPOURSYNTH_PORTABLE"] = str(self.vs_dir)

        # Prepend VapourSynth directory to PATH so DLLs are found first
        env["PATH"] = str(self.vs_dir) + os.pathsep + env.get("PATH", "")

        # Set plugin path
        if self.vs_plugin_dir.exists():
            env["VAPOURSYNTH_PLUGIN_PATH"] = str(self.vs_plugin_dir)

        # Set library paths for DLLs
        vsdll = self.vs_dir / "vapoursynth.dll"
        if vsdll.exists():
            env["VAPOURSYNTH_LIBRARY_PATH"] = str(vsdll)
        if self.vsscript_dll.exists():
            env["VSSCRIPT_LIBRARY_PATH"] = str(self.vsscript_dll)

        return env

    @property
    def autocrop_dll(self) -> Path:
        """Path to the autocrop plugin DLL."""
        return self.vs_plugin_dir / "autocrop.dll"

    def has_autocrop_plugin(self) -> bool:
        """Check if the autocrop plugin is available.

        Returns:
            True if autocrop.dll exists in the plugin directory.
        """
        return self.autocrop_dll.exists()


@dataclass(frozen=True)
class EncoderPaths:
    """Resolved paths to all encoding tools (x265 + VapourSynth).

    Usage:
        paths = EncoderPaths.from_cwd(cwd)
        paths.validate()  # Raises FileNotFoundError if any tool missing
        env = paths.vs_env.build_env()
    """

    x265_bin: Path
    vs_env: VapourSynthEnv

    @classmethod
    def from_cwd(cls, cwd: Path | None) -> "EncoderPaths":
        """Resolve all encoder paths from working directory.

        Args:
            cwd: Working directory. If None, uses relative paths.

        Returns:
            EncoderPaths with x265 and VapourSynth paths resolved.
        """
        x265_bin = Path(cwd) / X265_BIN_PATH if cwd else X265_BIN_PATH
        return cls(x265_bin=x265_bin, vs_env=VapourSynthEnv.from_cwd(cwd))

    def validate(self) -> None:
        """Validate that all required encoder files exist.

        Raises:
            FileNotFoundError: If x265.exe, VSScript.dll, or ffms2.dll is missing.
        """
        if not self.x265_bin.exists():
            raise FileNotFoundError(f"x265 encoder not found at: {self.x265_bin}")
        self.vs_env.validate()


@dataclass(frozen=True)
class SamplingParams:
    """Parameters for periodic frame sampling.

    Attributes:
        interval_frames: Sample every N frames
        region_frames: Number of consecutive frames per sample
        guard_start_frames: Frames to skip at start (intros/credits)
        guard_end_frames: Frames to skip at end (credits)
        total_frames: Total frames in source video
    """

    interval_frames: int
    region_frames: int
    guard_start_frames: int
    guard_end_frames: int
    total_frames: int

    def validate(self) -> None:
        """Validate sampling parameters.

        Raises:
            ValueError: If any parameter is invalid.
        """
        if self.interval_frames < 1:
            raise ValueError(
                f"interval_frames must be >= 1, got {self.interval_frames}"
            )
        if self.region_frames < 1:
            raise ValueError(f"region_frames must be >= 1, got {self.region_frames}")
        if self.guard_start_frames < 0:
            raise ValueError(
                f"guard_start_frames must be >= 0, got {self.guard_start_frames}"
            )
        if self.guard_end_frames < 0:
            raise ValueError(
                f"guard_end_frames must be >= 0, got {self.guard_end_frames}"
            )
        if self.total_frames < 1:
            raise ValueError(f"total_frames must be >= 1, got {self.total_frames}")


@dataclass(frozen=True)
class UsableRange:
    """Result of calculating usable frame range after guard bands.

    Attributes:
        start: First usable frame index
        end: Last usable frame index (exclusive)
        frame_count: Number of usable frames (end - start)
    """

    start: int
    end: int
    frame_count: int


def calculate_usable_range(params: SamplingParams) -> UsableRange:
    """Calculate usable frame range after excluding guard bands.

    Args:
        params: Sampling parameters with guard frame counts

    Returns:
        UsableRange with start, end, and frame_count

    Raises:
        ValueError: If guards leave no usable frames or fewer than region_frames
    """
    usable_start = params.guard_start_frames
    usable_end = params.total_frames - params.guard_end_frames

    if usable_end <= usable_start:
        raise ValueError(
            f"No usable frames after guards (start={usable_start}, end={usable_end})"
        )

    usable_frames = usable_end - usable_start
    if usable_frames < params.region_frames:
        raise ValueError(
            f"Usable frames ({usable_frames}) less than region size ({params.region_frames})"
        )

    return UsableRange(start=usable_start, end=usable_end, frame_count=usable_frames)


def calculate_sample_count(
    usable_frames: int, interval_frames: int, region_frames: int
) -> tuple[int, int]:
    """Calculate number of samples and total sampled frames.

    Args:
        usable_frames: Number of frames available for sampling
        interval_frames: Sample every N frames
        region_frames: Consecutive frames per sample

    Returns:
        Tuple of (num_samples, total_sampled_frames)
    """
    num_samples = (usable_frames + interval_frames - region_frames) // interval_frames
    total_sampled_frames = num_samples * region_frames
    return num_samples, total_sampled_frames


def build_sampling_vpy_script(
    source_path: Path,
    cache_file: Path,
    usable_range: UsableRange,
    interval_frames: int,
    region_frames: int,
    fps: float,
    cwd: Path | None = None,
    crop_values: "CropValues | None" = None,
) -> str:
    """Build VapourSynth script for periodic frame sampling.

    Generates a script that uses SelectEvery to sample frames at regular
    intervals from the usable range of the video.

    Args:
        source_path: Input source video path
        cache_file: FFMS2 cache file path
        usable_range: Frame range to sample from
        interval_frames: Sample every N frames
        region_frames: Consecutive frames per sample
        fps: Video framerate
        cwd: Working directory for path resolution
        crop_values: Optional crop values to apply

    Returns:
        VapourSynth script content as string
    """
    abs_source_path = resolve_absolute_path(source_path, cwd)
    abs_cache_file = resolve_absolute_path(cache_file, cwd)
    fps_num = int(fps * 1000)

    vpy_lines = [
        "import vapoursynth as vs",
        "core = vs.core",
        "",
        f'clip = core.ffms2.Source(r"{abs_source_path}", cachefile=r"{abs_cache_file}")',
        "",
        "# Trim to usable range (skip guard bands)",
        f"usable = clip[{usable_range.start}:{usable_range.end}]",
        "",
        "# Select periodic samples using SelectEvery",
        f"# Every {interval_frames} frames, take {region_frames} consecutive frames",
        f"offsets = list(range({region_frames}))",
        f"sampled = usable.std.SelectEvery({interval_frames}, offsets)",
        "",
        "# Reset FPS to original rate (SelectEvery preserves timestamps, creating gaps)",
        "# This renumbers frames sequentially at the original FPS",
        f"sampled = sampled.std.AssumeFPS(fpsnum={fps_num}, fpsden=1000)",
    ]

    # Apply autocrop if provided
    if crop_values is not None:
        if (
            crop_values.left > 0
            or crop_values.right > 0
            or crop_values.top > 0
            or crop_values.bottom > 0
        ):
            vpy_lines.append("")
            vpy_lines.append("# Apply autocrop to sampled frames")
            crop_line = (
                f"sampled = core.std.Crop(sampled, left={crop_values.left}, "
                f"right={crop_values.right}, top={crop_values.top}, bottom={crop_values.bottom})"
            )
            vpy_lines.append(crop_line)

    vpy_lines.append("")
    vpy_lines.append("sampled.set_output()")

    return "\n".join(vpy_lines)


def build_x265_command(
    paths: EncoderPaths,
    vpy_path: Path,
    hevc_path: Path,
    x265_params: list[str],
    preset: str | None,
    cwd: Path | None = None,
) -> list[str]:
    """Build x265 command line arguments.

    Args:
        paths: Resolved encoder paths
        vpy_path: Path to VapourSynth script
        hevc_path: Output HEVC path
        x265_params: Additional x265 parameters
        preset: x265 preset (or None)
        cwd: Working directory for path resolution

    Returns:
        List of command arguments for x265
    """
    abs_hevc_path = resolve_absolute_path(hevc_path, cwd)
    abs_vpy_path = resolve_absolute_path(vpy_path, cwd)

    x265_args = [
        str(paths.x265_bin),
        f"--reader-options=library={paths.vs_env.vsscript_dll}",
    ]

    if preset is not None:
        x265_args += ["--preset", preset]

    x265_args += ["--output", str(abs_hevc_path)]
    x265_args += x265_params
    x265_args += ["--input", str(abs_vpy_path)]

    return x265_args


def run_x265_encode(
    x265_args: list[str],
    vs_env: dict[str, str],
    cwd: Path | None,
    line_handler: Callable[[str], bool] | None,
    run_capture_fn: Callable[..., str],
) -> str:
    """Run x265 encoding process.

    Args:
        x265_args: Command arguments for x265
        vs_env: Environment with VapourSynth paths
        cwd: Working directory
        line_handler: Optional callback for progress
        run_capture_fn: Function to run subprocess (for dependency injection)

    Returns:
        Process output string

    Raises:
        Exception: If encoding fails
    """
    log = logging.getLogger(__name__)
    log.info("x265 cmd: %s", " ".join(shlex.quote(str(c)) for c in x265_args))

    return run_capture_fn(
        x265_args,
        cwd=cwd,
        env=vs_env,
        line_callback=line_handler,
    )


def mux_and_cleanup(
    hevc_path: Path,
    output_path: Path,
    vpy_path: Path,
    perform_mux: bool,
    mux_fn: Callable[..., None],
    mkvmerge_bin: str,
    cwd: Path | None,
    mux_handler: Callable[[str], bool] | None,
) -> Path:
    """Mux HEVC to MKV and cleanup temporary files.

    Args:
        hevc_path: Path to HEVC bitstream
        output_path: Output MKV path
        vpy_path: VapourSynth script path to cleanup
        perform_mux: Whether to mux to MKV
        mux_fn: Function to perform muxing
        mkvmerge_bin: Path to mkvmerge binary
        cwd: Working directory
        mux_handler: Optional callback for mux progress

    Returns:
        Final output path (MKV if muxed, HEVC if not)
    """
    log = logging.getLogger(__name__)

    try:
        if perform_mux:
            log.info("Muxing HEVC -> MKV")
            mux_fn(
                hevc_path=hevc_path,
                output_path=output_path,
                mkvmerge_bin=mkvmerge_bin,
                cwd=cwd,
                line_handler=mux_handler,
            )
            return output_path
        else:
            return hevc_path
    finally:
        if perform_mux and hevc_path.exists():
            hevc_path.unlink()
        if vpy_path.exists():
            vpy_path.unlink()
