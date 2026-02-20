"""Shared encoding utilities for VideoTuner."""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from io import TextIOWrapper
from pathlib import Path
from typing import IO, Callable

from .encoder_type import EncoderType

# Directory name for bundled VapourSynth portable installation
VAPOURSYNTH_PORTABLE_DIR = "vapoursynth-portable"

# Relative paths to bundled encoder binaries
X265_BIN_PATH = Path("tools") / "x265.exe"
X264_BIN_PATH = Path("tools") / "x264.exe"

# Map encoder types to their binary paths
ENCODER_BIN_PATHS: dict[EncoderType, Path] = {
    EncoderType.X265: X265_BIN_PATH,
    EncoderType.X264: X264_BIN_PATH,
}


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


def get_encoder_bin(encoder_type: EncoderType, cwd: Path | None = None) -> Path:
    """Get path to encoder binary.

    Args:
        encoder_type: Which encoder to locate
        cwd: Working directory (if None, uses relative path)

    Returns:
        Path to encoder binary
    """
    bin_path = ENCODER_BIN_PATHS[encoder_type]
    if cwd:
        return Path(cwd) / bin_path
    return bin_path


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
    encoder_type: EncoderType,
    temp_dir: Path | None = None,
    name: str = "encode",
) -> tuple[Path, Path]:
    """Create temporary VPY and bitstream file paths.

    Args:
        encoder_type: Encoder type (determines bitstream file extension)
        temp_dir: Directory for temporary files (None for system temp)
        name: Base filename (without extension)

    Returns:
        Tuple of (vpy_path, bitstream_path)
    """
    ext = encoder_type.bitstream_extension

    if temp_dir:
        from .utils import ensure_dir

        _ = ensure_dir(temp_dir)
        vpy_path = temp_dir / f"{name}.vpy"
        bitstream_path = temp_dir / f"{name}{ext}"
    else:
        with tempfile.NamedTemporaryFile(
            suffix=".vpy", prefix=f"{name}_", delete=False, mode="w", encoding="utf-8"
        ) as tmp:
            vpy_path = Path(tmp.name)
        with tempfile.NamedTemporaryFile(
            suffix=ext, prefix=f"{name}_", delete=False
        ) as tmp:
            bitstream_path = Path(tmp.name)

    return vpy_path, bitstream_path


@dataclass(frozen=True)
class VapourSynthEnv:
    """Unified VapourSynth environment for encoding and assessment tools.

    VapourSynth is mandatory for both x265 encoding and SSIMULACRA2 assessment.
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
    vspipe_bin: Path

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
            vspipe_bin=vs_dir / "vspipe.exe",
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
        vs_dir = (
            vs_dir_arg
            if vs_dir_arg is not None
            else repo_root / VAPOURSYNTH_PORTABLE_DIR
        )
        vs_plugin_dir = (
            vs_plugin_dir_arg
            if vs_plugin_dir_arg is not None
            else vs_dir / "vs-plugins"
        )
        return cls(
            vs_dir=vs_dir,
            vs_plugin_dir=vs_plugin_dir,
            vsscript_dll=vs_dir / "VSScript.dll",
            ffms2_dll=vs_plugin_dir / "ffms2.dll",
            vspipe_bin=vs_dir / "vspipe.exe",
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
        if not self.vspipe_bin.exists():
            raise FileNotFoundError(f"vspipe not found at: {self.vspipe_bin}")

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

        # Prepend VapourSynth directories to PATH so DLLs are found first
        # Include both vs_dir (for vapoursynth.dll, VSScript.dll) and
        # vs_plugin_dir (for ffms2.dll and other plugins)
        path_additions = str(self.vs_dir) + os.pathsep + str(self.vs_plugin_dir)
        env["PATH"] = path_additions + os.pathsep + env.get("PATH", "")

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


@dataclass(frozen=True)
class EncoderPaths:
    """Resolved paths to all encoding tools (encoder binary + VapourSynth).

    Usage:
        paths = EncoderPaths.from_cwd(cwd, encoder_type)
        paths.validate()  # Raises FileNotFoundError if any tool missing
        env = paths.vs_env.build_env()
    """

    encoder_type: EncoderType
    encoder_bin: Path
    vs_env: VapourSynthEnv

    @classmethod
    def from_cwd(cls, cwd: Path | None, encoder_type: EncoderType) -> "EncoderPaths":
        """Resolve all encoder paths from working directory.

        Args:
            cwd: Working directory. If None, uses relative paths.
            encoder_type: Which encoder to resolve paths for.

        Returns:
            EncoderPaths with encoder binary and VapourSynth paths resolved.
        """
        encoder_bin = get_encoder_bin(encoder_type, cwd)
        return cls(
            encoder_type=encoder_type,
            encoder_bin=encoder_bin,
            vs_env=VapourSynthEnv.from_cwd(cwd),
        )

    def validate(self) -> None:
        """Validate that all required encoder files exist.

        Raises:
            FileNotFoundError: If encoder binary, VSScript.dll, or ffms2.dll is missing.
        """
        if not self.encoder_bin.exists():
            raise FileNotFoundError(
                f"{self.encoder_type.value} encoder not found at: {self.encoder_bin}"
            )
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

    # Apply crop if provided
    if crop_values is not None:
        if (
            crop_values.left > 0
            or crop_values.right > 0
            or crop_values.top > 0
            or crop_values.bottom > 0
        ):
            vpy_lines.append("")
            vpy_lines.append("# Apply crop to sampled frames")
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
    output_path: Path,
    encoder_params: list[str],
    preset: str | None,
    cwd: Path | None = None,
) -> list[str]:
    """Build x265 command line arguments for piped y4m input.

    Args:
        paths: Resolved encoder paths
        output_path: Output bitstream path
        encoder_params: Additional x265 parameters
        preset: x265 preset (or None)
        cwd: Working directory for path resolution

    Returns:
        List of command arguments for x265
    """
    abs_output_path = resolve_absolute_path(output_path, cwd)

    args = [str(paths.encoder_bin)]

    if preset is not None:
        args += ["--preset", preset]

    args += ["--output", str(abs_output_path)]
    args += encoder_params
    args += ["--y4m", "--input", "-"]

    return args


def build_x264_command(
    paths: EncoderPaths,
    output_path: Path,
    encoder_params: list[str],
    preset: str | None,
    cwd: Path | None = None,
) -> list[str]:
    """Build x264 command line arguments for piped y4m input.

    Args:
        paths: Resolved encoder paths
        output_path: Output bitstream path
        encoder_params: Additional x264 parameters
        preset: x264 preset (or None)
        cwd: Working directory for path resolution

    Returns:
        List of command arguments for x264
    """
    abs_output_path = resolve_absolute_path(output_path, cwd)

    args = [str(paths.encoder_bin)]

    if preset is not None:
        args += ["--preset", preset]

    args += encoder_params
    args += ["--demuxer", "y4m", "--output", str(abs_output_path), "-"]

    return args


def build_encoder_command(
    paths: EncoderPaths,
    output_path: Path,
    encoder_params: list[str],
    preset: str | None,
    cwd: Path | None = None,
) -> list[str]:
    """Build encoder command line arguments based on encoder type.

    Dispatches to the appropriate command builder for x264 or x265.

    Args:
        paths: Resolved encoder paths (contains encoder_type)
        output_path: Output bitstream path
        encoder_params: Additional encoder parameters
        preset: Encoder preset (or None)
        cwd: Working directory for path resolution

    Returns:
        List of command arguments for the encoder
    """
    if paths.encoder_type == EncoderType.X264:
        return build_x264_command(paths, output_path, encoder_params, preset, cwd)
    return build_x265_command(paths, output_path, encoder_params, preset, cwd)


def build_vspipe_command(
    vs_env: VapourSynthEnv,
    vpy_path: Path,
    cwd: Path | None = None,
) -> list[str]:
    """Build vspipe command to output y4m to stdout.

    Args:
        vs_env: VapourSynth environment with vspipe path
        vpy_path: Path to VapourSynth script
        cwd: Working directory for path resolution

    Returns:
        List of command arguments for vspipe
    """
    abs_vpy_path = resolve_absolute_path(vpy_path, cwd)
    return [str(vs_env.vspipe_bin), "-c", "y4m", str(abs_vpy_path), "-"]


def run_vspipe_encode(
    vspipe_args: list[str],
    encoder_args: list[str],
    vs_env: dict[str, str],
    cwd: Path | None,
    line_handler: Callable[[str], bool] | None,
) -> str:
    """Run vspipe | encoder encoding pipeline.

    Pipes vspipe y4m output to encoder stdin, avoiding direct VapourSynth
    readout which has thread-safety bugs in the x265 VPY reader.

    Args:
        vspipe_args: Command arguments for vspipe
        encoder_args: Command arguments for encoder (x264 or x265)
        vs_env: Environment with VapourSynth paths
        cwd: Working directory
        line_handler: Optional callback for progress lines

    Returns:
        Combined process output string

    Raises:
        RuntimeError: If either process fails
    """
    from .utils import format_command_error, iter_stream_output

    log = logging.getLogger(__name__)
    log.info("vspipe cmd: %s", " ".join(shlex.quote(str(c)) for c in vspipe_args))
    log.info("encoder cmd: %s", " ".join(shlex.quote(str(c)) for c in encoder_args))

    cwd_str = str(cwd) if cwd else None
    captured: list[str] = []
    vspipe_errors: list[str] = []

    try:
        vspipe_proc = subprocess.Popen(
            vspipe_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd_str,
            env=vs_env,
        )
    except FileNotFoundError as e:
        raise RuntimeError(f"Command not found: {vspipe_args[0]}") from e

    try:
        encoder_proc = subprocess.Popen(
            encoder_args,
            stdin=vspipe_proc.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd_str,
            env=vs_env,
        )
    except FileNotFoundError as e:
        vspipe_proc.kill()
        _ = vspipe_proc.wait()
        raise RuntimeError(f"Command not found: {encoder_args[0]}") from e

    # Close vspipe stdout in parent so encoder gets EOF when vspipe finishes
    assert vspipe_proc.stdout is not None
    vspipe_proc.stdout.close()

    # Read encoder stderr for progress and vspipe stderr for errors
    assert encoder_proc.stdout is not None
    assert encoder_proc.stderr is not None
    assert vspipe_proc.stderr is not None

    def _read_encoder_stream(stream: IO[bytes]) -> None:
        wrapper = TextIOWrapper(stream, encoding="utf-8", errors="replace")
        try:
            for line in iter_stream_output(wrapper):
                captured.append(line)
                if line and line_handler is not None:
                    try:
                        _ = line_handler(line)
                    except Exception:
                        log.debug("Progress callback error", exc_info=True)
        finally:
            _ = wrapper.detach()

    def _read_vspipe_stderr(stream: IO[bytes]) -> None:
        wrapper = TextIOWrapper(stream, encoding="utf-8", errors="replace")
        try:
            for line in iter_stream_output(wrapper):
                if line:
                    vspipe_errors.append(line)
                    log.debug("vspipe: %s", line)
        finally:
            _ = wrapper.detach()

    threads = [
        threading.Thread(target=_read_encoder_stream, args=(encoder_proc.stdout,)),
        threading.Thread(target=_read_encoder_stream, args=(encoder_proc.stderr,)),
        threading.Thread(target=_read_vspipe_stderr, args=(vspipe_proc.stderr,)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    encoder_ret = encoder_proc.wait()
    vspipe_ret = vspipe_proc.wait()

    if encoder_ret != 0 or vspipe_ret != 0:
        encoder_output = "\n".join(captured) if captured else ""
        vspipe_detail = "\n".join(vspipe_errors) if vspipe_errors else ""

        # Prioritize encoder error: when the encoder dies, the pipe breaks,
        # causing vspipe to fail with fwrite(). The encoder error is the root cause.
        if encoder_ret != 0:
            error_msg = format_command_error(encoder_ret, encoder_args, encoder_output)
            if vspipe_ret != 0 and vspipe_detail:
                error_msg += f"\n\nvspipe also failed ({vspipe_ret}):\n{vspipe_detail}"
            raise RuntimeError(error_msg)

        # vspipe-only failure (e.g., VapourSynth script error)
        raise RuntimeError(format_command_error(vspipe_ret, vspipe_args, vspipe_detail))

    return "\n".join(captured)


def mux_and_cleanup(
    bitstream_path: Path,
    output_path: Path,
    vpy_path: Path,
    perform_mux: bool,
    mux_fn: Callable[..., None],
    mkvmerge_bin: str,
    cwd: Path | None,
    mux_handler: Callable[[str], bool] | None,
) -> Path:
    """Mux bitstream to MKV and cleanup temporary files.

    Args:
        bitstream_path: Path to raw bitstream (HEVC or H.264)
        output_path: Output MKV path
        vpy_path: VapourSynth script path to cleanup
        perform_mux: Whether to mux to MKV
        mux_fn: Function to perform muxing
        mkvmerge_bin: Path to mkvmerge binary
        cwd: Working directory
        mux_handler: Optional callback for mux progress

    Returns:
        Final output path (MKV if muxed, bitstream if not)
    """
    log = logging.getLogger(__name__)

    try:
        if perform_mux:
            log.info("Muxing bitstream -> MKV")
            mux_fn(
                bitstream_path=bitstream_path,
                output_path=output_path,
                mkvmerge_bin=mkvmerge_bin,
                cwd=cwd,
                line_handler=mux_handler,
            )
            return output_path
        else:
            return bitstream_path
    finally:
        if perform_mux and bitstream_path.exists():
            bitstream_path.unlink()
        if vpy_path.exists():
            vpy_path.unlink()
