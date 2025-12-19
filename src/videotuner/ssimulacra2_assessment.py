from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .utils import ensure_dir, run_capture
from .media import get_assessment_frame_count
from .encoding_utils import write_vpy_script
from .tool_parsers import FLOAT_PATTERN

if TYPE_CHECKING:
    from .profiles import Profile
    from .progress import PipelineDisplay


@dataclass(frozen=True)
class SSIM2Result:
    mean: float
    median: float
    p5_low: float
    p95_high: float
    std_dev: float
    count: int


def _parse_video_summary(text: str) -> SSIM2Result | None:
    mean = median = p5 = p95 = std_dev = None
    count = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.search(r"Video Score for\s+(\d+)\s+frames", line)
        if m:
            try:
                count = int(m.group(1))
            except Exception:
                pass
        low = line.lower()
        match = re.search(rf"mean\s*:\s*({FLOAT_PATTERN})", low, re.IGNORECASE)
        if match:
            mean = float(match.group(1))
            continue
        match = re.search(rf"median\s*:\s*({FLOAT_PATTERN})", low, re.IGNORECASE)
        if match:
            median = float(match.group(1))
            continue
        if low.startswith("5th percentile"):
            match = re.search(
                rf"5th percentile\s*:\s*({FLOAT_PATTERN})", low, re.IGNORECASE
            )
            if match:
                p5 = float(match.group(1))
            continue
        if low.startswith("95th percentile"):
            match = re.search(
                rf"95th percentile\s*:\s*({FLOAT_PATTERN})", low, re.IGNORECASE
            )
            if match:
                p95 = float(match.group(1))
            continue
        if low.startswith("std dev"):
            match = re.search(
                rf"std dev\s*:\s*({FLOAT_PATTERN})", low, re.IGNORECASE
            )
            if match:
                std_dev = float(match.group(1))
            continue
    if mean is None or median is None or p5 is None:
        return None
    if count is None:
        count = 0
    if p95 is None:
        p95 = float("nan")
    if std_dev is None:
        std_dev = float("nan")
    return SSIM2Result(
        mean=mean, median=median, p5_low=p5, p95_high=p95, std_dev=std_dev, count=count
    )


def assess_with_ssimulacra2_video(
    *,
    ssim2_bin: str,
    ref_path: Path,
    dis_path: Path,
    log_path: Path | None = None,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    line_handler: Callable[[str], bool] | None = None,
    _vs_plugin_dir: Path | None = None,
) -> SSIM2Result:
    """Run SSIMULACRA2 in 'video' mode directly on two inputs (MKV or .vpy).

    When line_handler is provided, uses --verbose flag to get per-frame scores
    and calls the handler for each frame line. Otherwise, lets native progress bar
    display to terminal.

    Args:
        ssim2_bin: Path to ssimulacra2 binary
        ref_path: Reference video path
        dis_path: Distorted video path
        log_path: Optional path to write JSON log
        cwd: Working directory for subprocess. If set, VapourSynth scripts will be
             generated to control index file locations.
        env: Environment variables for subprocess
        line_handler: Optional callback for progress updates. If provided, --verbose is used
                      and stderr is suppressed. Handler receives each line and returns True
                      if the line was consumed.
        vs_plugin_dir: Optional VapourSynth plugin directory path. Required when cwd is set
                       to explicitly load lsmas plugin in generated scripts.
    """
    log = logging.getLogger(__name__)
    import subprocess

    def _format_failure(code: int, stdout_data: str, stderr_data: str) -> str:
        parts = [f"SSIMULACRA2 video mode failed with exit code {code}."]
        stderr_data = (stderr_data or "").strip()
        stdout_data = (stdout_data or "").strip()
        if stderr_data:
            parts.append("stderr:\n" + stderr_data)
        if stdout_data:
            parts.append("stdout:\n" + stdout_data)
        if len(parts) == 1:
            parts.append("No output was captured.")
        return "\n".join(parts)

    stdout_capture = ""
    stderr_capture = ""

    # Generate VapourSynth scripts when cwd is set to control index file locations
    # This avoids Windows MAX_PATH issues with lsmas default encoded filenames
    cleanup_vpy_scripts = False
    if cwd is not None:
        cwd_path = ensure_dir(Path(cwd))

        # Generate .vpy scripts with explicit cachefile paths (short names in temp dir)
        ref_vpy = cwd_path / "ssim_ref.vpy"
        dis_vpy = cwd_path / "ssim_dis.vpy"

        # Create VapourSynth script for reference video
        # Note: lsmas plugin is auto-loaded via VAPOURSYNTH_PLUGIN_PATH env var
        ref_script = (
            f"from vapoursynth import core\n"
            f"clip = core.lsmas.LWLibavSource(\n"
            f"    source={str(ref_path.resolve())!r},\n"
            f"    cachefile={str(cwd_path / 'ssim_ref_index.lwi')!r}\n"
            f")\n"
            f"clip.set_output()\n"
        )
        write_vpy_script(ref_vpy, ref_script)

        # Create VapourSynth script for distorted video
        dis_script = (
            f"from vapoursynth import core\n"
            f"clip = core.lsmas.LWLibavSource(\n"
            f"    source={str(dis_path.resolve())!r},\n"
            f"    cachefile={str(cwd_path / 'ssim_dis_index.lwi')!r}\n"
            f")\n"
            f"clip.set_output()\n"
        )
        write_vpy_script(dis_vpy, dis_script)

        ref_arg = ref_vpy.name
        dis_arg = dis_vpy.name
        cleanup_vpy_scripts = True
    else:
        ref_arg = str(ref_path)
        dis_arg = str(dis_path)

    # Build command with --verbose if using line handler
    cmd = [ssim2_bin, "video"]
    if line_handler is not None:
        cmd.append("--verbose")
    cmd.extend([ref_arg, dis_arg])

    try:
        if line_handler is not None:
            # Use run_capture for line-by-line processing with callback
            # This handles threading for stdout/stderr and returns captured output
            out = run_capture(
                cmd,
                cwd=cwd,
                env=env,
                line_callback=line_handler,
            )
        else:
            # Use subprocess.run directly for simple capture without callback
            proc = subprocess.run(
                cmd,
                cwd=str(cwd) if cwd else None,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout_capture = proc.stdout or ""
            stderr_capture = proc.stderr or ""
            if proc.returncode != 0:
                raise RuntimeError(
                    _format_failure(proc.returncode, stdout_capture, stderr_capture)
                )
            out = stdout_capture

    except RuntimeError:
        # Re-raise RuntimeError (from run_capture or _format_failure) as-is
        raise
    except Exception as e:
        log.error("SSIMULACRA2 video call failed: %s", e)
        raise
    finally:
        # Cleanup generated VapourSynth scripts if they were created
        if cleanup_vpy_scripts and cwd is not None:
            try:
                (Path(cwd) / "ssim_ref.vpy").unlink(missing_ok=True)
                (Path(cwd) / "ssim_dis.vpy").unlink(missing_ok=True)
            except Exception:
                pass  # Ignore cleanup errors

    # Parse the summary from output
    result = _parse_video_summary(out)
    if result is None:
        raise RuntimeError("SSIMULACRA2 video mode produced no parseable scores")

    if log_path is not None:
        try:
            _ = ensure_dir(log_path.parent)
        except Exception:
            pass
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "tool": "ssimulacra2",
                    "mode": "video",
                    "count": result.count,
                    "mean": result.mean,
                    "median": result.median,
                    "p5_low": result.p5_low,
                    "p95_high": result.p95_high,
                    "std_dev": result.std_dev,
                },
                f,
                indent=2,
            )
    return result


def assess_ssim2_concatenated(
    reference_path: Path,
    distorted_path: Path,
    workdir: Path,
    temp_dir: Path,
    profile: Profile,
    ssim2_bin: str,
    vs_env: dict[str, str],
    vs_plugin_dir: Path | None,
    display: PipelineDisplay,
    log: logging.Logger,
    iteration: int,
) -> list[SSIM2Result]:
    """Run SSIMULACRA2 assessment on concatenated reference and distorted files.

    Args:
        reference_path: Path to concatenated lossless reference
        distorted_path: Path to concatenated distorted encode
        workdir: Working directory for output files
        temp_dir: Temporary directory for intermediate files
        profile: Encoding profile (used for output directory naming)
        ssim2_bin: Path to ssimulacra2_rs binary
        vs_env: Environment variables for VapourSynth
        vs_plugin_dir: VapourSynth plugin directory
        display: Progress display manager
        log: Logger instance
        iteration: Current iteration number

    Returns:
        List containing single SSIM2Result for the concatenated comparison
    """
    from .pipeline_types import get_ssim2_dir

    ssim2_log_path = (
        get_ssim2_dir(workdir, profile) / f"ssim2_concatenated_iter{iteration}.json"
    )

    # Get frame count for progress tracking
    total_frames = get_assessment_frame_count(reference_path)

    with display.stage(
        "Running SSIMULACRA2 assessment",
        total=total_frames,
        show_eta=True,
        transient=True,
        show_done=True,
    ) as stage:
        result = assess_with_ssimulacra2_video(
            ssim2_bin=ssim2_bin,
            ref_path=reference_path,
            dis_path=distorted_path,
            log_path=ssim2_log_path,
            cwd=temp_dir,
            env=vs_env,
            line_handler=stage.make_ssim_verbose_handler(total_frames=total_frames),
            _vs_plugin_dir=vs_plugin_dir,
        )

    log.info(
        "SSIMULACRA2: mean=%.2f, median=%.2f, 5%%=%.2f, 95%%=%.2f",
        result.mean,
        result.median,
        result.p5_low,
        result.p95_high,
    )
    return [result]
