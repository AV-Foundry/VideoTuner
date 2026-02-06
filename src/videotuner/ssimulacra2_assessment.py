from __future__ import annotations

import json
import logging
import statistics
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .utils import ensure_dir, run_capture
from .media import get_assessment_frame_count
from .encoding_utils import write_vpy_script, VapourSynthEnv

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


def _calculate_stats_from_scores(scores: list[float]) -> SSIM2Result:
    """Calculate summary statistics from a list of per-frame scores.

    Args:
        scores: List of SSIMULACRA2 scores (one per frame)

    Returns:
        SSIM2Result with calculated statistics

    Raises:
        RuntimeError: If scores list is empty
    """
    if not scores:
        raise RuntimeError("No scores provided for statistics calculation")

    scores_sorted = sorted(scores)
    count = len(scores)
    mean = statistics.mean(scores)
    median = statistics.median(scores)
    std_dev = statistics.stdev(scores) if count > 1 else 0.0

    # Calculate percentiles (5th and 95th)
    p5_idx = max(0, int(count * 0.05) - 1)
    p95_idx = min(count - 1, int(count * 0.95))
    p5_low = scores_sorted[p5_idx]
    p95_high = scores_sorted[p95_idx]

    return SSIM2Result(
        mean=mean,
        median=median,
        p5_low=p5_low,
        p95_high=p95_high,
        std_dev=std_dev,
        count=count,
    )


def _run_vszip_assessment(
    *,
    ref_path: Path,
    dis_path: Path,
    vs_env: VapourSynthEnv,
    temp_dir: Path,
    line_handler: Callable[[str], bool] | None = None,
) -> SSIM2Result:
    """Run SSIMULACRA2 assessment via vszip VapourSynth plugin.

    Creates a VapourSynth script that uses vszip.SSIMULACRA2 to compute
    per-frame scores, then calculates summary statistics.

    Args:
        ref_path: Reference video path
        dis_path: Distorted video path
        vs_env: VapourSynth environment configuration
        temp_dir: Directory for temporary files
        line_handler: Optional callback for progress updates

    Returns:
        SSIM2Result with calculated metrics

    Raises:
        RuntimeError: If vszip execution fails or produces no scores
    """
    log = logging.getLogger(__name__)

    python_exe = vs_env.vs_dir / "python.exe"
    if not python_exe.exists():
        raise FileNotFoundError(f"VapourSynth python.exe not found at: {python_exe}")

    # Create temporary script
    vpy_path = temp_dir / "vszip_ssim2.vpy"

    # Build VapourSynth script that calculates SSIMULACRA2 and prints scores
    # Using lsmas for video loading (same as our encoding pipeline)
    ref_cache = temp_dir / "vszip_ref.lwi"
    dis_cache = temp_dir / "vszip_dis.lwi"

    vpy_content = f'''import vapoursynth as vs
import sys

core = vs.core

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

try:
    print("[vszip] Loading reference video...", file=sys.stderr, flush=True)
    # Load reference and distorted videos
    ref = core.lsmas.LWLibavSource(
        source=r"{ref_path.resolve()}",
        cachefile=r"{ref_cache}"
    )
    print(f"[vszip] Reference: {{ref.width}}x{{ref.height}}, {{ref.format.name}}, {{len(ref)}} frames", file=sys.stderr, flush=True)

    print("[vszip] Loading distorted video...", file=sys.stderr, flush=True)
    dis = core.lsmas.LWLibavSource(
        source=r"{dis_path.resolve()}",
        cachefile=r"{dis_cache}"
    )
    print(f"[vszip] Distorted: {{dis.width}}x{{dis.height}}, {{dis.format.name}}, {{len(dis)}} frames", file=sys.stderr, flush=True)

    # Validate dimensions match
    if ref.width != dis.width or ref.height != dis.height:
        print(f"ERROR: Resolution mismatch - ref={{ref.width}}x{{ref.height}}, dis={{dis.width}}x{{dis.height}}", file=sys.stderr, flush=True)
        sys.exit(1)

    # Check if vszip plugin is available
    if not hasattr(core, 'vszip'):
        print("ERROR: vszip plugin not loaded", file=sys.stderr, flush=True)
        sys.exit(1)

    print("[vszip] Running SSIMULACRA2 comparison...", file=sys.stderr, flush=True)
    # vszip.Metrics handles format conversion internally
    # mode=0 selects SSIMULACRA2 metric
    result = ref.vszip.Metrics(dis, mode=0)

    total_frames = len(result)
    print(f"[vszip] Processing {{total_frames}} frames", file=sys.stderr, flush=True)

    if total_frames == 0:
        print("ERROR: SSIMULACRA2 returned empty result", file=sys.stderr, flush=True)
        sys.exit(1)

    # Iterate frames and collect scores
    for i in range(total_frames):
        frame = result.get_frame(i)
        score = frame.props.get("_SSIMULACRA2", None)
        if score is not None:
            # Print score to stdout for parsing
            print(f"SCORE:{{score}}", flush=True)
        else:
            print(f"WARNING: Frame {{i}} has no _SSIMULACRA2 property", file=sys.stderr, flush=True)
        # Progress to stderr for line_handler
        print(f"vszip progress: {{i + 1}}/{{total_frames}}", file=sys.stderr, flush=True)

    print("[vszip] Assessment complete", file=sys.stderr, flush=True)

except Exception as e:
    print(f"ERROR: {{e}}", file=sys.stderr, flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''

    write_vpy_script(vpy_path, vpy_content)
    log.debug("vszip script written to: %s", vpy_path)

    try:
        # Setup environment for VapourSynth portable
        env = vs_env.build_env()

        # Run VapourSynth script
        output = run_capture(
            [str(python_exe), str(vpy_path)],
            cwd=temp_dir,
            env=env,
            line_callback=line_handler,
        )

        # Parse scores from output (lines starting with "SCORE:")
        scores: list[float] = []
        for line in output.splitlines():
            line = line.strip()
            if line.startswith("SCORE:"):
                with suppress(ValueError):
                    scores.append(float(line[6:]))

        if not scores:
            # Include captured output in error for debugging
            log.error("vszip output:\n%s", output)
            raise RuntimeError("vszip produced no valid SSIMULACRA2 scores")

        return _calculate_stats_from_scores(scores)

    finally:
        # Clean up temporary files
        if vpy_path.exists():
            vpy_path.unlink()
        # Clean up cache files
        if ref_cache.exists():
            ref_cache.unlink()
        if dis_cache.exists():
            dis_cache.unlink()


def assess_ssim2_concatenated(
    reference_path: Path,
    distorted_path: Path,
    workdir: Path,
    temp_dir: Path,
    profile: Profile,
    vs_env: VapourSynthEnv,
    display: PipelineDisplay,
    log: logging.Logger,
    iteration: int,
) -> list[SSIM2Result]:
    """Run SSIMULACRA2 assessment on concatenated reference and distorted files.

    Uses vszip VapourSynth plugin for CPU-based SSIMULACRA2 calculation.

    Args:
        reference_path: Path to concatenated lossless reference
        distorted_path: Path to concatenated distorted encode
        workdir: Working directory for output files
        temp_dir: Temporary directory for intermediate files
        profile: Encoding profile (used for output directory naming)
        vs_env: VapourSynth environment configuration
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
        result = _run_vszip_assessment(
            ref_path=reference_path,
            dis_path=distorted_path,
            vs_env=vs_env,
            temp_dir=temp_dir,
            line_handler=stage.make_vszip_handler(total_frames=total_frames),
        )

    # Write JSON log
    try:
        _ = ensure_dir(ssim2_log_path.parent)
        with open(ssim2_log_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "tool": "vszip",
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
    except Exception:
        pass  # Log writing is optional

    log.info(
        "SSIMULACRA2: mean=%.2f, median=%.2f, 5%%=%.2f, 95%%=%.2f",
        result.mean,
        result.median,
        result.p5_low,
        result.p95_high,
    )
    return [result]
