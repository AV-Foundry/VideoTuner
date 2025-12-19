import logging
import math
import os
import shlex
import subprocess
import sys
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Callable, TextIO

from .constants import LOG_SEPARATOR_WIDTH, LOG_SEPARATOR_CHAR

logger = logging.getLogger(__name__)

LineCallback = Callable[[str], bool] | None


def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it doesn't exist, then return it.

    Args:
        path: Directory path to create

    Returns:
        The same path, for chaining
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def _is_nuitka_compiled() -> bool:
    """Check if running as Nuitka-compiled executable."""
    # Nuitka sets __compiled__ = True in every compiled module
    # Check via globals() to avoid NameError and type checker complaints
    return "__compiled__" in globals()


def get_app_root() -> Path:
    """Get application root directory - works in dev, PyInstaller, and Nuitka.

    In development: Returns the repo root (3 levels up from this file).
    When bundled: Returns the directory containing the executable.

    This allows bundled tools (x265, vapoursynth-portable) and config files
    (x265_profiles.yaml) to be found relative to the application.

    Returns:
        Path to the application root directory.
    """
    if getattr(sys, "frozen", False):
        # PyInstaller: sys.executable is the bundled exe
        return Path(sys.executable).parent
    elif _is_nuitka_compiled():
        # Nuitka: use sys.argv[0] which points to original exe location
        # (sys.executable may point to temp extraction folder in onefile mode)
        return Path(sys.argv[0]).resolve().parent
    else:
        # Development: src/videotuner/utils.py -> repo root
        return Path(__file__).resolve().parents[2]


def clean_float(val: float | None) -> float | None:
    """Clean a float value by rounding to 2 decimal places and handling NaN.

    Args:
        val: Float value to clean, or None

    Returns:
        Rounded float value, or None if input is None or NaN
    """
    if val is None:
        return None
    if math.isnan(val):
        return None
    return round(val, 2)


def make_relative_path(path: Path, cwd: Path | None) -> str:
    """Convert path to relative path if cwd is set, otherwise return absolute string.

    Args:
        path: Path to convert
        cwd: Working directory for relative path, or None for absolute

    Returns:
        Relative path string if cwd is set, otherwise absolute path string
    """
    return os.path.relpath(path, cwd) if cwd else str(path)


def format_command_error(returncode: int, cmd: list[str], output: str = "") -> str:
    """Format a consistent error message for failed subprocess commands.

    Args:
        returncode: Process return code
        cmd: Command and arguments that failed
        output: Optional stdout/stderr output

    Returns:
        Formatted error message string
    """
    msg = f"Command failed ({returncode}): {' '.join(shlex.quote(c) for c in cmd)}"
    if output:
        msg += f"\n\n{output}"
    return msg


def log_section(log: logging.Logger, title: str) -> None:
    """Log a visual section separator with a title."""
    separator = LOG_SEPARATOR_CHAR * LOG_SEPARATOR_WIDTH
    log.info("")
    log.info(separator)
    log.info(" %s", title.upper())
    log.info(separator)


def log_separator(log: logging.Logger, level: int = logging.INFO) -> None:
    """Log a visual separator line at the specified level.

    Use this for wrapping content blocks. For titled sections, use log_section instead.

    Args:
        log: Logger instance to use
        level: Logging level (default: logging.INFO)
    """
    separator = LOG_SEPARATOR_CHAR * LOG_SEPARATOR_WIDTH
    log.log(level, separator)


def iter_stream_output(stream: TextIO) -> Iterable[str]:
    """Iterate over lines from a stream, handling carriage returns for progress updates.

    Reads character-by-character to handle \r (carriage return) which is used by
    x265 and other tools for live progress updates.

    This function is public and can be reused by other modules that need to
    process streaming output with carriage returns.
    """
    buffer: list[str] = []
    while True:
        chunk = stream.read(1)
        if chunk == "":
            if buffer:
                yield "".join(buffer).replace("\x1b[K", "").strip()
            break
        if chunk in ("\r", "\n"):
            if buffer:
                yield "".join(buffer).replace("\x1b[K", "").strip()
                buffer.clear()
            continue
        buffer.append(chunk)


def _emit(line: str, callback: LineCallback, forward: TextIO | None) -> None:
    if not line:
        return
    consumed = False
    if callback is not None:
        try:
            consumed = bool(callback(line))
        except Exception:
            consumed = False
    if not consumed and forward is not None:
        _ = forward.write(line + "\n")
        forward.flush()


def run(
    cmd: list[str],
    live: bool = False,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    line_callback: LineCallback = None,
) -> None:
    """Execute a command and optionally stream output.

    Args:
        cmd: Command and arguments to execute
        live: If True, stream output line-by-line (processing both stdout and stderr)
        cwd: Working directory for the process
        env: Environment variables
        line_callback: Optional callback for each output line. If provided and returns True,
                      the line is considered consumed and won't be forwarded to stdout.
    """
    if live:
        try:
            with subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,  # Capture stderr separately
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                cwd=str(cwd) if cwd else None,
                env=env,
            ) as p:
                assert p.stdout is not None
                assert p.stderr is not None

                forward = None if line_callback is not None else sys.stdout

                # Process both stdout and stderr concurrently using threads
                # x265 writes progress to stderr, so we need to monitor both streams
                def _process_stream(stream: TextIO, _name: str) -> None:
                    for line in iter_stream_output(stream):
                        _emit(line, line_callback, forward)

                stdout_thread = threading.Thread(
                    target=_process_stream, args=(p.stdout, "stdout")
                )
                stderr_thread = threading.Thread(
                    target=_process_stream, args=(p.stderr, "stderr")
                )

                stdout_thread.start()
                stderr_thread.start()

                stdout_thread.join()
                stderr_thread.join()

                ret = p.wait()
        except FileNotFoundError as e:
            raise RuntimeError(f"Command not found: {cmd[0]}") from e
        if ret != 0:
            raise RuntimeError(format_command_error(ret, cmd))
        return
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(cwd) if cwd else None,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(format_command_error(proc.returncode, cmd, proc.stdout))


def run_capture(
    cmd: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    line_callback: LineCallback = None,
) -> str:
    """Execute a command and capture output, optionally calling a callback for each line.

    Args:
        cmd: Command and arguments to execute
        cwd: Working directory for the process
        env: Environment variables
        line_callback: Optional callback for each output line during execution

    Returns:
        Combined stdout and stderr output as a string
    """
    if line_callback is None:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(cwd) if cwd else None,
            env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError(format_command_error(proc.returncode, cmd, proc.stdout))
        return proc.stdout

    # When callback is provided, process both stdout and stderr concurrently
    captured: list[str] = []
    try:
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,  # Capture stderr separately for x265 progress
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            cwd=str(cwd) if cwd else None,
            env=env,
        ) as p:
            assert p.stdout is not None
            assert p.stderr is not None

            # Process both streams concurrently
            def _process_stream(stream: TextIO, _name: str) -> None:
                for line in iter_stream_output(stream):
                    captured.append(line)
                    _emit(line, line_callback, forward=None)

            stdout_thread = threading.Thread(
                target=_process_stream, args=(p.stdout, "stdout")
            )
            stderr_thread = threading.Thread(
                target=_process_stream, args=(p.stderr, "stderr")
            )

            stdout_thread.start()
            stderr_thread.start()

            stdout_thread.join()
            stderr_thread.join()

            ret = p.wait()
    except FileNotFoundError as e:
        raise RuntimeError(f"Command not found: {cmd[0]}") from e
    if ret != 0:
        raise RuntimeError(format_command_error(ret, cmd, "\n".join(captured)))
    return "\n".join(captured)


def parse_master_display_metadata(primaries_str: str, luminance_str: str) -> str | None:
    """Parse PyMediaInfo master display metadata and convert to x265 format.

    PyMediaInfo provides master display metadata in various formats that need to be
    parsed and converted to x265's expected format:
    "G(x,y)B(x,y)R(x,y)WP(x,y)L(max,min)"

    Args:
        primaries_str: Mastering display color primaries from PyMediaInfo
        luminance_str: Mastering display luminance from PyMediaInfo

    Returns:
        Formatted master display string for x265, or None if parsing fails

    Example x265 format:
        "G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)"
    """
    import re

    try:
        # Parse luminance: "min: 0.0050 cd/m2, max: 1000 cd/m2"
        luminance_pattern = r"min: ([\d.]+) cd/m2, max: ([\d.]+) cd/m2"
        lum_match = re.search(luminance_pattern, luminance_str)

        if not lum_match:
            logger.warning("Could not parse luminance string: %s", luminance_str)
            return None

        min_lum_raw = float(lum_match.group(1))
        max_lum_raw = float(lum_match.group(2))

        # Convert to x265 units (multiply by 10,000)
        min_lum = int(min_lum_raw * 10000)
        max_lum = int(max_lum_raw * 10000)

        # Color space coordinate mapping
        # Format: G(x,y)B(x,y)R(x,y)WP(x,y)
        COLOR_SPACE_COORDS = {
            "Display P3": "G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)",
            "DCI P3": "G(13250,34500)B(7500,3000)R(34000,16000)WP(15700,17550)",
            "BT.2020": "G(8500,39850)B(6550,2300)R(35400,14600)WP(15635,16450)",
        }

        # Look up color space coordinates
        if primaries_str not in COLOR_SPACE_COORDS:
            logger.warning(
                "Unknown color space '%s'. Supported: %s",
                primaries_str, list(COLOR_SPACE_COORDS.keys())
            )
            return None

        coords = COLOR_SPACE_COORDS[primaries_str]

        # Format: "G(x,y)B(x,y)R(x,y)WP(x,y)L(max,min)"
        master_display_str = f"{coords}L({max_lum},{min_lum})"

        logger.debug("Parsed master display: %s", master_display_str)
        return master_display_str

    except Exception as e:
        logger.error("Failed to parse master display metadata: %s", e)
        return None
