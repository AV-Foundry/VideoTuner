from __future__ import annotations

import re
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import TracebackType

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from .constants import PROGRESS_BAR_WIDTH
from .tool_parsers import (
    AUTOCROP_PROGRESS_RE,
    FFMPEG_FRAME_RE,
    FFMSINDEX_PROGRESS_RE,
    MKVMERGE_PCT_RE,
    SSIM_LINE_RE,
    VSZIP_PROGRESS_RE,
    X265_ENCODED_RE,
    X265_FRAME_RE,
    X265_PROGRESS_RE,
    clean_ansi,
)
from .version import __version__

LineHandler = Callable[[str], bool]


@dataclass
class PipelineDisplay:
    """Top-level helper that renders the app title and builds stages."""

    title: str = "VideoTuner"
    console: Console = field(default_factory=Console)
    show_title: bool = True

    def __post_init__(self) -> None:
        if self.show_title:
            self.console.print(
                Panel.fit(
                    f"[bold]{self.title}[/bold] [dim]v{__version__}[/dim]",
                    border_style="cyan",
                    padding=(0, 2),
                )
            )

    @contextmanager
    def stage(
        self,
        name: str,
        *,
        total: float | None = None,
        unit: str = "",
        show_eta: bool = True,
        transient: bool = True,
        show_done: bool = True,
        progress_indicator: str | None = None,
    ) -> Generator[Stage, None, None]:
        stage = Stage(
            console=self.console,
            name=name,
            total=total,
            unit=unit,
            show_eta=show_eta,
            transient=transient,
            show_done=show_done,
            progress_indicator=progress_indicator,
        )
        with stage:
            yield stage


class Stage:
    """Context manager representing a pipeline stage with Rich progress output."""

    console: Console
    name: str
    unit: str
    total: float | None
    show_eta: bool
    transient: bool
    show_done: bool
    progress_indicator: str | None
    done_suffix: str | None
    _progress: Progress | None
    _task_id: TaskID | None
    _closed: bool

    def __init__(
        self,
        *,
        console: Console,
        name: str,
        total: float | None,
        unit: str = "",
        show_eta: bool = True,
        transient: bool = True,
        show_done: bool = True,
        progress_indicator: str | None = None,
    ) -> None:
        self.console = console
        self.name = name
        self.unit = unit
        self.total = float(total) if total not in (None, 0) else None
        self.show_eta = show_eta
        self.transient = transient
        self.show_done = show_done
        self.progress_indicator = progress_indicator
        self.done_suffix = None
        self._progress = None
        self._task_id = None
        self._closed = False

    def __enter__(self) -> "Stage":
        # Only show header if not transient or if we want persistent headers
        if not self.transient:
            self.console.print(f"[cyan]{self.name}[/cyan]")

        # Build the description with optional progress indicator in magenta
        if self.progress_indicator:
            description = f"{self.name} [magenta]{self.progress_indicator}[/magenta]"
        else:
            description = self.name

        # Build columns based on whether we have a total (progress bar vs busy bar)
        if self.total is None:
            # Busy bar: show elapsed time instead of percentage
            progress = Progress(
                TextColumn("[cyan]{task.description}[/cyan]"),
                SpinnerColumn(),
                BarColumn(bar_width=PROGRESS_BAR_WIDTH),
                TimeElapsedColumn(),
                console=self.console,
                transient=self.transient,
                refresh_per_second=12,
            )
        else:
            # Progress bar: show percentage and time (ETA or elapsed)
            if self.show_eta:
                time_column = TimeRemainingColumn()
            else:
                time_column = TimeElapsedColumn()

            progress = Progress(
                TextColumn("[cyan]{task.description}[/cyan]"),
                SpinnerColumn(),
                BarColumn(bar_width=PROGRESS_BAR_WIDTH),
                TextColumn("{task.percentage:>5.1f}%"),
                time_column,
                console=self.console,
                transient=self.transient,
                refresh_per_second=12,
            )

        self._progress = progress
        _ = progress.__enter__()
        self._task_id = progress.add_task(description=description, total=self.total)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        success = exc_type is None
        self.complete(success=success)
        # Show "Done!" message if transient and successful
        if success and self.transient and self.show_done:
            suffix = f" {self.done_suffix}" if self.done_suffix else ""
            self.console.print(
                f"[cyan]{self.name}[/cyan] [bold green]Done![/bold green]{suffix}"
            )

    def set_total(self, total: float | None) -> None:
        if total in (None, 0):
            return
        self.total = float(total)
        if self._progress and self._task_id is not None:
            self._progress.update(self._task_id, total=self.total)

    def update(self, *, completed: float | None = None) -> None:
        if self._progress is None or self._task_id is None:
            return
        if completed is not None:
            bounded = completed
            if self.total is not None:
                bounded = min(max(0.0, completed), self.total)
            self._progress.update(self._task_id, completed=bounded)

    def advance(self, amount: float) -> None:
        if self._progress is None or self._task_id is None:
            return
        self._progress.advance(self._task_id, amount)

    def complete(self, *, success: bool = True) -> None:
        if self._closed:
            return
        self._closed = True
        if self._progress is not None:
            if success and self.total is not None and self._task_id is not None:
                self._progress.update(self._task_id, completed=self.total)
            self._progress.__exit__(None, None, None)
            self._progress = None

    # Parsers -----------------------------------------------------------------
    def make_ffmpeg_handler(self, *, total_frames: int | None = None) -> LineHandler:
        if total_frames:
            self.set_total(total_frames)

        def handler(line: str) -> bool:
            line = clean_ansi(line)
            if "frame=" not in line:
                return False
            match = FFMPEG_FRAME_RE.search(line)
            if not match:
                return False
            frame = int(match.group(1))
            self.update(completed=frame)
            return True

        return handler

    def make_percent_handler(self, *, total_percent: float = 100.0) -> LineHandler:
        self.set_total(total_percent)

        def handler(line: str) -> bool:
            line = clean_ansi(line)
            match = MKVMERGE_PCT_RE.search(line)
            if not match:
                return False
            pct = float(match.group(1))
            self.update(completed=pct)
            return True

        return handler

    def make_x265_handler(self, *, total_frames: int | None = None) -> LineHandler:
        if total_frames:
            self.set_total(total_frames)
        last_frame = 0

        def handler(line: str) -> bool:
            nonlocal last_frame
            line = clean_ansi(line)
            if not line:
                return False

            prog_match = X265_PROGRESS_RE.search(line)
            if prog_match:
                done = int(prog_match.group("done"))
                total = int(prog_match.group("total"))
                if total_frames is None:
                    self.set_total(total)
                last_frame = max(last_frame, done)
                self.update(completed=done)
                return True

            match = X265_FRAME_RE.search(line)
            if not match:
                match = X265_ENCODED_RE.search(line)
            if not match:
                return False

            frame = int(match.group(1))
            last_frame = max(last_frame, frame)
            if self.total is None and total_frames is None:
                self.set_total(frame)
            self.update(completed=frame)
            return True

        return handler

    def make_ssim_handler(
        self, *, total_frames: int | None = None, use_native_display: bool = False
    ) -> LineHandler:
        """Create a handler for SSIMULACRA2 output.

        Args:
            total_frames: Expected total frame count
            use_native_display: If True, don't update Rich progress (let SSIMULACRA2's native display show)
        """
        if total_frames:
            self.set_total(total_frames)

        def handler(line: str) -> bool:
            line = clean_ansi(line)
            match = SSIM_LINE_RE.search(line)
            if not match:
                return False

            if not use_native_display:
                done = float(match.group("done"))
                total = float(match.group("total"))
                self.set_total(total)
                # Always use the exact 'done' count since we have it (e.g., "481/481")
                # rather than recalculating from percentage which may be less accurate
                self.update(completed=done)
            return True  # Mark as consumed so it doesn't print

        return handler

    def make_ffmsindex_handler(self) -> LineHandler:
        """Create a handler for ffmsindex progress output.

        ffmsindex outputs progress in the format:
            "Indexing, please wait... 0%"
            "Indexing, please wait... 50%"
            "Indexing, please wait... 100%"
        """

        def handler(line: str) -> bool:
            line = clean_ansi(line)
            match = FFMSINDEX_PROGRESS_RE.search(line)
            if not match:
                return False

            percent = int(match.group(1))

            # Set total to 100 on first match (percentage-based progress)
            if self.total is None or self.total == 1:
                self.set_total(100)

            self.update(completed=percent)
            return True

        return handler

    def make_autocrop_handler(self) -> LineHandler:
        """Create a handler for autocrop progress output.

        Autocrop outputs progress in the format:
            "AutoCrop progress: 5/10 (50%)"
        """

        def handler(line: str) -> bool:
            line = clean_ansi(line)
            match = AUTOCROP_PROGRESS_RE.search(line)
            if not match:
                return False

            current = int(match.group(1))
            total = int(match.group(2))

            # Set total on first match
            if self.total is None or self.total == 1:
                self.set_total(total)

            self.update(completed=current)
            return True

        return handler

    def make_ssim_verbose_handler(
        self, *, total_frames: int | None = None
    ) -> LineHandler:
        """Create a handler for SSIMULACRA2 --verbose output.

        The --verbose flag outputs per-frame scores in the format:
            Frame 0: 91.63643741
            Frame 1: 86.03377399
            ...

        Args:
            total_frames: Expected total frame count
        """
        if total_frames:
            self.set_total(total_frames)

        # Pattern to match tolerant variants such as:
        # "Frame 0: 91.6", "Frame 12/480: 88.1", etc.
        frame_pattern = re.compile(r"Frame\s+(\d+)(?:\s*/\s*(\d+))?", re.IGNORECASE)
        last_frame = -1

        def handler(line: str) -> bool:
            nonlocal last_frame
            line = clean_ansi(line)
            match = frame_pattern.search(line)
            if not match:
                return False

            frame_num = int(match.group(1))
            total_reported = match.group(2)
            if total_reported:
                try:
                    self.set_total(int(total_reported))
                except ValueError:
                    pass
            elif total_frames:
                self.set_total(total_frames)

            if frame_num < last_frame:
                frame_index = last_frame
            else:
                frame_index = frame_num
                last_frame = frame_num

            # Frame numbers are 0-indexed, so completed = frame_index + 1
            self.update(completed=frame_index + 1)
            return True  # Mark as consumed so it doesn't print

        return handler

    def make_vszip_handler(self, *, total_frames: int | None = None) -> LineHandler:
        """Create a handler for vszip SSIMULACRA2 progress output.

        vszip outputs progress in the format:
            "vszip progress: 50/100"

        Args:
            total_frames: Expected total frame count
        """
        if total_frames:
            self.set_total(total_frames)

        def handler(line: str) -> bool:
            line = clean_ansi(line)

            match = VSZIP_PROGRESS_RE.search(line)
            if not match:
                return False

            current = int(match.group(1))
            total = int(match.group(2))

            if total > 0:
                self.set_total(total)
            self.update(completed=current)

            return True

        return handler
