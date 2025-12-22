# Contributing to VideoTuner

Thank you for your interest in contributing to VideoTuner! This document provides guidelines and instructions for contributing.

## Code of Conduct

This project adheres to the [Contributor Covenant 3.0 Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to <avfoundry@pm.me>.

## Getting Started

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- External tools on PATH: FFmpeg (with libvmaf, libplacebo), ffprobe, mkvmerge

### Development Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/AV-Foundry/VideoTuner.git
   cd VideoTuner
   ```

2. **Install with development dependencies:**

   Using uv (recommended):

   ```bash
   uv sync --extra dev
   ```

   Using pip:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -e ".[dev]"
   ```

3. **Set up VapourSynth portable environment:**

   Download `Install-Portable-VapourSynth-R72.ps1` from [VapourSynth R72 releases](https://github.com/vapoursynth/vapoursynth/releases/tag/R72) and run from the repository root:

   ```powershell
   powershell -ExecutionPolicy Bypass -File Install-Portable-VapourSynth-R72.ps1 -TargetFolder vapoursynth-portable
   ```

4. **Download required VapourSynth plugins:**

   Download and place in `vapoursynth-portable/vs-plugins/`:
   - **ffms2** 5.0: Download `ffms2-5.0-msvc.7z` from [ffms2 releases](https://github.com/FFMS/ffms2/releases/tag/5.0), extract `x64/ffms2.dll` and `x64/ffmsindex.exe`
   - **LSMASHSource** 1266.0.0.0: Download `L-SMASH-Works-r1266.0.0.0.7z` from [L-SMASH-Works releases](https://github.com/HomeOfAviSynthPlusEvolution/L-SMASH-Works/releases/tag/1266.0.0.0), extract `x64/LSMASHSource.dll`
   - **vszip** R11: Download `vapoursynth-zip-r11-windows-x86_64.zip` from [vapoursynth-zip releases](https://github.com/dnjulek/vapoursynth-zip/releases/tag/R11), extract `vszip.dll`
   - **autocrop**: Download from [vapoursynth-autocrop](https://github.com/Irrational-Encoding-Wizardry/vapoursynth-autocrop), extract `autocrop.dll`

5. **Download x265 encoder:**

   Download and place in `tools/`:
   - **x265** 4.1+191+33: Download from [x265-Mod-by-Patman releases](https://github.com/Patman86/x265-Mod-by-Patman/releases/tag/4.1%2B191%2B33), extract `x265.exe` to `tools/x265.exe`

## Development Workflow

### Running Tests

```bash
pytest                    # Run all tests
pytest tests/test_file.py # Run a specific test file
pytest -v -x              # Verbose output, stop on first failure
```

### Type Checking

The project uses basedpyright for type checking:

```bash
basedpyright src tests
```

### Linting and Formatting

The project uses ruff for linting and formatting:

```bash
ruff check .    # Check for linting issues
ruff check . --fix  # Auto-fix linting issues
ruff format .   # Format code
```

### Running the Application

```bash
# Via entry point
videotuner

# Direct execution
python main.py "<input>.mkv"
```

## Code Conventions

- Use `from __future__ import annotations` in all modules
- Full type annotations for function signatures and class attributes
- Dataclasses for structured data
- Protocol types for duck typing
- Google-style docstrings
- Callback-based progress via `LineHandler` type for subprocess output parsing

## Submitting Changes

### Reporting Issues

Before creating an issue:

1. Search existing issues to avoid duplicates
2. Use the appropriate issue template if available
3. Provide clear reproduction steps for bugs
4. Include relevant system information (OS, Python version, FFmpeg version)

### Pull Requests

1. **Fork and branch:** Create a feature branch from `main`

   ```bash
   git checkout -b feat/your-feature-name
   ```

   Branch names must follow the convention: `<type>/<description>`

   Valid types: `feat/`, `fix/`, `docs/`, `style/`, `refactor/`, `test/`, `chore/`

2. **Make your changes:**
   - Follow the existing code style and conventions
   - Add tests for new functionality
   - Update documentation if needed

3. **Verify your changes:**

   ```bash
   pytest                    # All tests pass
   basedpyright src tests    # No type errors
   ruff check .              # No linting issues
   ruff format .             # Code is formatted
   ```

4. **Commit your changes:**
   - Write clear, descriptive commit messages
   - Reference related issues in commits when applicable

5. **Submit a pull request:**
   - Provide a clear description of the changes
   - Link to any related issues
   - Be responsive to review feedback

## Architecture Overview

The codebase follows a pipeline architecture in `src/videotuner/`:

| Module | Purpose |
| ------ | ------- |
| `pipeline.py` | Main orchestration and CLI entry point |
| `pipeline_*.py` | Pipeline modules (CLI, iteration, validation, etc.) |
| `crf_search.py` | Interpolated binary search for optimal CRF |
| `profiles.py` | YAML-based profile loading and validation |
| `encoder_params.py` | x265 parameter building with auto-detection |
| `media.py` | Video metadata extraction |
| `create_encodes.py` | VapourSynth script generation and encoding |
| `vmaf_assessment.py` | VMAF quality assessment |
| `ssimulacra2_assessment.py` | SSIMULACRA2 quality assessment |
| `progress.py` | Rich console progress display |
| `constants.py` | Centralized constants (CRF limits, thread counts, etc.) |
| `utils.py` | Shared utilities (subprocess execution, file operations) |
| `encoding_utils.py` | Encoding utilities (HDR detection, path dataclasses) |

## Questions?

If you have questions about contributing, feel free to open a discussion or issue on GitHub.
