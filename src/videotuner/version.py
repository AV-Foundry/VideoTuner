"""Version information and release notes for VideoTuner."""

__version__ = "0.3.0"

RELEASE_NOTES = """
## 0.3.0

### Breaking Changes

- Rename `--no-autocrop` to `--no-cropdetect`
- Remove VapourSynth `autocrop` plugin dependency; crop detection now uses FFmpeg's native `cropdetect` filter

### Features

- Add unified tonemapping module with automatic GPU acceleration (Vulkan/libplacebo) and CPU fallback (zscale/hable)
- Replace VapourSynth-based autocrop with FFmpeg `cropdetect` for more robust, dependency-free crop detection
- Add HDR tonemapping support for crop detection, ensuring consistent results across SDR and HDR sources
- Add `cropdetect` CLI argument group with full parameter control: `--cropdetect-interval`, `--cropdetect-mode`, `--cropdetect-limit`, `--cropdetect-round`, `--cropdetect-mv-threshold`, `--cropdetect-low`, `--cropdetect-high`
- Support two cropdetect modes: `black` (pixel threshold, default) and `mvedges` (motion vector + edge detection)
- Integrate unified tonemapping into VMAF assessment with automatic GPU/CPU branching
- Change default crop detection sampling interval to 30 seconds for denser frame sampling

### Fixes

- Resolve CodeQL code quality alerts: replace self-comparison NaN checks with `math.isnan()`, narrow broad exception clauses, remove dead variable assignments, replace empty except blocks with `contextlib.suppress()` or appropriate logging

## 0.2.4

### Fixes

- Fix Unicode 17.0.0 compatibility by updating Rich to >=14.3.2

### Build

- Add explicit rich._unicode_data package inclusion for Nuitka builds

## 0.2.3

### Fixes

- Prevent duplicate CRF testing when interpolation rounds to an already-tested value
- Add exact match early termination when all targets are met and a score exactly equals its target

## 0.2.2

### Fixes

- Correct metric rounding to match display precision, fixing false negatives where visually-passing values failed programmatically

### Features

- Add `--metric-decimals` CLI argument to control metric display and comparison precision (default: 2)

### Build

- Update bundled x265 encoder to 4.1+212+35 (Patman's Mod)

## 0.2.1

### Features

- Optimize encoding by sharing samples when VMAF and SSIM2 use identical sampling parameters (default behavior)

## 0.2.0

### Breaking Changes

- Removed `--ssim2-bin` CLI argument; ssimulacra2_rs is no longer supported

### Features

- Migrate SSIMULACRA2 to vszip VapourSynth plugin for improved performance and integration
- Align SSIM2 sampling defaults with VMAF parameters for consistent sample density

### Build

- Externalize bundled dependencies to auto-download at build time with SHA256 verification

## 0.1.0

Initial release.

- CRF optimization using VMAF and SSIMULACRA2 quality metrics
- Interpolated binary search algorithm for finding optimal CRF values
- YAML-based encoding profiles with HDR/SDR conditional parameters
- Automated sample extraction and quality assessment
- Multi-profile comparison mode
- Rich console progress display
""".strip()
