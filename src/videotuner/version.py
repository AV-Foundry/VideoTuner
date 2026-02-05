"""Version information and release notes for VideoTuner."""

__version__ = "0.2.3"

RELEASE_NOTES = """
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
