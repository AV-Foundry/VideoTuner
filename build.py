"""Build script for creating VideoTuner releases with Nuitka."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

# Import version from the package
sys.path.insert(0, str(Path(__file__).parent / "src"))
from videotuner.version import __version__

REPO_ROOT = Path(__file__).parent
DIST_DIR = REPO_ROOT / "dist"
RELEASE_NAME = f"VideoTuner-v{__version__}"
RELEASE_DIR = DIST_DIR / RELEASE_NAME

# External dependency URLs
VSZIP_URL = "https://github.com/dnjulek/vapoursynth-zip/releases/download/R11/vapoursynth-zip-r11-windows-x86_64.zip"
VSZIP_DLL = "vszip.dll"


def clean_previous_build() -> None:
    """Remove previous build artifacts."""
    if RELEASE_DIR.exists():
        print(f"Cleaning previous release: {RELEASE_DIR}")
        shutil.rmtree(RELEASE_DIR)

    # Clean Nuitka build cache for fresh builds (optional - comment out for faster rebuilds)
    # nuitka_cache = DIST_DIR / "pipeline.build"
    # if nuitka_cache.exists():
    #     shutil.rmtree(nuitka_cache)


def download_vszip(plugin_dir: Path) -> None:
    """Download and extract vszip plugin to the plugin directory."""
    dest_dll = plugin_dir / VSZIP_DLL
    if dest_dll.exists():
        print(f"  {VSZIP_DLL} already exists, skipping download")
        return

    print(f"Downloading {VSZIP_DLL} from vapoursynth-zip...")

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "vszip.zip"

        # Download the zip file
        try:
            _ = urllib.request.urlretrieve(VSZIP_URL, zip_path)
        except Exception as e:
            print(f"ERROR: Failed to download vszip: {e}")
            sys.exit(1)

        # Extract vszip.dll from the zip
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                # Find vszip.dll in the archive (may be in a subdirectory)
                dll_found = False
                for name in zf.namelist():
                    if name.endswith(VSZIP_DLL):
                        # Extract to temp dir then move to destination
                        _ = zf.extract(name, tmpdir)
                        extracted_path = Path(tmpdir) / name
                        _ = shutil.copy2(extracted_path, dest_dll)
                        dll_found = True
                        print(f"  Extracted {VSZIP_DLL} to {plugin_dir}")
                        break

                if not dll_found:
                    print(f"ERROR: {VSZIP_DLL} not found in downloaded archive")
                    sys.exit(1)

        except zipfile.BadZipFile as e:
            print(f"ERROR: Invalid zip file: {e}")
            sys.exit(1)


def run_nuitka() -> Path:
    """Run Nuitka to build the executable."""
    print("Building with Nuitka (this may take several minutes)...")

    cmd = [
        sys.executable,
        "-m",
        "nuitka",
        "--onefile",
        "--assume-yes-for-downloads",  # Auto-accept dependency downloads in CI
        f"--output-dir={DIST_DIR}",
        "--output-filename=VideoTuner.exe",
        # Compile as a package run with -m (uses __main__.py automatically)
        "--python-flag=-m",
        "--nofollow-import-to=pytest",
        "--nofollow-import-to=tests",
        "--windows-console-mode=force",
        # Optional: Add version info to the exe
        f"--product-version={__version__}",
        f"--file-version={__version__}",
        "--product-name=VideoTuner",
        "--company-name=AVFoundry",
        "--copyright=Copyright 2025 AVFoundry",
        "--file-description=CRF optimization and encoder benchmarking tool",
        # Point to the package directory (not __main__.py)
        "src/videotuner",
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=REPO_ROOT)

    if result.returncode != 0:
        print("Nuitka build failed!")
        sys.exit(1)

    # Nuitka names output after source file when using --output-filename
    exe_path = DIST_DIR / "VideoTuner.exe"
    if not exe_path.exists():
        # Fallback: check alternate names if --output-filename didn't work
        for alt_name in ["videotuner.exe", "__main__.exe"]:
            alt_path = DIST_DIR / alt_name
            if alt_path.exists():
                _ = alt_path.rename(exe_path)
                break
        else:
            print(f"Expected exe not found at: {exe_path}")
            print("Checked: VideoTuner.exe, videotuner.exe, __main__.exe")
            sys.exit(1)

    return exe_path


def assemble_release(exe_path: Path) -> None:
    """Assemble the release folder with exe and required files."""
    print(f"Assembling release: {RELEASE_DIR}")

    RELEASE_DIR.mkdir(parents=True, exist_ok=True)

    # Copy the executable
    _ = shutil.copy2(exe_path, RELEASE_DIR / "VideoTuner.exe")

    # Copy tools folder
    tools_src = REPO_ROOT / "tools"
    tools_dst = RELEASE_DIR / "tools"
    if tools_src.exists():
        print("Copying tools/ ...")
        _ = shutil.copytree(tools_src, tools_dst)
    else:
        print(f"WARNING: tools/ not found at {tools_src}")

    # Copy vapoursynth-portable folder
    vs_src = REPO_ROOT / "vapoursynth-portable"
    vs_dst = RELEASE_DIR / "vapoursynth-portable"
    if vs_src.exists():
        print("Copying vapoursynth-portable/ ...")
        _ = shutil.copytree(vs_src, vs_dst)

        # Download external plugins to the release plugin directory
        plugin_dir = vs_dst / "vs-plugins"
        plugin_dir.mkdir(parents=True, exist_ok=True)
        download_vszip(plugin_dir)
    else:
        print(f"WARNING: vapoursynth-portable/ not found at {vs_src}")

    # Copy sample profile config
    sample_config = REPO_ROOT / "x265_profiles.yaml.sample"
    if sample_config.exists():
        _ = shutil.copy2(sample_config, RELEASE_DIR / "x265_profiles.yaml.sample")

    # Copy README
    readme = REPO_ROOT / "README.md"
    if readme.exists():
        _ = shutil.copy2(readme, RELEASE_DIR / "README.md")

    # Copy license files
    for license_file in ["LICENSE", "THIRD_PARTY_LICENSES.md"]:
        src = REPO_ROOT / license_file
        if src.exists():
            _ = shutil.copy2(src, RELEASE_DIR / license_file)

    # Copy licenses folder (third-party license texts)
    licenses_src = REPO_ROOT / "licenses"
    licenses_dst = RELEASE_DIR / "licenses"
    if licenses_src.exists():
        print("Copying licenses/ ...")
        _ = shutil.copytree(licenses_src, licenses_dst)
    else:
        print(f"WARNING: licenses/ not found at {licenses_src}")

    # Clean up the standalone exe from dist root (now in release folder)
    exe_path.unlink()


def print_summary() -> None:
    """Print build summary."""
    print()
    print("=" * 60)
    print(f"BUILD COMPLETE: {RELEASE_NAME}")
    print("=" * 60)
    print()
    print(f"Release folder: {RELEASE_DIR}")
    print()
    print("Contents:")
    for item in sorted(RELEASE_DIR.iterdir()):
        if item.is_dir():
            # Count files in directory
            file_count = sum(1 for _ in item.rglob("*") if _.is_file())
            print(f"  {item.name}/  ({file_count} files)")
        else:
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"  {item.name}  ({size_mb:.1f} MB)")
    print()
    print("Next steps:")
    print(f"  1. Test: cd {RELEASE_DIR} && VideoTuner.exe --help")
    print(f"  2. Package: zip -r {RELEASE_NAME}.zip {RELEASE_NAME}/")
    print()


def main() -> None:
    """Main build entry point."""
    print(f"Building VideoTuner v{__version__}")
    print()

    clean_previous_build()
    exe_path = run_nuitka()
    assemble_release(exe_path)
    print_summary()


if __name__ == "__main__":
    main()
