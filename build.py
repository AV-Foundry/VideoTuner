"""Build script for creating VideoTuner releases with Nuitka."""

from __future__ import annotations

import hashlib
import hashlib
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

# External dependency URLs and versions
# x265 encoder
X265_VERSION = "4.1+191+33"
X265_URL = "https://github.com/Patman86/x265-Mod-by-Patman/releases/download/4.1%2B191%2B33/x265-4.1+191+33-61d0a57b3-.Mod-by-Patman.-x64-avx2-clang2111.7z"

# VapourSynth portable environment
VAPOURSYNTH_VERSION = "R72"
VAPOURSYNTH_INSTALLER_URL = f"https://github.com/vapoursynth/vapoursynth/releases/download/{VAPOURSYNTH_VERSION}/Install-Portable-VapourSynth-{VAPOURSYNTH_VERSION}.ps1"

# VapourSynth plugins (all x64)
FFMS2_VERSION = "5.0"
FFMS2_URL = f"https://github.com/FFMS/ffms2/releases/download/{FFMS2_VERSION}/ffms2-{FFMS2_VERSION}-msvc.7z"

LSMASH_VERSION = "1266.0.0.0"
LSMASH_URL = f"https://github.com/HomeOfAviSynthPlusEvolution/L-SMASH-Works/releases/download/{LSMASH_VERSION}/L-SMASH-Works-r{LSMASH_VERSION}.7z"

VSZIP_VERSION = "R11"
VSZIP_URL = f"https://github.com/dnjulek/vapoursynth-zip/releases/download/{VSZIP_VERSION}/vapoursynth-zip-r11-windows-x86_64.zip"
# External dependency URLs and versions
# x265 encoder
X265_VERSION = "4.1+191+33"
X265_URL = "https://github.com/Patman86/x265-Mod-by-Patman/releases/download/4.1%2B191%2B33/x265-4.1+191+33-61d0a57b3-.Mod-by-Patman.-x64-avx2-clang2111.7z"

# VapourSynth portable environment
VAPOURSYNTH_VERSION = "R72"
VAPOURSYNTH_INSTALLER_URL = f"https://github.com/vapoursynth/vapoursynth/releases/download/{VAPOURSYNTH_VERSION}/Install-Portable-VapourSynth-{VAPOURSYNTH_VERSION}.ps1"

# VapourSynth plugins (all x64)
FFMS2_VERSION = "5.0"
FFMS2_URL = f"https://github.com/FFMS/ffms2/releases/download/{FFMS2_VERSION}/ffms2-{FFMS2_VERSION}-msvc.7z"

LSMASH_VERSION = "1266.0.0.0"
LSMASH_URL = f"https://github.com/HomeOfAviSynthPlusEvolution/L-SMASH-Works/releases/download/{LSMASH_VERSION}/L-SMASH-Works-r{LSMASH_VERSION}.7z"

VSZIP_VERSION = "R11"
VSZIP_URL = f"https://github.com/dnjulek/vapoursynth-zip/releases/download/{VSZIP_VERSION}/vapoursynth-zip-r11-windows-x86_64.zip"
VSZIP_DLL = "vszip.dll"

# SHA256 checksums for integrity verification (protects against compromised downloads)
# To update: download file, run: python -c "import hashlib; print(hashlib.sha256(open('file','rb').read()).hexdigest())"
CHECKSUMS = {
    "vapoursynth_installer": "ac9faf0f009f81fe108018b1ca424355627b93c145bee43f7f8ef62279fa5170",
    "x265": "97de51546e6f76365c3fd7900205d376ada87a45b7877c4fec9df6a25b1fd592",
    "ffms2": "e867a3df7262865107df40f230f5b8e1455905eba9b8852e6f35b1227537caeb",
    "lsmash": "7189f299730c82e2cef025082095628c1028effb7e7276eae5fb5c9c3f1aef00",
    "vszip": "99cfaba01ce50d414dfd371d8c9b261f164e398a98b49bed2983fa770a143cdd",
}


def verify_checksum(file_path: Path, expected_hash: str, name: str) -> None:
    """Verify SHA256 checksum of a downloaded file.

    Args:
        file_path: Path to the file to verify
        expected_hash: Expected SHA256 hex digest
        name: Human-readable name for error messages

    Raises:
        SystemExit: If checksum doesn't match
    """
    if expected_hash == "PLACEHOLDER":
        # Skip verification if hash not yet set - print actual hash for user to add
        actual = hashlib.sha256(file_path.read_bytes()).hexdigest()
        print(f"  WARNING: No checksum for {name}. Actual SHA256: {actual}")
        return

    actual = hashlib.sha256(file_path.read_bytes()).hexdigest()
    if actual != expected_hash:
        print(f"ERROR: Checksum mismatch for {name}!")
        print(f"  Expected: {expected_hash}")
        print(f"  Actual:   {actual}")
        print("This could indicate a compromised or corrupted download.")
        sys.exit(1)


# SHA256 checksums for integrity verification (protects against compromised downloads)
# To update: download file, run: python -c "import hashlib; print(hashlib.sha256(open('file','rb').read()).hexdigest())"
CHECKSUMS = {
    "vapoursynth_installer": "ac9faf0f009f81fe108018b1ca424355627b93c145bee43f7f8ef62279fa5170",
    "x265": "97de51546e6f76365c3fd7900205d376ada87a45b7877c4fec9df6a25b1fd592",
    "ffms2": "e867a3df7262865107df40f230f5b8e1455905eba9b8852e6f35b1227537caeb",
    "lsmash": "7189f299730c82e2cef025082095628c1028effb7e7276eae5fb5c9c3f1aef00",
    "vszip": "99cfaba01ce50d414dfd371d8c9b261f164e398a98b49bed2983fa770a143cdd",
}


def verify_checksum(file_path: Path, expected_hash: str, name: str) -> None:
    """Verify SHA256 checksum of a downloaded file.

    Args:
        file_path: Path to the file to verify
        expected_hash: Expected SHA256 hex digest
        name: Human-readable name for error messages

    Raises:
        SystemExit: If checksum doesn't match
    """
    if expected_hash == "PLACEHOLDER":
        # Skip verification if hash not yet set - print actual hash for user to add
        actual = hashlib.sha256(file_path.read_bytes()).hexdigest()
        print(f"  WARNING: No checksum for {name}. Actual SHA256: {actual}")
        return

    actual = hashlib.sha256(file_path.read_bytes()).hexdigest()
    if actual != expected_hash:
        print(f"ERROR: Checksum mismatch for {name}!")
        print(f"  Expected: {expected_hash}")
        print(f"  Actual:   {actual}")
        print("This could indicate a compromised or corrupted download.")
        sys.exit(1)


def clean_previous_build() -> None:
    """Remove previous build artifacts."""
    if RELEASE_DIR.exists():
        print(f"Cleaning previous release: {RELEASE_DIR}")
        shutil.rmtree(RELEASE_DIR)

    # Clean Nuitka build cache for fresh builds
    nuitka_cache = DIST_DIR / "pipeline.build"
    if nuitka_cache.exists():
        shutil.rmtree(nuitka_cache)


def install_vapoursynth_portable(target_dir: Path) -> None:
    """Download and run VapourSynth portable installer.

    Downloads the official installer script from GitHub and executes it
    to create a portable VapourSynth environment at the target directory.
    """
    if target_dir.exists():
        print(f"  VapourSynth portable already exists at {target_dir}, skipping")
        return

    print(f"Installing VapourSynth {VAPOURSYNTH_VERSION} portable environment...")

    with tempfile.TemporaryDirectory() as tmpdir:
        installer_path = (
            Path(tmpdir) / f"Install-Portable-VapourSynth-{VAPOURSYNTH_VERSION}.ps1"
        )

        # Download the installer script
        print(f"  Downloading installer from {VAPOURSYNTH_INSTALLER_URL}")
        try:
            _ = urllib.request.urlretrieve(VAPOURSYNTH_INSTALLER_URL, installer_path)
        except Exception as e:
            print(f"ERROR: Failed to download VapourSynth installer: {e}")
            sys.exit(1)

        verify_checksum(
            installer_path, CHECKSUMS["vapoursynth_installer"], "VapourSynth installer"
        )

        # Run the PowerShell installer with target folder
        # The installer creates the folder relative to its working directory,
        # so we run it from the parent of the target and specify the folder name
        target_parent = target_dir.parent
        target_name = target_dir.name

        cmd = [
            "powershell.exe",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(installer_path),
            "-TargetFolder",
            target_name,
        ]

        print(f"  Running installer (this may take a minute)...")
        result = subprocess.run(cmd, cwd=target_parent)

        if result.returncode != 0:
            print(f"ERROR: VapourSynth installer failed with code {result.returncode}")
            sys.exit(1)

        if not target_dir.exists():
            print(f"ERROR: VapourSynth installation did not create {target_dir}")
            sys.exit(1)

        print(f"  VapourSynth {VAPOURSYNTH_VERSION} installed to {target_dir}")


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

        verify_checksum(zip_path, CHECKSUMS["vszip"], "vszip")

        verify_checksum(zip_path, CHECKSUMS["vszip"], "vszip")

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


def extract_from_7z(
    archive_path: Path,
    files_to_extract: list[str],
    dest_dir: Path,
    sevenzip_exe: Path,
) -> None:
    """Extract specific files from a 7z archive.

    Args:
        archive_path: Path to the .7z archive
        files_to_extract: List of file paths within the archive to extract
        dest_dir: Destination directory for extracted files
        sevenzip_exe: Path to 7z.exe
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Extract specified files to temp directory
        cmd = [
            str(sevenzip_exe),
            "e",  # extract without directory structure
            str(archive_path),
            f"-o{tmpdir}",
            "-y",  # yes to all prompts
            *files_to_extract,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ERROR: 7z extraction failed: {result.stderr}")
            sys.exit(1)

        # Move extracted files to destination
        for file_path in files_to_extract:
            filename = Path(file_path).name
            src = Path(tmpdir) / filename
            if src.exists():
                _ = shutil.copy2(src, dest_dir / filename)
            else:
                print(f"ERROR: Expected file not found after extraction: {filename}")
                sys.exit(1)


def download_ffms2(plugin_dir: Path, sevenzip_exe: Path) -> None:
    """Download and extract ffms2 plugin."""
    dest_dll = plugin_dir / "ffms2.dll"
    if dest_dll.exists():
        print(f"  ffms2.dll already exists, skipping download")
        return

    print(f"Downloading ffms2 {FFMS2_VERSION}...")

    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = Path(tmpdir) / "ffms2.7z"

        try:
            _ = urllib.request.urlretrieve(FFMS2_URL, archive_path)
        except Exception as e:
            print(f"ERROR: Failed to download ffms2: {e}")
            sys.exit(1)

        verify_checksum(archive_path, CHECKSUMS["ffms2"], "ffms2")

        # Extract ffms2.dll and ffmsindex.exe from x64 folder
        extract_from_7z(
            archive_path,
            [
                f"ffms2-{FFMS2_VERSION}-msvc/x64/ffms2.dll",
                f"ffms2-{FFMS2_VERSION}-msvc/x64/ffmsindex.exe",
            ],
            plugin_dir,
            sevenzip_exe,
        )
        print(f"  Extracted ffms2.dll and ffmsindex.exe to {plugin_dir}")


def download_lsmashsource(plugin_dir: Path, sevenzip_exe: Path) -> None:
    """Download and extract LSMASHSource plugin."""
    dest_dll = plugin_dir / "LSMASHSource.dll"
    if dest_dll.exists():
        print(f"  LSMASHSource.dll already exists, skipping download")
        return

    print(f"Downloading LSMASHSource {LSMASH_VERSION}...")

    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = Path(tmpdir) / "lsmash.7z"

        try:
            _ = urllib.request.urlretrieve(LSMASH_URL, archive_path)
        except Exception as e:
            print(f"ERROR: Failed to download LSMASHSource: {e}")
            sys.exit(1)

        verify_checksum(archive_path, CHECKSUMS["lsmash"], "LSMASHSource")

        # Extract LSMASHSource.dll from x64 folder
        extract_from_7z(
            archive_path,
            ["x64/LSMASHSource.dll"],
            plugin_dir,
            sevenzip_exe,
        )
        print(f"  Extracted LSMASHSource.dll to {plugin_dir}")


def download_x265(tools_dir: Path, sevenzip_exe: Path) -> None:
    """Download and extract x265 encoder."""
    dest_exe = tools_dir / "x265.exe"
    if dest_exe.exists():
        print(f"  x265.exe already exists, skipping download")
        return

    print(f"Downloading x265 {X265_VERSION}...")

    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = Path(tmpdir) / "x265.7z"

        try:
            _ = urllib.request.urlretrieve(X265_URL, archive_path)
        except Exception as e:
            print(f"ERROR: Failed to download x265: {e}")
            sys.exit(1)

        verify_checksum(archive_path, CHECKSUMS["x265"], "x265")

        # Extract x265.exe from archive root
        extract_from_7z(
            archive_path,
            ["x265.exe"],
            tools_dir,
            sevenzip_exe,
        )
        print(f"  Extracted x265.exe to {tools_dir}")


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

    # Install vapoursynth-portable from official source (needed for 7z.exe)
    # Install vapoursynth-portable from official source (needed for 7z.exe)
    vs_dst = RELEASE_DIR / "vapoursynth-portable"
    install_vapoursynth_portable(vs_dst)
    sevenzip_exe = vs_dst / "7z.exe"

    # Download x265 encoder to tools folder
    tools_dst = RELEASE_DIR / "tools"
    tools_dst.mkdir(parents=True, exist_ok=True)
    download_x265(tools_dst, sevenzip_exe)
    install_vapoursynth_portable(vs_dst)
    sevenzip_exe = vs_dst / "7z.exe"

    # Download x265 encoder to tools folder
    tools_dst = RELEASE_DIR / "tools"
    tools_dst.mkdir(parents=True, exist_ok=True)
    download_x265(tools_dst, sevenzip_exe)

    # Download plugins to the plugin directory
    plugin_dir = vs_dst / "vs-plugins"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    download_ffms2(plugin_dir, sevenzip_exe)
    download_lsmashsource(plugin_dir, sevenzip_exe)
    download_vszip(plugin_dir)
    # Download plugins to the plugin directory
    plugin_dir = vs_dst / "vs-plugins"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    download_ffms2(plugin_dir, sevenzip_exe)
    download_lsmashsource(plugin_dir, sevenzip_exe)
    download_vszip(plugin_dir)

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
