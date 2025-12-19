"""Tests for pipeline path management utilities."""

from __future__ import annotations

import tempfile
from pathlib import Path

from videotuner.pipeline_types import (
    get_distorted_dir,
    get_reference_dir,
    get_ssim2_dir,
    get_vmaf_dir,
)
from videotuner.profiles import Profile


class TestGetReferenceDir:
    """Tests for reference directory path management."""

    def test_creates_reference_directory(self):
        """Test creates reference directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            ref_dir = get_reference_dir(workdir)

            assert ref_dir.exists()
            assert ref_dir.is_dir()
            assert ref_dir == workdir / "reference"

    def test_returns_existing_reference_directory(self):
        """Test returns existing reference directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            expected_dir = workdir / "reference"
            expected_dir.mkdir(parents=True, exist_ok=True)

            ref_dir = get_reference_dir(workdir)

            assert ref_dir == expected_dir
            assert ref_dir.exists()

    def test_creates_parent_directories(self):
        """Test creates parent directories if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir) / "nested" / "path"
            ref_dir = get_reference_dir(workdir)

            assert ref_dir.exists()
            assert workdir.exists()


class TestGetDistortedDir:
    """Tests for distorted directory path management."""

    def test_creates_distorted_directory_for_profile(self):
        """Test creates distorted directory with profile name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            profile = Profile(name="TestProfile", description="Test", settings={})

            dist_dir = get_distorted_dir(workdir, profile)

            assert dist_dir.exists()
            assert dist_dir.is_dir()
            assert dist_dir == workdir / "distorted" / "profile_TestProfile"

    def test_sanitizes_profile_name_with_spaces(self):
        """Test replaces spaces in profile name with underscores."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            profile = Profile(name="Film High Quality", description="Test", settings={})

            dist_dir = get_distorted_dir(workdir, profile)

            assert dist_dir == workdir / "distorted" / "profile_Film_High_Quality"
            assert dist_dir.exists()

    def test_sanitizes_profile_name_with_forward_slash(self):
        """Test replaces forward slashes in profile name with underscores."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            profile = Profile(name="Film/Animation", description="Test", settings={})

            dist_dir = get_distorted_dir(workdir, profile)

            assert dist_dir == workdir / "distorted" / "profile_Film_Animation"
            assert dist_dir.exists()

    def test_sanitizes_profile_name_with_backslash(self):
        """Test replaces backslashes in profile name with underscores."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            profile = Profile(name="Film\\Animation", description="Test", settings={})

            dist_dir = get_distorted_dir(workdir, profile)

            # Backslash should be replaced with underscore
            assert "profile_Film_Animation" in str(dist_dir)
            assert dist_dir.exists()

    def test_returns_existing_distorted_directory(self):
        """Test returns existing distorted directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            profile = Profile(name="TestProfile", description="Test", settings={})
            expected_dir = workdir / "distorted" / "profile_TestProfile"
            expected_dir.mkdir(parents=True, exist_ok=True)

            dist_dir = get_distorted_dir(workdir, profile)

            assert dist_dir == expected_dir
            assert dist_dir.exists()


class TestGetVmafDir:
    """Tests for VMAF directory path management."""

    def test_creates_vmaf_directory_for_profile(self):
        """Test creates VMAF directory with profile name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            profile = Profile(name="TestProfile", description="Test", settings={})

            vmaf_dir = get_vmaf_dir(workdir, profile)

            assert vmaf_dir.exists()
            assert vmaf_dir.is_dir()
            assert vmaf_dir == workdir / "vmaf" / "TestProfile_profile"

    def test_sanitizes_profile_name_with_spaces(self):
        """Test replaces spaces in profile name with underscores."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            profile = Profile(name="Film High Quality", description="Test", settings={})

            vmaf_dir = get_vmaf_dir(workdir, profile)

            assert vmaf_dir == workdir / "vmaf" / "Film_High_Quality_profile"
            assert vmaf_dir.exists()

    def test_sanitizes_profile_name_with_forward_slash(self):
        """Test replaces forward slashes in profile name with underscores."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            profile = Profile(name="Film/Animation", description="Test", settings={})

            vmaf_dir = get_vmaf_dir(workdir, profile)

            assert vmaf_dir == workdir / "vmaf" / "Film_Animation_profile"
            assert vmaf_dir.exists()

    def test_sanitizes_profile_name_with_backslash(self):
        """Test replaces backslashes in profile name with underscores."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            profile = Profile(name="Test\\Profile", description="Test", settings={})

            vmaf_dir = get_vmaf_dir(workdir, profile)

            assert "Test_Profile_profile" in str(vmaf_dir)
            assert vmaf_dir.exists()

    def test_returns_existing_vmaf_directory(self):
        """Test returns existing VMAF directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            profile = Profile(name="TestProfile", description="Test", settings={})
            expected_dir = workdir / "vmaf" / "TestProfile_profile"
            expected_dir.mkdir(parents=True, exist_ok=True)

            vmaf_dir = get_vmaf_dir(workdir, profile)

            assert vmaf_dir == expected_dir
            assert vmaf_dir.exists()


class TestGetSsim2Dir:
    """Tests for SSIMULACRA2 directory path management."""

    def test_creates_ssim2_directory_for_profile(self):
        """Test creates SSIMULACRA2 directory with profile name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            profile = Profile(name="TestProfile", description="Test", settings={})

            ssim2_dir = get_ssim2_dir(workdir, profile)

            assert ssim2_dir.exists()
            assert ssim2_dir.is_dir()
            assert ssim2_dir == workdir / "ssimulacra2" / "TestProfile_profile"

    def test_sanitizes_profile_name_with_spaces(self):
        """Test replaces spaces in profile name with underscores."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            profile = Profile(name="Animation Ultra", description="Test", settings={})

            ssim2_dir = get_ssim2_dir(workdir, profile)

            assert ssim2_dir == workdir / "ssimulacra2" / "Animation_Ultra_profile"
            assert ssim2_dir.exists()

    def test_sanitizes_profile_name_with_forward_slash(self):
        """Test replaces forward slashes in profile name with underscores."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            profile = Profile(name="CGI/Anime", description="Test", settings={})

            ssim2_dir = get_ssim2_dir(workdir, profile)

            assert ssim2_dir == workdir / "ssimulacra2" / "CGI_Anime_profile"
            assert ssim2_dir.exists()

    def test_sanitizes_profile_name_with_backslash(self):
        """Test replaces backslashes in profile name with underscores."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            profile = Profile(name="Test\\Profile", description="Test", settings={})

            ssim2_dir = get_ssim2_dir(workdir, profile)

            assert "Test_Profile_profile" in str(ssim2_dir)
            assert ssim2_dir.exists()

    def test_returns_existing_ssim2_directory(self):
        """Test returns existing SSIMULACRA2 directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            profile = Profile(name="TestProfile", description="Test", settings={})
            expected_dir = workdir / "ssimulacra2" / "TestProfile_profile"
            expected_dir.mkdir(parents=True, exist_ok=True)

            ssim2_dir = get_ssim2_dir(workdir, profile)

            assert ssim2_dir == expected_dir
            assert ssim2_dir.exists()


class TestPathConsistency:
    """Tests for consistent behavior across all path functions."""

    def test_all_functions_create_directories(self):
        """Test all functions create their directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            profile = Profile(name="Test", description="Test", settings={})

            ref_dir = get_reference_dir(workdir)
            dist_dir = get_distorted_dir(workdir, profile)
            vmaf_dir = get_vmaf_dir(workdir, profile)
            ssim2_dir = get_ssim2_dir(workdir, profile)

            assert ref_dir.exists()
            assert dist_dir.exists()
            assert vmaf_dir.exists()
            assert ssim2_dir.exists()

    def test_all_functions_handle_same_profile_consistently(self):
        """Test all functions handle the same profile consistently."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            profile = Profile(name="Film/High Quality", description="Test", settings={})

            dist_dir = get_distorted_dir(workdir, profile)
            vmaf_dir = get_vmaf_dir(workdir, profile)
            ssim2_dir = get_ssim2_dir(workdir, profile)

            # All should sanitize the profile name the same way
            assert "Film_High_Quality" in str(dist_dir)
            assert "Film_High_Quality" in str(vmaf_dir)
            assert "Film_High_Quality" in str(ssim2_dir)
