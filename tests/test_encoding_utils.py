"""Tests for encoding utilities module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from videotuner.encoding_utils import (
    HDR_TRANSFER_CHARACTERISTICS,
    create_temp_encode_paths,
    get_vapoursynth_portable_dir,
    get_x265_bin,
    is_hdr_video,
    resolve_absolute_path,
    calculate_usable_frames,
    VapourSynthEnv,
    EncoderPaths,
)


class TestHDRDetection:
    """Tests for HDR transfer characteristic detection."""

    def test_pq_transfer_is_hdr(self):
        """Test PQ transfer is detected as HDR."""
        assert is_hdr_video("pq") is True

    def test_smpte2084_transfer_is_hdr(self):
        """Test SMPTE 2084 transfer is detected as HDR."""
        assert is_hdr_video("smpte2084") is True

    def test_smpte2084_with_spaces_is_hdr(self):
        """Test SMPTE 2084 with spaces is detected as HDR."""
        assert is_hdr_video("smpte 2084") is True

    def test_hlg_transfer_is_hdr(self):
        """Test HLG transfer is detected as HDR."""
        assert is_hdr_video("hlg") is True

    def test_arib_std_b67_transfer_is_hdr(self):
        """Test ARIB STD-B67 transfer is detected as HDR."""
        assert is_hdr_video("arib-std-b67") is True

    def test_arib_std_b67_with_space_is_hdr(self):
        """Test ARIB STD-B67 with space is detected as HDR."""
        assert is_hdr_video("arib std-b67") is True

    def test_bt709_transfer_is_not_hdr(self):
        """Test BT.709 transfer is not HDR."""
        assert is_hdr_video("bt709") is False

    def test_none_transfer_is_not_hdr(self):
        """Test None transfer is not HDR."""
        assert is_hdr_video(None) is False

    def test_empty_string_is_not_hdr(self):
        """Test empty string is not HDR."""
        assert is_hdr_video("") is False

    def test_unknown_transfer_is_not_hdr(self):
        """Test unknown transfer is not HDR."""
        assert is_hdr_video("unknown") is False

    def test_case_insensitive_pq(self):
        """Test PQ detection is case insensitive."""
        assert is_hdr_video("PQ") is True
        assert is_hdr_video("Pq") is True
        assert is_hdr_video("pQ") is True

    def test_case_insensitive_hlg(self):
        """Test HLG detection is case insensitive."""
        assert is_hdr_video("HLG") is True
        assert is_hdr_video("Hlg") is True

    def test_hdr_transfer_characteristics_constant(self):
        """Test HDR_TRANSFER_CHARACTERISTICS contains expected values."""
        assert "pq" in HDR_TRANSFER_CHARACTERISTICS
        assert "smpte2084" in HDR_TRANSFER_CHARACTERISTICS
        assert "smpte 2084" in HDR_TRANSFER_CHARACTERISTICS
        assert "hlg" in HDR_TRANSFER_CHARACTERISTICS
        assert "arib-std-b67" in HDR_TRANSFER_CHARACTERISTICS
        assert "arib std-b67" in HDR_TRANSFER_CHARACTERISTICS


class TestGetX265Bin:
    """Tests for x265 binary path resolution."""

    def test_returns_relative_path_when_no_cwd(self):
        """Test returns relative path when cwd is None."""
        result = get_x265_bin(cwd=None)
        assert result == Path("tools") / "x265.exe"
        assert not result.is_absolute()

    def test_returns_absolute_path_when_cwd_provided(self):
        """Test returns absolute path when cwd is provided."""
        cwd = Path("C:/test/dir")
        result = get_x265_bin(cwd=cwd)
        assert result == Path("C:/test/dir") / "tools" / "x265.exe"

    def test_preserves_cwd_path_type(self):
        """Test preserves the type of cwd (Path or string-like)."""
        cwd = Path("/home/user/project")
        result = get_x265_bin(cwd=cwd)
        assert isinstance(result, Path)
        # Check the path ends with the expected components (platform-agnostic)
        assert result.name == "x265.exe"
        assert result.parent.name == "tools"


class TestGetVapourSynthPortableDir:
    """Tests for VapourSynth portable directory path resolution."""

    def test_returns_relative_path_when_no_cwd(self):
        """Test returns relative path when cwd is None."""
        result = get_vapoursynth_portable_dir(cwd=None)
        assert result == Path("vapoursynth-portable")
        assert not result.is_absolute()

    def test_returns_absolute_path_when_cwd_provided(self):
        """Test returns absolute path when cwd is provided."""
        cwd = Path("C:/test/dir")
        result = get_vapoursynth_portable_dir(cwd=cwd)
        assert result == Path("C:/test/dir") / "vapoursynth-portable"

    def test_preserves_cwd_path_type(self):
        """Test preserves the type of cwd."""
        cwd = Path("/opt/videotuner")
        result = get_vapoursynth_portable_dir(cwd=cwd)
        assert isinstance(result, Path)
        # Check the path ends with the expected component (platform-agnostic)
        assert result.name == "vapoursynth-portable"


class TestResolveAbsolutePath:
    """Tests for absolute path resolution."""

    def test_returns_absolute_path_unchanged(self):
        """Test absolute paths are returned unchanged."""
        # Use a fully qualified path that works on both Windows and Unix
        abs_path = Path("C:/absolute/path/to/file.mkv").resolve()
        result = resolve_absolute_path(abs_path, cwd=None)
        assert result == abs_path
        assert result.is_absolute()

    def test_resolves_relative_path_with_cwd(self):
        """Test relative paths are resolved relative to cwd."""
        rel_path = Path("videos/input.mkv")
        cwd = Path("/home/user/project")
        result = resolve_absolute_path(rel_path, cwd=cwd)
        assert result == Path("/home/user/project") / "videos" / "input.mkv"

    def test_resolves_relative_path_without_cwd(self):
        """Test relative paths are resolved without cwd (uses resolve())."""
        rel_path = Path("test.mkv")
        result = resolve_absolute_path(rel_path, cwd=None)
        # Should call resolve() which makes it absolute
        assert result.is_absolute()

    def test_handles_windows_absolute_path(self):
        """Test Windows absolute paths are returned unchanged."""
        win_path = Path("C:/Users/test/video.mkv")
        result = resolve_absolute_path(win_path, cwd=None)
        assert result == win_path


class TestCreateTempEncodePaths:
    """Tests for temporary encode path creation."""

    def test_creates_paths_in_system_temp_by_default(self):
        """Test creates paths in system temp when temp_dir is None."""
        vpy_path, hevc_path = create_temp_encode_paths(temp_dir=None)

        # Should be in system temp directory
        assert vpy_path.suffix == ".vpy"
        assert hevc_path.suffix == ".hevc"
        assert vpy_path.is_absolute()
        assert hevc_path.is_absolute()

        # Clean up
        try:
            vpy_path.unlink(missing_ok=True)
            hevc_path.unlink(missing_ok=True)
        except Exception:
            pass

    def test_creates_paths_with_name_in_system_temp(self):
        """Test name is used as prefix when temp_dir is None."""
        vpy_path, hevc_path = create_temp_encode_paths(temp_dir=None, name="mytest")

        # When using system temp, name is used as prefix, but files still use random names
        # Just verify they have the right extensions and the prefix is included
        assert vpy_path.suffix == ".vpy"
        assert hevc_path.suffix == ".hevc"
        assert "mytest" in vpy_path.name

        # Clean up
        try:
            vpy_path.unlink(missing_ok=True)
            hevc_path.unlink(missing_ok=True)
        except Exception:
            pass

    def test_creates_paths_in_custom_temp_dir(self):
        """Test creates paths in custom temp directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir)
            vpy_path, hevc_path = create_temp_encode_paths(temp_dir=temp_dir)

            assert vpy_path.parent == temp_dir
            assert hevc_path.parent == temp_dir
            assert vpy_path.name == "encode.vpy"
            assert hevc_path.name == "encode.hevc"

    def test_creates_paths_in_custom_temp_dir_with_custom_name(self):
        """Test creates paths in custom directory with custom name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir)
            vpy_path, hevc_path = create_temp_encode_paths(
                temp_dir=temp_dir, name="iteration_5"
            )

            assert vpy_path.parent == temp_dir
            assert hevc_path.parent == temp_dir
            assert vpy_path.name == "iteration_5.vpy"
            assert hevc_path.name == "iteration_5.hevc"

    def test_creates_temp_dir_if_not_exists(self):
        """Test creates temp directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir) / "subdir" / "nested"
            assert not temp_dir.exists()

            vpy_path, hevc_path = create_temp_encode_paths(temp_dir=temp_dir)

            assert temp_dir.exists()
            assert vpy_path.parent == temp_dir
            assert hevc_path.parent == temp_dir

    def test_returns_tuple_of_paths(self):
        """Test returns tuple of (vpy_path, hevc_path)."""
        result = create_temp_encode_paths(temp_dir=None)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], Path)
        assert isinstance(result[1], Path)

        # Clean up
        try:
            result[0].unlink(missing_ok=True)
            result[1].unlink(missing_ok=True)
        except Exception:
            pass


class TestVapourSynthEnv:
    """Tests for VapourSynthEnv dataclass."""

    def test_from_cwd_with_none(self):
        """Test from_cwd returns relative paths when cwd is None."""
        env = VapourSynthEnv.from_cwd(None)
        assert env.vs_dir == Path("vapoursynth-portable")
        assert env.vsscript_dll == Path("vapoursynth-portable") / "VSScript.dll"
        assert (
            env.ffms2_dll == Path("vapoursynth-portable") / "vs-plugins" / "ffms2.dll"
        )
        assert env.vs_plugin_dir == Path("vapoursynth-portable") / "vs-plugins"

    def test_from_cwd_with_path(self):
        """Test from_cwd returns paths relative to cwd."""
        cwd = Path("C:/project")
        env = VapourSynthEnv.from_cwd(cwd)
        assert env.vs_dir == Path("C:/project/vapoursynth-portable")
        assert env.vsscript_dll == Path("C:/project/vapoursynth-portable/VSScript.dll")
        assert env.ffms2_dll == Path(
            "C:/project/vapoursynth-portable/vs-plugins/ffms2.dll"
        )
        assert env.vs_plugin_dir == Path("C:/project/vapoursynth-portable/vs-plugins")

    def test_from_args_uses_provided_paths(self):
        """Test from_args uses CLI-provided paths."""
        vs_dir = Path("C:/custom/vs")
        vs_plugin_dir = Path("C:/custom/plugins")
        env = VapourSynthEnv.from_args(vs_dir, vs_plugin_dir, Path("C:/repo"))
        assert env.vs_dir == Path("C:/custom/vs")
        assert env.vs_plugin_dir == Path("C:/custom/plugins")

    def test_from_args_falls_back_to_repo_root(self):
        """Test from_args falls back to repo root when args are None."""
        env = VapourSynthEnv.from_args(None, None, Path("C:/repo"))
        assert env.vs_dir == Path("C:/repo/vapoursynth-portable")
        assert env.vs_plugin_dir == Path("C:/repo/vapoursynth-portable/vs-plugins")

    def test_validate_raises_when_vsscript_missing(self):
        """Test validate raises FileNotFoundError when VSScript.dll is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            env = VapourSynthEnv.from_cwd(cwd)

            with pytest.raises(FileNotFoundError) as exc_info:
                env.validate()
            assert "VSScript.dll" in str(exc_info.value)

    def test_validate_raises_when_ffms2_missing(self):
        """Test validate raises FileNotFoundError when ffms2.dll is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            vs_dir = cwd / "vapoursynth-portable"
            vs_dir.mkdir(parents=True)
            (vs_dir / "VSScript.dll").touch()

            env = VapourSynthEnv.from_cwd(cwd)

            with pytest.raises(FileNotFoundError) as exc_info:
                env.validate()
            assert "ffms2.dll" in str(exc_info.value)

    def test_validate_passes_when_files_exist(self):
        """Test validate passes when all required files exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            vs_dir = cwd / "vapoursynth-portable"
            plugins_dir = vs_dir / "vs-plugins"
            plugins_dir.mkdir(parents=True)
            (vs_dir / "VSScript.dll").touch()
            (plugins_dir / "ffms2.dll").touch()

            env = VapourSynthEnv.from_cwd(cwd)
            # Should not raise
            env.validate()

    def test_build_env_sets_vapoursynth_portable(self):
        """Test build_env sets VAPOURSYNTH_PORTABLE environment variable."""
        vs_env = VapourSynthEnv.from_cwd(Path("C:/project"))
        env = vs_env.build_env({})
        assert "VAPOURSYNTH_PORTABLE" in env
        assert env["VAPOURSYNTH_PORTABLE"] == str(
            Path("C:/project/vapoursynth-portable")
        )

    def test_build_env_prepends_to_path(self):
        """Test build_env prepends VS directory to PATH."""
        vs_env = VapourSynthEnv.from_cwd(Path("C:/project"))
        base_env = {"PATH": "/usr/bin", "HOME": "/home/user"}
        env = vs_env.build_env(base_env)
        assert env["PATH"].startswith(str(Path("C:/project/vapoursynth-portable")))
        assert "/usr/bin" in env["PATH"]
        assert env["HOME"] == "/home/user"

    def test_frozen_dataclass(self):
        """Test VapourSynthEnv is frozen (immutable)."""
        env = VapourSynthEnv.from_cwd(None)
        with pytest.raises(AttributeError):
            setattr(env, "vs_dir", Path("new/path"))


class TestEncoderPaths:
    """Tests for EncoderPaths dataclass."""

    def test_from_cwd_with_none(self):
        """Test from_cwd returns relative paths when cwd is None."""
        paths = EncoderPaths.from_cwd(None)
        assert paths.x265_bin == Path("tools") / "x265.exe"
        assert paths.vs_env.vs_dir == Path("vapoursynth-portable")

    def test_from_cwd_with_path(self):
        """Test from_cwd returns paths relative to cwd."""
        cwd = Path("C:/project")
        paths = EncoderPaths.from_cwd(cwd)
        assert paths.x265_bin == Path("C:/project/tools/x265.exe")
        assert paths.vs_env.vs_dir == Path("C:/project/vapoursynth-portable")

    def test_validate_raises_when_x265_missing(self):
        """Test validate raises FileNotFoundError when x265.exe is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            paths = EncoderPaths.from_cwd(cwd)

            with pytest.raises(FileNotFoundError) as exc_info:
                paths.validate()
            assert "x265" in str(exc_info.value)

    def test_validate_raises_when_vapoursynth_missing(self):
        """Test validate raises FileNotFoundError when VapourSynth is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            tools_dir = cwd / "tools"
            tools_dir.mkdir(parents=True)
            (tools_dir / "x265.exe").touch()

            paths = EncoderPaths.from_cwd(cwd)

            with pytest.raises(FileNotFoundError) as exc_info:
                paths.validate()
            # Should fail on VapourSynth validation
            assert "VSScript.dll" in str(exc_info.value)

    def test_validate_passes_when_all_files_exist(self):
        """Test validate passes when all required files exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            # Create x265
            tools_dir = cwd / "tools"
            tools_dir.mkdir(parents=True)
            (tools_dir / "x265.exe").touch()
            # Create VapourSynth
            vs_dir = cwd / "vapoursynth-portable"
            plugins_dir = vs_dir / "vs-plugins"
            plugins_dir.mkdir(parents=True)
            (vs_dir / "VSScript.dll").touch()
            (plugins_dir / "ffms2.dll").touch()

            paths = EncoderPaths.from_cwd(cwd)
            # Should not raise
            paths.validate()

    def test_nested_vs_env_accessible(self):
        """Test that VapourSynthEnv is accessible via vs_env."""
        paths = EncoderPaths.from_cwd(Path("C:/project"))
        assert isinstance(paths.vs_env, VapourSynthEnv)
        assert paths.vs_env.vsscript_dll == Path(
            "C:/project/vapoursynth-portable/VSScript.dll"
        )

    def test_frozen_dataclass(self):
        """Test EncoderPaths is frozen (immutable)."""
        paths = EncoderPaths.from_cwd(None)
        with pytest.raises(AttributeError):
            setattr(paths, "x265_bin", Path("new/path"))


class TestCalculateUsableFrames:
    """Tests for calculate_usable_frames function."""

    def test_calculates_usable_frames(self):
        """Test basic usable frames calculation."""
        result = calculate_usable_frames(
            total_frames=1000,
            guard_start_frames=100,
            guard_end_frames=100,
        )
        assert result == 800

    def test_no_guards(self):
        """Test with zero guard frames."""
        result = calculate_usable_frames(
            total_frames=1000,
            guard_start_frames=0,
            guard_end_frames=0,
        )
        assert result == 1000

    def test_asymmetric_guards(self):
        """Test with different start and end guards."""
        result = calculate_usable_frames(
            total_frames=1000,
            guard_start_frames=50,
            guard_end_frames=150,
        )
        assert result == 800

    def test_raises_when_total_frames_zero(self):
        """Test raises ValueError when total_frames is zero."""
        with pytest.raises(ValueError) as exc_info:
            _ = calculate_usable_frames(
                total_frames=0,
                guard_start_frames=0,
                guard_end_frames=0,
            )
        assert "total_frames must be >= 1" in str(exc_info.value)

    def test_raises_when_total_frames_negative(self):
        """Test raises ValueError when total_frames is negative."""
        with pytest.raises(ValueError) as exc_info:
            _ = calculate_usable_frames(
                total_frames=-10,
                guard_start_frames=0,
                guard_end_frames=0,
            )
        assert "total_frames must be >= 1" in str(exc_info.value)

    def test_raises_when_guard_start_negative(self):
        """Test raises ValueError when guard_start_frames is negative."""
        with pytest.raises(ValueError) as exc_info:
            _ = calculate_usable_frames(
                total_frames=1000,
                guard_start_frames=-10,
                guard_end_frames=0,
            )
        assert "guard_start_frames must be >= 0" in str(exc_info.value)

    def test_raises_when_guard_end_negative(self):
        """Test raises ValueError when guard_end_frames is negative."""
        with pytest.raises(ValueError) as exc_info:
            _ = calculate_usable_frames(
                total_frames=1000,
                guard_start_frames=0,
                guard_end_frames=-10,
            )
        assert "guard_end_frames must be >= 0" in str(exc_info.value)

    def test_raises_when_guards_exceed_total(self):
        """Test raises ValueError when guards exceed total frames."""
        with pytest.raises(ValueError) as exc_info:
            _ = calculate_usable_frames(
                total_frames=100,
                guard_start_frames=60,
                guard_end_frames=60,
            )
        assert "No usable frames after guards" in str(exc_info.value)

    def test_raises_when_guards_equal_total(self):
        """Test raises ValueError when guards equal total frames."""
        with pytest.raises(ValueError) as exc_info:
            _ = calculate_usable_frames(
                total_frames=100,
                guard_start_frames=50,
                guard_end_frames=50,
            )
        assert "No usable frames after guards" in str(exc_info.value)

    def test_minimum_usable_frames(self):
        """Test with guards leaving exactly 1 usable frame."""
        result = calculate_usable_frames(
            total_frames=101,
            guard_start_frames=50,
            guard_end_frames=50,
        )
        assert result == 1
