"""Tests for vmaf_assessment build_vmaf_filter with GPU/CPU branching."""

from __future__ import annotations

from videotuner.vmaf_assessment import build_vmaf_filter


class TestBuildVmafFilter:
    """Tests for build_vmaf_filter GPU/CPU branching."""

    def test_tonemap_gpu_uses_libplacebo(self) -> None:
        result = build_vmaf_filter(
            ref_needs_tonemap=True,
            _dis_needs_tonemap=True,
            use_gpu=True,
        )
        assert "libplacebo=" in result

    def test_tonemap_cpu_uses_zscale(self) -> None:
        result = build_vmaf_filter(
            ref_needs_tonemap=True,
            _dis_needs_tonemap=True,
            use_gpu=False,
        )
        assert "zscale=" in result
        assert "tonemap=hable" in result

    def test_tonemap_cpu_no_libplacebo(self) -> None:
        result = build_vmaf_filter(
            ref_needs_tonemap=True,
            _dis_needs_tonemap=True,
            use_gpu=False,
        )
        assert "libplacebo" not in result

    def test_tonemap_off_ignores_gpu(self) -> None:
        result = build_vmaf_filter(
            ref_needs_tonemap=True,
            _dis_needs_tonemap=True,
            tonemap_policy="off",
            use_gpu=True,
        )
        assert "libplacebo" not in result
        assert "scale=" in result

    def test_no_tonemap_uses_scale(self) -> None:
        result = build_vmaf_filter(
            ref_needs_tonemap=False,
            _dis_needs_tonemap=False,
            use_gpu=True,
        )
        assert "scale=" in result
        assert "libplacebo" not in result

    def test_default_use_gpu_true(self) -> None:
        result = build_vmaf_filter(
            ref_needs_tonemap=True,
            _dis_needs_tonemap=True,
        )
        # Default use_gpu=True should produce libplacebo
        assert "libplacebo=" in result

    def test_filter_graph_structure(self) -> None:
        result = build_vmaf_filter(
            ref_needs_tonemap=True,
            _dis_needs_tonemap=True,
            use_gpu=True,
        )
        assert "[0:v]" in result
        assert "[1:v]" in result
        assert "[ref]" in result
        assert "[dis]" in result
        assert "[dis][ref]" in result

    def test_tonemap_force_policy(self) -> None:
        result = build_vmaf_filter(
            ref_needs_tonemap=False,
            _dis_needs_tonemap=False,
            tonemap_policy="force",
            use_gpu=True,
        )
        assert "libplacebo=" in result

    def test_tonemap_force_cpu(self) -> None:
        result = build_vmaf_filter(
            ref_needs_tonemap=False,
            _dis_needs_tonemap=False,
            tonemap_policy="force",
            use_gpu=False,
        )
        assert "tonemap=hable" in result
        assert "libplacebo" not in result

    def test_pixel_format_8bit(self) -> None:
        result = build_vmaf_filter(
            ref_needs_tonemap=False,
            _dis_needs_tonemap=False,
            dis_bit_depth=8,
        )
        assert "format=yuv420p," in result

    def test_pixel_format_10bit(self) -> None:
        result = build_vmaf_filter(
            ref_needs_tonemap=False,
            _dis_needs_tonemap=False,
            dis_bit_depth=10,
        )
        assert "format=yuv420p10le," in result
