"""Tests for utils module."""

from __future__ import annotations

from videotuner.utils import parse_master_display_metadata


class TestParseMasterDisplayMetadata:
    """Tests for master display metadata parsing."""

    def test_display_p3_parses_correctly(self):
        """Test Display P3 color space parsing."""
        primaries = "Display P3"
        luminance = "min: 0.0050 cd/m2, max: 1000 cd/m2"

        result = parse_master_display_metadata(primaries, luminance)

        assert result is not None
        # Display P3 coordinates
        assert "G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)" in result
        # Luminance: max=1000 -> 10000000, min=0.0050 -> 50
        assert result.endswith("L(10000000,50)")

    def test_bt2020_parses_correctly(self):
        """Test BT.2020 color space parsing."""
        primaries = "BT.2020"
        luminance = "min: 0.0001 cd/m2, max: 4000 cd/m2"

        result = parse_master_display_metadata(primaries, luminance)

        assert result is not None
        # BT.2020 coordinates
        assert "G(8500,39850)B(6550,2300)R(35400,14600)WP(15635,16450)" in result
        # Luminance: max=4000 -> 40000000, min=0.0001 -> 1
        assert result.endswith("L(40000000,1)")

    def test_dci_p3_parses_correctly(self):
        """Test DCI P3 color space parsing."""
        primaries = "DCI P3"
        luminance = "min: 0.0050 cd/m2, max: 1000 cd/m2"

        result = parse_master_display_metadata(primaries, luminance)

        assert result is not None
        # DCI P3 has same primaries as Display P3 but different white point
        assert "G(13250,34500)B(7500,3000)R(34000,16000)WP(15700,17550)" in result
        assert result.endswith("L(10000000,50)")

    def test_unknown_color_space_returns_none(self):
        """Test that unknown color space returns None."""
        primaries = "Unknown Color Space"
        luminance = "min: 0.0050 cd/m2, max: 1000 cd/m2"

        result = parse_master_display_metadata(primaries, luminance)

        assert result is None

    def test_invalid_luminance_format_returns_none(self):
        """Test that invalid luminance format returns None."""
        primaries = "Display P3"
        luminance = "invalid format"

        result = parse_master_display_metadata(primaries, luminance)

        assert result is None

    def test_luminance_unit_conversion(self):
        """Test that luminance values are converted to x265 units (multiply by 10000)."""
        primaries = "Display P3"
        luminance = "min: 0.0100 cd/m2, max: 500 cd/m2"

        result = parse_master_display_metadata(primaries, luminance)

        assert result is not None
        # min: 0.01 * 10000 = 100
        # max: 500 * 10000 = 5000000
        assert result.endswith("L(5000000,100)")

    def test_zero_min_luminance(self):
        """Test handling of zero minimum luminance."""
        primaries = "BT.2020"
        luminance = "min: 0.0000 cd/m2, max: 1000 cd/m2"

        result = parse_master_display_metadata(primaries, luminance)

        assert result is not None
        assert result.endswith("L(10000000,0)")

    def test_high_max_luminance(self):
        """Test handling of high maximum luminance (10000 nits)."""
        primaries = "BT.2020"
        luminance = "min: 0.0050 cd/m2, max: 10000 cd/m2"

        result = parse_master_display_metadata(primaries, luminance)

        assert result is not None
        assert result.endswith("L(100000000,50)")
