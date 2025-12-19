"""Tests for create_encodes module."""

from __future__ import annotations

import pytest

from videotuner.create_encodes import CropConfig
from videotuner.encoding_utils import CropValues


class TestCropValues:
    """Tests for CropValues dataclass."""

    def test_create_crop_values(self):
        """Test creating CropValues with all fields."""
        crop = CropValues(left=10, right=20, top=5, bottom=15)
        assert crop.left == 10
        assert crop.right == 20
        assert crop.top == 5
        assert crop.bottom == 15

    def test_frozen_dataclass(self):
        """Test CropValues is frozen (immutable)."""
        crop = CropValues(left=10, right=20, top=5, bottom=15)
        with pytest.raises(AttributeError):
            setattr(crop, "left", 100)


class TestCropConfig:
    """Tests for CropConfig dataclass."""

    def test_default_values(self):
        """Test CropConfig default values."""
        config = CropConfig()
        assert config.enabled is False
        assert config.values is None

    def test_disabled_factory(self):
        """Test CropConfig.disabled() factory method."""
        config = CropConfig.disabled()
        assert config.enabled is False
        assert config.values is None

    def test_with_values_factory(self):
        """Test CropConfig.with_values() factory method."""
        crop = CropValues(left=10, right=20, top=5, bottom=15)
        config = CropConfig.with_values(crop)
        assert config.enabled is True
        assert config.values is crop
        assert config.values is not None and config.values.left == 10

    def test_auto_factory(self):
        """Test CropConfig.auto() factory method."""
        config = CropConfig.auto()
        assert config.enabled is True
        assert config.values is None

    def test_frozen_dataclass(self):
        """Test CropConfig is frozen (immutable)."""
        config = CropConfig()
        with pytest.raises(AttributeError):
            setattr(config, "enabled", True)

    def test_explicit_enabled_with_values(self):
        """Test creating CropConfig with explicit enabled and values."""
        crop = CropValues(left=0, right=0, top=100, bottom=100)
        config = CropConfig(enabled=True, values=crop)
        assert config.enabled is True
        assert config.values is not None
        assert config.values.top == 100
        assert config.values.bottom == 100

    def test_disabled_with_values_ignored(self):
        """Test that values can be set even when disabled (not enforced)."""
        crop = CropValues(left=10, right=10, top=10, bottom=10)
        # This is technically valid even though semantically odd
        config = CropConfig(enabled=False, values=crop)
        assert config.enabled is False
        assert config.values is crop
