"""Tests for profile management functionality."""

import pytest

from videotuner.encoder_type import EncoderType
from videotuner.profiles import (
    Profile,
    ProfileError,
    get_all_groups,
    get_profiles_by_groups,
    validate_groups_exist,
)


class TestProfileGroups:
    """Tests for profile group functionality."""

    def _create_test_profiles(self) -> dict[str, Profile]:
        """Create a set of test profiles with groups."""
        return {
            "Film": Profile(
                name="Film",
                description="Live action film",
                settings={"preset": "slow"},
                encoder=EncoderType.X265,
                groups=["Default", "live-action"],
            ),
            "Grain": Profile(
                name="Grain",
                description="Grainy content",
                settings={"preset": "slow"},
                encoder=EncoderType.X265,
                groups=["Default", "live-action"],
            ),
            "CGI": Profile(
                name="CGI",
                description="CGI movies",
                settings={"preset": "slow"},
                encoder=EncoderType.X265,
                groups=["Default", "animation"],
            ),
            "Anime": Profile(
                name="Anime",
                description="Anime content",
                settings={"preset": "slow"},
                encoder=EncoderType.X265,
                groups=["animation"],
            ),
            "Ungrouped": Profile(
                name="Ungrouped",
                description="No groups",
                settings={"preset": "slow"},
                encoder=EncoderType.X265,
                groups=[],
            ),
        }

    def test_get_all_groups_returns_unique_groups(self):
        """Test that get_all_groups returns all unique group names."""
        profiles = self._create_test_profiles()
        groups = get_all_groups(profiles)

        assert groups == {"Default", "live-action", "animation"}

    def test_get_all_groups_empty_when_no_groups(self):
        """Test get_all_groups with profiles that have no groups."""
        profiles = {
            "Test": Profile(
                name="Test",
                description="Test",
                settings={"preset": "slow"},
                encoder=EncoderType.X265,
                groups=[],
            )
        }
        groups = get_all_groups(profiles)

        assert groups == set()

    def test_validate_groups_exist_passes_for_valid_groups(self):
        """Test that validate_groups_exist passes for existing groups."""
        profiles = self._create_test_profiles()

        # Should not raise
        validate_groups_exist(profiles, ["Default"])
        validate_groups_exist(profiles, ["Default", "animation"])
        validate_groups_exist(profiles, ["live-action", "animation"])

    def test_validate_groups_exist_raises_for_invalid_group(self):
        """Test that validate_groups_exist raises ProfileError for non-existent groups."""
        profiles = self._create_test_profiles()

        with pytest.raises(ProfileError) as exc_info:
            validate_groups_exist(profiles, ["nonexistent"])

        assert "Profile group 'nonexistent' not found" in str(exc_info.value)
        assert "Default" in str(exc_info.value)  # Should list available groups

    def test_get_profiles_by_groups_returns_all_when_no_filter(self):
        """Test that get_profiles_by_groups returns all profiles when no groups specified."""
        profiles = self._create_test_profiles()

        result = get_profiles_by_groups(profiles, None)
        assert len(result) == 5

        result = get_profiles_by_groups(profiles, [])
        assert len(result) == 5

    def test_get_profiles_by_groups_filters_by_single_group(self):
        """Test filtering by a single group."""
        profiles = self._create_test_profiles()

        result = get_profiles_by_groups(profiles, ["animation"])
        names = [p.name for p in result]

        assert len(result) == 2
        assert "CGI" in names
        assert "Anime" in names

    def test_get_profiles_by_groups_filters_by_multiple_groups(self):
        """Test filtering by multiple groups (union)."""
        profiles = self._create_test_profiles()

        result = get_profiles_by_groups(profiles, ["live-action", "animation"])
        names = [p.name for p in result]

        # Should include profiles from either group
        assert len(result) == 4
        assert "Film" in names
        assert "Grain" in names
        assert "CGI" in names
        assert "Anime" in names

    def test_get_profiles_by_groups_excludes_ungrouped(self):
        """Test that ungrouped profiles are excluded when filtering."""
        profiles = self._create_test_profiles()

        result = get_profiles_by_groups(profiles, ["Default"])
        names = [p.name for p in result]

        assert "Ungrouped" not in names

    def test_get_profiles_by_groups_preserves_order(self):
        """Test that profiles are returned in definition order."""
        profiles = self._create_test_profiles()

        result = get_profiles_by_groups(profiles, None)
        names = [p.name for p in result]

        # Should maintain dict insertion order
        assert names == ["Film", "Grain", "CGI", "Anime", "Ungrouped"]

    def test_profile_with_multiple_groups_returned_once(self):
        """Test that a profile in multiple groups is only returned once."""
        profiles = self._create_test_profiles()

        # Film is in both Default and live-action
        result = get_profiles_by_groups(profiles, ["Default", "live-action"])
        film_count = sum(1 for p in result if p.name == "Film")

        assert film_count == 1


class TestProfileClass:
    """Tests for Profile class with groups."""

    def test_profile_with_groups(self):
        """Test creating a profile with groups."""
        profile = Profile(
            name="Test",
            description="Test profile",
            settings={"preset": "slow"},
            encoder=EncoderType.X265,
            groups=["group1", "group2"],
        )

        assert profile.groups == ["group1", "group2"]

    def test_profile_without_groups(self):
        """Test creating a profile without groups defaults to empty list."""
        profile = Profile(
            name="Test",
            description="Test profile",
            settings={"preset": "slow"},
            encoder=EncoderType.X265,
        )

        assert profile.groups == []

    def test_profile_with_none_groups(self):
        """Test creating a profile with None groups defaults to empty list."""
        profile = Profile(
            name="Test",
            description="Test profile",
            settings={"preset": "slow"},
            encoder=EncoderType.X265,
            groups=None,
        )

        assert profile.groups == []

    def test_profile_is_preset_defaults_false(self):
        """Test creating a profile without is_preset defaults to False."""
        profile = Profile(
            name="Test",
            description="Test profile",
            settings={"preset": "slow"},
            encoder=EncoderType.X265,
        )

        assert profile.is_preset is False
        assert profile.display_label == "Profile"
        assert profile.display_name == "Test"

    def test_preset_display_properties(self):
        """Test that preset profiles have correct display properties."""
        profile = Profile(
            name="preset-slow",
            description="x265 slow preset",
            settings={"preset": "slow"},
            encoder=EncoderType.X265,
            is_preset=True,
        )

        assert profile.is_preset is True
        assert profile.display_label == "Preset"
        assert profile.display_name == "slow"

    def test_preset_without_prefix_keeps_full_name(self):
        """Test that preset without 'preset-' prefix keeps full name."""
        profile = Profile(
            name="custom-slow",
            description="Custom preset",
            settings={"preset": "slow"},
            encoder=EncoderType.X265,
            is_preset=True,
        )

        assert profile.display_label == "Preset"
        assert profile.display_name == "custom-slow"

    def test_profile_encoder_x265(self):
        """Test creating a profile with x265 encoder."""
        profile = Profile(
            name="Test",
            description="Test profile",
            settings={"preset": "slow"},
            encoder=EncoderType.X265,
        )

        assert profile.encoder == EncoderType.X265

    def test_profile_encoder_x264(self):
        """Test creating a profile with x264 encoder."""
        profile = Profile(
            name="Test",
            description="Test profile",
            settings={"preset": "slow"},
            encoder=EncoderType.X264,
        )

        assert profile.encoder == EncoderType.X264
