"""Profile management for x265 encoding settings.

This module handles loading and parsing x265 encoding profiles from YAML configuration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, cast

import yaml

from .utils import get_app_root

if TYPE_CHECKING:
    from .media import VideoFormat, VideoInfo

logger = logging.getLogger(__name__)


class ProfileError(Exception):
    """Exception raised for profile-related errors."""

    pass


class Profile:
    """Represents an x265 encoding profile."""

    def __init__(
        self,
        name: str,
        description: str,
        settings: dict[str, object],
        groups: list[str] | None = None,
        is_preset: bool = False,
    ) -> None:
        """Initialize a profile.

        Args:
            name: Profile name
            description: Profile description
            settings: Dictionary of x265 settings
            groups: List of group names this profile belongs to (optional)
            is_preset: True if this is a preset-based profile (not from YAML)
        """
        self.name: str = name
        self.description: str = description
        self.settings: dict[str, object] = settings
        self.groups: list[str] = groups if groups is not None else []
        self.is_preset: bool = is_preset

        # Validate required fields
        if not self.name:
            raise ProfileError("Profile name is required")

        # Validate settings
        self._validate_settings()

    def _validate_settings(self) -> None:
        """Validate profile settings for invalid parameters.

        Raises:
            ProfileError: If validation fails
        """
        from .encoder_params import VALID_PRESETS

        # Check for inverse boolean syntax (error - suggest fix)
        # Common x265 boolean parameters that users might try to set to false
        boolean_params = {
            "open-gop": "no-open-gop",
            "strong-intra-smoothing": "no-strong-intra-smoothing",
            "cutree": "no-cutree",
            "b-adapt": "no-b-adapt",
            "b-pyramid": "no-b-pyramid",
            "weightp": "no-weightp",
            "weightb": "no-weightb",
            "rect": "no-rect",
            "amp": "no-amp",
            "early-skip": "no-early-skip",
            "fast-intra": "no-fast-intra",
            "deblock": "no-deblock",
            "sao": "no-sao",
            "signhide": "no-signhide",
            "tskip": "no-tskip",
            "tskip-fast": "no-tskip-fast",
        }

        for positive_param, negative_param in boolean_params.items():
            if positive_param in self.settings:
                value = self.settings[positive_param]
                if isinstance(value, bool) and value is False:
                    raise ProfileError(
                        f"Profile '{self.name}': Invalid boolean syntax: '{positive_param}: false' should be '{negative_param}: true'"
                    )

        # Validate preset parameter
        if "preset" in self.settings:
            preset_value = str(self.settings["preset"])
            if preset_value not in VALID_PRESETS:
                raise ProfileError(
                    f"Profile '{self.name}': Invalid preset '{preset_value}'. Valid presets: {', '.join(VALID_PRESETS)}"
                )

        # Validate bitrate and pass parameters
        if "bitrate" in self.settings and "crf" in self.settings:
            raise ProfileError(
                f"Profile '{self.name}': 'bitrate' and 'crf' are mutually exclusive. "
                + "Remove 'crf' from profile settings (CRF is controlled by --crf-start-value CLI argument)."
            )

        # Validate bitrate value
        if "bitrate" in self.settings:
            bitrate_value = self.settings["bitrate"]
            if not isinstance(bitrate_value, (int, float)):
                raise ProfileError(
                    f"Profile '{self.name}': 'bitrate' must be a number, got {type(bitrate_value).__name__}"
                )
            if bitrate_value <= 0:
                raise ProfileError(
                    f"Profile '{self.name}': 'bitrate' must be positive, got {bitrate_value}"
                )

        # Validate pass parameter
        if "pass" in self.settings:
            pass_value = self.settings["pass"]

            # Validate pass is an integer
            if not isinstance(pass_value, int):
                raise ProfileError(
                    f"Profile '{self.name}': 'pass' must be an integer (1, 2, or 3), got {type(pass_value).__name__}"
                )

            # Validate pass value range
            if pass_value not in (1, 2, 3):
                raise ProfileError(
                    f"Profile '{self.name}': 'pass' must be 1, 2, or 3, got {pass_value}"
                )

            # Validate bitrate is present when pass is specified
            if "bitrate" not in self.settings:
                raise ProfileError(
                    f"Profile '{self.name}': 'pass' requires 'bitrate' to be specified"
                )

        # Validate multi-pass optimization parameters
        opt_analysis = self.settings.get("multi-pass-opt-analysis")
        opt_distortion = self.settings.get("multi-pass-opt-distortion")

        if opt_analysis is not None:
            if not isinstance(opt_analysis, bool):
                raise ProfileError(
                    f"Profile '{self.name}': 'multi-pass-opt-analysis' must be a boolean, got {type(opt_analysis).__name__}"
                )
            # Require multi-pass encoding (any pass value 1, 2, or 3)
            pass_value = self.settings.get("pass")
            if opt_analysis and pass_value is None:
                raise ProfileError(
                    f"Profile '{self.name}': 'multi-pass-opt-analysis' requires multi-pass encoding (pass: 1, 2, or 3)"
                )

        if opt_distortion is not None:
            if not isinstance(opt_distortion, bool):
                raise ProfileError(
                    f"Profile '{self.name}': 'multi-pass-opt-distortion' must be a boolean, got {type(opt_distortion).__name__}"
                )
            # Require multi-pass encoding (any pass value 1, 2, or 3)
            pass_value = self.settings.get("pass")
            if opt_distortion and pass_value is None:
                raise ProfileError(
                    f"Profile '{self.name}': 'multi-pass-opt-distortion' requires multi-pass encoding (pass: 1, 2, or 3)"
                )

        # Note: VBV parameters (vbv-maxrate/vbv-bufsize) can be used with both bitrate
        # and CRF modes. In CRF mode, VBV will constrain the encode to meet the buffer
        # requirements, which is valid for streaming use cases.

    @property
    def preset(self) -> str | None:
        """Get the x265 preset for this profile, or None if not specified."""
        if "preset" not in self.settings:
            return None
        return str(self.settings["preset"])

    @property
    def is_bitrate_mode(self) -> bool:
        """Check if this profile uses bitrate mode instead of CRF."""
        return "bitrate" in self.settings

    @property
    def bitrate(self) -> int | None:
        """Get the bitrate value in kbps, or None if CRF mode."""
        if "bitrate" not in self.settings:
            return None
        bitrate_val = self.settings["bitrate"]
        # Convert to int (may be float in YAML)
        return int(bitrate_val) if isinstance(bitrate_val, (int, float)) else None

    @property
    def pass_number(self) -> int | None:
        """Get the pass number (1, 2, or 3), or None if not specified."""
        if "pass" not in self.settings:
            return None
        pass_val = self.settings["pass"]
        return int(pass_val) if isinstance(pass_val, int) else None

    @property
    def pass_mode_description(self) -> str:
        """Get a human-readable description of the pass mode."""
        pass_num = self.pass_number or 1
        if pass_num == 3:
            return "3-Pass"
        elif pass_num == 2:
            return "2-Pass"
        return "1-Pass"

    @property
    def display_label(self) -> str:
        """Get 'Preset' or 'Profile' label for display."""
        return "Preset" if self.is_preset else "Profile"

    @property
    def display_name(self) -> str:
        """Get the name for display, extracting preset name if applicable.

        For presets: returns just the preset name (e.g., 'slow' from 'preset-slow')
        For profiles: returns the full name
        """
        if self.is_preset and self.name.startswith("preset-"):
            return self.name[7:]  # Remove 'preset-' prefix
        return self.name

    def to_x265_params(
        self,
        crf: float,
        video_format: "VideoFormat",
        video_info: "VideoInfo",
        is_lossless: bool = False,
        stats_file: Path | None = None,
        analysis_file: Path | None = None,
    ) -> list[str]:
        """Convert profile settings to x265 command line parameters (standalone x265.exe).

        Format-specific parameters (color, HDR) are auto-detected from source video
        unless explicitly overridden in the profile settings.

        Args:
            crf: The CRF value to use (ignored if is_lossless=True or bitrate mode)
            video_format: The target video format
            video_info: Video metadata for extracting MaxCLL, master display, etc.
            is_lossless: If True, uses --lossless flag instead of CRF
            stats_file: Path to stats file for multi-pass bitrate encoding (required for pass 2/3)
            analysis_file: Path to analysis file for multi-pass optimization (optional)

        Returns:
            List of x265 CLI arguments (e.g., ["--preset", "slow", "--aud", "--crf", "16"])
        """
        from .encoder_params import GLOBAL_X265_PARAMS, build_global_x265_params
        from .media import VideoFormat

        # Determine which global params the profile overrides
        skip_params = set(self.settings.keys()) & GLOBAL_X265_PARAMS

        # 1. Get global parameters from centralized builder (already in CLI format)
        chroma_location = None
        if video_info and video_info.chroma_location is not None:
            chroma_location = video_info.chroma_location

        params = build_global_x265_params(
            video_info=video_info,
            is_lossless=is_lossless,
            chroma_location=chroma_location,
            skip_params=skip_params,
        )

        # 2. Add user-configurable parameters from profile settings
        is_hdr = video_format != VideoFormat.SDR

        logger.debug("Building x265 params: video_format=%s, is_hdr=%s", video_format, is_hdr)

        for key, value in self.settings.items():
            # Skip preset (handled separately in command building)
            if key == "preset":
                continue

            # Skip bitrate, pass, and multi-pass optimization params (handled in rate control section)
            if key in (
                "bitrate",
                "pass",
                "multi-pass-opt-analysis",
                "multi-pass-opt-distortion",
            ):
                continue

            # Handle conditional parameters (dict with 'hdr' and 'sdr' keys)
            if isinstance(value, dict):
                if "hdr" in value and "sdr" in value:
                    # Use conditional value based on format
                    conditional_value = cast(
                        str | int | float | bool | None,
                        value["hdr"] if is_hdr else value["sdr"],
                    )
                    logger.debug(
                        f"Conditional param '{key}': is_hdr={is_hdr}, using value={conditional_value} (hdr={value['hdr']}, sdr={value['sdr']})"
                    )
                    if isinstance(conditional_value, bool):
                        if conditional_value:
                            params.append(f"--{key}")
                    elif conditional_value is not None:
                        params.extend([f"--{key}", str(conditional_value)])
                else:
                    # Unknown dict structure, skip with warning
                    logger.warning(
                        "Skipping parameter '%s': unrecognized dict structure (expected 'hdr' and 'sdr' keys)",
                        key,
                    )
                continue

            # Handle universal parameters (simple values)
            if isinstance(value, bool):
                if value:
                    params.append(f"--{key}")
            elif value is not None:
                params.extend([f"--{key}", str(value)])

        # 3. Add rate control parameters (lossless, bitrate, or CRF)
        if is_lossless:
            params.append("--lossless")
        elif self.is_bitrate_mode:
            # Bitrate mode: add --bitrate, --pass, and --stats
            bitrate_kbps = self.bitrate
            if bitrate_kbps is not None:
                params.extend(["--bitrate", str(bitrate_kbps)])

            pass_num = self.pass_number
            # Only add --pass for multi-pass encoding (pass 2/3, or pass 1 with stats file)
            if pass_num is not None and (pass_num in (2, 3) or stats_file is not None):
                params.extend(["--pass", str(pass_num)])

                # Add stats file for pass 2 and 3
                if pass_num in (2, 3):
                    if stats_file is None:
                        raise ProfileError(
                            f"Profile '{self.name}': pass {pass_num} requires stats_file parameter"
                        )
                    params.extend(["--stats", str(stats_file)])
                elif pass_num == 1 and stats_file is not None:
                    # Pass 1 in multi-pass: output stats file
                    params.extend(["--stats", str(stats_file)])

            # Add multi-pass optimization parameters
            # No pass-specific logic needed - x265 handles read/write based on pass number
            opt_analysis = self.settings.get("multi-pass-opt-analysis", False)
            opt_distortion = self.settings.get("multi-pass-opt-distortion", False)

            if opt_analysis:
                params.append("--multi-pass-opt-analysis")
                logger.debug("Enabled multi-pass analysis optimization")

            if opt_distortion:
                params.append("--multi-pass-opt-distortion")
                logger.debug("Enabled multi-pass distortion optimization")

            # Add analysis-reuse-file if either optimization flag is enabled
            if (opt_analysis or opt_distortion) and analysis_file is not None:
                params.extend(["--analysis-reuse-file", str(analysis_file)])
                logger.debug("Analysis reuse file: %s", analysis_file)
        else:
            # CRF mode (default)
            params.extend(["--crf", str(crf)])

        return params


# Keys to exclude from Pass 1 settings (they only apply to later passes)
_PASS1_EXCLUDED_KEYS = frozenset(("multi-pass-opt-analysis", "multi-pass-opt-distortion"))


def create_multipass_profile(base_profile: "Profile", pass_num: int) -> "Profile":
    """Create a multi-pass variant of a profile.

    For pass 1, excludes multi-pass optimization keys since they only apply
    to subsequent passes. For passes 2 and 3, copies all settings.

    Args:
        base_profile: The base profile to derive from
        pass_num: Pass number (1, 2, or 3)

    Returns:
        A new Profile configured for the specified pass

    Raises:
        ValueError: If pass_num is not 1, 2, or 3
    """
    if pass_num not in (1, 2, 3):
        raise ValueError(f"pass_num must be 1, 2, or 3, got {pass_num}")

    if pass_num == 1:
        # Filter out multi-pass optimization keys for pass 1
        settings = {
            k: v
            for k, v in base_profile.settings.items()
            if k not in _PASS1_EXCLUDED_KEYS
        }
    else:
        # For passes 2 and 3, copy all settings
        settings = {**base_profile.settings}

    settings["pass"] = pass_num

    return Profile(
        name=f"{base_profile.name}_pass{pass_num}",
        description=f"Pass {pass_num} for {base_profile.name}",
        settings=settings,
        groups=base_profile.groups,
    )


def load_profiles(profile_file: Path | None = None) -> dict[str, Profile]:
    """Load profiles from YAML configuration file.

    Args:
        profile_file: Path to profiles YAML file. If None, searches for x265_profiles.yaml
                     in the current working directory and project root.

    Returns:
        Dictionary mapping profile names to Profile objects

    Raises:
        ProfileError: If the profile file is missing, malformed, contains invalid data,
                     or if multiple profile files are found in different locations
    """
    # Default to x265_profiles.yaml - check multiple locations
    if profile_file is None:
        config_filename = "x265_profiles.yaml"
        candidates: list[Path] = []

        # Check current working directory
        cwd_path = Path.cwd() / config_filename
        if cwd_path.exists():
            candidates.append(cwd_path)

        # Check project/app root (works in dev, PyInstaller, and Nuitka)
        project_root = get_app_root()
        project_path = project_root / config_filename
        if project_path.exists() and project_path.resolve() != cwd_path.resolve():
            candidates.append(project_path)

        if len(candidates) > 1:
            paths_str = "\n  - ".join(str(p) for p in candidates)
            raise ProfileError(
                f"Multiple profile files found:\n  - {paths_str}\n"
                + "Please specify which file to use with --profile-file, or remove duplicates."
            )
        elif len(candidates) == 1:
            profile_file = candidates[0]
        else:
            raise ProfileError(
                f"Profile file not found. Searched:\n  - {cwd_path}\n  - {project_path}\n"
                + "Please create an x265_profiles.yaml file or specify --profile-file."
            )

    # Check if explicitly specified file exists
    if not profile_file.exists():
        raise ProfileError(f"Profile file not found: {profile_file}")

    # Load and parse YAML
    try:
        with open(profile_file, "r", encoding="utf-8") as f:
            data = cast(object, yaml.safe_load(f))
    except yaml.YAMLError as e:
        raise ProfileError(f"Failed to parse profile file: {e}")
    except Exception as e:
        raise ProfileError(f"Failed to read profile file: {e}")

    # Validate structure
    if not isinstance(data, dict):
        raise ProfileError("Profile file must contain a dictionary at root level")

    if "profiles" not in data:
        raise ProfileError("Profile file must contain a 'profiles' key")

    if not isinstance(data["profiles"], list):
        raise ProfileError("'profiles' must be a list")

    # Parse profiles
    profiles: dict[str, Profile] = {}
    profiles_list = cast(list[object], data["profiles"])
    for profile_data in profiles_list:
        if not isinstance(profile_data, dict):
            raise ProfileError("Each profile must be a dictionary")

        profile_dict = cast(dict[str, object], profile_data)
        name_val = profile_dict.get("name")
        if not isinstance(name_val, str) or not name_val:
            raise ProfileError("Profile missing 'name' field or name is not a string")
        name: str = name_val

        description_val = profile_dict.get("description", "")
        description: str = description_val if isinstance(description_val, str) else ""

        settings_val = profile_dict.get("settings", {})
        if not isinstance(settings_val, dict):
            raise ProfileError(f"Profile '{name}' settings must be a dictionary")
        settings: dict[str, object] = cast(dict[str, object], settings_val)

        groups_val = profile_dict.get("groups", [])
        if not isinstance(groups_val, list):
            raise ProfileError(f"Profile '{name}' groups must be a list")
        groups_list = cast(list[object], groups_val)
        groups: list[str] = []
        for group in groups_list:
            if not isinstance(group, str):
                raise ProfileError(f"Profile '{name}' group names must be strings")
            groups.append(group)

        # Check for duplicate profile names
        if name in profiles:
            raise ProfileError(f"Duplicate profile name '{name}' found")

        # Validate groups are strings
        # Create profile
        try:
            profile = Profile(name=name, description=description, settings=settings, groups=groups)
            profiles[name] = profile
        except ProfileError as e:
            raise ProfileError(f"Invalid profile '{name}': {e}")

    if not profiles:
        raise ProfileError("No profiles found in profile file")

    # Validate profile and group name uniqueness
    validate_profile_and_group_names(profiles)

    return profiles


def list_profiles(profiles: dict[str, Profile]) -> str:
    """Generate a formatted list of available profiles.

    Args:
        profiles: Dictionary of profiles from load_profiles()

    Returns:
        Formatted string listing all profiles with descriptions
    """
    lines = ["Available x265 encoding profiles:", ""]

    for name, profile in profiles.items():
        lines.append(f"  {name}")
        if profile.description:
            lines.append(f"    {profile.description}")
        lines.append("")

    return "\n".join(lines)


def get_profile(profiles: dict[str, Profile], name: str) -> Profile:
    """Get a specific profile by name.

    Args:
        profiles: Dictionary of profiles from load_profiles()
        name: Name of the profile to retrieve

    Returns:
        The requested Profile object

    Raises:
        ProfileError: If the profile name is not found
    """
    if name not in profiles:
        available = ", ".join(profiles.keys())
        raise ProfileError(f"Profile '{name}' not found.\nAvailable profiles: {available}")

    return profiles[name]


def get_all_groups(profiles: dict[str, Profile]) -> set[str]:
    """Get all unique group names across all profiles.

    Args:
        profiles: Dictionary of profiles from load_profiles()

    Returns:
        Set of all unique group names
    """
    groups: set[str] = set()
    for profile in profiles.values():
        groups.update(profile.groups)
    return groups


def validate_groups_exist(profiles: dict[str, Profile], groups: list[str]) -> None:
    """Validate that all specified groups exist in the profiles.

    Args:
        profiles: Dictionary of profiles from load_profiles()
        groups: List of group names to validate

    Raises:
        ProfileError: If any group does not exist
    """
    available_groups = get_all_groups(profiles)
    for group in groups:
        if group not in available_groups:
            available = ", ".join(sorted(available_groups)) if available_groups else "(none)"
            raise ProfileError(f"Profile group '{group}' not found.\nAvailable groups: {available}")


def validate_profile_and_group_names(profiles: dict[str, Profile]) -> None:
    """Validate that profile names don't collide with group names.

    Multiple profiles can share the same group. We only need to ensure no profile
    name matches any group name to avoid ambiguity in --multi-profile-search.

    Checks:
    1. No profile name matches any group name

    Args:
        profiles: Dictionary of profiles from load_profiles()

    Raises:
        ProfileError: If any profile name matches a group name
    """
    # Collect all unique group names
    all_groups: set[str] = set()
    for profile in profiles.values():
        all_groups.update(profile.groups)

    # Check for profile name / group name collisions
    profile_names = set(profiles.keys())
    collisions = profile_names & all_groups
    if collisions:
        raise ProfileError(
            f"Name collision between profile(s) and group(s): {', '.join(sorted(collisions))}. "
            + "Profile names and group names must be distinct."
        )


def get_profiles_by_groups(
    profiles: dict[str, Profile],
    groups: list[str] | None = None,
) -> list[Profile]:
    """Get profiles filtered by group membership.

    Args:
        profiles: Dictionary of profiles from load_profiles()
        groups: List of group names to filter by. If None or empty,
                returns all profiles (including ungrouped ones).

    Returns:
        List of Profile objects matching the filter criteria,
        in YAML definition order (dict insertion order).
    """
    if not groups:
        # No filter - return all profiles in definition order
        return list(profiles.values())

    # Filter by group membership - profile matches if it belongs to ANY specified group
    matching: list[Profile] = []
    groups_set = set(groups)
    for profile in profiles.values():
        if groups_set.intersection(profile.groups):
            matching.append(profile)

    return matching
