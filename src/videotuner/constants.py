"""Constants used throughout VideoTuner."""

from __future__ import annotations

# =============================================================================
# CRF Search Constants
# =============================================================================

# x265 CRF range (0 = lossless, 51 = worst quality)
CRF_MIN: float = 0.0
CRF_MAX: float = 51.0

# Maximum iterations for CRF search (safety limit to prevent infinite loops)
CRF_SEARCH_MAX_ITERATIONS: int = 20

# CRF value used when quality floor is reached
CRF_FLOOR_VALUE: float = 51.0

# Tolerance for CRF floor comparison (floating point comparison)
CRF_FLOOR_TOLERANCE: float = 0.01

# =============================================================================
# Guard Band Constants
# =============================================================================

# Maximum combined guard percentage (start + end must be less than this)
MAX_COMBINED_GUARD_PERCENT: float = 0.99

# =============================================================================
# Bitrate Warning Constants
# =============================================================================

# Valid range for bitrate warning percentage
BITRATE_WARNING_PERCENT_MIN: float = 1.0
BITRATE_WARNING_PERCENT_MAX: float = 100.0

# =============================================================================
# VMAF Assessment Constants
# =============================================================================

# Thread calculation: use this fraction of CPU cores
VMAF_THREAD_CPU_FRACTION: float = 0.5

# Maximum threads for VMAF calculation
VMAF_MAX_THREADS: int = 16

# =============================================================================
# Resolution Constants
# =============================================================================

# Resolution classification tolerance (3% relative)
RESOLUTION_RELATIVE_TOLERANCE: float = 0.03

# Minimum accepted crop fraction (55% of expected dimension)
MIN_CROP_FRACTION: float = 0.55

# Confidence threshold for resolution classification
RESOLUTION_CONFIDENCE_THRESHOLD: float = 0.5

# =============================================================================
# Display Constants
# =============================================================================

# Decimal places for quality metrics (VMAF, SSIMULACRA2, etc.)
# Used for both display formatting and target comparison rounding
METRIC_DECIMALS: int = 2

# Progress bar width in characters
PROGRESS_BAR_WIDTH: int = 40

# Log section separator width
LOG_SEPARATOR_WIDTH: int = 60

# Log section separator character
LOG_SEPARATOR_CHAR: str = "="

# =============================================================================
# Frame Rate Constants
# =============================================================================

# Denominator used for FPS calculation (fps * 1000 / 1000)
FPS_DENOMINATOR: int = 1000

# =============================================================================
# Percentile Constants
# =============================================================================

# 1st percentile (bottom 1%)
PERCENTILE_1PCT: float = 0.01

# 5th percentile (bottom 5%)
PERCENTILE_5PCT: float = 0.05

# 95th percentile (top 5%)
PERCENTILE_95PCT: float = 0.95
