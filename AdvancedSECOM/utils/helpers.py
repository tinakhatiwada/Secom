"""
AdvancedSECOM/utils/helpers.py
Shared static helpers.
"""


def fmt(v, spec=".4f", fallback="N/A"):
    """Format a numeric value, or return fallback if None."""
    return format(v, spec) if v is not None else fallback
