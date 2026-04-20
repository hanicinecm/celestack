"""Custom exceptions for celestack."""


class CelestackError(Exception):
    """Base exception for celestack."""


class TileReadError(CelestackError):
    """Raised when tile-reading is unavailable for a frame."""


class MaskError(CelestackError):
    """Raised when a mask operation fails."""


class StarDetectorError(CelestackError):
    """Raised when star detection fails."""


class ProxyMetadataError(CelestackError):
    """Raised when a proxy sidecar metadata file is missing or malformed."""
