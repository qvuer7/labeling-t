"""labeling_t — batch auto-labeling backend with an owned neutral schema."""

from .schema import BBox, Detection, ImageLabels

__all__ = ["BBox", "Detection", "ImageLabels"]
