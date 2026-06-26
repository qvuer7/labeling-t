"""COCO export adapter — neutral schema -> COCO via supervision.

We do NOT hand-write COCO. `supervision` owns the serialization (and gives us
YOLO/VOC export + ByteTrack tracking for free later). Our job is only the
boundary conversion: ImageLabels -> sv.Detections.

    ImageLabels (abs-pixel xyxy)  ──►  sv.Detections (abs-pixel xyxy)  ──►  COCO json
                                        via sv.DetectionDataset.as_coco

Precondition: every `ImageLabels.image_path` must exist on disk — supervision
reads each file once to record its width/height in the COCO `images` array.
This keeps export memory-light (no pixel data held in RAM).
"""

from __future__ import annotations

import numpy as np
import supervision as sv

from ..schema import ImageLabels


def _stable_classes(images: list[ImageLabels]) -> list[str]:
    """Deterministic class list = sorted unique categories. Sorting makes
    category_id stable across runs so two exports of the same project agree."""
    cats: set[str] = set()
    for img in images:
        for d in img.detections:
            cats.add(d.category)
    return sorted(cats)


def _to_sv_detections(img: ImageLabels, class_index: dict[str, int]) -> sv.Detections:
    if not img.detections:
        return sv.Detections.empty()
    xyxy = np.array(
        [[d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2] for d in img.detections],
        dtype=float,
    )
    class_id = np.array([class_index[d.category] for d in img.detections], dtype=int)
    scores = [d.score for d in img.detections]
    confidence = (
        np.array(scores, dtype=float) if all(s is not None for s in scores) else None
    )
    return sv.Detections(xyxy=xyxy, class_id=class_id, confidence=confidence)


def to_coco(
    images: list[ImageLabels],
    annotations_path: str,
    classes: list[str] | None = None,
) -> tuple[int, int]:
    """Write a COCO annotations JSON for `images`.

    classes: pass the project's full category set to pin category_id ordering.
             If None, derived as sorted-unique from the data (stable, but a
             category absent from this batch won't get an id).
    Returns supervision's (image_count, annotation_count).
    """
    if classes is None:
        classes = _stable_classes(images)
    class_index = {c: i for i, c in enumerate(classes)}

    missing = {
        d.category
        for img in images
        for d in img.detections
        if d.category not in class_index
    }
    if missing:
        raise ValueError(f"categories not in class list: {sorted(missing)}")

    annotations = {img.image_path: _to_sv_detections(img, class_index) for img in images}
    image_paths = [img.image_path for img in images]
    dataset = sv.DetectionDataset(
        classes=classes, images=image_paths, annotations=annotations
    )
    # supervision's as_coco returns the NEXT (image_id, annotation_id), not
    # counts. Return honest counts computed from the input instead.
    dataset.as_coco(annotations_path=annotations_path)
    n_anns = sum(len(img.detections) for img in images)
    return len(images), n_anns
