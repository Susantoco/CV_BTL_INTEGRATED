"""Draw detection/tracking overlays on frames."""

import colorsys

import cv2
import numpy as np

from src.detection import Detection


def _color_for_id(id_: int) -> tuple[int, int, int]:
    """Deterministic, well-separated BGR color for a given id."""
    hue = (id_ * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw bbox + label (+ track_id if present) for each detection, in place."""
    for d in detections:
        x1, y1, x2, y2 = map(int, d.bbox)
        color = _color_for_id(d.track_id if d.track_id is not None else d.class_id)
        label = f"{d.class_name} {d.confidence:.2f}"
        if d.track_id is not None:
            label = f"#{d.track_id} {label}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y1 = max(y1 - th - 6, 0)
        cv2.rectangle(frame, (x1, label_y1), (x1 + tw + 4, label_y1 + th + 6), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 2, label_y1 + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return frame
