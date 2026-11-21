"""Traditional (non-deep-learning) object detection: skin-color segmentation.

Classic pipeline straight from the course slides:
- Chap10_11 (Image Segmentation): thresholding to isolate objects by color —
  slide gives "Detecting/Tracking faces based on skin color" as the
  application example.
- Chap06_07 (Morphological Image Processing): opening/closing to clean the
  binary mask, then "Extraction of Connected Components" to pull out blobs.

No neural network involved: a frame is thresholded in YCrCb space, cleaned
with morphology, and blobs become bounding boxes. "Tracking" is a simple
nearest-centroid matcher across frames — no motion model, no re-ID.
"""

import math

import cv2
import numpy as np

from src.detection import Detection

# Classic YCrCb skin-tone range (Chai & Ngan, 1999) — Y is left unconstrained
# on purpose since skin brightness varies a lot with lighting.
_SKIN_LOWER = np.array([0, 135, 85], dtype=np.uint8)
_SKIN_UPPER = np.array([255, 180, 135], dtype=np.uint8)


class SkinColorDetector:
    """Detects skin-colored blobs (faces/hands/arms) via color thresholding.

    Duck-types `ObjectDetector`'s interface (`detect`, `track`,
    `reset_tracker`) so it drops into `process_video` unchanged.
    """

    def __init__(self, min_area: int = 1500, max_track_distance: float = 80.0):
        self.min_area = min_area
        self.max_track_distance = max_track_distance
        self._next_track_id = 0
        self._tracks: dict[int, tuple[float, float]] = {}  # track_id -> last centroid

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Threshold + morphology + connected components, no tracking."""
        blobs = self._find_blobs(frame)
        detections = [
            Detection(
                bbox=(x, y, x + w, y + h),
                confidence=1.0,
                class_id=0,
                class_name="skin-region",
            )
            for (x, y, w, h, _cx, _cy) in blobs
        ]
        return detections

    def track(self, frame: np.ndarray) -> list[Detection]:
        """Same blob extraction as `detect`, plus nearest-centroid matching
        against the previous frame's blobs to assign a persistent track_id."""
        blobs = self._find_blobs(frame)

        detections = []
        new_tracks: dict[int, tuple[float, float]] = {}
        used_track_ids: set[int] = set()
        for x, y, w, h, cx, cy in blobs:
            best_id = None
            best_dist = self.max_track_distance
            for tid, (px, py) in self._tracks.items():
                if tid in used_track_ids:
                    continue
                dist = math.hypot(cx - px, cy - py)
                if dist < best_dist:
                    best_dist = dist
                    best_id = tid
            if best_id is None:
                best_id = self._next_track_id
                self._next_track_id += 1

            used_track_ids.add(best_id)
            new_tracks[best_id] = (cx, cy)
            detections.append(
                Detection(
                    bbox=(x, y, x + w, y + h),
                    confidence=1.0,
                    class_id=0,
                    class_name="skin-region",
                    track_id=best_id,
                )
            )

        self._tracks = new_tracks
        return detections

    def reset_tracker(self) -> None:
        """Clear centroid-tracking state (call before tracking a new video)."""
        self._tracks = {}
        self._next_track_id = 0

    def _find_blobs(
        self, frame: np.ndarray
    ) -> list[tuple[int, int, int, int, float, float]]:
        """Return (x, y, w, h, centroid_x, centroid_y) for each blob above
        `min_area`, largest first."""
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, _SKIN_LOWER, _SKIN_UPPER)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        blobs = []
        for label in range(1, num_labels):  # label 0 = background
            area = stats[label, cv2.CC_STAT_AREA]
            if area < self.min_area:
                continue
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            cx, cy = centroids[label]
            blobs.append((x, y, w, h, float(cx), float(cy)))

        blobs.sort(key=lambda b: b[2] * b[3], reverse=True)
        return blobs
