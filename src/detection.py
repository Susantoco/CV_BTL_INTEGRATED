"""Object detection on video frames using a pretrained YOLOv8 model."""

from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixel coords
    confidence: float
    class_id: int
    class_name: str
    track_id: int | None = None  # set when produced by `track()`, else None


class ObjectDetector:
    """Wraps a pretrained YOLOv8 model for per-frame object detection.

    Output is plain `Detection` objects so downstream steps (tracking,
    overlay) don't need to know anything about ultralytics internals.
    """

    def __init__(
        self,
        model_path: str = "yolov8s.pt",
        device: str | None = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ):
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = YOLO(model_path)
        self.model.to(self.device)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run detection (no tracking) on a single BGR frame (as read by cv2)."""
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )
        return self._to_detections(results[0])

    def track(self, frame: np.ndarray) -> list[Detection]:
        """Run detection + ByteTrack tracking on a single BGR frame.

        Call this on consecutive frames of the *same* video to keep track IDs
        consistent across frames (relies on internal tracker state via
        `persist=True`). Call `reset_tracker()` before starting a new video.
        """
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )
        return self._to_detections(results[0])

    def reset_tracker(self) -> None:
        """Clear ByteTrack's internal state (call before tracking a new video)."""
        if hasattr(self.model, "predictor") and self.model.predictor is not None:
            self.model.predictor.trackers = []

    def _to_detections(self, result) -> list[Detection]:
        names = result.names
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            track_id = int(box.id[0]) if box.id is not None else None
            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    class_id=class_id,
                    class_name=names[class_id],
                    track_id=track_id,
                )
            )
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections
