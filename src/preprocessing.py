"""Preprocessing applied to frames before they're fed into YOLO detect/track.

Default is pass-through: measured brightness/contrast/blur on the actual
input video (`data/image1.mp4`) showed no noise/darkness/blur issues, so no
denoising is applied unless explicitly enabled. Resize/denoise exist as
opt-in knobs for the planned FPS comparison experiments.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PreprocessConfig:
    resize_width: int | None = None  # None = keep original size
    denoise: bool = True


def preprocess_frame(frame: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    """Apply resize/denoise to a single BGR frame per `config`. Color space is
    kept as BGR (what cv2 reads and what YOLO expects) — no conversion."""
    if config.resize_width is not None and config.resize_width != frame.shape[1]:
        scale = config.resize_width / frame.shape[1]
        new_height = int(round(frame.shape[0] * scale))
        frame = cv2.resize(
            frame, (config.resize_width, new_height), interpolation=cv2.INTER_AREA
        )

    if config.denoise:
        frame = cv2.fastNlMeansDenoisingColored(frame, None, 5, 5, 7, 21)

    return frame
