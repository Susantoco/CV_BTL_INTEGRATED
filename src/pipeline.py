"""Run detect(+track) over a video and write an overlaid output video."""

import time

import cv2

from src.detection import ObjectDetector
from src.overlay import draw_detections
from src.preprocessing import PreprocessConfig, preprocess_frame


def process_video(
    input_path: str,
    output_path: str,
    detector: ObjectDetector,
    tracking: bool = True,
    preprocess_config: PreprocessConfig | None = None,
    show: bool = False,
) -> dict:
    """Run detection (optionally with tracking) frame-by-frame, write overlaid video.

    If `show` is True, also preview the overlaid frames live in a window
    (press 'q' to stop early). Returns stats (frame_count, elapsed_sec,
    avg_fps) for FPS comparison experiments (e.g. tracking vs no-tracking,
    resize vs no-resize).
    """
    preprocess_config = preprocess_config or PreprocessConfig()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if preprocess_config.resize_width is not None:
        scale = preprocess_config.resize_width / orig_width
        width, height = preprocess_config.resize_width, int(round(orig_height * scale))
    else:
        width, height = orig_width, orig_height

    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    if tracking:
        detector.reset_tracker()

    frame_count = 0
    start = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = preprocess_frame(frame, preprocess_config)
        detections = detector.track(frame) if tracking else detector.detect(frame)
        draw_detections(frame, detections)
        writer.write(frame)
        frame_count += 1

        if show:
            cv2.imshow("detect+track preview (press q to stop)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    elapsed = time.perf_counter() - start

    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()

    return {
        "frame_count": frame_count,
        "elapsed_sec": elapsed,
        "avg_fps": frame_count / elapsed if elapsed > 0 else 0.0,
    }
