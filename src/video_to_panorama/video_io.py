
from pathlib import Path
import cv2


def extract_sampled_frames(
    video_path: str | Path,
    output_dir: str | Path,
    sample_interval: float = 0.2,
    max_frames: int = 0,
    max_width: int = 1200,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
    rotation: int = 0,
) -> tuple[list, dict]:
    """Lấy frame theo khoảng thời gian trong video."""

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Kiểm tra tham số đầu vào.
    if sample_interval <= 0:
        raise ValueError("sample_interval phải lớn hơn 0.")

    if start_seconds < 0:
        raise ValueError("start_seconds không được âm.")

    if end_seconds is not None and end_seconds <= start_seconds:
        raise ValueError("end_seconds phải lớn hơn start_seconds.")

    if max_frames < 0:
        raise ValueError("max_frames không được âm.")

    if max_width < 0:
        raise ValueError("max_width không được âm.")

    if rotation not in (0, 90, 180, 270):
        raise ValueError(
            "rotation phải là 0, 90, 180 hoặc 270 độ theo chiều kim đồng hồ."
        )

    if not video_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy video: {video_path}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0:
            raise RuntimeError("Không xác định được FPS của video.")

        duration = frame_count / fps if frame_count > 0 else 0.0

        if start_seconds >= duration and duration > 0:
            raise ValueError(
                f"start_seconds={start_seconds} vượt hoặc bằng "
                f"thời lượng video ({duration:.2f}s)."
            )

        if end_seconds is not None and duration > 0:
            if end_seconds > duration:
                raise ValueError(
                    f"end_seconds={end_seconds} vượt thời lượng video "
                    f"({duration:.2f}s)."
                )

        # Xóa các frame được tạo từ lần chạy trước.
        for old_frame in output_dir.glob("frame_*.jpg"):
            old_frame.unlink()

        start_frame = int(start_seconds * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        timestamps = []

        next_sample_time = start_seconds
        frame_index = start_frame

        while True:
            ok, frame = cap.read()

            if not ok:
                break

            timestamp = frame_index / fps

            # Nếu có end_seconds thì chỉ lấy frame đến thời điểm đó.
            if end_seconds is not None and timestamp > end_seconds:
                break

            if timestamp + 1e-6 >= next_sample_time:

                if rotation == 90:
                    frame = cv2.rotate(
                        frame,
                        cv2.ROTATE_90_CLOCKWISE,
                    )
                elif rotation == 180:
                    frame = cv2.rotate(
                        frame,
                        cv2.ROTATE_180,
                    )
                elif rotation == 270:
                    frame = cv2.rotate(
                        frame,
                        cv2.ROTATE_90_COUNTERCLOCKWISE,
                    )

                # Resize nếu frame vượt quá max_width.
                if max_width > 0 and frame.shape[1] > max_width:
                    scale = max_width / frame.shape[1]

                    frame = cv2.resize(
                        frame,
                        (
                            max_width,
                            max(1, round(frame.shape[0] * scale)),
                        ),
                        interpolation=cv2.INTER_AREA,
                    )

                out_path = output_dir / f"frame_{len(frames):03d}.jpg"

                if not cv2.imwrite(str(out_path), frame):
                    raise RuntimeError(f"Không ghi được frame: {out_path}")

                frames.append(frame)
                timestamps.append(round(timestamp, 3))

                next_sample_time += sample_interval

                # 0 = không giới hạn số frame.
                if max_frames > 0 and len(frames) >= max_frames:
                    break

            frame_index += 1

        info = {
            "video_path": str(video_path),
            "fps": fps,
            "frame_count": frame_count,
            "original_width": width,
            "original_height": height,
            "frame_rotation_clockwise_degrees": rotation,
            "duration_seconds": round(duration, 3),
            "sample_interval_seconds": sample_interval,
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
            "sampled_timestamps_seconds": timestamps,
            "sampled_frame_count": len(frames),
        }

        if len(frames) < 2:
            raise RuntimeError(
                f"Chỉ lấy được {len(frames)} frame trong khoảng đã chọn. "
                "Cần ít nhất 2 frame."
            )

        return frames, info

    finally:
        cap.release()
