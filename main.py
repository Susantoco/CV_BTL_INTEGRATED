"""CLI entry point: run detect(+track)+overlay on a video.

Examples:
    python main.py --show
    python main.py --input data/image1.mp4 --output outputs/result.mp4
    python main.py --no-tracking --resize-width 320
    python main.py --method traditional --output outputs/result_traditional.mp4
"""

import argparse
import os

from src.pipeline import process_video
from src.preprocessing import PreprocessConfig


def build_detector(method: str):
    if method == "yolo":
        from src.detection import ObjectDetector

        return ObjectDetector()
    from src.traditional import SkinColorDetector

    return SkinColorDetector()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/image1.mp4")
    parser.add_argument("--output", default="outputs/result.mp4")
    parser.add_argument(
        "--method",
        choices=["yolo", "traditional"],
        default="yolo",
        help=(
            "yolo = YOLOv8s+ByteTrack (deep learning, pretrained). "
            "traditional = YCrCb skin-color threshold + morphology + "
            "connected components, with nearest-centroid tracking."
        ),
    )
    parser.add_argument(
        "--no-tracking", action="store_true", help="Detect only, no tracking."
    )
    parser.add_argument("--resize-width", type=int, default=None)
    parser.add_argument("--denoise", action="store_true")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Preview live in a window while processing (press q to stop early).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    detector = build_detector(args.method)
    config = PreprocessConfig(resize_width=args.resize_width, denoise=args.denoise)

    stats = process_video(
        args.input,
        args.output,
        detector,
        tracking=not args.no_tracking,
        preprocess_config=config,
        show=args.show,
    )
    print(f"Done: {stats}")
    print(f"Output video written to: {args.output}")


if __name__ == "__main__":
    main()
