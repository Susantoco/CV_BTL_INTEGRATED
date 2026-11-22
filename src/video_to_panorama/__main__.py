import traceback
import argparse

from .features import METHODS
from .pipeline import run_pipeline


def build_parser():
    parser = argparse.ArgumentParser(
        description="Tạo panorama từ video bằng feature matching + Homography."
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Đường dẫn video đầu vào."
    )

    parser.add_argument(
        "--method",
        choices=METHODS,
        default="sift"
    )

    parser.add_argument(
        "--engine",
        choices=("custom", "opencv"),
        default="custom",
        help="Engine ghép; OpenCV Stitcher phù hợp với góc lia rộng.",
    )

    parser.add_argument(
        "--sample-interval",
        type=float,
        default=0.5,
        help="Khoảng thời gian giữa hai frame được lấy mẫu (giây)."
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Số frame tối đa. 0 = không giới hạn."
    )

    parser.add_argument(
        "--max-width",
        type=int,
        default=1200,
        help="Chiều rộng tối đa của frame. 0 = giữ nguyên kích thước."
    )

    parser.add_argument(
        "--start-seconds",
        type=float,
        default=0.0,
        help="Thời điểm bắt đầu xử lý video."
    )

    parser.add_argument(
        "--end-seconds",
        type=float,
        default=None,
        help="Thời điểm kết thúc. Bỏ trống = đến hết video."
    )

    parser.add_argument(
        "--rotation",
        type=int,
        choices=(0, 90, 180, 270),
        default=0,
        help="Xoay frame theo chiều kim đồng hồ (mặc định: 0 độ)."
    )

    parser.add_argument("--min-matches", type=int, default=12)
    parser.add_argument("--min-keypoints", type=int, default=20)

    parser.add_argument(
        "--min-sharpness",
        type=float,
        default=8.0,
        help="Ngưỡng variance Laplacian để lọc frame nhòe."
    )

    parser.add_argument("--ratio", type=float, default=0.80)
    parser.add_argument("--ransac-threshold", type=float, default=4.0)

    parser.add_argument(
        "--no-denoise",
        action="store_true"
    )

    parser.add_argument(
        "--results-root",
        default="results"
    )

    return parser


def main():
    args = build_parser().parse_args()

    if args.start_seconds < 0:
        raise SystemExit("[ERROR] start-seconds phải >= 0.")

    if args.end_seconds is not None:
        if args.end_seconds <= args.start_seconds:
            raise SystemExit(
                "[ERROR] end-seconds phải lớn hơn start-seconds."
            )

    if args.sample_interval <= 0:
        raise SystemExit(
            "[ERROR] sample-interval phải lớn hơn 0."
        )

    if args.max_frames < 0:
        raise SystemExit(
            "[ERROR] max-frames phải >= 0."
        )

    try:
        panorama_path, metrics = run_pipeline(
            video_path=args.video,
            method=args.method,
            engine=args.engine,
            sample_interval=args.sample_interval,
            max_frames=args.max_frames,
            max_width=args.max_width,
            min_matches=args.min_matches,
            min_keypoints=args.min_keypoints,
            min_sharpness=args.min_sharpness,
            ratio=args.ratio,
            ransac_threshold=args.ransac_threshold,
            denoise=not args.no_denoise,
            start_seconds=args.start_seconds,
            end_seconds=args.end_seconds,
            rotation=args.rotation,
            results_root=args.results_root,
        )

    except Exception as exc:
        traceback.print_exc()
        raise SystemExit(f"[ERROR] {exc}")

    print("\n=== VIDEO TO PANORAMA ===")

    if metrics.get("engine") == "opencv_stitcher":
        print("Engine: OpenCV Stitcher (feature matching + homography)")
        print(f"Candidate frames: {metrics['candidate_frame_count']}")
        print(
            "Frames passed to stitcher: "
            f"{metrics['stitcher_input_frame_count']}"
        )
        print(f"Total time: {metrics['total_time_ms']:.1f} ms")
        print(f"Panorama: {panorama_path}")
        return

    print(f"Method: {metrics['method'].upper()}")
    print(f"Candidate frames: {metrics['candidate_frame_count']}")
    print(f"Accepted frames: {metrics['accepted_frame_count']}")
    print(f"Skipped frames: {metrics['skipped_frame_count']}")
    print(f"Average keypoints: {metrics['average_keypoints']:.1f}")
    print(f"Average good matches: {metrics['average_good_matches']:.1f}")
    print(f"Average inlier ratio: {metrics['average_inlier_ratio']:.3f}")
    print(
        "Average reprojection error: "
        f"{metrics['average_reprojection_error_px']:.3f} px"
    )
    print(f"Total time: {metrics['total_time_ms']:.1f} ms")
    print(f"Panorama: {panorama_path}")


if __name__ == "__main__":
    main()
