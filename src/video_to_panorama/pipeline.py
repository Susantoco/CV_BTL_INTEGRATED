
from pathlib import Path
import json
import time

import cv2
import numpy as np

from .features import (
    create_feature_extractor,
    preprocess_frame,
    detect_and_describe,
)
from .matching import (
    match_descriptors,
    estimate_pair_homography,
)
from .stitching import (
    build_centered_global_transforms,
    optimize_global_transforms,
    create_panorama,
    create_opencv_panorama,
)
from .video_io import extract_sampled_frames


# ==========================================================
# 1. LOGGING
# ==========================================================

def log(message, level="INFO"):
    """Hiển thị log thống nhất."""
    print(f"[{level}] {message}")


# ==========================================================
# 2. GENERAL UTILITIES
# ==========================================================

def safe_average(values):
    """Tính trung bình an toàn, trả về 0 nếu danh sách rỗng."""
    return sum(values) / len(values) if values else 0.0


def calculate_overlap_ratio(corners, reference_shape):
    """
    Ước lượng tỷ lệ chồng lấp giữa frame hiện tại sau biến đổi
    và frame tham chiếu.

    Sử dụng bounding box nên đây là tỷ lệ overlap xấp xỉ,
    không phải diện tích giao chính xác giữa hai đa giác.
    """
    ref_h, ref_w = reference_shape[:2]

    x_min = float(corners[:, 0].min())
    y_min = float(corners[:, 1].min())
    x_max = float(corners[:, 0].max())
    y_max = float(corners[:, 1].max())

    projected_width = x_max - x_min
    projected_height = y_max - y_min

    projected_area = projected_width * projected_height
    reference_area = float(ref_w * ref_h)

    if (
        not np.isfinite(projected_area)
        or projected_area <= 0
        or reference_area <= 0
    ):
        return 0.0

    intersection_width = max(
        0.0,
        min(x_max, float(ref_w)) - max(x_min, 0.0),
    )

    intersection_height = max(
        0.0,
        min(y_max, float(ref_h)) - max(y_min, 0.0),
    )

    intersection_area = (
        intersection_width * intersection_height
    )

    denominator = min(
        projected_area,
        reference_area,
    )

    if denominator <= 0:
        return 0.0

    return float(intersection_area / denominator)


# ==========================================================
# 3. HOMOGRAPHY VALIDATION
# ==========================================================

def validate_pair_homography(
    H,
    frame_shape,
    min_scale=0.50,
    max_scale=2.00,
    min_edge_ratio=0.50,
    max_edge_ratio=2.00,
):
    """
    Kiểm tra biến dạng hình học của Homography giữa hai frame.

    H ánh xạ frame hiện tại về frame tham chiếu.
    """
    h, w = frame_shape[:2]

    if H is None or H.shape != (3, 3):
        return False, "invalid_homography", {}

    if not np.isfinite(H).all():
        return False, "non_finite_homography", {}

    corners = np.float64([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h],
    ])

    homogeneous = np.column_stack([
        corners,
        np.ones(4, dtype=np.float64),
    ])

    denominators = (H @ homogeneous.T)[2]

    if (
        not np.isfinite(denominators).all()
        or np.any(np.abs(denominators) < 1e-8)
        or not (
            np.all(denominators > 0)
            or np.all(denominators < 0)
        )
    ):
        return False, "projective_singularity", {}

    projected = cv2.perspectiveTransform(
        corners.reshape(1, -1, 2),
        H,
    ).reshape(-1, 2)

    if not np.isfinite(projected).all():
        return False, "invalid_projected_corners", {}

    edges = np.roll(projected, -1, axis=0) - projected
    edge_lengths = np.linalg.norm(edges, axis=1)

    original_edges = np.array([
        w,
        h,
        w,
        h,
    ], dtype=np.float64)

    if np.any(edge_lengths <= 1e-6):
        return False, "degenerate_projected_edges", {}

    edge_ratios = edge_lengths / original_edges

    if (
        np.any(edge_ratios < min_edge_ratio)
        or np.any(edge_ratios > max_edge_ratio)
    ):
        return False, "excessive_edge_distortion", {
            "edge_ratios": edge_ratios.tolist(),
        }

    area = abs(
        cv2.contourArea(projected.astype(np.float32))
    )

    original_area = float(w * h)

    if not np.isfinite(area) or area <= 1e-6:
        return False, "degenerate_projected_area", {}

    area_ratio = area / original_area

    if (
        area_ratio < min_scale ** 2
        or area_ratio > max_scale ** 2
    ):
        return False, "excessive_area_distortion", {
            "area_ratio": float(area_ratio),
        }

    contour = projected.astype(
        np.float32
    ).reshape(-1, 1, 2)

    if not cv2.isContourConvex(contour):
        return False, "non_convex_projected_frame", {}

    return True, None, {
        "edge_ratios": edge_ratios.tolist(),
        "area_ratio": float(area_ratio),
        "projected_corners": projected.tolist(),
    }


def validate_composed_homography(
    previous_transform,
    H,
    frame_shape,
    min_denominator_margin=1e-8,
):
    """
    Kiểm tra Global Transform sau khi ghép Homography mới.

    previous_transform:
        Global Transform của frame tham chiếu.

    H:
        Homography ánh xạ frame hiện tại về frame tham chiếu.
    """
    h, w = frame_shape[:2]

    corners = np.float64([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h],
    ])

    homogeneous = np.column_stack([
        corners,
        np.ones(4, dtype=np.float64),
    ])

    composed = previous_transform @ H

    projected_h = (composed @ homogeneous.T).T
    denominators = projected_h[:, 2]

    denominator_scales = (
        np.abs(composed[2, 0] * corners[:, 0])
        + np.abs(composed[2, 1] * corners[:, 1])
        + abs(composed[2, 2])
        + 1e-12
    )

    margins = (
        np.abs(denominators) / denominator_scales
    )

    same_sign = (
        np.all(denominators > 0)
        or np.all(denominators < 0)
    )

    valid = (
        np.isfinite(denominators).all()
        and same_sign
        and np.all(margins > min_denominator_margin)
    )

    stats = {
        "denominators": denominators.tolist(),
        "minimum_margin": float(margins.min()),
        "same_sign": bool(same_sign),
    }

    if not valid:
        return False, "global_projective_horizon", stats

    return True, None, stats


# ==========================================================
# 4. MAIN PIPELINE
# ==========================================================

def run_pipeline(
    video_path,
    method="sift",
    engine="custom",
    sample_interval=0.5,
    max_frames=0,
    max_width=1200,
    min_matches=12,
    ratio=0.80,
    ransac_threshold=4.0,
    denoise=True,
    start_seconds=0.0,
    end_seconds=None,
    rotation=0,
    min_keypoints=20,
    min_sharpness=8.0,
    min_overlap_ratio=0.10,
    max_overlap_ratio=0.85,
    max_canvas_side=12000,
    max_canvas_pixels=25_000_000,
    results_root="results",
):
    """
    Chuyển video thành panorama.

    Các bước:
        1. Trích xuất frame theo thời gian.
        2. Tiền xử lý và trích xuất đặc trưng.
        3. Lọc frame chất lượng thấp.
        4. Matching giữa các frame.
        5. Ước lượng Homography bằng RANSAC.
        6. Kiểm tra chất lượng và overlap.
        7. Tạo adjacent và loop constraints.
        8. Tối ưu Global Transform.
        9. Tạo và lưu panorama.
    """

    total_start = time.perf_counter()

    video_path = Path(video_path)

    if not isinstance(method, str):
        raise TypeError("method phải là chuỗi.")

    if not isinstance(engine, str):
        raise TypeError("engine phải là chuỗi.")

    method = method.strip().lower()
    engine = engine.strip().lower()

    if engine not in ("custom", "opencv"):
        raise ValueError("engine phải là 'custom' hoặc 'opencv'.")

    if not 0 < min_overlap_ratio <= 1:
        raise ValueError(
            "min_overlap_ratio phải nằm trong khoảng (0, 1]."
        )

    if (
        not 0 < max_overlap_ratio <= 1
        or min_overlap_ratio >= max_overlap_ratio
    ):
        raise ValueError(
            "max_overlap_ratio phải lớn hơn min_overlap_ratio "
            "và không vượt quá 1."
        )

    if min_keypoints < 0:
        raise ValueError("min_keypoints không được âm.")

    if min_sharpness < 0:
        raise ValueError("min_sharpness không được âm.")

    if max_canvas_side <= 0 or max_canvas_pixels <= 0:
        raise ValueError(
            "Giới hạn canvas phải lớn hơn 0."
        )

    # ------------------------------------------------------
    # Khởi tạo thư mục kết quả
    # ------------------------------------------------------

    run_dir = (
        Path(results_root)
        / video_path.stem
        / (method if engine == "custom" else "opencv_stitcher")
    )

    frame_dir = run_dir / "sampled_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    log(f"Processing video: {video_path}")
    log(
        f"Panorama engine: "
        f"{'OpenCV Stitcher' if engine == 'opencv' else method.upper()}"
    )

    # ======================================================
    # 1. EXTRACT FRAMES
    # ======================================================

    candidate_frames, video_info = extract_sampled_frames(
        video_path=video_path,
        output_dir=frame_dir,
        sample_interval=sample_interval,
        max_frames=max_frames,
        max_width=max_width,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        rotation=rotation,
    )

    if not candidate_frames:
        raise RuntimeError(
            "Không trích xuất được frame nào từ video."
        )

    log(
        f"Extracted {len(candidate_frames)} candidate frames."
    )

    if engine == "opencv":
        panorama, selected_indices = create_opencv_panorama(
            candidate_frames,
            max_canvas_side=max_canvas_side,
            max_canvas_pixels=max_canvas_pixels,
        )

        panorama_path = run_dir / "panorama.jpg"

        if not cv2.imwrite(str(panorama_path), panorama):
            raise RuntimeError(
                f"Không ghi được panorama: {panorama_path}"
            )

        total_ms = (
            time.perf_counter() - total_start
        ) * 1000

        selected_timestamps = [
            video_info["sampled_timestamps_seconds"][index]
            for index in selected_indices
        ]

        metrics = {
            "engine": "opencv_stitcher",
            "candidate_frame_count": len(candidate_frames),
            "stitcher_input_frame_count": len(selected_indices),
            "selected_candidate_indices": selected_indices,
            "selected_timestamps_seconds": selected_timestamps,
            "total_time_ms": total_ms,
        }

        with (run_dir / "metrics.json").open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                metrics,
                f,
                ensure_ascii=False,
                indent=2,
            )

        run_info = {
            **video_info,
            "engine": "opencv_stitcher",
            "parameters": {
                "sample_interval": sample_interval,
                "max_frames": max_frames,
                "max_width": max_width,
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
                "rotation_clockwise_degrees": rotation,
                "stitcher_max_frames": 16,
                "max_canvas_side": max_canvas_side,
                "max_canvas_pixels": max_canvas_pixels,
            },
            "stitcher_input_frame_count": len(selected_indices),
            "selected_candidate_indices": selected_indices,
            "selected_timestamps_seconds": selected_timestamps,
            "panorama_width": int(panorama.shape[1]),
            "panorama_height": int(panorama.shape[0]),
            "panorama_path": str(panorama_path),
        }

        with (run_dir / "run_info.json").open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                run_info,
                f,
                ensure_ascii=False,
                indent=2,
            )

        log(
            f"Stitcher input frames: {len(selected_indices)} "
            f"of {len(candidate_frames)}."
        )
        log(
            f"Panorama size: "
            f"{panorama.shape[1]} x {panorama.shape[0]}"
        )
        log(f"Output: {panorama_path}")

        return panorama_path, metrics

    # ======================================================
    # 2. FEATURE EXTRACTION
    # ======================================================

    extractor, norm_type = create_feature_extractor(method)

    candidates = []

    preprocessing_ms = 0.0
    detection_ms = 0.0

    for index, frame in enumerate(candidate_frames):

        start = time.perf_counter()

        color, gray = preprocess_frame(
            frame,
            denoise=denoise,
        )

        preprocessing_ms += (
            time.perf_counter() - start
        ) * 1000

        sharpness = float(
            cv2.Laplacian(
                gray,
                cv2.CV_64F,
            ).var()
        )

        start = time.perf_counter()

        keypoints, descriptors = detect_and_describe(
            gray,
            extractor,
        )

        detection_ms += (
            time.perf_counter() - start
        ) * 1000

        candidates.append({
            "index": index,
            "frame": color,
            "keypoints": keypoints,
            "descriptors": descriptors,
            "sharpness": sharpness,
        })

    log("Feature extraction completed.")

    # ======================================================
    # 3. MATCHING AND KEYFRAME SELECTION
    # ======================================================

    accepted = []

    pair_metrics = []
    pairwise_homographies = []
    pose_constraints = []

    rejected_frames = []

    for candidate in candidates:

        kp_curr = candidate["keypoints"]
        desc_curr = candidate["descriptors"]
        candidate_index = candidate["index"]

        # --------------------------------------------------
        # 3.1. Kiểm tra số lượng keypoints
        # --------------------------------------------------

        if len(kp_curr) < min_keypoints:
            rejected_frames.append({
                "candidate_index": candidate_index,
                "reason": "too_few_keypoints",
                "keypoints": len(kp_curr),
                "sharpness": candidate["sharpness"],
            })
            continue

        # --------------------------------------------------
        # 3.2. Kiểm tra độ sắc nét
        # --------------------------------------------------

        if candidate["sharpness"] < min_sharpness:
            rejected_frames.append({
                "candidate_index": candidate_index,
                "reason": "too_blurry",
                "keypoints": len(kp_curr),
                "sharpness": candidate["sharpness"],
            })
            continue

        # Frame đủ chất lượng đầu tiên.
        if not accepted:
            accepted.append(candidate)

            log(
                f"Keyframe 0 accepted "
                f"(candidate={candidate_index})."
            )

            continue

        previous = accepted[-1]

        kp_prev = previous["keypoints"]
        desc_prev = previous["descriptors"]

        # --------------------------------------------------
        # 3.3. Descriptor matching
        # --------------------------------------------------

        try:
            matches, matching_ms = match_descriptors(
                desc_prev,
                desc_curr,
                norm_type,
                ratio=ratio,
            )

        except cv2.error:
            rejected_frames.append({
                "candidate_index": candidate_index,
                "reason": "matching_failed",
            })
            continue

        if len(matches) < min_matches:
            rejected_frames.append({
                "candidate_index": candidate_index,
                "reason": "not_enough_matches_to_previous_accepted_frame",
                "previous_candidate_index": previous["index"],
                "good_matches": len(matches),
                "required_matches": min_matches,
                "keypoints": len(kp_curr),
                "sharpness": candidate["sharpness"],
            })
            continue

        # --------------------------------------------------
        # 3.4. Estimate Homography
        # --------------------------------------------------

        try:
            H, stats = estimate_pair_homography(
                kp_prev,
                kp_curr,
                matches,
                min_matches=min_matches,
                ransac_threshold=ransac_threshold,
            )

        except (RuntimeError, ValueError, cv2.error) as exc:
            rejected_frames.append({
                "candidate_index": candidate_index,
                "reason": "homography_failed",
                "previous_candidate_index": previous["index"],
                "good_matches": len(matches),
                "error": str(exc),
            })
            continue

        # --------------------------------------------------
        # 3.5. Validate local geometry
        # --------------------------------------------------

        valid_geometry, geometry_reason, geometry_stats = (
            validate_pair_homography(
                H,
                candidate["frame"].shape,
            )
        )

        if not valid_geometry:
            rejected_frames.append({
                "candidate_index": candidate_index,
                "reason": geometry_reason,
                "previous_candidate_index": previous["index"],
                "good_matches": len(matches),
                "inliers": stats["inliers"],
                "inlier_ratio": stats["inlier_ratio"],
                "reprojection_error_px": stats[
                    "reprojection_error_px"
                ],
                "geometry": geometry_stats,
            })
            continue

        corners = np.asarray(
            geometry_stats["projected_corners"],
            dtype=np.float64,
        )

        # --------------------------------------------------
        # 3.6. Validate RANSAC quality
        # --------------------------------------------------

        reasonable_matches = (
            stats["inliers"] >= 4
            and stats["inlier_ratio"] >= 0.15
            and stats["reprojection_error_px"] <= 5.0
        )

        if not reasonable_matches:
            rejected_frames.append({
                "candidate_index": candidate_index,
                "reason": "poor_matching_quality",
                "previous_candidate_index": previous["index"],
                "good_matches": len(matches),
                "inliers": stats["inliers"],
                "inlier_ratio": stats["inlier_ratio"],
                "reprojection_error_px": stats[
                    "reprojection_error_px"
                ],
            })
            continue

        # --------------------------------------------------
        # 3.7. Validate overlap
        # --------------------------------------------------

        overlap_ratio = calculate_overlap_ratio(
            corners,
            previous["frame"].shape,
        )

        if overlap_ratio < min_overlap_ratio:
            rejected_frames.append({
                "candidate_index": candidate_index,
                "reason": "insufficient_overlap",
                "previous_candidate_index": previous["index"],
                "overlap_ratio": overlap_ratio,
            })
            continue

        if overlap_ratio > max_overlap_ratio:
            rejected_frames.append({
                "candidate_index": candidate_index,
                "reason": "redundant_viewpoint",
                "previous_candidate_index": previous["index"],
                "overlap_ratio": overlap_ratio,
                "required_max_overlap_ratio": max_overlap_ratio,
            })
            continue

        # --------------------------------------------------
        # 3.8. Accept keyframe
        # --------------------------------------------------

        stats.update({
            "previous_candidate_index": previous["index"],
            "current_candidate_index": candidate_index,
            "previous_keypoints": len(kp_prev),
            "current_keypoints": len(kp_curr),
            "matching_time_ms": matching_ms,
            "previous_sharpness": previous["sharpness"],
            "current_sharpness": candidate["sharpness"],
            "overlap_ratio": overlap_ratio,
            "geometry": {
                "edge_ratios": geometry_stats["edge_ratios"],
                "area_ratio": geometry_stats["area_ratio"],
            },
        })

        previous_position = len(accepted) - 1

        accepted.append(candidate)

        current_position = len(accepted) - 1

        pair_metrics.append(stats)
        pairwise_homographies.append(H)

        pose_constraints.append({
            "i": previous_position,
            "j": current_position,
            "H": H,
            "type": "adjacent",
        })

        log(
            f"Keyframe {current_position} accepted "
            f"(candidate={candidate_index}, "
            f"inliers={stats['inliers']}, "
            f"overlap={overlap_ratio:.2f})."
        )

        # ==================================================
        # 4. LOOP CONSTRAINTS
        # ==================================================

        anchor_positions = [
            current_position - 2,
            current_position - 4,
            current_position - 8,
        ]

        for anchor_position in anchor_positions:

            if anchor_position < 0:
                continue

            anchor = accepted[anchor_position]

            try:
                extra_matches, _ = match_descriptors(
                    anchor["descriptors"],
                    desc_curr,
                    norm_type,
                    ratio=ratio,
                )

                if len(extra_matches) < min_matches:
                    continue

                extra_H, extra_stats = estimate_pair_homography(
                    anchor["keypoints"],
                    kp_curr,
                    extra_matches,
                    min_matches=min_matches,
                    ransac_threshold=ransac_threshold,
                )

                valid_extra, _, extra_geometry_stats = (
                    validate_pair_homography(
                        extra_H,
                        candidate["frame"].shape,
                    )
                )

                if not valid_extra:
                    continue

                if (
                    extra_stats["inlier_ratio"] < 0.20
                    or extra_stats["reprojection_error_px"] > 4.0
                    or extra_stats["inliers"] < min_matches
                ):
                    continue

                extra_corners = np.asarray(
                    extra_geometry_stats["projected_corners"],
                    dtype=np.float64,
                )

                extra_overlap = calculate_overlap_ratio(
                    extra_corners,
                    anchor["frame"].shape,
                )

                if extra_overlap < min_overlap_ratio:
                    continue

                pose_constraints.append({
                    "i": anchor_position,
                    "j": current_position,
                    "H": extra_H,
                    "type": "loop",
                })

                log(
                    f"Loop constraint added: "
                    f"{anchor_position} -> {current_position} "
                    f"(inliers={extra_stats['inliers']}, "
                    f"error={extra_stats['reprojection_error_px']:.2f}px).",
                    "LOOP",
                )

            except (RuntimeError, ValueError, cv2.error):
                continue

    # ======================================================
    # 5. VALIDATE ACCEPTED KEYFRAMES
    # ======================================================

    if len(accepted) < 2:
        raise RuntimeError(
            f"Chỉ giữ lại được {len(accepted)} frame phù hợp; "
            "cần ít nhất 2. Hãy điều chỉnh các ngưỡng lọc "
            "hoặc kiểm tra video."
        )

    selected_frames = [
        item["frame"] for item in accepted
    ]

    if len(pairwise_homographies) != len(accepted) - 1:
        raise RuntimeError(
            "Số Homography liên tiếp không khớp số keyframe."
        )

    if len(pose_constraints) < len(accepted) - 1:
        raise RuntimeError(
            "Không đủ pose constraints."
        )

    # ======================================================
    # 6. GLOBAL TRANSFORM OPTIMIZATION
    # ======================================================

    anchor_index = len(selected_frames) // 2
    centered_transforms = build_centered_global_transforms(
        pairwise_homographies,
        frame_count=len(selected_frames),
        anchor_index=anchor_index,
    )

    frame_order = (
        [anchor_index]
        + list(range(anchor_index + 1, len(selected_frames)))
        + list(range(anchor_index - 1, -1, -1))
    )
    original_to_ordered = {
        original_index: ordered_index
        for ordered_index, original_index in enumerate(frame_order)
    }

    optimization_frames = [
        selected_frames[index]
        for index in frame_order
    ]
    initial_transforms = [
        centered_transforms[index]
        for index in frame_order
    ]

    optimization_constraints = []

    for constraint in pose_constraints:
        i = original_to_ordered[constraint["i"]]
        j = original_to_ordered[constraint["j"]]
        H = constraint["H"]

        if i > j:
            i, j = j, i
            H = np.linalg.inv(H)

        optimization_constraints.append({
            **constraint,
            "i": i,
            "j": j,
            "H": H,
        })

    log(
        f"Starting global optimization "
        f"({len(accepted)} keyframes, "
        f"{len(pose_constraints)} constraints, "
        f"anchor={anchor_index})."
    )

    transforms, optimization_metrics = optimize_global_transforms(
        initial_transforms=initial_transforms,
        frames=optimization_frames,
        pose_constraints=optimization_constraints,
        max_nfev=300,
        loss="soft_l1",
        f_scale=3.0,
        #debug=True,
    )

    if len(transforms) != len(optimization_frames):
        raise RuntimeError(
            "Số Global Transform sau tối ưu không khớp keyframe."
        )

    if optimization_metrics["success"]:
        log("Global optimization completed.")
    else:
        log(
            "Global optimization reached its evaluation limit; "
            "using the best validated transforms.",
            "WARNING",
        )

    log(
        f"Pose error: "
        f"{optimization_metrics['initial_mean_absolute_error_px']:.4f}px "
        f"-> "
        f"{optimization_metrics['final_mean_absolute_error_px']:.4f}px."
    )

    # ======================================================
    # 7. CREATE PANORAMA
    # ======================================================

    panorama = create_panorama(
        optimization_frames,
        transforms,
        max_canvas_side=max_canvas_side,
        max_canvas_pixels=max_canvas_pixels,
    )

    panorama_path = run_dir / "panorama.jpg"

    if not cv2.imwrite(str(panorama_path), panorama):
        raise RuntimeError(
            f"Không ghi được panorama: {panorama_path}"
        )

    # ======================================================
    # 8. CALCULATE METRICS
    # ======================================================

    total_ms = (
        time.perf_counter() - total_start
    ) * 1000

    accepted_indices = [
        item["index"] for item in accepted
    ]

    adjacent_count = sum(
        item["type"] == "adjacent"
        for item in pose_constraints
    )

    loop_count = sum(
        item["type"] == "loop"
        for item in pose_constraints
    )

    metrics = {
        "method": method,

        "candidate_frame_count": len(candidate_frames),
        "accepted_frame_count": len(accepted),
        "skipped_frame_count": (
            len(candidate_frames) - len(accepted)
        ),

        "average_keypoints": safe_average([
            len(item["keypoints"])
            for item in accepted
        ]),

        "average_good_matches": safe_average([
            item["good_matches"]
            for item in pair_metrics
        ]),

        "average_inlier_ratio": safe_average([
            item["inlier_ratio"]
            for item in pair_metrics
        ]),

        "average_reprojection_error_px": safe_average([
            item["reprojection_error_px"]
            for item in pair_metrics
        ]),

        "average_overlap_ratio": safe_average([
            item["overlap_ratio"]
            for item in pair_metrics
        ]),

        "preprocessing_time_ms_total": preprocessing_ms,
        "feature_detection_time_ms_total": detection_ms,
        "total_time_ms": total_ms,

        "accepted_candidate_indices": accepted_indices,

        "accepted_timestamps_seconds": [
            video_info["sampled_timestamps_seconds"][index]
            for index in accepted_indices
        ],

        "pairs": pair_metrics,
        "rejected_frames": rejected_frames,

        "global_optimization": {
            **optimization_metrics,
            "pose_constraint_count": len(pose_constraints),
            "adjacent_constraint_count": adjacent_count,
            "loop_constraint_count": loop_count,
        },
    }

    with (run_dir / "metrics.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ======================================================
    # 9. SAVE RUN INFORMATION
    # ======================================================

    run_info = {
        **video_info,

        "method": method,

        "parameters": {
            "sample_interval": sample_interval,
            "max_frames": max_frames,
            "max_width": max_width,
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
            "rotation_clockwise_degrees": rotation,
            "min_matches": min_matches,
            "ratio": ratio,
            "ransac_threshold": ransac_threshold,
            "denoise": denoise,
            "min_keypoints": min_keypoints,
            "min_sharpness": min_sharpness,
            "min_overlap_ratio": min_overlap_ratio,
            "max_overlap_ratio": max_overlap_ratio,
            "max_canvas_side": max_canvas_side,
            "max_canvas_pixels": max_canvas_pixels,
        },

        "accepted_frame_count": len(accepted),
        "skipped_frame_count": (
            len(candidate_frames) - len(accepted)
        ),

        "panorama_width": int(panorama.shape[1]),
        "panorama_height": int(panorama.shape[0]),
        "panorama_path": str(panorama_path),
    }

    with (run_dir / "run_info.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            run_info,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ======================================================
    # 10. FINAL SUMMARY
    # ======================================================

    log("=" * 50)
    log("PANORAMA GENERATION COMPLETED")
    log(f"Method: {method.upper()}")
    log(f"Candidate frames: {len(candidate_frames)}")
    log(f"Accepted frames: {len(accepted)}")
    log(f"Rejected frames: {len(rejected_frames)}")

    log(
        f"Pose constraints: {len(pose_constraints)} "
        f"(adjacent={adjacent_count}, loop={loop_count})"
    )

    log(
        f"Panorama size: "
        f"{panorama.shape[1]} x {panorama.shape[0]}"
    )

    log(f"Total time: {total_ms / 1000:.2f}s")
    log(f"Output: {panorama_path}")
    log("=" * 50)

    return panorama_path, metrics
