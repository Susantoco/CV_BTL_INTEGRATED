
import cv2
import numpy as np
import time


def match_descriptors(desc_a, desc_b, norm_type, ratio: float = 0.75):
    """
    Ghép descriptor giữa hai frame.

    desc_a: descriptor của frame tham chiếu (frame trước).
    desc_b: descriptor của frame hiện tại.

    queryIdx thuộc frame trước.
    trainIdx thuộc frame hiện tại.
    """
    if not 0 < ratio < 1:
        raise ValueError("ratio phải nằm trong khoảng (0, 1).")

    if (
        desc_a is None
        or desc_b is None
        or len(desc_a) < 2
        or len(desc_b) < 2
    ):
        return [], 0.0

    matcher = cv2.BFMatcher(norm_type, crossCheck=False)

    start = time.perf_counter()
    pairs = matcher.knnMatch(desc_a, desc_b, k=2)
    elapsed_ms = (time.perf_counter() - start) * 1000

    good = []

    for pair in pairs:
        if len(pair) != 2:
            continue

        first, second = pair

        if first.distance < ratio * second.distance:
            good.append(first)

    return good, elapsed_ms


def estimate_pair_homography(
    keypoints_previous,
    keypoints_current,
    matches,
    min_matches: int = 12,
    ransac_threshold: float = 4.0,
):
    """
    Ước lượng Homography ánh xạ frame hiện tại về frame trước.

    H: current frame -> previous frame.

    Lưu ý:
        matches được tạo từ:
            match_descriptors(desc_previous, desc_current)

        Do đó:
            queryIdx = frame trước
            trainIdx = frame hiện tại
    """
    if min_matches < 4:
        raise ValueError("min_matches phải lớn hơn hoặc bằng 4.")

    if ransac_threshold <= 0:
        raise ValueError("ransac_threshold phải lớn hơn 0.")

    if keypoints_previous is None or keypoints_current is None:
        raise RuntimeError("Keypoints không hợp lệ.")

    if len(matches) < min_matches:
        raise RuntimeError(
            f"Không đủ good matches: {len(matches)} < {min_matches}."
        )

    # Nguồn: frame hiện tại.
    src = np.float32([
        keypoints_current[m.trainIdx].pt
        for m in matches
    ]).reshape(-1, 1, 2)

    # Đích: frame trước.
    dst = np.float32([
        keypoints_previous[m.queryIdx].pt
        for m in matches
    ]).reshape(-1, 1, 2)

    if not np.isfinite(src).all() or not np.isfinite(dst).all():
        raise RuntimeError("Tọa độ keypoint chứa NaN hoặc Inf.")

    start = time.perf_counter()

    try:
        H, mask = cv2.findHomography(
            src,
            dst,
            cv2.RANSAC,
            ransac_threshold,
        )
    except cv2.error as exc:
        raise RuntimeError(
            f"OpenCV không thể ước lượng Homography: {exc}"
        ) from exc

    elapsed_ms = (time.perf_counter() - start) * 1000

    if H is None or mask is None:
        raise RuntimeError("RANSAC không tìm được Homography.")

    H = np.asarray(H, dtype=np.float64)

    if H.shape != (3, 3):
        raise RuntimeError("Homography không có kích thước 3x3.")

    if not np.isfinite(H).all():
        raise RuntimeError("Homography chứa NaN hoặc Inf.")

    if np.linalg.matrix_rank(H) < 3:
        raise RuntimeError("Homography bị suy biến.")

    # Chuẩn hóa Homography.
    scale = H[2, 2]

    if abs(scale) > 1e-12:
        H /= scale
    else:
        norm = np.linalg.norm(H)

        if norm <= 1e-12:
            raise RuntimeError("Homography không thể chuẩn hóa.")

        H /= norm

    inlier_mask = mask.ravel().astype(bool)

    if len(inlier_mask) != len(matches):
        raise RuntimeError("Kích thước RANSAC mask không khớp matches.")

    inlier_count = int(inlier_mask.sum())

    if inlier_count < 4:
        raise RuntimeError(
            f"Homography chỉ có {inlier_count} inliers."
        )

    # Tính reprojection error trên inliers.
    projected = cv2.perspectiveTransform(src, H).reshape(-1, 2)
    target = dst.reshape(-1, 2)

    errors = np.linalg.norm(
        projected[inlier_mask] - target[inlier_mask],
        axis=1,
    )

    if not np.isfinite(errors).all():
        raise RuntimeError("Sai số reprojection không hợp lệ.")

    reprojection_error = float(errors.mean())

    return H, {
        "good_matches": len(matches),
        "inliers": inlier_count,
        "inlier_ratio": inlier_count / len(matches),
        "reprojection_error_px": reprojection_error,
        "homography_time_ms": elapsed_ms,
    }
