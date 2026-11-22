
import cv2
import numpy as np

from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


def _debug_print(debug, *args, **kwargs):
    """Chỉ in thông tin khi debug được bật."""
    if debug:
        print(*args, **kwargs)


# ==========================================================
# 1. HOMOGRAPHY UTILITIES
# ==========================================================

def _normalize_homography(H):
    """Chuẩn hóa và kiểm tra Homography."""

    H = np.asarray(H, dtype=np.float64)

    if H.shape != (3, 3):
        raise RuntimeError(
            "Homography phải có kích thước 3x3."
        )

    if not np.isfinite(H).all():
        raise RuntimeError(
            "Homography chứa NaN hoặc Inf."
        )

    scale = np.max(np.abs(H))

    if not np.isfinite(scale) or scale < 1e-12:
        raise RuntimeError(
            "Homography không thể chuẩn hóa."
        )

    H = H / scale

    singular_values = np.linalg.svd(
        H,
        compute_uv=False,
    )

    if (
        not np.isfinite(singular_values).all()
        or singular_values[-1] <= singular_values[0] * 1e-12
    ):
        raise RuntimeError(
            "Homography gần như suy biến."
        )

    return H


def _transformed_corners(image, transform, debug=False):
    """
    Tính tọa độ bốn góc ảnh sau phép biến đổi.

    Phát hiện trường hợp projective horizon
    cắt qua frame.
    """

    if image is None or image.size == 0:
        raise RuntimeError("Frame không hợp lệ.")

    h, w = image.shape[:2]

    corners = np.float64([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h],
    ])

    H = _normalize_homography(transform)

    homogeneous = np.column_stack([
        corners,
        np.ones(4, dtype=np.float64),
    ])

    projected_h = (
        H @ homogeneous.T
    ).T

    denominators = projected_h[:, 2]

    _debug_print(
        debug,
        f"\n[HORIZON DEBUG]"
        f"\nDenominators: {denominators}"
        f"\nTransform:\n{H}"
    )

    if (
        not np.isfinite(denominators).all()
        or np.any(np.abs(denominators) < 1e-8)
        or not (
            np.all(denominators > 0)
            or np.all(denominators < 0)
        )
    ):
        raise RuntimeError(
            "Phép chiếu không hợp lệ: "
            "frame có thể đi qua projective horizon."
        )

    result = (
        projected_h[:, :2]
        / denominators[:, None]
    )

    if not np.isfinite(result).all():
        raise RuntimeError(
            "Tọa độ góc ảnh sau biến đổi không hợp lệ."
        )

    return result


def _get_frame_geometry(image, transform, debug=False):
    """Lấy thông tin hình học của frame sau biến đổi."""

    h, w = image.shape[:2]

    corners = _transformed_corners(
        image,
        transform,
        debug=debug,
    )

    edge_vectors = (
        np.roll(corners, -1, axis=0) - corners
    )

    edge_lengths = np.linalg.norm(
        edge_vectors,
        axis=1,
    )

    original_edge_lengths = np.array([
        w,
        h,
        w,
        h,
    ], dtype=np.float64)

    if np.any(edge_lengths <= 1e-8):
        raise RuntimeError(
            "Frame bị suy biến sau phép biến đổi."
        )

    edge_ratios = (
        edge_lengths / original_edge_lengths
    )

    width = float(
        corners[:, 0].max() - corners[:, 0].min()
    )

    height = float(
        corners[:, 1].max() - corners[:, 1].min()
    )

    area = abs(
        cv2.contourArea(
            corners.astype(np.float32)
        )
    )

    area_ratio = area / float(w * h)

    return {
        "corners": corners,
        "width": width,
        "height": height,
        "edge_ratios": edge_ratios.tolist(),
        "area_ratio": float(area_ratio),
    }


# ==========================================================
# 2. CANVAS UTILITIES
# ==========================================================

def _get_canvas_bounds(frames, transforms, debug=False):
    """Tính bounding box bao phủ toàn bộ frame."""

    if (
        frames is None
        or transforms is None
        or len(frames) == 0
        or len(frames) != len(transforms)
    ):
        raise RuntimeError(
            "Số lượng frame và transform không khớp."
        )

    all_corners = []

    for index, (frame, transform) in enumerate(
        zip(frames, transforms)
    ):
        try:
            corners = _transformed_corners(
                frame,
                transform,
                debug=debug,
            )

        except (RuntimeError, cv2.error) as exc:
            raise RuntimeError(
                f"Không thể tính tọa độ frame "
                f"{index}: {exc}"
            ) from exc

        all_corners.append(corners)

    all_corners = np.concatenate(
        all_corners,
        axis=0,
    )

    min_xy = np.floor(
        all_corners.min(axis=0)
    ).astype(np.int64)

    max_xy = np.ceil(
        all_corners.max(axis=0)
    ).astype(np.int64)

    width, height = (
        max_xy - min_xy
    ).tolist()

    return min_xy, max_xy, width, height


def _validate_canvas(
    width,
    height,
    max_canvas_side,
    max_canvas_pixels,
    context="Canvas",
):
    """Kiểm tra kích thước canvas trước khi cấp phát."""

    if width <= 0 or height <= 0:
        raise RuntimeError(
            f"{context} có kích thước không hợp lệ: "
            f"{width}x{height}."
        )

    if (
        width > max_canvas_side
        or height > max_canvas_side
    ):
        raise RuntimeError(
            f"{context} quá lớn: {width}x{height}. "
            f"Giới hạn mỗi cạnh là {max_canvas_side}."
        )

    total_pixels = width * height

    if total_pixels > max_canvas_pixels:
        raise RuntimeError(
            f"{context} cần {total_pixels:,} pixels, "
            f"vượt giới hạn {max_canvas_pixels:,} pixels."
        )


# ==========================================================
# 3. GLOBAL TRANSFORMS
# ==========================================================

def build_global_transforms(
    pairwise_homographies,
    frames=None,
    max_canvas_side=12000,
    max_canvas_pixels=25_000_000,
    debug=False,
):
    """
    Tích lũy Homography từ các keyframe.

    pairwise_homographies[i]:
        Ánh xạ frame i+1 -> frame i.

    T[i]:
        Ánh xạ frame i -> hệ tọa độ frame 0.
    """

    if frames is not None:

        if len(frames) < 2:
            raise RuntimeError(
                "Cần ít nhất 2 frame để tạo panorama."
            )

        if (
            len(pairwise_homographies)
            != len(frames) - 1
        ):
            raise RuntimeError(
                "Số Homography phải bằng số frame trừ 1."
            )

    transforms = [
        np.eye(3, dtype=np.float64)
    ]

    if frames is not None:

        first_corners = _transformed_corners(
            frames[0],
            transforms[0],
            debug=debug,
        )

        min_xy = first_corners.min(axis=0)
        max_xy = first_corners.max(axis=0)

    for index, H in enumerate(pairwise_homographies):

        H = _normalize_homography(H)

        # H: frame hiện tại -> frame trước.
        # T_previous: frame trước -> frame đầu tiên.
        # T_current: frame hiện tại -> frame đầu tiên.

        current_transform = _normalize_homography(
            transforms[-1] @ H
        )

        transforms.append(current_transform)

        _debug_print(
            debug,
            f"\n[GLOBAL TRANSFORM DEBUG]"
            f"\nFrame index: {index + 1}"
            f"\nPairwise H:\n{H}"
            f"\nGlobal T:\n{current_transform}"
        )

        if frames is None:
            continue

        current_frame = frames[index + 1]

        h, w = current_frame.shape[:2]

        corners_h = np.float64([
            [0, 0, 1],
            [w, 0, 1],
            [w, h, 1],
            [0, h, 1],
        ])

        projected_h = (
            current_transform @ corners_h.T
        ).T

        denominators = projected_h[:, 2]

        _debug_print(
            debug,
            f"\n[GLOBAL HORIZON CHECK]"
            f"\nFrame: {index + 1}"
            f"\nDenominators: {denominators}"
        )

        if not (
            np.all(denominators > 0)
            or np.all(denominators < 0)
        ):
            _debug_print(
                debug,
                f"[GLOBAL HORIZON WARNING] "
                f"Frame {index + 1} "
                f"has crossed projective horizon."
            )

        projected_previous = (
            H @ corners_h.T
        ).T

        pairwise_denominators = (
            projected_previous[:, 2]
        )

        projected_global = (
            transforms[-2]
            @ projected_previous.T
        ).T

        global_denominators = (
            projected_global[:, 2]
        )

        _debug_print(
            debug,
            f"\n[TRANSFORM DEBUG] Frame {index + 1}"
            f"\nPairwise denominators: "
            f"{pairwise_denominators}"
            f"\nGlobal denominators: "
            f"{global_denominators}"
        )

        try:
            geometry = _get_frame_geometry(
                current_frame,
                current_transform,
                debug=debug,
            )

        except (RuntimeError, cv2.error) as exc:
            raise RuntimeError(
                f"Global transform tại keyframe "
                f"{index + 1} không hợp lệ: {exc}"
            ) from exc

        current_corners = geometry["corners"]

        min_xy = np.minimum(
            min_xy,
            current_corners.min(axis=0),
        )

        max_xy = np.maximum(
            max_xy,
            current_corners.max(axis=0),
        )

        canvas_min = np.floor(
            min_xy
        ).astype(np.int64)

        canvas_max = np.ceil(
            max_xy
        ).astype(np.int64)

        width, height = (
            canvas_max - canvas_min
        ).tolist()

        _debug_print(
            debug,
            f"Projected size: "
            f"{geometry['width']:.2f} x "
            f"{geometry['height']:.2f}"
        )

        _debug_print(
            debug,
            f"Edge ratios: "
            f"{np.round(geometry['edge_ratios'], 4)}"
        )

        _debug_print(
            debug,
            f"Area ratio: "
            f"{geometry['area_ratio']:.4f}"
        )

        _debug_print(
            debug,
            f"Canvas: {width}x{height} "
            f"({width * height:,} pixels)"
        )

        _validate_canvas(
            width,
            height,
            max_canvas_side,
            max_canvas_pixels,
            context=f"Canvas tại keyframe {index + 1}",
        )

    return transforms


def build_centered_global_transforms(
    pairwise_homographies,
    frame_count,
    anchor_index,
):
    """Tích lũy Homography hai phía quanh frame neo."""

    if frame_count < 2:
        raise RuntimeError("Cần ít nhất 2 frame.")

    if len(pairwise_homographies) != frame_count - 1:
        raise RuntimeError(
            "Số Homography phải bằng số frame trừ 1."
        )

    if not 0 <= anchor_index < frame_count:
        raise ValueError("anchor_index không hợp lệ.")

    transforms = [None] * frame_count
    transforms[anchor_index] = np.eye(3, dtype=np.float64)

    for index in range(anchor_index + 1, frame_count):
        H = _normalize_homography(
            pairwise_homographies[index - 1]
        )
        transforms[index] = _normalize_homography(
            transforms[index - 1] @ H
        )

    for index in range(anchor_index - 1, -1, -1):
        H = _normalize_homography(
            pairwise_homographies[index]
        )

        try:
            inverse_H = np.linalg.inv(H)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                f"Homography {index} không thể đảo ngược."
            ) from exc

        transforms[index] = _normalize_homography(
            transforms[index + 1] @ inverse_H
        )

    return transforms


# ==========================================================
# 4. PANORAMA GENERATION
# ==========================================================

def create_panorama(
    frames,
    transforms,
    max_canvas_side=12000,
    max_canvas_pixels=25_000_000,
    debug=False,
):
    """Ghép các keyframe thành panorama bằng feather blending."""

    if (
        frames is None
        or len(frames) == 0
        or len(frames) != len(transforms)
    ):
        raise RuntimeError(
            "Số lượng frame và transform không khớp."
        )

    print("\n[ORIENTATION DEBUG]")

    for index in sorted(set([
        0,
        len(frames) // 2,
        len(frames) - 1,
    ])):
        frame = frames[index]
        h, w = frame.shape[:2]

        corners = np.float32([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1],
        ]).reshape(-1, 1, 2)

        H = _normalize_homography(transforms[index])

        projected = cv2.perspectiveTransform(
            corners,
            H,
        ).reshape(-1, 2)

        print(f"\nFrame {index}:")
        print(f"Original size: {w}x{h}")
        print("Transform:")
        print(H)
        print("Projected corners:")
        print(projected)

    min_xy, max_xy, width, height = _get_canvas_bounds(
        frames,
        transforms,
        debug=debug,
    )

    _validate_canvas(
        width,
        height,
        max_canvas_side,
        max_canvas_pixels,
        context="Panorama",
    )

    translation = np.array([
        [1, 0, -min_xy[0]],
        [0, 1, -min_xy[1]],
        [0, 0, 1],
    ], dtype=np.float64)

    accumulator = np.zeros(
        (height, width, 3),
        dtype=np.float32,
    )

    weight_sum = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    for index, (frame, transform) in enumerate(
        zip(frames, transforms)
    ):

        if frame is None or frame.size == 0:
            raise RuntimeError(
                f"Frame {index} không hợp lệ."
            )

        warp_matrix = (
            translation
            @ _normalize_homography(transform)
        )

        warped = cv2.warpPerspective(
            frame,
            warp_matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
        )

        mask = np.full(
            frame.shape[:2],
            255,
            dtype=np.uint8,
        )

        warped_mask = cv2.warpPerspective(
            mask,
            warp_matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
        )

        valid = warped_mask > 0

        source_mask = np.pad(
            np.ones(frame.shape[:2], dtype=np.uint8),
            1,
        )
        source_weights = cv2.distanceTransform(
            source_mask,
            cv2.DIST_L2,
            3,
        )[1:-1, 1:-1]
        warped_weights = cv2.warpPerspective(
            source_weights,
            warp_matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
        )
        warped_weights = np.maximum(warped_weights, 1e-3)

        accumulator[valid] += (
            warped[valid].astype(np.float32)
            * warped_weights[valid, None]
        )

        weight_sum[valid] += warped_weights[valid]

        _debug_print(
            debug,
            f"[BLENDING] Frame {index + 1}/{len(frames)}"
        )

    valid = weight_sum > 0

    panorama = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    panorama[valid] = np.clip(
        accumulator[valid]
        / weight_sum[valid, None],
        0,
        255,
    ).astype(np.uint8)

    return panorama


def create_opencv_panorama(
    frames,
    max_frames=16,
    max_canvas_side=12000,
    max_canvas_pixels=25_000_000,
):
    """Ghép panorama rộng bằng bộ Stitcher của OpenCV."""

    if frames is None or len(frames) < 2:
        raise RuntimeError(
            "Cần ít nhất 2 frame để tạo panorama."
        )

    if max_frames < 2:
        raise ValueError("max_frames phải lớn hơn hoặc bằng 2.")

    if len(frames) > max_frames:
        frame_indices = np.linspace(
            0,
            len(frames) - 1,
            num=max_frames,
            dtype=np.int64,
        )
    else:
        frame_indices = np.arange(len(frames), dtype=np.int64)

    selected_frames = [
        frames[int(index)]
        for index in frame_indices
    ]

    stitcher = cv2.Stitcher_create(
        cv2.Stitcher_PANORAMA
    )

    try:
        status, panorama = stitcher.stitch(selected_frames)
    except cv2.error as exc:
        raise RuntimeError(
            f"OpenCV Stitcher không thể ghép panorama: {exc}"
        ) from exc

    if status != cv2.Stitcher_OK or panorama is None:
        raise RuntimeError(
            "OpenCV Stitcher không thể ghép các frame. "
            f"Status: {status}."
        )

    _validate_canvas(
        panorama.shape[1],
        panorama.shape[0],
        max_canvas_side,
        max_canvas_pixels,
        context="Panorama OpenCV",
    )

    return panorama, frame_indices.tolist()


# ==========================================================
# 5. HOMOGRAPHY PARAMETERIZATION
# ==========================================================

def _homography_to_parameters(H):
    """
    Chuyển Homography 3x3 thành 8 tham số.

    Chọn phần tử có trị tuyệt đối lớn nhất làm pivot.
    Pivot được chuẩn hóa thành +1.
    """

    H = _normalize_homography(H)

    flat = H.reshape(-1)

    pivot_index = int(
        np.argmax(np.abs(flat))
    )

    pivot = flat[pivot_index]

    if abs(pivot) < 1e-12:
        raise RuntimeError(
            "Không thể tham số hóa Homography."
        )

    H = H / pivot

    parameters = np.delete(
        H.reshape(-1),
        pivot_index,
    )

    return parameters, pivot_index


def _parameters_to_homography(
    parameters,
    pivot_index,
):
    """Khôi phục Homography từ 8 tham số."""

    parameters = np.asarray(
        parameters,
        dtype=np.float64,
    )

    if parameters.shape != (8,):
        raise RuntimeError(
            "Homography cần đúng 8 tham số."
        )

    if not np.isfinite(parameters).all():
        raise RuntimeError(
            "Tham số Homography chứa NaN hoặc Inf."
        )

    if not 0 <= pivot_index < 9:
        raise RuntimeError(
            "Pivot index không hợp lệ."
        )

    flat = np.insert(
        parameters,
        pivot_index,
        1.0,
    )

    H = flat.reshape(3, 3)

    if not np.isfinite(H).all():
        raise RuntimeError(
            "Homography sau giải mã không hợp lệ."
        )

    return H


# ==========================================================
# 6. SAFE POINT PROJECTION
# ==========================================================

def _project_points(points, H):
    """
    Chiếu điểm qua Homography.

    Dùng cho các phép chiếu cần kiểm tra nghiêm ngặt.
    """

    points = np.asarray(points, dtype=np.float64)
    points = points.reshape(-1, 2)

    H = np.asarray(
        H,
        dtype=np.float64,
    )

    if H.shape != (3, 3):
        raise RuntimeError(
            "Homography phải có kích thước 3x3."
        )

    if not np.isfinite(H).all():
        raise RuntimeError(
            "Homography chứa NaN hoặc Inf."
        )

    homogeneous = np.column_stack([
        points,
        np.ones(len(points), dtype=np.float64),
    ])

    projected_h = (
        H @ homogeneous.T
    ).T

    if not np.isfinite(projected_h).all():
        raise RuntimeError(
            "Phép chiếu tạo ra NaN hoặc Inf."
        )

    denominator = projected_h[:, 2]

    denominator_scale = np.sum(
        np.abs(
            H[2][None, :] * homogeneous
        ),
        axis=1,
    )

    threshold = np.maximum(
        1e-12,
        denominator_scale * 1e-10,
    )

    if np.any(
        np.abs(denominator) <= threshold
    ):
        raise RuntimeError(
            "Mẫu số phối cảnh quá gần 0."
        )

    projected = (
        projected_h[:, :2]
        / denominator[:, None]
    )

    if not np.isfinite(projected).all():
        raise RuntimeError(
            "Tọa độ sau phép chiếu không hợp lệ."
        )

    return projected


def _project_points_for_optimization(points, H):
    """
    Chiếu điểm an toàn trong quá trình tối ưu.

    Cho phép denominator đổi dấu trong nghiệm trung gian.
    Horizon được xử lý bằng residual riêng.
    """

    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    H = np.asarray(H, dtype=np.float64)

    homogeneous = np.column_stack([
        points,
        np.ones(len(points), dtype=np.float64),
    ])

    projected_h = (
        H @ homogeneous.T
    ).T

    numerator = projected_h[:, :2]
    denominator = projected_h[:, 2]

    denominator_scale = np.sum(
        np.abs(
            H[2][None, :] * homogeneous
        ),
        axis=1,
    )

    threshold = np.maximum(
        1e-10,
        denominator_scale * 1e-10,
    )

    valid = (
        np.isfinite(projected_h).all(axis=1)
        & (np.abs(denominator) > threshold)
    )

    safe_denominator = np.copysign(
        np.maximum(
            np.abs(denominator),
            threshold,
        ),
        np.where(
            denominator == 0,
            1.0,
            denominator,
        ),
    )

    projected = (
        numerator
        / safe_denominator[:, None]
    )

    projected = np.nan_to_num(
        projected,
        nan=1e5,
        posinf=1e5,
        neginf=-1e5,
    )

    projected = np.clip(
        projected,
        -1e5,
        1e5,
    )

    return projected, valid


# ==========================================================
# 7. COORDINATE NORMALIZATION
# ==========================================================

def _get_coordinate_normalization(frame):
    """
    Tạo ma trận chuẩn hóa tọa độ pixel về [-1, 1].

    N maps pixel coordinates -> normalized coordinates.
    """

    if frame is None or frame.size == 0:
        raise RuntimeError("Frame không hợp lệ.")

    h, w = frame.shape[:2]

    if w <= 0 or h <= 0:
        raise RuntimeError(
            "Kích thước frame không hợp lệ."
        )

    return np.array([
        [2.0 / w, 0.0, -1.0],
        [0.0, 2.0 / h, -1.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def _convert_homography_coordinates(
    H,
    source_normalization,
    target_normalization,
):
    """
    Chuyển Homography từ tọa độ pixel sang normalized.

    H: source -> target.

    H_normalized =
        N_target @ H @ inv(N_source)
    """

    H = np.asarray(
        H,
        dtype=np.float64,
    )

    H_normalized = (
        target_normalization
        @ H
        @ np.linalg.inv(source_normalization)
    )

    return _normalize_homography(
        H_normalized
    )


# ==========================================================
# 8. POSE ERROR METRIC
# ==========================================================

def _calculate_pose_error_pixel(
    transforms,
    prepared_constraints,
):
    """
    Tính Pose MAE theo đơn vị pixel.

    transforms:
        Global transforms trong pixel coordinates.

    prepared_constraints:
        Các pose constraints đã được chuẩn bị.
    """

    residuals = []

    for constraint in prepared_constraints:

        i = constraint["i"]
        j = constraint["j"]

        H = constraint["H_pixel"]
        points = constraint["points_pixel"]

        projected_global, valid_global = (
            _project_points_for_optimization(
                points,
                transforms[j],
            )
        )

        reference_transform = (
            transforms[i] @ H
        )

        projected_reference, valid_reference = (
            _project_points_for_optimization(
                points,
                reference_transform,
            )
        )

        difference = (
            projected_global - projected_reference
        )

        invalid = (
            ~valid_global | ~valid_reference
        )

        if np.any(invalid):
            difference[invalid] = 1e5

        residuals.extend(
            difference.reshape(-1)
        )

    residuals = np.asarray(
        residuals,
        dtype=np.float64,
    )

    if len(residuals) == 0:
        raise RuntimeError(
            "Không có residual để đánh giá Pose Error."
        )

    return float(
        np.mean(np.abs(residuals))
    )


# ==========================================================
# 9. GLOBAL HOMOGRAPHY OPTIMIZATION
# ==========================================================

def optimize_global_transforms(
    initial_transforms,
    frames,
    pose_constraints,
    max_nfev=1000,
    loss="soft_l1",
    f_scale=3.0,
    debug=False,
):
    """
    Tối ưu đồng thời Global Homography của tất cả keyframe.

    H trong pose_constraints:
        Ánh xạ frame j -> frame i.

    T[i]:
        Ánh xạ frame i -> frame 0.

    Frame 0 được cố định Identity.

    Bao gồm:
        - Coordinate normalization.
        - Pose constraints.
        - Horizon constraints.
        - Sparse Jacobian.
        - Kiểm tra nghiệm sau tối ưu.
    """

    # --------------------------------------------------
    # 0. VALIDATION
    # --------------------------------------------------

    if frames is None:
        raise RuntimeError("Frames không hợp lệ.")

    frame_count = len(frames)

    if frame_count < 2:
        raise RuntimeError(
            "Cần ít nhất 2 frame để tối ưu."
        )

    if len(initial_transforms) != frame_count:
        raise RuntimeError(
            "Số lượng initial transforms không khớp frames."
        )

    if not pose_constraints:
        raise RuntimeError(
            "Không có pose constraints để tối ưu."
        )

    if max_nfev < 1:
        raise ValueError(
            "max_nfev phải lớn hơn 0."
        )

    if f_scale <= 0:
        raise ValueError(
            "f_scale phải lớn hơn 0."
        )

    for index, frame in enumerate(frames):

        if frame is None or frame.size == 0:
            raise RuntimeError(
                f"Frame {index} không hợp lệ."
            )

    # --------------------------------------------------
    # 1. COORDINATE NORMALIZATION
    # --------------------------------------------------

    normalizations = [
        _get_coordinate_normalization(frame)
        for frame in frames
    ]

    inverse_normalizations = [
        np.linalg.inv(N)
        for N in normalizations
    ]

    # --------------------------------------------------
    # 2. NORMALIZE INITIAL TRANSFORMS
    # --------------------------------------------------

    normalized_initial_transforms = []

    for index, transform in enumerate(initial_transforms):

        transform = _normalize_homography(
            transform
        )

        normalized_transform = (
            normalizations[0]
            @ transform
            @ inverse_normalizations[index]
        )

        normalized_transform = _normalize_homography(
            normalized_transform
        )

        normalized_initial_transforms.append(
            normalized_transform
        )

    # --------------------------------------------------
    # 3. PARAMETERIZE INITIAL TRANSFORMS
    # --------------------------------------------------

    initial_parameters = []
    pivot_indices = []

    for index, transform in enumerate(
        normalized_initial_transforms[1:],
        start=1,
    ):

        try:
            parameters, pivot_index = (
                _homography_to_parameters(transform)
            )

        except RuntimeError as exc:
            raise RuntimeError(
                f"Không thể tham số hóa frame "
                f"{index}: {exc}"
            ) from exc

        initial_parameters.extend(parameters)
        pivot_indices.append(pivot_index)

    initial_parameters = np.asarray(
        initial_parameters,
        dtype=np.float64,
    )

    parameter_count = (
        (frame_count - 1) * 8
    )

    if len(initial_parameters) != parameter_count:
        raise RuntimeError(
            "Số lượng tham số Homography không hợp lệ."
        )

    # --------------------------------------------------
    # 4. DECODE PARAMETERS
    # --------------------------------------------------

    def decode_parameters(parameters):
        """Giải mã tham số thành normalized transforms."""

        transforms = [
            np.eye(3, dtype=np.float64)
        ]

        for index in range(frame_count - 1):

            start = index * 8
            end = start + 8

            H = _parameters_to_homography(
                parameters[start:end],
                pivot_indices[index],
            )

            transforms.append(H)

        return transforms

    decoded_initial = decode_parameters(
        initial_parameters
    )

    # --------------------------------------------------
    # 5. PREPARE HORIZON CONSTRAINTS
    # --------------------------------------------------

    horizon_margin = 0.02
    horizon_weight = 30.0
    horizon_softness = 20.0

    horizon_constraints = []

    normalized_corners = np.float64([
        [-1.0, -1.0],
        [1.0, -1.0],
        [1.0, 1.0],
        [-1.0, 1.0],
    ])

    homogeneous_corners = np.column_stack([
        normalized_corners,
        np.ones(4, dtype=np.float64),
    ])

    center = np.array([
        [0.0, 0.0, 1.0]
    ])

    for index, transform in enumerate(decoded_initial[1:], start=1):
        H = transform

        projected_h = (
            H @ homogeneous_corners.T
        ).T

        denominators = projected_h[:, 2]

        denominator_scale = np.sum(
            np.abs(
                H[2][None, :] * homogeneous_corners
            ),
            axis=1,
        )

        if (
            not np.isfinite(denominators).all()
            or not np.isfinite(denominator_scale).all()
        ):
            raise RuntimeError(
                f"Transform frame {index} "
                "có denominator không hợp lệ."
            )

        normalized_denominators = (
            denominators
            / np.maximum(
                denominator_scale,
                1e-12,
            )
        )

        center_denominator = (
            H @ center.T
        ).T[0, 2]

        if abs(center_denominator) < 1e-12:
            center_denominator = float(
                np.median(denominators)
            )

        if abs(center_denominator) < 1e-12:
            raise RuntimeError(
                f"Không thể xác định chiều Horizon "
                f"cho frame {index}."
            )

        sign = (
            1.0
            if center_denominator > 0
            else -1.0
        )

        initial_margin = (
            sign * normalized_denominators
        )

        horizon_constraints.append({
            "index": index,
            "points": homogeneous_corners.copy(),
            "sign": sign,
            "initial_margin": initial_margin,
            "required_margin": horizon_margin,
        })

    _debug_print(
        debug,
        f"\n[OPTIMIZATION]"
        f"\nHorizon constraints: "
        f"{len(horizon_constraints)}"
    )

    for constraint in horizon_constraints:

        index = constraint["index"]
        margins = constraint["initial_margin"]

        _debug_print(
            debug,
            f"[HORIZON INITIAL] "
            f"Frame {index}: "
            f"minimum margin = {margins.min():.6f}"
        )

    # --------------------------------------------------
    # 6. PREPARE POSE CONSTRAINTS
    # --------------------------------------------------

    prepared_constraints = []

    for constraint_index, constraint in enumerate(pose_constraints):
        try:
            i = int(constraint["i"])
            j = int(constraint["j"])
            H = _normalize_homography(constraint["H"])

        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Constraint {constraint_index} không hợp lệ: {exc}") from exc

        if not (0 <= i < frame_count and 0 <= j < frame_count):
            raise RuntimeError(f"Constraint không hợp lệ: {i} -> {j}.")

        if i >= j:
            raise RuntimeError(f"Constraint phải có i < j: {i} -> {j}.")

        frame_j = frames[j]
        h, w = frame_j.shape[:2]

        xs = np.linspace(0, w - 1, 5)
        ys = np.linspace(0, h - 1, 5)

        grid_x, grid_y = np.meshgrid(xs, ys)
        points_pixel = np.column_stack([grid_x.ravel(), grid_y.ravel()])

        try:
            projected_reference = _project_points(
                points_pixel,
                H,
            )

        except RuntimeError:
            _debug_print(debug, f"[OPTIMIZATION WARNING] Skip invalid constraint {i} -> {j}.")
            continue

        reference_h, reference_w = (frames[i].shape[:2])

        inside = (
            (projected_reference[:, 0] >= 0)
            & (projected_reference[:, 0] < reference_w)
            & (projected_reference[:, 1] >= 0)
            & (projected_reference[:, 1] < reference_h)
        )

        points_pixel = points_pixel[inside]

        if len(points_pixel) < 4:
            _debug_print(
                debug,
                f"[OPTIMIZATION WARNING] "
                f"Skip constraint {i} -> {j}: "
                f"only {len(points_pixel)} overlapping points."
            )
            continue

        # Pixel coordinates -> normalized coordinates.
        points_homogeneous = np.column_stack([
            points_pixel,
            np.ones(
                len(points_pixel),
                dtype=np.float64,
            ),
        ])

        points_normalized_h = (normalizations[j] @ points_homogeneous.T).T
        points_normalized = (points_normalized_h[:, :2] / points_normalized_h[:, 2:3])

        # Pairwise H:
        # frame j normalized -> frame i normalized.
        H_normalized = _convert_homography_coordinates(H, normalizations[j], normalizations[i])

        prepared_constraints.append({
            "i": i,
            "j": j,

            # Dữ liệu normalized dùng cho optimizer.
            "H": H_normalized,
            "points": points_normalized,

            # Dữ liệu pixel dùng để đánh giá sai số.
            "H_pixel": H,
            "points_pixel": points_pixel,
        })

    if not prepared_constraints:
        raise RuntimeError(
            "Không có constraint hợp lệ để tối ưu."
        )

    _debug_print(
        debug,
        f"[OPTIMIZATION] "
        f"Prepared pose constraints: "
        f"{len(prepared_constraints)}"
    )

    # --------------------------------------------------
    # 7. SPARSE JACOBIAN
    # --------------------------------------------------

    pose_residual_count = sum(
        len(item["points"]) * 2
        for item in prepared_constraints
    )

    horizon_residual_count = (
        len(horizon_constraints) * 4
    )

    residual_count = (
        pose_residual_count
        + horizon_residual_count
    )

    jac_sparsity = lil_matrix(
        (residual_count, parameter_count),
        dtype=np.int8,
    )

    row_offset = 0

    # Pose residual dependencies.
    for constraint in prepared_constraints:

        i = constraint["i"]
        j = constraint["j"]

        rows = len(
            constraint["points"]
        ) * 2

        if i > 0:

            start = (i - 1) * 8

            jac_sparsity[
                row_offset:row_offset + rows,
                start:start + 8,
            ] = 1

        if j > 0:

            start = (j - 1) * 8

            jac_sparsity[
                row_offset:row_offset + rows,
                start:start + 8,
            ] = 1

        row_offset += rows

    # Horizon residual dependencies.
    for constraint in horizon_constraints:

        index = constraint["index"]

        if index > 0:

            start = (index - 1) * 8

            jac_sparsity[
                row_offset:row_offset + 4,
                start:start + 8,
            ] = 1

        row_offset += 4

    jac_sparsity = jac_sparsity.tocsr()

    # --------------------------------------------------
    # 8. RESIDUAL FUNCTION
    # --------------------------------------------------

    def calculate_residuals(parameters):

        transforms = decode_parameters(
            parameters
        )

        pose_residuals = []
        horizon_residuals = []

        # ==========================================
        # A. POSE RESIDUALS
        # ==========================================

        reference_height, reference_width = (frames[0].shape[:2])

        for constraint in prepared_constraints:
            i = constraint["i"]
            j = constraint["j"]

            H = constraint["H"]
            points = np.asarray(constraint["points"], dtype=np.float64)

            if points.size % 2 != 0:
                print("\n[PREPARED CONSTRAINT ERROR]")
                print("Constraint:", constraint["i"], "->", constraint["j"])
                print("Points shape:", points.shape)
                print("Points size:", points.size)
                print("Points:", points)

                raise ValueError(
                    f"Constraint {constraint['i']}->{constraint['j']} "
                    f"có points không hợp lệ: {points.size} phần tử."
                )

            points = points.reshape(-1, 2)

            projected_global, valid_global = (
                _project_points_for_optimization(
                    points,
                    transforms[j],
                )
            )

            reference_transform = (transforms[i] @ H)

            projected_reference, valid_reference = (
                _project_points_for_optimization(
                    points,
                    reference_transform,
                )
            )

            difference = (
                projected_global - projected_reference
            )

            # Normalized residual -> pixel-equivalent residual.
            #
            # Cả hai phép chiếu đều nằm trong hệ tọa độ
            # normalized của frame 0.

            difference[:, 0] *= (
                reference_width / 2.0
            )

            difference[:, 1] *= (
                reference_height / 2.0
            )

            invalid = (
                ~valid_global | ~valid_reference
            )

            if np.any(invalid):
                invalid_count = np.count_nonzero(invalid)

                _debug_print(debug, f"[INVALID PROJECTION] {i}->{j}: {invalid_count}/{len(points)}")

                difference[invalid] = np.clip(
                    difference[invalid],
                    -1e5,
                    1e5,
                )

                invalid_penalty = (
                    100.0
                    * (
                        ~valid_global[invalid]
                        | ~valid_reference[invalid]
                    )
                )

                difference[invalid] += (
                    invalid_penalty[:, None]
                )

            pose_residuals.extend(difference.reshape(-1))

        # ==========================================
        # B. HORIZON RESIDUALS
        # ==========================================

        for constraint in horizon_constraints:

            index = constraint["index"]
            points = constraint["points"]
            sign = constraint["sign"]

            required_margin = (constraint["required_margin"])
            H = transforms[index]
            projected_h = (H @ points.T).T

            denominator = projected_h[:, 2]
            denominator_scale = np.sum(np.abs(H[2][None, :] * points), axis=1)

            normalized_margin = (
                sign * denominator
                / np.maximum(
                    denominator_scale,
                    1e-12,
                )
            )

            violation = (
                required_margin - normalized_margin
            )

            penalty = (
                np.logaddexp(
                    0.0,
                    horizon_softness * violation,
                )
                / horizon_softness
            )

            horizon_residual = (
                horizon_weight * penalty
            )

            horizon_residuals.extend(
                horizon_residual
            )

        pose_residuals = np.asarray(
            pose_residuals,
            dtype=np.float64,
        )

        horizon_residuals = np.asarray(
            horizon_residuals,
            dtype=np.float64,
        )

        pose_residuals = np.nan_to_num(
            pose_residuals,
            nan=1e5,
            posinf=1e5,
            neginf=-1e5,
        )

        horizon_residuals = np.nan_to_num(
            horizon_residuals,
            nan=1e5,
            posinf=1e5,
            neginf=1e5,
        )

        if debug:
            print(
                "\n[RESIDUAL DEBUG]"
                f"\nPose count: {len(pose_residuals)}"
                f"\nPose norm: {np.linalg.norm(pose_residuals):.6f}"
                f"\nPose max: {np.max(np.abs(pose_residuals)):.6f}"
                f"\nHorizon count: {len(horizon_residuals)}"
                f"\nHorizon norm: {np.linalg.norm(horizon_residuals):.6f}"
                f"\nHorizon max: {np.max(np.abs(horizon_residuals)):.6f}"
            )

        return pose_residuals, horizon_residuals

    def residual_function(parameters):

        pose_residuals, horizon_residuals = (
            calculate_residuals(parameters)
        )

        return np.concatenate([
            pose_residuals,
            horizon_residuals,
        ])

    # --------------------------------------------------
    # 9. INITIAL METRICS
    # --------------------------------------------------

    initial_pose_residuals, initial_horizon_residuals = (
        calculate_residuals(initial_parameters)
    )

    initial_pose_error = _calculate_pose_error_pixel(
        initial_transforms,
        prepared_constraints,
    )

    initial_horizon_error = float(
        np.mean(
            np.abs(initial_horizon_residuals)
        )
    )

    _debug_print(
        debug,
        f"\n[OPTIMIZATION INITIAL]"
        f"\nPose MAE: {initial_pose_error:.4f} px"
        f"\nHorizon penalty: {initial_horizon_error:.6f}"
        f"\nFrames: {frame_count}"
        f"\nConstraints: {len(prepared_constraints)}"
        f"\nParameters: {parameter_count}"
    )

    # --------------------------------------------------
    # 10. RUN OPTIMIZATION
    # --------------------------------------------------

    initial_residuals = residual_function(initial_parameters)

    print(
        "[INITIAL RESIDUAL]",
        "Count:", len(initial_residuals),
        "Norm:", np.linalg.norm(initial_residuals),
        "Max:", np.max(np.abs(initial_residuals)),
    )

    result = least_squares(
        residual_function,
        initial_parameters,
        method="trf",
        loss=loss,
        f_scale=f_scale,
        x_scale="jac",
        jac_sparsity=jac_sparsity,
        tr_solver="lsmr",
        max_nfev=max_nfev,
        verbose=2 if debug else 0,
    )

    print("\n[OPTIMIZATION RESULT]")
    print("Success:", result.success)
    print("Status:", result.status)
    print("Message:", result.message)
    print("Function evaluations:", result.nfev)
    print("Jacobian evaluations:", result.njev)
    print("Final cost:", result.cost)
    print("Optimality:", result.optimality)
    print("Termination status:", result.status)
    print("Optimality:", result.optimality)
    print("Parameters finite:", np.isfinite(result.x).all())
    print("Residuals finite:", np.isfinite(result.fun).all())
    print("max_nfev:", max_nfev)
    print("loss:", loss)
    print("f_scale:", f_scale)

    if not np.isfinite(result.x).all():
        raise RuntimeError(
            "Optimizer tạo ra tham số NaN hoặc Inf."
        )

    if not result.success and result.status != 0:
        raise RuntimeError(
            "Global Optimization chưa hội tụ. "
            f"Status: {result.status}. "
            f"Message: {result.message}. "
            f"Function evaluations: {result.nfev}."
        )

    if not result.success:
        print(
            "[OPTIMIZATION WARNING] Maximum function evaluations "
            "reached; validating and using the best finite iterate."
        )

    # --------------------------------------------------
    # 11. DECODE OPTIMIZED TRANSFORMS
    # --------------------------------------------------

    optimized_normalized_transforms = decode_parameters(
        result.x
    )

    optimized_transforms = []

    for index, transform in enumerate(
        optimized_normalized_transforms
    ):

        # Normalized transform -> pixel transform.
        pixel_transform = (
            inverse_normalizations[0]
            @ transform
            @ normalizations[index]
        )

        pixel_transform = _normalize_homography(
            pixel_transform
        )

        optimized_transforms.append(
            pixel_transform
        )

    # --------------------------------------------------
    # 12. FINAL METRICS
    # --------------------------------------------------

    final_normalized_pose_residuals, final_horizon_residuals = (
        calculate_residuals(result.x)
    )

    final_pose_error = _calculate_pose_error_pixel(
        optimized_transforms,
        prepared_constraints,
    )

    final_horizon_error = float(
        np.mean(
            np.abs(final_horizon_residuals)
        )
    )

    if (
        not np.isfinite(final_pose_error)
        or not np.isfinite(final_horizon_error)
    ):
        raise RuntimeError(
            "Kết quả Optimization chứa sai số không hợp lệ."
        )

    # --------------------------------------------------
    # 13. VALIDATE OPTIMIZED TRANSFORMS
    # --------------------------------------------------

    horizon_margins = []

    for constraint in horizon_constraints:

        index = constraint["index"]

        # Dùng normalized transform với normalized points.
        H = optimized_normalized_transforms[index]

        points = constraint["points"]
        sign = constraint["sign"]

        projected_h = (
            H @ points.T
        ).T

        denominator = projected_h[:, 2]

        scale = np.sum(
            np.abs(
                H[2][None, :] * points
            ),
            axis=1,
        )

        margin = (
            sign * denominator
            / np.maximum(
                scale,
                1e-12,
            )
        )

        horizon_margins.extend(
            margin.tolist()
        )

        if not np.all(margin > 1e-6):
            raise RuntimeError(
                f"Transform frame {index} "
                "vẫn bị projective horizon cắt qua "
                "sau Optimization. "
                f"Margins: {margin}."
            )

        # Kiểm tra hình học thực tế trong pixel coordinates.
        try:
            optimized_transforms[index] = (
                _normalize_homography(
                    optimized_transforms[index]
                )
            )

            _transformed_corners(
                frames[index],
                optimized_transforms[index],
                debug=debug,
            )

        except (RuntimeError, cv2.error) as exc:
            raise RuntimeError(
                f"Transform frame {index} "
                f"không hợp lệ sau Optimization: {exc}"
            ) from exc

    minimum_horizon_margin = float(
        np.min(horizon_margins)
    )

    # --------------------------------------------------
    # 14. CHECK POSE QUALITY
    # --------------------------------------------------

    allowed_increase = max(
        initial_pose_error * 0.10,
        initial_pose_error + 5.0,
    )

    if final_pose_error > allowed_increase:
        raise RuntimeError(
            "Pose error tăng quá mức cho phép sau "
            "Global Optimization. "
            f"Initial: {initial_pose_error:.4f}px, "
            f"Final: {final_pose_error:.4f}px."
        )

    # --------------------------------------------------
    # 15. SUMMARY
    # --------------------------------------------------

    _debug_print(
        debug,
        f"\n[OPTIMIZATION SUMMARY]"
        f"\nFrames: {frame_count}"
        f"\nPose constraints: {len(prepared_constraints)}"
        f"\nInitial pose MAE: {initial_pose_error:.4f} px"
        f"\nFinal pose MAE: {final_pose_error:.4f} px"
        f"\nInitial horizon penalty: {initial_horizon_error:.6f}"
        f"\nFinal horizon penalty: {final_horizon_error:.6f}"
        f"\nMinimum horizon margin: {minimum_horizon_margin:.6f}"
        f"\nFunction evaluations: {result.nfev}"
        f"\nOptimizer status: {result.message}"
        f"\nCost: {result.cost:.6f}"
        f"\nOptimality: {result.optimality:.6e}"
    )

    return optimized_transforms, {
        "success": bool(result.success),

        "message": str(result.message),

        "initial_mean_absolute_error_px": (
            initial_pose_error
        ),

        "final_mean_absolute_error_px": (
            final_pose_error
        ),

        "initial_horizon_penalty": (
            initial_horizon_error
        ),

        "final_horizon_penalty": (
            final_horizon_error
        ),

        "constraint_count": len(
            prepared_constraints
        ),

        "horizon_constraint_count": len(
            horizon_constraints
        ),

        "minimum_horizon_margin": (
            minimum_horizon_margin
        ),

        "function_evaluations": int(
            result.nfev
        ),

        "cost": float(
            result.cost
        ),

        "optimality": float(
            result.optimality
        ),
    }
