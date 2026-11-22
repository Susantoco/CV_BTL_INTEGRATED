
import cv2


METHODS = ("sift", "orb", "akaze", "brisk")


def create_feature_extractor(method: str):
    """Tạo bộ trích xuất đặc trưng và loại norm tương ứng."""

    if not isinstance(method, str):
        raise TypeError("method phải là chuỗi.")

    method = method.strip().lower()

    if method == "sift":
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError(
                "OpenCV hiện tại không hỗ trợ SIFT_create."
            )

        return cv2.SIFT_create(
            nfeatures=5000
        ), cv2.NORM_L2

    if method == "orb":
        return cv2.ORB_create(
            nfeatures=8000,
            scaleFactor=1.15,
            nlevels=12,
            fastThreshold=7,
        ), cv2.NORM_HAMMING

    if method == "akaze":
        return cv2.AKAZE_create(), cv2.NORM_HAMMING

    if method == "brisk":
        return cv2.BRISK_create(), cv2.NORM_HAMMING

    raise ValueError(
        f"Phương pháp '{method}' không hợp lệ. "
        f"Chọn một trong: {METHODS}"
    )


def preprocess_frame(frame, denoise: bool = True):
    """
    Tiền xử lý frame.

    Returns:
        color: ảnh màu dùng cho bước stitching.
        gray: ảnh grayscale dùng cho feature extraction.
    """

    if frame is None:
        raise ValueError("Frame đầu vào không hợp lệ.")

    if frame.size == 0:
        raise ValueError("Frame đầu vào rỗng.")

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(
            "Frame phải là ảnh màu BGR có 3 kênh."
        )

    if frame.dtype != "uint8":
        raise ValueError(
            "Frame phải có kiểu dữ liệu uint8."
        )

    color = frame.copy()

    if denoise:
        color = cv2.GaussianBlur(
            color,
            (3, 3),
            0,
        )

    gray = cv2.cvtColor(
        color,
        cv2.COLOR_BGR2GRAY,
    )

    return color, gray


def detect_and_describe(gray, extractor):
    """
    Phát hiện keypoints và tính descriptors.
    """

    if gray is None or gray.size == 0:
        raise ValueError("Ảnh grayscale không hợp lệ.")

    if gray.ndim != 2:
        raise ValueError(
            "Đầu vào detect_and_describe phải là ảnh grayscale."
        )

    if gray.dtype != "uint8":
        raise ValueError(
            "Ảnh grayscale phải có kiểu dữ liệu uint8."
        )

    if extractor is None:
        raise ValueError("Feature extractor không hợp lệ.")

    keypoints, descriptors = extractor.detectAndCompute(
        gray,
        None,
    )

    if keypoints is None:
        keypoints = []

    return keypoints, descriptors
