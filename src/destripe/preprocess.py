import numbers

import cv2
import numpy as np

# Rec. 601 luma coefficients (standard for NTSC/JPEG grayscale conversion)
_LUMA_R = 0.2989
_LUMA_G = 0.5870
_LUMA_B = 0.1140


def rgb_to_luma(image: np.ndarray) -> np.ndarray:
    return (
        _LUMA_R * image[..., 0]
        + _LUMA_G * image[..., 1]
        + _LUMA_B * image[..., 2]
    )


def validate_process_size(process_size: int | None) -> int | None:
    if process_size is None:
        return None
    if isinstance(process_size, bool) or not isinstance(process_size, numbers.Integral):
        raise ValueError("process_size must be None or an integer greater than 1.")
    value = int(process_size)
    if value <= 1:
        raise ValueError("process_size must be None or an integer greater than 1.")
    return value


def solver_gray(*, gray: np.ndarray, process_size: int | None) -> np.ndarray:
    size = process_shape(gray.shape, process_size)
    if size == gray.shape:
        return gray
    return np.clip(resize_2d(gray, size=size, mode="lanczos"), 0.0, 1.0)


def process_shape(
    shape: tuple[int, int], process_size: int | None
) -> tuple[int, int]:
    if process_size is None:
        return shape
    h, w = shape
    long_edge = max(h, w)
    if process_size >= long_edge:
        return shape
    scale = process_size / long_edge
    if h >= w:
        return process_size, _scaled_dim(w, scale)
    return _scaled_dim(h, scale), process_size


def resize_2d(
    image: np.ndarray,
    *,
    size: tuple[int, int],
    mode: str,
) -> np.ndarray:
    if image.shape == size:
        return np.asarray(image, dtype=np.float64).copy()

    if mode != "lanczos":
        raise ValueError(f"unsupported resize mode: {mode}")

    array = np.asarray(image, dtype=np.float64)
    resized = cv2.resize(
        array,
        dsize=(size[1], size[0]),
        interpolation=cv2.INTER_LANCZOS4,
    )
    return np.asarray(resized, dtype=np.float64).reshape(size)


def _scaled_dim(dim: int, scale: float) -> int:
    if dim <= 1:
        return dim
    return max(2, int(np.floor(dim * scale + 0.5)))
