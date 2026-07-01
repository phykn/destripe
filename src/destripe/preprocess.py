import numbers

import cv2
import numpy as np

# Rec. 601 keeps RGB stripe estimation aligned with conventional luma.
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


def prepare_solver_gray(*, gray: np.ndarray, process_size: int | None) -> np.ndarray:
    shape = compute_solver_shape(gray.shape, process_size)
    if shape == gray.shape:
        return gray
    return np.clip(resize_lanczos(gray, shape=shape), 0.0, 1.0)


def compute_solver_shape(
    shape: tuple[int, int], process_size: int | None
) -> tuple[int, int]:
    if process_size is None:
        return shape
    height, width = shape
    long_edge = max(height, width)
    if process_size >= long_edge:
        return shape
    scale = process_size / long_edge
    if height >= width:
        return process_size, _scale_dim(width, scale)
    return _scale_dim(height, scale), process_size


def resize_lanczos(
    image: np.ndarray,
    *,
    shape: tuple[int, int],
) -> np.ndarray:
    if image.shape == shape:
        return np.asarray(image, dtype=np.float64).copy()

    array = np.asarray(image, dtype=np.float64)
    resized = cv2.resize(
        array,
        dsize=(shape[1], shape[0]),
        interpolation=cv2.INTER_LANCZOS4,
    )
    return np.asarray(resized, dtype=np.float64).reshape(shape)


def _scale_dim(dim: int, scale: float) -> int:
    if dim <= 1:
        return dim
    return max(2, int(np.floor(dim * scale + 0.5)))
