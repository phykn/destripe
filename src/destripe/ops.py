import numpy as np

from .automatic import automatic_clean
from .preprocess import (
    prepare_solver_gray,
    resize_to_shape,
    rgb_to_luma,
    validate_process_size,
)


def destripe(
    image: np.ndarray,
    *,
    process_size: int | None = None,
    proj: bool = True,
) -> np.ndarray:
    """Remove stripe noise from a NumPy image.

    Supports grayscale, single-channel, and RGB arrays. RGB stripes are
    estimated from luma and subtracted from every channel.

    Args:
        image: Input image.
        process_size: Long-edge analysis size; None keeps original resolution.
        proj: Project output to the input intensity range.

    Returns:
        Destriped image with the same shape and dtype.
    """
    input_array = np.asarray(image)
    if not np.issubdtype(input_array.dtype, np.number) or np.iscomplexobj(input_array):
        raise ValueError("image must contain numeric values.")
    if (
        input_array.ndim not in {2, 3}
        or 0 in input_array.shape
        or (input_array.ndim == 3 and input_array.shape[2] not in {1, 3})
    ):
        raise ValueError(
            f"image must have shape (H, W) or (H, W, C) with C in {{1, 3}}, "
            f"got {input_array.shape}."
        )
    if not np.isfinite(input_array).all():
        raise ValueError("image must not contain NaN or Inf values.")

    process_size_value = validate_process_size(process_size)
    orig_dtype = input_array.dtype
    normalized = input_array.astype(np.float64)

    min_value = float(normalized.min())
    max_value = float(normalized.max())
    intensity_scale = max_value - min_value
    if intensity_scale == 0.0:
        return input_array.copy()
    restore_magnitude: float | None = None
    restore_offset = min_value
    restore_scale = intensity_scale
    if np.isfinite(intensity_scale):
        normalized = (normalized - min_value) / intensity_scale
    else:
        restore_magnitude = max(abs(min_value), abs(max_value))
        scaled = normalized / restore_magnitude
        restore_offset = min_value / restore_magnitude
        restore_scale = max_value / restore_magnitude - restore_offset
        normalized = (scaled - restore_offset) / restore_scale

    if normalized.ndim == 2:
        gray = normalized
    elif normalized.shape[2] == 3:
        gray = rgb_to_luma(normalized)
    else:
        gray = normalized[..., 0]

    processed_gray = prepare_solver_gray(
        gray=gray,
        process_size=process_size_value,
    )
    automatic_result = automatic_clean(processed_gray, proj=proj)
    correction = processed_gray - automatic_result.clean
    if processed_gray.shape != gray.shape:
        correction = resize_to_shape(correction, shape=gray.shape)
    if not np.any(correction):
        return input_array.copy()

    if normalized.ndim == 2:
        clean = gray - correction
    else:
        clean = normalized - correction[..., np.newaxis]
    if proj:
        clean = np.clip(clean, 0.0, 1.0)

    result = clean * restore_scale + restore_offset
    if restore_magnitude is not None:
        result *= restore_magnitude
    if np.issubdtype(orig_dtype, np.integer):
        return _restore_integer(result, dtype=orig_dtype)
    return result.astype(orig_dtype)


def _restore_integer(result: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    info = np.iinfo(dtype)
    values = np.asarray(result, dtype=np.float64)
    lower_bound = float(info.min)
    upper_bound = float(info.max)
    lower_endpoint = values <= lower_bound
    upper_endpoint = values >= upper_bound
    safe_upper = np.nextafter(upper_bound, -np.inf)
    safe = np.clip(values, lower_bound, safe_upper)
    restored = safe.astype(dtype)
    restored[lower_endpoint] = info.min
    restored[upper_endpoint] = info.max
    return restored
