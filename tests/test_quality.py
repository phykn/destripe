import cv2
import numpy as np
import pytest

from destripe.automatic import automatic_clean


@pytest.mark.parametrize("process_size", (None, 64))
@pytest.mark.parametrize("proj", (False, True))
def test_final_output_removes_local_stripes_and_preserves_scene(
    process_size: int | None,
    proj: bool,
) -> None:
    _, cols = np.indices((96, 128))
    target = (
        0.25 + 0.3 * ((cols >= 24) & (cols < 48)) + 0.2 * ((cols >= 80) & (cols < 104))
    )
    contaminated = cols >= 64
    stripe = 0.025 * np.sin(2 * np.pi * 8 * cols / 128) * contaminated
    observed = target + stripe

    result = automatic_clean(observed, process_size=process_size, proj=proj)

    assert result.directions == (0,)
    error = result.clean - target
    input_rms = float(np.sqrt(np.mean(stripe[contaminated] ** 2)))
    residual_rms = float(np.sqrt(np.mean(error[contaminated] ** 2)))
    untouched_rms = float(np.sqrt(np.mean(error[~contaminated] ** 2)))
    # A no-op fails removal, and flattening the image fails preservation.
    assert residual_rms < 0.5 * input_rms
    assert untouched_rms < 1e-3

    edges = [23, 47, 79, 103]
    target_contrast = np.diff(target, axis=1)[:, edges]
    result_contrast = np.diff(result.clean, axis=1)[:, edges]
    np.testing.assert_allclose(result_contrast, target_contrast, atol=0.01)


@pytest.mark.parametrize("process_size", (None, 64))
@pytest.mark.parametrize("scene", ("text", "texture"))
def test_final_output_preserves_clean_text_and_texture(
    process_size: int | None,
    scene: str,
) -> None:
    rows, cols = np.indices((96, 128))
    if scene == "text":
        target = np.full((96, 128), 0.35)
        cv2.putText(target, "TEST", (7, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, 0.65, 2)
    else:
        target = 0.4 + 0.06 * np.sin(2 * np.pi * rows / 15) * np.sin(
            2 * np.pi * cols / 19
        )

    result = automatic_clean(target, process_size=process_size, proj=True)

    assert result.directions == ()
    np.testing.assert_array_equal(result.clean, target)
