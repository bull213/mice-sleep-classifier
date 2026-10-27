"""The original TensorFlow non-flat morphology, retained for parity checks."""
import numpy as np
from scipy.ndimage import generate_binary_structure, iterate_structure


def reference_masks(tf, probabilities):
    signed = (2 * probabilities - 1)[..., None]
    for radius, opening in ((4, True), (5, False)):
        filt = tf.constant(iterate_structure(generate_binary_structure(2, 1), radius)[..., None], tf.float32)

        def erode(value):
            return tf.nn.erosion2d(value, filt, [1, 1, 1, 1], "SAME", "NHWC", [1, 1, 1, 1])

        def dilate(value):
            return tf.nn.dilation2d(value, filt, [1, 1, 1, 1], "SAME", "NHWC", [1, 1, 1, 1])

        signed = dilate(erode(signed)) if opening else erode(dilate(signed))
    return signed[..., 0] > 0.5


def probability_cases(height, width):
    """Include threshold rounding, edges, holes, isolated noise, and soft masks."""
    rng = np.random.default_rng(42)
    cases = [np.full((height, width), value, np.float32) for value in (
        0, 0.5, np.nextafter(np.float32(0.75), np.float32(0)), 0.75,
        np.nextafter(np.float32(0.75), np.float32(1)), 1,
    )]
    y, x = np.mgrid[:height, :width]
    cases.append((0.5 + 0.5 * np.sin(x / 8) * np.cos(y / 8)).astype(np.float32))
    cases.append(rng.random((height, width), dtype=np.float32))
    mask = np.full((height, width), 0.6, np.float32)
    mask[height // 4:3 * height // 4, width // 4:3 * width // 4] = 1
    mask[height // 2, width // 2] = 0
    mask[:height // 3, :width // 3] = 0.99
    cases.extend([mask, np.flip(mask).copy()])
    return np.stack(cases)
