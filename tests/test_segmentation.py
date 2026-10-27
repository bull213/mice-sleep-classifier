import numpy as np
import pytest
from pathlib import Path
from mouse_sleep.config import MODEL_STEPS, Settings
from mouse_sleep.segmentation import Segmenter, clean_masks, refine_mask
from mouse_sleep.video import square_image
from tests.reference_morphology import probability_cases, reference_masks


def test_white_fur_is_preserved_and_padding_is_excluded():
    image = np.full((80, 160), 220, np.uint8)
    square, valid = square_image(image)
    assert square.shape == (160, 160)
    assert valid.sum() == image.size
    mask = np.ones((480, 480), np.uint8)
    assert refine_mask(mask, square, valid, "off").sum() > 0
    assert refine_mask(mask, square, valid, "dark", 30).sum() == 0
    assert Settings(appearance="white").refinement_mode == "off"
    assert Settings(appearance="white").model_name == "FullModel"


@pytest.fixture(scope="module")
def tensorflow_morphology():
    import tensorflow as tf

    graph = tf.Graph()
    with graph.as_default():
        probabilities = tf.compat.v1.placeholder(tf.float32, [None, None, None])
        masks = reference_masks(tf, probabilities)
    with tf.compat.v1.Session(graph=graph) as session:
        yield lambda values: session.run(masks, {probabilities: values})


@pytest.mark.parametrize("shape", [(1, 1), (3, 7), (35, 51), (64, 64), (480, 480)])
def test_cleanup_matches_original_nonflat_morphology(shape: tuple[int, int], tensorflow_morphology):
    probabilities = probability_cases(*shape)
    np.testing.assert_array_equal(clean_masks(probabilities), tensorflow_morphology(probabilities))


@pytest.mark.requires_models
@pytest.mark.parametrize("model_name", MODEL_STEPS)
def test_prediction_masks_match_original_for_pretrained_models(model_name, tensorflow_morphology):
    checkpoint = Settings(model=model_name).checkpoint
    required = [Path(str(checkpoint) + suffix) for suffix in (".index", ".data-00000-of-00001")]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        pytest.skip(f"{model_name}: missing {', '.join(missing)}; see docs/INSTALLATION.md")
    rng = np.random.default_rng(52)
    images = rng.normal(130, 12, (2, 480, 480)).astype(np.float32)
    images[0, 150:260, 200:270] = 15
    images[1, 150:260, 200:270] = 240
    with Segmenter(checkpoint) as model:
        probabilities = model.session.run(model.probabilities, {model.inputs: images[..., None]})
        predicted = model.predict(images)
    assert predicted.dtype == bool
    np.testing.assert_array_equal(predicted, tensorflow_morphology(probabilities))


def test_enhanced_still_mouse_lighting_boundary_and_missing_detection():
    import cv2
    from mouse_sleep.backgrounds import CageGeometry, build_backgrounds
    from mouse_sleep.masking import EnhancedMasker
    from mouse_sleep.segmentation import FramePreparer
    settings = Settings(cage_polygon=((20, 20), (450, 20), (450, 450), (20, 450)))
    geometry = CageGeometry(settings, (480, 480))
    samples = np.stack([np.full((480, 480), level, np.uint8) for level in [180] * 8 + [100] * 8])
    backgrounds = build_backgrounds(samples, geometry, {})
    detector = EnhancedMasker(settings, backgrounds, geometry)
    prepare = FramePreparer()
    empty = np.zeros((480, 480), bool)
    for i in range(301):
        frame = np.full((480, 480), 180 if i < 150 else 100, np.uint8)
        cv2.ellipse(frame, (240, 240), (25, 16), 0, 0, 360, 10, -1)
        cv2.ellipse(frame, (80, 80), (12, 7), 0, 0, 360, 10, -1)  # Changed bedding.
        frame[100:160, :18] = 0  # A larger dark reflection outside the allowed boundary.
        result = detector.process(empty, prepare(frame), i / 30)
        if i not in (150, 151):
            assert result.source == "background" and result.mask[540, 540] == 1
            assert not result.mask[:, :40].any()
            assert not result.mask[180, 180]
    assert detector.counts["lighting_changes"] == 1
    result = detector.process(empty, prepare(np.full((480, 480), 100, np.uint8)), 301 / 30)
    assert result.source == "rejected" and not result.mask.any()
    competing = np.full((480, 480), 100, np.uint8)
    for center in ((240, 240), (310, 240)):
        cv2.ellipse(competing, center, (25, 16), 0, 0, 360, 10, -1)
    result = detector.process(empty, prepare(competing), 302 / 30)
    assert result.reason == "ambiguous_candidates" and not result.mask.any()
    # A mouse embedded in every automatic reference sample still has neural evidence.
    stationary = np.full((480, 480), 180, np.uint8)
    cv2.ellipse(stationary, (240, 240), (25, 16), 0, 0, 360, 10, -1)
    reference = build_backgrounds(np.repeat(stationary[None], 16, axis=0), geometry, {})
    result = EnhancedMasker(settings, reference, geometry).process(stationary < 50, prepare(stationary), 0)
    assert result.source == "neural" and result.mask[540, 540] == 1
