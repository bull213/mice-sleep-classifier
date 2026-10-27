"""Modern inference port of KumarLabJax/MouseTracking construct_segsoft_v5 (MIT)."""
from __future__ import annotations

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import generate_binary_structure, iterate_structure

from .config import MEASUREMENT_SIZE
from .video import SquareCrop


_MORPHOLOGY_KERNELS = tuple(
    (np.ones((2 * radius + 1, 2 * radius + 1), np.uint8),
     iterate_structure(generate_binary_structure(2, 1), radius).astype(np.uint8))
    for radius in (4, 5)
)


def _erode_scores(scores, rectangle, diamond):
    return np.minimum(cv2.erode(scores, rectangle), cv2.erode(scores, diamond) - 1)


def _dilate_scores(scores, rectangle, diamond):
    return np.maximum(cv2.dilate(scores, rectangle), cv2.dilate(scores, diamond) + 1)


def clean_masks(probabilities):
    """Apply the original non-flat opening/closing to a batch of probabilities.

    The TensorFlow filters have height 1 inside a diamond and height 0 in
    the surrounding square; those zero entries are *not* excluded pixels.
    A non-flat erosion is therefore min(erode(square), erode(diamond) - 1),
    and dilation is max(dilate(square), dilate(diamond) + 1). OpenCV performs
    these flat operations much faster on CPU. Its default morphology border
    values are neutral, matching TensorFlow's SAME padding.
    """
    masks = np.empty(probabilities.shape, dtype=bool)
    for index, probability in enumerate(probabilities):
        scores = 2 * probability - 1
        rectangle, diamond = _MORPHOLOGY_KERNELS[0]
        scores = _dilate_scores(_erode_scores(scores, rectangle, diamond), rectangle, diamond)
        rectangle, diamond = _MORPHOLOGY_KERNELS[1]
        scores = _erode_scores(_dilate_scores(scores, rectangle, diamond), rectangle, diamond)
        masks[index] = scores > 0.5  # original MouseSleep signed-score threshold
    return masks


class Segmenter:
    """Restore only segmentation parameters; legacy training and angle graphs are unnecessary."""

    def __init__(self, checkpoint):
        import tensorflow as tf
        self.tf = tf
        self.checkpoint = str(checkpoint)
        if not Path(self.checkpoint + ".index").is_file():
            raise FileNotFoundError(f"Segmentation checkpoint is missing: {checkpoint}")
        reader = tf.train.load_checkpoint(self.checkpoint)
        names = reader.get_variable_to_shape_map()
        self.graph = tf.Graph()
        self.loaded_names = []
        with self.graph.as_default():
            self.inputs = tf.compat.v1.placeholder(tf.float32, [None, 480, 480, 1], name="images")
            image = tf.map_fn(tf.image.per_image_standardization, self.inputs, fn_output_signature=tf.float32)

            def parameter(name):
                full = "Network/" + name
                if full not in names:
                    raise ValueError(f"Checkpoint does not match MouseTracking architecture: missing {full}")
                self.loaded_names.append(full)
                return tf.constant(reader.get_tensor(full), dtype=tf.float32, name=full.replace("/", "_"))

            encoded = []
            for i in range(7):
                scope = "SegmentEncoder/" + ("Conv" if i == 0 else f"Conv_{i}")
                image = tf.nn.conv2d(image, parameter(scope + "/weights"), strides=1, padding="SAME")
                bn = scope + "/BatchNorm/"
                image = tf.nn.batch_normalization(image, parameter(bn + "moving_mean"), parameter(bn + "moving_variance"), parameter(bn + "beta"), parameter(bn + "gamma"), variance_epsilon=0.001)
                image = tf.nn.relu(image)
                encoded.append(image)
                if i < 6:
                    size = 3 if i == 5 else 2
                    image = tf.nn.max_pool2d(image, ksize=size, strides=size, padding="VALID")
            for i, skip in enumerate(reversed(encoded[:-1])):
                scope = "SegmentDecoder/" + ("Conv2d_transpose" if i == 0 else f"Conv2d_transpose_{i}")
                step = 3 if i == 0 else 2
                image = tf.nn.conv2d_transpose(image, parameter(scope + "/weights"), output_shape=tf.shape(skip), strides=[1, step, step, 1], padding="SAME")
                image = tf.nn.bias_add(image, parameter(scope + "/biases")) + skip
            scope = "SegmentDecoder/seg/"
            # tf.contrib.slim's default activation on the final convolution was ReLU.
            logits = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(image, parameter(scope + "weights"), strides=1, padding="SAME"), parameter(scope + "biases")))
            self.probabilities = tf.nn.softmax(logits, axis=-1)[..., 0]
        config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=min(os.cpu_count() or 4, 12), inter_op_parallelism_threads=2)
        self.session = tf.compat.v1.Session(graph=self.graph, config=config)

    def predict(self, images):
        array = np.asarray(images, dtype=np.float32)
        if array.ndim == 3:
            array = array[..., None]
        if array.shape[1:] != (480, 480, 1):
            raise ValueError("Network input must be a batch of 480×480 grayscale images.")
        probabilities = self.session.run(self.probabilities, {self.inputs: array})
        return clean_masks(probabilities)

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def largest_mask(mask):
    contours, _ = cv2.findContours(np.uint8(mask), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    result = np.zeros(mask.shape, np.uint8)
    if contours:
        cv2.drawContours(result, [max(contours, key=cv2.contourArea)], -1, 1, -1)
    return result


class MaskRefiner:
    """Per-consumer scratch space. The returned mask is valid until the next call."""

    def __init__(self):
        self.small = np.empty((480, 480), np.float32)
        self.scaled = np.empty((MEASUREMENT_SIZE, MEASUREMENT_SIZE), np.float32)
        self.output = np.empty(self.scaled.shape, np.uint8)
        self.native = None

    def __call__(self, mask480, square, valid, mode="off", threshold=30):
        if self.native is None or self.native.shape != square.shape:
            self.native = np.empty(square.shape, np.float32)
            self.selected = np.empty(square.shape, bool)
            self.intensity = np.empty(square.shape, bool)
        np.copyto(self.small, largest_mask(mask480))
        cv2.resize(self.small, (square.shape[1], square.shape[0]), dst=self.native,
                   interpolation=cv2.INTER_LINEAR)
        np.greater(self.native, 0, out=self.selected)
        np.logical_and(self.selected, valid, out=self.selected)
        if mode in ("dark", "bright"):
            compare = np.less_equal if mode == "dark" else np.greater_equal
            compare(square, threshold, out=self.intensity)
            np.logical_and(self.selected, self.intensity, out=self.selected)
        elif mode != "off":
            raise ValueError("Unknown mask refinement mode.")
        np.copyto(self.native, self.selected)
        # Preserve the original interpolation and >0 threshold, including holes.
        cv2.resize(self.native, (MEASUREMENT_SIZE, MEASUREMENT_SIZE), dst=self.scaled,
                   interpolation=cv2.INTER_LINEAR)
        np.greater(self.scaled, 0, out=self.output)
        return self.output


def refine_mask(mask480, square, valid, mode="off", threshold=30):
    return MaskRefiner()(mask480, square, valid, mode, threshold)


class FramePreparer:
    """Reuse crop geometry without sharing mutable image buffers between batch items."""

    def __init__(self, roi=None):
        self.roi = roi
        self.crop = None

    def __call__(self, gray):
        if self.crop is None or self.crop.shape != gray.shape[:2]:
            self.crop = SquareCrop(gray.shape, self.roi)
        square = self.crop.apply(gray)
        return cv2.resize(square, (480, 480), interpolation=cv2.INTER_AREA), square, self.crop.valid


def network_image(gray, roi=None):
    return FramePreparer(roi)(gray)


def overlay(gray, mask480, roi, mode, threshold, *, prepared=None):
    image, square, valid = network_image(gray, roi) if prepared is None else prepared
    mask = refine_mask(mask480, square, valid, mode, threshold)
    view = cv2.cvtColor(cv2.resize(square, (720, 720)), cv2.COLOR_GRAY2RGB)
    display_mask = cv2.resize(mask, (720, 720), interpolation=cv2.INTER_NEAREST).astype(bool)
    view[display_mask] = (view[display_mask] * 0.5 + np.array([30, 230, 100]) * 0.5).astype(np.uint8)
    return view, mask
