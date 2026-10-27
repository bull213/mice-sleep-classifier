"""Synthetic software fixtures; these do not validate biological accuracy."""
import cv2
import numpy as np


def make_video(path, count=75, fps=30):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (96, 64))
    assert writer.isOpened()
    for i in range(count):
        frame = np.full((64, 96, 3), 180, np.uint8)
        cv2.ellipse(frame, (45 + int(4 * np.sin(i / 9)), 32), (12, 8 + i % 3), i % 180, 0, 360, (10, 10, 10), -1)
        writer.write(frame)
    writer.release()
    return path


class FakeClassifier:
    schema = {"model_sha256": "test-classifier"}
    transition = np.full((3, 3), 1 / 3)
    initial = np.full(3, 1 / 3)

    def __init__(self, *args):
        pass

    def predict(self, feature):
        return np.array([[0.02, 0.95, 0.03]])


class FakeSegmenter:
    def __init__(self, *args):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def predict(self, images):
        return np.asarray(images) < 50
