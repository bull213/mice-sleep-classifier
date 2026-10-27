"""Visual smoke test on the authors' annotated albino demonstration.

Supplementary Movie 4, Geuther et al. (2019), CC BY 4.0.
This is not a sleep accuracy benchmark or unannotated FLIR footage.
"""
import cv2
from mouse_sleep.config import ROOT, Settings, atomic_json
from mouse_sleep.segmentation import Segmenter, network_image, overlay

directory = ROOT / "verification"
video = directory / "supplementary_albino.mp4"
capture = cv2.VideoCapture(str(video))
if not capture.isOpened():
    raise FileNotFoundError("Place the authors' Supplementary Movie 4 in verification/supplementary_albino.mp4 first.")
samples = []
for second in (2, 15, 35, 55):
    capture.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
    okay, frame = capture.read()
    assert okay
    samples.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
capture.release()
report = []
for name in ("FullModel", "AllCoatColor_RegularOFA"):
    settings = Settings(appearance="white", model=name)
    with Segmenter(settings.checkpoint) as model:
        masks = model.predict([network_image(gray)[0] for gray in samples])
    tiles, areas = [], []
    for gray, mask in zip(samples, masks, strict=True):
        rendered, refined = overlay(gray, mask, None, "off", 30)
        tiles.append(cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
        areas.append(float(refined.mean()))
    montage = cv2.vconcat([cv2.hconcat(tiles[:2]), cv2.hconcat(tiles[2:])])
    cv2.imwrite(str(directory / ("white_reference_" + name + ".png")), montage)
    report.append({"model": name, "mask_area_fractions_at_2_15_35_55_seconds": areas})
    print(report[-1], flush=True)
atomic_json(directory / "white_reference.json", {"source": "Geuther et al. 2019 Supplementary Movie 4; annotated demonstration only", "checks": report})
