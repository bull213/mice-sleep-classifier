"""Restore and exercise all supplied segmentation checkpoints."""
import json
import numpy as np
from mouse_sleep.config import Settings
from mouse_sleep.segmentation import Segmenter

results = []
for name in ("24Hr_OFA", "FullModel", "AllCoatColor_RegularOFA"):
    with Segmenter(Settings(model=name).checkpoint) as model:
        output = model.predict(np.full((1, 480, 480), 180, dtype=np.uint8))
        results.append({"model": name, "restored_parameters": len(model.loaded_names), "shape": list(output.shape), "mask_pixels": int(output.sum())})
        print(json.dumps(results[-1]), flush=True)
