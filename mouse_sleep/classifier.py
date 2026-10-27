from __future__ import annotations

from pathlib import Path
import json

import numpy as np
from xgboost import XGBClassifier

from .config import ROOT, SEED, Cancelled, atomic_json, sha256
from .features import FEATURE_NAMES
from .tasks import CancelCallback, ProgressCallback, check_cancelled, ignore_progress


def verify_dataset(path, cancel=None):
    from .dataset import verify_dataset as verify
    return verify(path, cancel)


def download_dataset(directory=None, progress=ignore_progress, cancel=None):
    from .dataset import download_dataset as download
    return download(directory, progress, cancel)


def viterbi(probabilities, transition, initial):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if len(probabilities) == 0:
        return np.empty(0, dtype=int)
    log_t = np.log(np.maximum(transition, 1e-12))
    log_p = np.log(np.maximum(probabilities, 1e-12))
    scores = np.log(np.maximum(initial, 1e-12)) + log_p[0]
    back = np.zeros((len(probabilities), 3), np.uint8)
    for i in range(1, len(probabilities)):
        choices = scores[:, None] + log_t
        back[i] = choices.argmax(axis=0)
        scores = choices.max(axis=0) + log_p[i]
    states = np.zeros(len(probabilities), dtype=int)
    states[-1] = scores.argmax()
    for i in range(len(states) - 1, 0, -1):
        states[i - 1] = back[i, states[i]]
    return states


def ordered_sequences(df):
    """Split per-animal sequences at missing 10-second epochs."""
    import pandas as pd

    for _, group in df.groupby("video", sort=True):
        parsed = pd.to_datetime(group["unique_epoch_id"].str.extract(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?)", expand=False), errors="coerce")
        if parsed.isna().any():
            raise ValueError("Epoch IDs must contain timestamps so temporal transitions can be trained without inventing order.")
        group = group.assign(_time=parsed).sort_values("_time")
        if group["_time"].duplicated().any():
            raise ValueError("Duplicate epoch timestamps within an animal.")
        seconds = group["_time"].astype("int64").to_numpy() / 1e9
        boundaries = np.r_[0, np.flatnonzero(np.abs(np.diff(seconds) - 10) > 0.01) + 1, len(group)]
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            yield group.iloc[start:stop]


def transitions(df):
    counts = np.full((3, 3), 1e-3, dtype=np.float64)
    starts = np.full(3, 1e-3)
    for group in ordered_sequences(df):
        states = group.Stage.to_numpy(dtype=int)
        starts[states[0]] += 1
        if len(states) > 1:
            np.add.at(counts, (states[:-1], states[1:]), 1)
    return counts / counts.sum(axis=1, keepdims=True), starts / starts.sum()


def balanced_indices(labels, seed=SEED):
    rng = np.random.RandomState(seed)
    pools = [np.flatnonzero(labels == i) for i in range(3)]
    minimum = min(map(len, pools))
    if not minimum:
        raise ValueError("Training requires Wake, NREM and REM labels.")
    return np.sort(np.concatenate([rng.choice(pool, minimum, replace=False) for pool in pools]))


def build_model():
    return XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.3, subsample=1.0, colsample_bytree=1.0,
                         objective="multi:softprob", num_class=3, tree_method="hist", random_state=SEED, n_jobs=8)


def _load_training_data(csv_path):
    import pandas as pd

    columns = ["unique_epoch_id", "video", "Stage", *FEATURE_NAMES]
    df = pd.read_csv(csv_path, usecols=columns)
    df[FEATURE_NAMES] = df[FEATURE_NAMES].replace([np.inf, -np.inf], np.nan)
    if not df.Stage.isin([0, 1, 2]).all() or df.video.isna().any():
        raise ValueError("Published training data contains invalid animal or stage labels.")
    usable = df[FEATURE_NAMES].notna().mean(axis=1) >= 0.8
    dropped = int((~usable).sum())
    df = df[usable].reset_index(drop=True)
    return df, dropped


def _validation_report(df, dropped, progress, cancel):
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split

    animals = sorted(df.video.unique())
    if len(animals) < 4:
        raise ValueError("At least four animals are needed for held-out validation.")
    train_animals, test_animals = train_test_split(animals, test_size=0.3, random_state=SEED)
    training = df[df.video.isin(train_animals)].copy()
    validation = df[df.video.isin(test_animals)].copy()
    selection = balanced_indices(training.Stage.to_numpy())
    model = build_model()
    progress({"message": f"Training classifier on {len(train_animals)} animals; validating on {len(test_animals)} held-out animals"})
    model.fit(training[FEATURE_NAMES].iloc[selection].to_numpy(), training.Stage.iloc[selection].to_numpy())
    transition, initial = transitions(training)
    actual, raw, filtered = [], [], []
    for group in ordered_sequences(validation):
        if cancel is not None and cancel():
            raise Cancelled()
        probabilities = model.predict_proba(group[FEATURE_NAMES].to_numpy())
        actual.extend(group.Stage)
        raw.extend(probabilities.argmax(axis=1))
        filtered.extend(viterbi(probabilities, transition, initial))
    actual, raw, filtered = map(np.asarray, (actual, raw, filtered))
    report = {"validation_type": "held-out animals before class balancing", "seed": SEED,
              "training_animals": list(train_animals), "validation_animals": list(test_animals),
              "epochs": len(df), "discarded_invalid_epochs": dropped, "balanced_training_epochs": len(selection),
              "raw_three_state": classification_report(actual, raw, output_dict=True, zero_division=0),
              "smoothed_three_state": classification_report(actual, filtered, output_dict=True, zero_division=0),
              "sleep_awake": classification_report(actual > 0, filtered > 0, target_names=["awake", "sleep"], output_dict=True, zero_division=0),
              "sleep_awake_confusion_matrix": confusion_matrix(actual > 0, filtered > 0).tolist(),
              "limitation": "Published reference-data validation does not establish accuracy on FLIR footage or white mice."}
    return report


def _fit_final_model(df, progress, cancel):
    progress({"message": "Saving reusable classifier trained on all reference animals"})
    selection = balanced_indices(df.Stage.to_numpy())
    final = build_model()
    final.fit(df[FEATURE_NAMES].iloc[selection].to_numpy(), df.Stage.iloc[selection].to_numpy())
    transition, initial = transitions(df)
    if cancel is not None and cancel():
        raise Cancelled()
    return final, transition, initial


def _save_classifier(output, final, transition, initial, report):
    import importlib.metadata
    from .dataset import DATASET_RECORD, DATASET_MD5

    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "model.pending.json"
    final.save_model(temporary)
    temporary.replace(output / "model.json")
    schema = {"schema_version": 1, "features": FEATURE_NAMES, "sample_rate": 30, "epoch_seconds": 10,
              "measurement_size": 1080, "classes": ["awake", "NREM", "REM"], "seed": SEED,
              "transition": transition.tolist(), "initial": initial.tolist(), "dataset": DATASET_RECORD,
              "dataset_md5": DATASET_MD5, "model_sha256": sha256(output / "model.json"),
              "versions": {name: importlib.metadata.version(name) for name in ["xgboost", "scikit-learn", "scipy", "numpy"]},
              "training_cohort": "Published C57BL/6J reference animals; white/light mouse transfer is experimental"}
    atomic_json(output / "schema.json", schema)
    atomic_json(output / "validation.json", report)


def train(csv_path=None, output=None, progress: ProgressCallback = ignore_progress, cancel: CancelCallback = None):
    output = Path(output or ROOT / "classifier")
    csv_path = verify_dataset(csv_path, cancel) if csv_path else download_dataset(progress=progress, cancel=cancel)
    check_cancelled(cancel)
    progress({"message": "Reading selected training features"})
    df, dropped = _load_training_data(csv_path)
    report = _validation_report(df, dropped, progress, cancel)
    check_cancelled(cancel)
    final, transition, initial = _fit_final_model(df, progress, cancel)
    _save_classifier(output, final, transition, initial, report)
    progress({"message": f"Classifier ready. Held-out sleep/awake accuracy: {report['sleep_awake']['accuracy']:.1%}", "fraction": 1.0})
    return report


class SleepClassifier:
    def __init__(self, directory):
        self.directory = Path(directory)
        if not (self.directory / "schema.json").is_file() or not (self.directory / "model.json").is_file():
            raise FileNotFoundError("Sleep classifier is not set up yet. Open Classifier setup and train from the published data first.")
        self.schema = json.loads((self.directory / "schema.json").read_text(encoding="utf-8"))
        expected = {"schema_version": 1, "features": FEATURE_NAMES, "sample_rate": 30, "epoch_seconds": 10, "measurement_size": 1080}
        if any(self.schema.get(key) != value for key, value in expected.items()):
            raise ValueError("Classifier feature schema is incompatible with this analyzer.")
        if sha256(self.directory / "model.json") != self.schema.get("model_sha256"):
            raise ValueError("Classifier checksum failed; repeat classifier setup.")
        self.model = XGBClassifier()
        self.model.load_model(self.directory / "model.json")
        self.transition = np.asarray(self.schema["transition"], dtype=float)
        self.initial = np.asarray(self.schema["initial"], dtype=float)
        if self.transition.shape != (3, 3) or self.initial.shape != (3,) or not np.isfinite(self.transition).all() or not np.isfinite(self.initial).all() or np.any(self.transition < 0) or np.any(self.initial < 0) or not np.allclose(self.transition.sum(axis=1), 1) or not np.isclose(self.initial.sum(), 1):
            raise ValueError("Classifier contains invalid temporal probabilities.")

    def predict(self, features):
        return self.model.predict_proba(np.asarray(features).reshape(-1, len(FEATURE_NAMES)))
