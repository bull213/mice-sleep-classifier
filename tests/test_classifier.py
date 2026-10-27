import pandas as pd
import json
import numpy as np
import pytest
from mouse_sleep.classifier import transitions, ordered_sequences
from mouse_sleep import classifier
from mouse_sleep.config import Cancelled, sha256
from mouse_sleep.features import FEATURE_NAMES


@pytest.fixture
def training_fixture(tmp_path, monkeypatch):
    rows = []
    for animal in range(6):
        for epoch in range(6):
            features = dict.fromkeys(FEATURE_NAMES, 1.0)
            features[FEATURE_NAMES[0]] = epoch % 3
            features[FEATURE_NAMES[1]] = animal
            rows.append({"video": f"animal{animal}", "Stage": epoch % 3,
                         "unique_epoch_id": f"animal{animal} 2020-01-01 00:00:{epoch * 10:02d}", **features})
    path = tmp_path / "reference.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    monkeypatch.setattr(classifier, "verify_dataset", lambda *args: path)
    models = []

    class Model:
        def __init__(self):
            models.append(self)

        def fit(self, features, labels):
            self.features, self.labels = features.copy(), labels.copy()

        def predict_proba(self, features):
            return np.eye(3)[features[:, 0].astype(int)] * 0.97 + 0.01

        def save_model(self, destination):
            destination.write_text("{}")

    monkeypatch.setattr(classifier, "build_model", Model)
    return path, models, Model


def test_training_keeps_held_out_animals_and_saves_compatible_artifacts(training_fixture, tmp_path):
    path, models, _ = training_fixture
    output = tmp_path / "classifier"
    report = classifier.train(path, output)
    assert len(models) == 2
    fitted_animals = {f"animal{int(value)}" for value in models[0].features[:, 1]}
    assert fitted_animals == set(report["training_animals"])
    assert fitted_animals.isdisjoint(report["validation_animals"])
    assert set(models[1].features[:, 1]) == set(range(6))
    assert len(set(np.bincount(models[0].labels))) == 1
    schema = json.loads((output / "schema.json").read_text())
    assert schema["features"] == FEATURE_NAMES and schema["schema_version"] == 1
    assert schema["model_sha256"] == sha256(output / "model.json")
    assert json.loads((output / "validation.json").read_text()) == report


def test_cancelling_final_fit_does_not_publish_artifacts(training_fixture, tmp_path, monkeypatch):
    path, models, Model = training_fixture
    cancelled = False
    fit = Model.fit

    def cancel_final(self, features, labels):
        nonlocal cancelled
        fit(self, features, labels)
        cancelled = len(models) == 2

    monkeypatch.setattr(Model, "fit", cancel_final)
    output = tmp_path / "classifier"
    with pytest.raises(Cancelled):
        classifier.train(path, output, cancel=lambda: cancelled)
    assert not output.exists()


def test_temporal_training_splits_animals_and_gaps():
    df = pd.DataFrame({"video": ["a", "a", "a", "b", "b"], "Stage": [0, 1, 2, 2, 2],
                       "unique_epoch_id": ["a 2020-01-01 10:00:00", "a 2020-01-01 10:00:10", "a 2020-01-01 10:01:00", "b 2020-01-01 10:00:00", "b 2020-01-01 10:00:10"]})
    assert [len(group) for group in ordered_sequences(df)] == [2, 1, 2]
    transition, _ = transitions(df)
    assert transition[0, 1] > 0.99
    assert transition[1, 2] == pytest.approx(1 / 3)  # no learned transition across the gap
    assert transition[2, 2] > 0.99
