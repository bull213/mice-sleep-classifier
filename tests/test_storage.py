import pytest
import json
import sqlite3
from types import SimpleNamespace

from mouse_sleep.config import Settings
from mouse_sleep.storage import AnalysisStore
from mouse_sleep.epochs import EpochRecord
from mouse_sleep.video import VideoInfo
from mouse_sleep.locking import OutputLock


def make_store(tmp_path, masking_mode="legacy", **kwargs):
    settings = Settings(output=str(tmp_path), masking_mode=masking_mode)
    return AnalysisStore(settings, VideoInfo(96, 64, 30, 20, 600), SimpleNamespace(manifest=lambda: {}),
                         20, {"settings": {"roi": [0, 0, 96, 64]}}, **kwargs)


def test_initialization_failure_closes_database(tmp_path, monkeypatch):
    handles = []
    connect = sqlite3.connect

    def track(*args, **kwargs):
        connection = connect(*args, **kwargs)
        handles.append(connection)
        return connection

    def fail(*args):
        raise OSError("settings write failed")

    monkeypatch.setattr("mouse_sleep.storage.sqlite3.connect", track)
    monkeypatch.setattr(Settings, "save", fail)
    with pytest.raises(OSError, match="settings write failed"):
        with make_store(tmp_path):
            pytest.fail("Initialization should have failed")
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        handles[0].execute("SELECT 1")
    assert json.loads((tmp_path / "run.json").read_text())["status"] == "error"


def test_committed_database_recovers_from_stale_manifest(tmp_path):
    with make_store(tmp_path) as store:
        store.persist(EpochRecord(0, 0, 10, 0, "mouse_not_tracked", None, None))
    manifest = json.loads((tmp_path / "run.json").read_text())
    manifest["completed_epochs"] = 0
    (tmp_path / "run.json").write_text(json.dumps(manifest))
    with make_store(tmp_path, resume=True) as resumed:
        assert resumed.next_epoch == 1
        assert len(resumed.records()) == 1


@pytest.mark.parametrize("missing", ["run.json", "analysis.sqlite3"])
def test_resume_rejects_missing_artifacts(tmp_path, missing):
    with make_store(tmp_path):
        pass
    (tmp_path / missing).unlink()
    with pytest.raises(ValueError, match="manifest|database"):
        with make_store(tmp_path, resume=True):
            pass


def test_output_lock_prevents_concurrent_writers_and_releases(tmp_path):
    with OutputLock(tmp_path):
        with pytest.raises(ValueError, match="Another analysis"):
            with OutputLock(tmp_path):
                pass
    with OutputLock(tmp_path):
        pass


def test_enhanced_checkpoint_failure_rolls_back_epoch(tmp_path):
    with make_store(tmp_path, masking_mode="enhanced") as store:
        with pytest.raises(ValueError):
            store.persist(EpochRecord(0, 0, 10, 0, "missing", None, None),
                          masking={"state": {}, "counts": {"invalid": float("nan")}})
        assert store.connection.execute("SELECT COUNT(*) FROM epochs").fetchone()[0] == 0
        assert store.connection.execute("SELECT COUNT(*) FROM masking_epochs").fetchone()[0] == 0
