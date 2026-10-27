import hashlib
from types import SimpleNamespace

import pytest

from mouse_sleep import dataset
from mouse_sleep.config import Cancelled


class Response:
    def __init__(self, payload, status=206, content_range=""):
        self.payload, self.status_code = payload, status
        self.headers = {"Content-Range": content_range}
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def raise_for_status(self):
        pass

    def iter_content(self, size):
        yield self.payload


@pytest.fixture
def payload(monkeypatch):
    value = b"small verified dataset fixture"
    monkeypatch.setattr(dataset, "DATASET_SIZE", len(value))
    monkeypatch.setattr(dataset, "DATASET_MD5", hashlib.md5(value).hexdigest())
    return value


def test_range_download_resumes_and_reuses_verified_cache(tmp_path, monkeypatch, payload):
    partial = tmp_path / (dataset.DATASET_NAME + ".part")
    partial.write_bytes(payload[:5])
    responses = []

    def get(url, **kwargs):
        assert kwargs["headers"]["Range"] == f"bytes=5-{len(payload) - 1}"
        response = Response(payload[5:], content_range=f"bytes 5-{len(payload) - 1}/{len(payload)}")
        responses.append(response)
        return response

    monkeypatch.setattr(dataset.requests, "get", get)
    path = dataset.download_dataset(tmp_path)
    assert path.read_bytes() == payload and not partial.exists()
    assert len(responses) == 1 and responses[0].closed
    monkeypatch.setattr(dataset.requests, "get", lambda *a, **k: pytest.fail("Cache should avoid network"))
    assert dataset.download_dataset(tmp_path) == path


@pytest.mark.parametrize("status", [200, 206])
def test_streaming_fallback_preserves_resume_semantics(tmp_path, monkeypatch, payload, status):
    partial = tmp_path / (dataset.DATASET_NAME + ".part")
    partial.write_bytes(payload[:5])

    def fail_range(*args):
        raise ValueError("range unavailable")

    response = Response(payload if status == 200 else payload[5:], status,
                        f"bytes 5-{len(payload) - 1}/{len(payload)}")

    class Session:
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, url, **kwargs):
            assert kwargs["headers"]["Range"] == "bytes=5-"
            return response

    monkeypatch.setattr(dataset, "_range_download", fail_range)
    monkeypatch.setattr(dataset.requests, "Session", Session)
    assert dataset.download_dataset(tmp_path).read_bytes() == payload
    assert response.closed


def test_range_retry_and_cached_chunk(tmp_path, monkeypatch, payload):
    downloader = dataset._RangeDownload(tmp_path / "partial", lambda event: None, None)
    monkeypatch.setattr(downloader.stopped, "wait", lambda delay: False)
    responses = [Response(b"bad", content_range="wrong range"),
                 Response(payload, content_range=f"bytes 0-{len(payload) - 1}/{len(payload)}")]
    calls = []

    def get(*args, **kwargs):
        calls.append(kwargs)
        return responses[len(calls) - 1]

    monkeypatch.setattr(dataset.requests, "get", get)
    cached = downloader.fetch((0, len(payload)))
    assert cached.read_bytes() == payload and len(calls) == 2
    assert downloader.fetch((0, len(payload))) == cached
    assert len(calls) == 2 and all(response.closed for response in responses)


def test_stream_cancellation_keeps_saved_prefix_and_closes_response(tmp_path, payload):
    partial = tmp_path / "partial"
    partial.write_bytes(payload[:5])
    response = Response(payload[5:], content_range=f"bytes 5-{len(payload) - 1}/{len(payload)}")
    session = SimpleNamespace(get=lambda *a, **k: response)
    with pytest.raises(Cancelled):
        dataset._stream_download(session, "unused", partial, lambda event: None, lambda: True)
    assert partial.read_bytes() == payload[:5] and response.closed


def test_bad_checksum_is_not_promoted(tmp_path, monkeypatch, payload):
    partial = tmp_path / (dataset.DATASET_NAME + ".part")
    partial.write_bytes(b"x" * len(payload))
    with pytest.raises(ValueError, match="checksum"):
        dataset.download_dataset(tmp_path)
    assert partial.exists() and not (tmp_path / dataset.DATASET_NAME).exists()
