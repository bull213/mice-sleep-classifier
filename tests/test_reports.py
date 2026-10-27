import json
from mouse_sleep.reports import scored_rows, summaries
from tests.helpers import FakeClassifier


def test_unknown_breaks_sleep_bouts_and_smoothing():
    records = [dict(epoch=i, start=i * 10, end=(i + 1) * 10, coverage=1, reason="tracking_gap" if i == 1 else "",
                    probabilities=None if i == 1 else json.dumps([0.01, 0.98, 0.01])) for i in range(3)]
    rows = scored_rows(records, FakeClassifier(), 0.6)
    assert [row["state"] for row in rows] == ["sleep", "unknown", "sleep"]
    bouts, hourly, totals = summaries(rows)
    assert len(bouts) == 2 and totals["unknown"] == 10 and totals["sleep"] == 20
    assert hourly[0]["sleep_seconds"] == 20
