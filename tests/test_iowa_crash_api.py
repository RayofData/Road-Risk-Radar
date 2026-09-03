import pytest
import pandas as pd

from src.apis.iowa_crash_api import batched, request_json
from src.etl.extract_iowa_crash_data import validate_download


def test_batched_splits_values_and_keeps_remainder():
    values = [1, 2, 3, 4, 5]

    result = list(batched(values, batch_size=2))

    assert result == [[1, 2], [3, 4], [5]]


def test_batched_rejects_zero_batch_size():
    with pytest.raises(ValueError):
        list(batched([1, 2, 3], batch_size=0))


def test_batched_rejects_negative_batch_size():
    with pytest.raises(ValueError):
        list(batched([1, 2, 3], batch_size=-1))


def test_batched_rejects_noninteger_batch_size():
    with pytest.raises(ValueError):
        list(batched([1, 2, 3], batch_size=3.14))


def test_validate_download_rejects_missing_ids():
    crashes = pd.DataFrame({
        "OBJECTID": [1,2,4]
    })

    expected_object_ids = [1,2,3,4]

    with pytest.raises(RuntimeError, match="object IDs were not downloaded"):
        validate_download(crashes, expected_object_ids)


def test_request_json_raises_for_api_error(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass 

        def json(self):
            return {
                "error": {
                    "message": "Invalid query",
                    "details": ["Bad parameter"]
                }
            }

    def fake_get(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr(
        "src.apis.iowa_crash_api.requests.get",
        fake_get
    )

    with pytest.raises(RuntimeError, match="Invalid query"):
        request_json({"where": "1=1"})


def test_validate_download_rejects_unexpected_ids():
    crashes = pd.DataFrame({
        "OBJECTID": [1,2,3,4,5]
    })

    expected_object_ids = [1,2,3,4]

    with pytest.raises(RuntimeError, match="object IDs were downloaded"):
        validate_download(crashes, expected_object_ids)


def test_validate_download_rejects_duplicate_ids():
    crashes = pd.DataFrame({
        "OBJECTID": [1,2,3,3]
    })

    expected_object_ids = [1,2,3]

    with pytest.raises(RuntimeError, match="duplicate OBJECTIDs"):
        validate_download(crashes, expected_object_ids)