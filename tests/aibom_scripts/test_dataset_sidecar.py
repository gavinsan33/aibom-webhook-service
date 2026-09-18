import hashlib
import hmac
import json

import dataset_sidecar as ds


def hmac_hex(key, payload):
    return hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()


# ---------------------------------------------------------------------------
# sign_payload()
# ---------------------------------------------------------------------------


def test_sign_payload_returns_none_when_key_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "_SIGNING_KEY_PATH", str(tmp_path / "does-not-exist"))
    assert ds.sign_payload("payload") is None


def test_sign_payload_returns_none_when_key_file_empty(tmp_path, monkeypatch):
    key_path = tmp_path / "hmac-key"
    key_path.write_bytes(b"")
    monkeypatch.setattr(ds, "_SIGNING_KEY_PATH", str(key_path))
    assert ds.sign_payload("payload") is None


def test_sign_payload_matches_expected_hmac(tmp_path, monkeypatch):
    key_path = tmp_path / "hmac-key"
    key_path.write_bytes(b"test-key")
    monkeypatch.setattr(ds, "_SIGNING_KEY_PATH", str(key_path))

    got = ds.sign_payload("payload")

    assert got == hmac_hex(b"test-key", "payload")


# ---------------------------------------------------------------------------
# _run_once()
# ---------------------------------------------------------------------------


def _setup(tmp_path, monkeypatch, *, signing_key=None):
    output_path = tmp_path / "dataset_detected.json"
    monkeypatch.setattr(ds, "_OUTPUT_PATH", str(output_path))
    monkeypatch.setattr(ds, "_POD_NAME", "train-job-abc123")
    monkeypatch.setattr(ds, "_POD_NAMESPACE", "gavin-test")
    monkeypatch.setattr(ds, "_DATA_CONFIGMAP", "train-job-aibom-postprocess-data")
    if signing_key is not None:
        key_path = tmp_path / "hmac-key"
        key_path.write_bytes(signing_key)
        monkeypatch.setattr(ds, "_SIGNING_KEY_PATH", str(key_path))
    else:
        monkeypatch.setattr(ds, "_SIGNING_KEY_PATH", str(tmp_path / "no-such-key"))
    return output_path


def test_run_once_skips_when_file_absent(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(ds.k8s_api, "patch_configmap", lambda *a, **k: calls.append((a, k)))

    result = ds._run_once(None)

    assert result is None
    assert calls == []


def test_run_once_publishes_signed_payload_on_change(tmp_path, monkeypatch):
    output_path = _setup(tmp_path, monkeypatch, signing_key=b"shared-secret")
    output_path.write_text(json.dumps({"datasets": [{"dataset_name": "tatsu-lab/alpaca"}]}))
    calls = []
    monkeypatch.setattr(ds.k8s_api, "patch_configmap", lambda ns, name, updates: calls.append((ns, name, updates)))

    result = ds._run_once(None)

    assert result is not None
    assert len(calls) == 1
    ns, name, updates = calls[0]
    assert ns == "gavin-test"
    assert name == "train-job-aibom-postprocess-data"
    payload = updates["dataset-train-job-abc123.json"]
    assert updates["dataset-train-job-abc123.sig"] == hmac_hex(b"shared-secret", payload)
    assert json.loads(payload) == {"datasets": [{"dataset_name": "tatsu-lab/alpaca"}]}


def test_run_once_publishes_unsigned_when_no_key_configured(tmp_path, monkeypatch):
    output_path = _setup(tmp_path, monkeypatch)
    output_path.write_text(json.dumps({"datasets": []}))
    calls = []
    monkeypatch.setattr(ds.k8s_api, "patch_configmap", lambda ns, name, updates: calls.append(updates))

    ds._run_once(None)

    updates = calls[0]
    assert "dataset-train-job-abc123.json" in updates
    assert "dataset-train-job-abc123.sig" not in updates


def test_run_once_skips_unchanged_file(tmp_path, monkeypatch):
    output_path = _setup(tmp_path, monkeypatch, signing_key=b"key")
    output_path.write_text(json.dumps({"datasets": []}))
    calls = []
    monkeypatch.setattr(ds.k8s_api, "patch_configmap", lambda *a, **k: calls.append(1))

    mtime = ds._run_once(None)
    ds._run_once(mtime)

    assert len(calls) == 1


def test_run_once_does_not_advance_mtime_on_write_failure(tmp_path, monkeypatch):
    output_path = _setup(tmp_path, monkeypatch, signing_key=b"key")
    output_path.write_text(json.dumps({"datasets": []}))

    def failing_patch(*a, **k):
        raise RuntimeError("api unavailable")

    monkeypatch.setattr(ds.k8s_api, "patch_configmap", failing_patch)

    result = ds._run_once(None)

    assert result is None
