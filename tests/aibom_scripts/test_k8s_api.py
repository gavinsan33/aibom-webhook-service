import urllib.error

import pytest

import k8s_api


def test_resolve_data_configmap_name_prefers_env_var(monkeypatch):
    monkeypatch.setenv("AIBOM_DATA_CONFIGMAP", "train-job-aibom-postprocess-data")
    monkeypatch.setenv("POD_NAME", "train-job-pod")
    assert k8s_api.resolve_data_configmap_name() == "train-job-aibom-postprocess-data"


def test_resolve_data_configmap_name_derives_from_pod_name_when_env_var_absent(monkeypatch):
    # Reproduces the bare/ReplicaSet-owned pod case (e.g. a KServe predictor):
    # the webhook can't bake in AIBOM_DATA_CONFIGMAP statically, since the
    # pod's own name isn't assigned yet at admission time — see
    # mutator.go's dataConfigMapEnvVar.
    monkeypatch.delenv("AIBOM_DATA_CONFIGMAP", raising=False)
    monkeypatch.setenv("POD_NAME", "granite-model-predictor-58f446b5c6-6mmc7")
    assert (
        k8s_api.resolve_data_configmap_name()
        == "granite-model-predictor-58f446b5c6-6mmc7-aibom-postprocess-data"
    )


def test_resolve_data_configmap_name_empty_when_neither_set(monkeypatch):
    monkeypatch.delenv("AIBOM_DATA_CONFIGMAP", raising=False)
    monkeypatch.delenv("POD_NAME", raising=False)
    assert k8s_api.resolve_data_configmap_name() == ""


def test_get_cluster_object_builds_expected_path_and_returns_result(monkeypatch):
    calls = {}

    def fake_request(method, path, body=None, content_type="application/json"):
        calls["method"] = method
        calls["path"] = path
        return {"kind": "Image"}

    monkeypatch.setattr(k8s_api, "_request", fake_request)
    result = k8s_api.get_cluster_object("image.openshift.io", "v1", "images", "sha256:abc")
    assert result == {"kind": "Image"}
    assert calls == {"method": "GET", "path": "/apis/image.openshift.io/v1/images/sha256:abc"}


def test_get_cluster_object_returns_none_on_404_or_403(monkeypatch):
    for code in (404, 403):
        def fake_request(method, path, body=None, content_type="application/json", code=code):
            raise urllib.error.HTTPError(path, code, "error", hdrs=None, fp=None)

        monkeypatch.setattr(k8s_api, "_request", fake_request)
        assert k8s_api.get_cluster_object("image.openshift.io", "v1", "images", "sha256:missing") is None


def test_get_cluster_object_reraises_other_http_errors(monkeypatch):
    def fake_request(method, path, body=None, content_type="application/json"):
        raise urllib.error.HTTPError(path, 500, "server error", hdrs=None, fp=None)

    monkeypatch.setattr(k8s_api, "_request", fake_request)
    with pytest.raises(urllib.error.HTTPError):
        k8s_api.get_cluster_object("image.openshift.io", "v1", "images", "sha256:x")


def test_resolve_data_configmap_name_truncates_long_pod_names(monkeypatch):
    # Mirrors aibomdata.PostprocessJobName's truncation in watcher.go: names
    # are cut to fit Kubernetes' 63-character limit before the suffix.
    monkeypatch.delenv("AIBOM_DATA_CONFIGMAP", raising=False)
    monkeypatch.setenv("POD_NAME", "a" * 63)
    max_base = 63 - len("-aibom-postprocess")
    assert (
        k8s_api.resolve_data_configmap_name()
        == ("a" * max_base) + "-aibom-postprocess-data"
    )
