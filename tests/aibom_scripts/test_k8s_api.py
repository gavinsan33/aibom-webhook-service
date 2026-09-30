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
    # mutator.go's dataConfigMapEnvVar. Pod name is 39 chars (the truncation
    # budget itself -- see test below), so it survives untruncated here; a
    # name one character longer would already lose its last character, per
    # aibomdata.go's truncatedTriggerBase.
    monkeypatch.delenv("AIBOM_DATA_CONFIGMAP", raising=False)
    monkeypatch.setenv("POD_NAME", "granite-model-predictor-58f446b5c-6mmc7")
    assert (
        k8s_api.resolve_data_configmap_name()
        == "granite-model-predictor-58f446b5c-6mmc7-aibom-postprocess-data"
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


def test_set_custom_object_owner_merge_patches_owner_references(monkeypatch):
    calls = {}

    def fake_request(method, path, body=None, content_type="application/json"):
        calls.update(method=method, path=path, body=body, content_type=content_type)

    monkeypatch.setattr(k8s_api, "_request", fake_request)
    owner = {"apiVersion": "aibom.io/v1alpha1", "kind": "AIBOM", "name": "x", "uid": "u"}
    k8s_api.set_custom_object_owner("ns", "aibom.io", "v1alpha1", "aibomtelemetries", "tel", owner)
    assert calls["method"] == "PATCH"
    assert calls["path"] == "/apis/aibom.io/v1alpha1/namespaces/ns/aibomtelemetries/tel"
    assert calls["body"] == {"metadata": {"ownerReferences": [owner]}}
    assert calls["content_type"] == "application/merge-patch+json"


def test_k8s_api_exposes_no_delete_helper():
    # aibom-postprocess is deliberately not granted delete on anything it
    # creates (rbac.yaml); a helper for it would only ever 403.
    assert not hasattr(k8s_api, "delete_custom_object")


def test_resolve_data_configmap_name_truncates_long_pod_names(monkeypatch):
    # Mirrors aibomdata.ConfigMapName's truncation in aibomdata.go: budgeted
    # against WorkloadIdentitySuffix's length (the narrowest of the three
    # suffixes aibomdata.go's truncatedTriggerBase supports), not this
    # module's own (shorter) _POSTPROCESS_SUFFIX -- see
    # _WORKLOAD_IDENTITY_SUFFIX_LEN's comment in k8s_api.py for why a past
    # mismatch here silently broke every long-named bare pod's AIBOM.
    monkeypatch.delenv("AIBOM_DATA_CONFIGMAP", raising=False)
    monkeypatch.setenv("POD_NAME", "a" * 63)
    max_base = 63 - len("-aibom-workload-identity")
    assert (
        k8s_api.resolve_data_configmap_name()
        == ("a" * max_base) + "-aibom-postprocess-data"
    )
