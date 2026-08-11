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
