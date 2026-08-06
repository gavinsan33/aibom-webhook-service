# Minimal, dependency-free in-cluster Kubernetes REST client.
#
# Uses only the Python stdlib so it can run inside arbitrary user application
# images (runtime_detector.py runs as usercustomize.py in the workload's own
# container) as well as the pinned discovery image, without requiring the
# `kubernetes` pip package to be installed anywhere it's imported.

import json
import os
import ssl
import urllib.error
import urllib.request

_SA_DIR = "/var/run/secrets/kubernetes.io/serviceaccount"

# Mirrors aibomdata's constants in watcher.go — the two must stay in sync,
# since the Go watcher and resolve_data_configmap_name() below each
# independently compute the same deterministic name for the same trigger.
_MAX_JOB_NAME_LENGTH = 63
_POSTPROCESS_SUFFIX = "-aibom-postprocess"
_CONFIGMAP_SUFFIX = "-data"


def _api_server():
    host = os.environ["KUBERNETES_SERVICE_HOST"]
    port = os.environ["KUBERNETES_SERVICE_PORT"]
    return f"https://{host}:{port}"


def _token():
    with open(os.path.join(_SA_DIR, "token")) as f:
        return f.read().strip()


def _ssl_context():
    return ssl.create_default_context(cafile=os.path.join(_SA_DIR, "ca.crt"))


def _request(method, path, body=None, content_type="application/json"):
    url = _api_server() + path
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {_token()}")
    req.add_header("Content-Type", content_type)
    req.add_header("Accept", "application/json")
    with urllib.request.urlopen(req, context=_ssl_context(), timeout=15) as resp:
        raw = resp.read()
        return json.loads(raw) if raw else None


def patch_configmap(namespace, name, data_updates):
    """Merge `data_updates` into ConfigMap `name`'s `data`, creating the
    ConfigMap if it doesn't exist yet. Safe to call concurrently from
    multiple pods/containers targeting the same ConfigMap — each caller only
    touches its own keys, and a lost create race is retried as a patch.
    """
    path = f"/api/v1/namespaces/{namespace}/configmaps/{name}"
    patch_body = {"data": data_updates}
    try:
        return _request(
            "PATCH", path, body=patch_body, content_type="application/merge-patch+json"
        )
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        create_body = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {"name": name, "namespace": namespace},
            "data": data_updates,
        }
        try:
            return _request("POST", f"/api/v1/namespaces/{namespace}/configmaps", body=create_body)
        except urllib.error.HTTPError as create_err:
            if create_err.code != 409:
                raise
            # Lost the create race against a sibling pod/container — it exists now.
            return _request(
                "PATCH", path, body=patch_body, content_type="application/merge-patch+json"
            )


def create_custom_object(namespace, group, version, plural, body):
    """POST a namespaced custom resource, e.g. an AIBOM (aibom.io/v1alpha1)."""
    path = f"/apis/{group}/{version}/namespaces/{namespace}/{plural}"
    return _request("POST", path, body=body)


def get_custom_object(namespace, group, version, plural, name):
    """GET a namespaced custom resource, e.g. a KServe InferenceService
    (serving.kserve.io/v1beta1). Returns None if it doesn't exist (or has
    already been deleted) rather than raising, since callers use this for
    best-effort enrichment, not anything that should fail the caller outright.
    """
    path = f"/apis/{group}/{version}/namespaces/{namespace}/{plural}/{name}"
    try:
        return _request("GET", path)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def _postprocess_job_name(trigger_name):
    max_base = _MAX_JOB_NAME_LENGTH - len(_POSTPROCESS_SUFFIX)
    trigger_name = trigger_name[:max_base].rstrip("-")
    return trigger_name + _POSTPROCESS_SUFFIX


def resolve_data_configmap_name():
    """Returns the data ConfigMap name discovery/dataset scripts should write
    into. Prefers the webhook-injected AIBOM_DATA_CONFIGMAP env var, which is
    reliable for Job/JobSet/PyTorchJob/RayJob-owned pods (their trigger name
    comes from ownerReferences, already set before admission).

    Falls back to deriving it from POD_NAME for bare/ReplicaSet-owned pods
    (e.g. KServe predictors): their own pod name isn't assigned yet when the
    mutating webhook runs (the API server hasn't resolved generateName to a
    real name at that point), so the webhook can't bake in a static value for
    them — see mutator.go's dataConfigMapEnvVar. POD_NAME, a downward API
    value resolved later by the kubelet, is reliable by the time this
    actually runs.
    """
    configmap_name = os.environ.get("AIBOM_DATA_CONFIGMAP", "")
    if configmap_name:
        return configmap_name
    pod_name = os.environ.get("POD_NAME", "")
    if not pod_name:
        return ""
    return _postprocess_job_name(pod_name) + _CONFIGMAP_SUFFIX
