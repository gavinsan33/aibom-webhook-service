# Minimal, dependency-free in-cluster Kubernetes REST client.
#
# Uses only the Python stdlib so it can run inside arbitrary user application
# images (dataset_detector.py runs as usercustomize.py in the workload's own
# container) as well as the pinned discovery image, without requiring the
# `kubernetes` pip package to be installed anywhere it's imported.

import json
import os
import ssl
import urllib.error
import urllib.request

_SA_DIR = "/var/run/secrets/kubernetes.io/serviceaccount"


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
