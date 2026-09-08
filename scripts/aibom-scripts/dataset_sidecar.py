"""
AIBOM Dataset Sidecar - signs and publishes dataset detection data from
outside the workload's own application container.

runtime_detector.py (usercustomize.py, running inside the app container)
only ever writes its detection results to a local file on the shared
aibom-data volume ($AIBOM_DATASET_OUTPUT) -- it never talks to the
Kubernetes API. This script runs as a Kubernetes native sidecar container
(an init container with restartPolicy: Always, see mutator.go's
buildDatasetSidecarContainer), which keeps it running for the pod's whole
lifetime without delaying startup of the main containers, and guarantees
it's only terminated after they exit.

That separation is the point: the discovery signing key for
discovery-<pod-name>.json/storage-<pod-name>.json is never mounted into the
app container specifically so it can't produce a validly-signed forgery of
its own hardware/storage claims (see CLAUDE.md's Discovery Data Signing
section). The same protection didn't previously exist for
dataset-<pod-name>.json, since the app container itself performed that
ConfigMap write using whatever credentials it held. Moving the write (and
signing) here means the app container no longer holds any Kubernetes API
credentials for this purpose at all -- see #47.

A separate signing key from discovery/storage data (aibom-dataset-hmac-key,
mounted only into this container) is used deliberately, per the same
compromise-domain-separation reasoning as the discovery key itself: this
sidecar and the discovery init container are different processes with
different inputs, so a compromise of one doesn't need to also invalidate
trust in data signed by the other.

This is NOT the same trust level as discovery data. The underlying content
still comes entirely from runtime_detector.py's in-process hooks, which the
app container's own (compromised or malicious) code could still fabricate
at the source -- this only prevents the ConfigMap-write step itself from
being forged by something other than this sidecar. See #47's write-up for
what closing that deeper gap would actually require.
"""

import hashlib
import hmac
import json
import os
import signal
import sys
import time

import k8s_api

_OUTPUT_PATH = os.environ.get("AIBOM_DATASET_OUTPUT", "/tmp/aibom/dataset_detected.json")
_DEBUG = os.environ.get("AIBOM_DEBUG", "0") == "1"
_POD_NAME = os.environ.get("POD_NAME", "")
_POD_NAMESPACE = os.environ.get("POD_NAMESPACE", "")
_DATA_CONFIGMAP = os.environ.get("AIBOM_DATA_CONFIGMAP") or k8s_api.resolve_data_configmap_name()

_SIGNING_KEY_PATH = "/var/run/secrets/aibom/dataset-signing/hmac-key"

# How often to check _OUTPUT_PATH for changes. Detection typically flushes
# once, at process exit (runtime_detector.py's atexit hook), so this doesn't
# need to be tight -- it only has to catch up before the pod actually
# terminates, and native sidecars get a final chance to run after the main
# container(s) exit (see _run_once's use at shutdown, below).
_POLL_INTERVAL_S = float(os.environ.get("AIBOM_DATASET_SIDECAR_POLL_INTERVAL", "5"))

_shutdown_requested = False


def _dbg(msg):
    if _DEBUG:
        print(f"[AIBOM-DATASET-SIDECAR-DEBUG] {msg}", file=sys.stderr, flush=True)


def _handle_sigterm(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    _dbg("received SIGTERM, will do one final pass before exiting")


def sign_payload(payload):
    """HMAC-SHA256 payload's canonical bytes with the per-namespace dataset
    signing key. Mirrors generate_snapshot.py's sign_payload exactly, but
    reads a separate key file -- see the module docstring for why this
    isn't shared with the discovery/storage signing key.

    Returns None (unsigned) if the key isn't mounted -- e.g. a namespace
    whose aibom-workload-namespace chart install predates the dataset
    signing.yaml addition -- so a missing key degrades to "unverifiable"
    rather than crashing this sidecar.
    """
    try:
        with open(_SIGNING_KEY_PATH, "rb") as f:
            key = f.read().strip()
    except OSError:
        return None
    if not key:
        return None
    return hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def _read_output_file():
    try:
        with open(_OUTPUT_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception:
        _dbg(f"could not parse {_OUTPUT_PATH}, skipping this pass")
        return None


def _run_once(last_mtime):
    """Checks _OUTPUT_PATH for a change since last_mtime; if changed, signs
    and publishes it. Returns the mtime observed this pass (unchanged from
    last_mtime if nothing was read).
    """
    try:
        mtime = os.stat(_OUTPUT_PATH).st_mtime
    except FileNotFoundError:
        return last_mtime

    if mtime == last_mtime:
        return last_mtime

    output = _read_output_file()
    if output is None:
        return last_mtime

    if not _POD_NAME or not _POD_NAMESPACE or not _DATA_CONFIGMAP:
        _dbg("POD_NAME/POD_NAMESPACE/AIBOM_DATA_CONFIGMAP not set, skipping ConfigMap write")
        return mtime

    # Canonical (sorted-key, no incidental whitespace) serialization: this
    # exact string is what gets signed and later re-hashed by the watcher,
    # so it must be reproduced byte-for-byte -- see generate_snapshot.py's
    # identical requirement for discovery/storage payloads.
    payload = json.dumps(output, sort_keys=True, separators=(",", ":"), default=str)
    data_updates = {f"dataset-{_POD_NAME}.json": payload}
    signature = sign_payload(payload)
    if signature:
        data_updates[f"dataset-{_POD_NAME}.sig"] = signature

    try:
        k8s_api.patch_configmap(_POD_NAMESPACE, _DATA_CONFIGMAP, data_updates)
        _dbg(f"published dataset data to ConfigMap {_DATA_CONFIGMAP}")
    except Exception as e:
        print(f"WARNING: dataset sidecar could not write to ConfigMap {_DATA_CONFIGMAP}: {e}", file=sys.stderr)
        # Don't advance last_mtime -- retry this same content next pass
        # rather than silently dropping it.
        return last_mtime

    return mtime


def main():
    signal.signal(signal.SIGTERM, _handle_sigterm)
    _dbg(f"watching {_OUTPUT_PATH} for pod {_POD_NAMESPACE}/{_POD_NAME}, configmap={_DATA_CONFIGMAP}")

    last_mtime = None
    while not _shutdown_requested:
        last_mtime = _run_once(last_mtime)
        time.sleep(_POLL_INTERVAL_S)

    # Native sidecars (restartPolicy: Always init containers) are only sent
    # SIGTERM after every main container has already exited, so this is the
    # last chance to catch a final atexit-triggered flush from
    # runtime_detector.py before the pod terminates.
    _run_once(last_mtime)
    _dbg("shutting down")


if __name__ == "__main__":
    main()
