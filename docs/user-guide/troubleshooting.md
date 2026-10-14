# Troubleshooting

## Pods have no `aibom-discovery` init container

The webhook fails open: if it can't be reached, pods are created normally and no error is raised. The API server, not a pod, calls the webhook, so a NetworkPolicy in the install namespace that admits only same-namespace ingress makes every admission call time out silently. Workloads then run uninstrumented and no AIBOM is produced, with nothing logged.

The chart ships a NetworkPolicy named `aibom-webhook-admission` that allows ingress to the webhook pod's port 8443. It is on by default; set `networkPolicy.enabled=false` to opt out.

To check, run a server-side dry run of a bare GPU pod:

```bash
oc create --dry-run=server -o yaml -f <pod.yaml>
```

The result should carry the `aibom.io/instrumented` label, and a `mutating pod ...` line should appear in the webhook's log.

## Pods stay in `Init:ImagePullBackOff`

The `aibom-discovery` init container can't pull its image. Unless `image.discovery` is set, it is the OpenShift AI `runtime-pytorch` ImageStream where OpenShift AI is installed and `pytorch/pytorch` from Docker Hub elsewhere, which needs egress and can hit Docker Hub's pull rate limit. Check the event:

```bash
oc describe pod <pod-name> -n <namespace>
```

Set `image.discovery.repository` and `image.discovery.tag` to an image the cluster can pull, as shown in [Getting Started](getting-started.md#cluster-requirements), and upgrade the webhook chart. Pods already created keep the old image, so recreate them.

A common trigger on OpenShift AI is an **offline render** (e.g. Argo CD with `helm template`): the discovery-image default keys on the OpenShift AI API, which an offline render can't see, so it falls through to `pytorch/pytorch` even with `platform=openshift` set. Setting `platform` alone doesn't fix this — set `image.discovery` by hand too (see the offline note in [Getting Started](getting-started.md#cluster-requirements)).

## Pods fail to start in an enabled namespace

A stale `aibom-scripts` ConfigMap can fail pod startup for every instrumented workload in the namespace. The dataset detector hook mounts `runtime_detector.py` from that ConfigMap, and the `aibom-discovery` and `aibom-dataset-sidecar` init containers each run a script from it.

Re-run the workload-namespace install (see [Getting Started](getting-started.md)) to refresh it.

## An AIBOM has no stored time series, or `AIBOMTelemetry` objects are left over

An AIBOM is still created without `spec.data.telemetry_series_ref` when Prometheus returned no data, the series would exceed the 900 KB cap, or creating the `AIBOMTelemetry` object was denied, for example by a ResourceQuota or a namespace whose chart predates the resource. The postprocess Job logs a warning in each case.

The series object is created before the AIBOM. If the AIBOM create then fails, the object is left behind: the postprocess Job can't delete these objects, by design. The Job logs the delete command, and its retry creates a fresh object. Leftovers have no owner, so the AIBOM column is empty:

```bash
oc get aibomtel -A
oc delete aibomtel <name> -n <namespace>
```

## Telemetry is empty

Outside OpenShift, `prometheus.url` has no default, so telemetry is off until you set it. Check what the webhook was given:

```bash
oc get deployment aibom-webhook -n project-aibom -o jsonpath='{.spec.template.spec.containers[0].args}'
```

On OpenShift, reading platform metrics from Thanos Querier requires a `cluster-monitoring-view` ClusterRoleBinding, which the webhook chart creates once, for the `aibom-postprocess` ServiceAccount in its own namespace, and which needs cluster-admin permission.

If your account doesn't have it, install the webhook chart with `--set monitoringAccess.enabled=false` and have a cluster-admin apply the `ClusterRoleBinding` in `charts/aibom-webhook/templates/postprocess.yaml` once. Telemetry comes back empty until then; the install itself does not fail.
