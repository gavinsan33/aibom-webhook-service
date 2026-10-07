# Troubleshooting

## Pods have no `aibom-discovery` init container

The webhook fails open: if it can't be reached, pods are created normally and no error is raised. The API server, not a pod, calls the webhook, so a NetworkPolicy in the install namespace that admits only same-namespace ingress makes every admission call time out silently. Workloads then run uninstrumented and no AIBOM is produced, with nothing logged.

The chart ships a NetworkPolicy named `aibom-webhook-admission` that allows ingress to the webhook pod's port 8443. It is on by default; set `networkPolicy.enabled=false` to opt out.

To check, run a server-side dry run of a bare GPU pod:

```bash
oc create --dry-run=server -o yaml -f <pod.yaml>
```

The result should carry the `aibom.io/instrumented` label, and a `mutating pod ...` line should appear in the webhook's log.

## Pods fail to start in an enabled namespace

A stale `aibom-scripts` ConfigMap can fail pod startup for every instrumented workload in the namespace. The dataset detector hook mounts `runtime_detector.py` from that ConfigMap, and the `aibom-discovery` and `aibom-dataset-sidecar` init containers each run a script from it.

Re-run the workload-namespace install (see [Getting Started](getting-started.md)) to refresh it.

## Telemetry is empty

Reading platform metrics from Thanos Querier requires a `cluster-monitoring-view` ClusterRoleBinding, which `just setup-namespace` creates per namespace by default and which needs cluster-admin permission.

If your account doesn't have it, pass `--skip-monitoring-access` and have a cluster-admin apply the `ClusterRoleBinding` in `charts/aibom-workload-namespace/templates/monitoring.yaml` once. Telemetry comes back empty for that namespace until then; the install itself does not fail.
