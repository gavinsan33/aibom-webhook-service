# AIBOM Schema

## Storage

Completed AIBOMs are stored as namespaced `AIBOM` custom resources (`aiboms.aibom.io`), one per completed workload, created in the namespace the workload ran in. The CRD is defined in `charts/aibom-webhook/crds/aibom-crd.yaml`.

Because `AIBOM` is namespaced, it follows ordinary Kubernetes RBAC: a user with `get` and `list` on `aiboms` in `team-a` cannot see `team-b`'s AIBOMs.

`spec` is immutable once created. The CRD rejects any update that changes it, even from a user with `update` or `patch` on `aiboms.aibom.io`. Deletion is still governed by ordinary `delete` RBAC.

Preferred: the `oc aibom` plugin from [`oc-aibom`](https://github.com/gavinsan33/oc-aibom) (a plugin subcommand, not built into `oc`), or the console plugin in the web UI:

```bash
# Requires the oc-aibom plugin
oc aibom list -n <namespace>
oc aibom describe <name> -n <namespace>
```

Fallback, with no plugin needed (standard `oc`/`kubectl`):

```bash
# List AIBOMs in a namespace
oc get aiboms -n <namespace>

# Inspect one, including the full compiled AIBOM under spec.data
oc get aibom <name> -n <namespace> -o yaml
```

## Spec fields

| Field | Contents |
|-------|----------|
| `spec.data` | The complete AIBOM JSON, exactly as `postprocess.py` produced it |
| `spec.jobName` | Summary field for printer columns |
| `spec.modelName` | Summary field for printer columns |
| `spec.experimentIntent` | Summary field for printer columns |
| `spec.collectedAt` | Summary field for printer columns |

## Telemetry

The postprocess Job queries Prometheus (Thanos Querier) directly for the AIBOM's summary stats. It authenticates with its own ServiceAccount token and trusts the cluster's service-ca bundle. Set the endpoint with the chart's `prometheus.url` value, which defaults to OpenShift's `https://thanos-querier.openshift-monitoring.svc:9091`. Leave it empty to disable telemetry collection.

A downsampled time series for each metric (about 200 points) is stored in an `AIBOMTelemetry` custom resource (`aibomtelemetries.aibom.io`, short name `aibomtel`). It is owned by the AIBOM and referenced from `spec.data.telemetry_series_ref`, so charts keep working after Prometheus's retention window. It is optional: AIBOMs without it simply have no such field.

For the field-by-field detection reference, see [Detected Fields](../CAPABILITIES.md).
