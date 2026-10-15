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

### Stored time series

The AIBOM itself holds only summary statistics per metric. A downsampled copy of each metric's time series is stored next to it in an `AIBOMTelemetry` custom resource (`aibomtelemetries.aibom.io`, short name `aibomtel`), so charts keep working after Prometheus's retention window. It is optional: an AIBOM without one has no `spec.data.telemetry_series_ref`, and readers fall back to live Prometheus.

**Finding it.** The object is in the AIBOM's namespace, labeled `aibom.io/job-name`, with a random name. Use the name in `spec.data.telemetry_series_ref`, not a naming convention:

```bash
oc get aibom <name> -n <namespace> -o jsonpath='{.spec.data.telemetry_series_ref.name}'
oc get aibomtel <series-name> -n <namespace> -o jsonpath='{.spec.seriesJson}'
```

It is owned by the AIBOM, so deleting the AIBOM deletes it. Like the AIBOM's, its `spec` can't be changed after creation. Users who can read AIBOMs can read these too, through the `aibom-view` role the webhook chart adds to `view`, `edit` and `admin`.

**Contents.** `spec.seriesJson` is a JSON string, kept as a string so it can be hashed exactly as stored. It holds one entry per metric that returned data, with the same metric names as `resource_utilization.metrics`, plus the vLLM metrics under `inference.performance.metrics` for vLLM pods. Each entry has:

- `aggregate`: the line for the whole run, summed across pods and containers, or averaged for utilization, KV-cache usage and latency.
- `aggregate_max`: for gauge metrics, the peak within each point's interval, so a spike shorter than the step still shows.
- `series`: one line per pod, and per container, network interface or GPU where the metric has them. It's left out, with `series_omitted: true`, when a metric has more than 64 series.

Values are in raw base units (bytes, bytes per second, cores, MiB, watts, percent, seconds), not the scaled GiB, Mbps and MBps the AIBOM reports. Timestamps are Unix seconds. Per-GPU series need dcgm-exporter's pod mapping; without it the GPU metrics are absent.

**Resolution.** The series covers each pod's full run at up to about 200 points per metric; on very large runs the per-pod `series` may be dropped to fit the size cap.

**Integrity.** The object isn't signed itself. The AIBOM's signed `spec.data.telemetry_series_ref` holds the SHA-256 and size of `spec.seriesJson`, and `oc aibom describe` checks the object against them on its `Telemetry Series:` line. It reports a match as fully verified only when the AIBOM's own signature is valid, since without that the digest could be forged along with the series. Only the postprocess Job's ServiceAccount can create these objects.

AIBOMs from before this resource existed reference a ConfigMap instead, with `configmap` and `key` fields in place of `kind` and `name`. They aren't migrated.

For the field-by-field detection reference, see [Detected Fields](../CAPABILITIES.md).
