# Configuration

Set chart values with `--set key=value` or `-f values.yaml` on `helm upgrade --install`. Each table lists the value, its default and what it does. The commented defaults are in `charts/aibom-webhook/values.yaml` and `charts/aibom-workload-namespace/values.yaml`.

## `aibom-webhook` chart

### Platform

| Value | Default | Description |
|-------|---------|-------------|
| `platform` | `auto` | `auto`, `openshift` or `kubernetes`. `auto` detects OpenShift by its `image.openshift.io` API, which `helm template` without a cluster can't see, so set it explicitly when rendering offline. It picks the `prometheus.url` default, whether the `Image` read RBAC is installed, and whether `build.enabled` is allowed. |

### Images

| Value | Default | Description |
|-------|---------|-------------|
| `image.webhook.repository` / `.tag` | `quay.io/gsanders/aibom-webhook-service` / `latest` | The webhook and watcher image. |
| `image.postprocess.repository` / `.tag` | `quay.io/gsanders/aibom-postprocess` / `latest` | The image the postprocess Job runs. |
| `image.discovery.repository` / `.tag` | `""` / `""` | The `aibom-discovery` init container's image. It needs only `python3` and `bash`. `nvidia-smi` is injected at runtime by the NVIDIA Container Toolkit on GPU pods. When empty, it is OpenShift AI's `image-registry.openshift-image-registry.svc:5000/redhat-ods-applications/runtime-pytorch:pytorch` where the OpenShift AI API (`datasciencecluster.opendatahub.io/v1`) exists, and `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime` otherwise. Setting `repository` requires `tag`. |
| `image.pullPolicy` | `Always` | Pull policy for the webhook and postprocess images. |

### Webhook and admission

| Value | Default | Description |
|-------|---------|-------------|
| `replicaCount` | `1` | Webhook replicas. |
| `resources` | 50m CPU / 64Mi requested, 200m / 128Mi limit | Webhook container resources. |
| `webhook.failurePolicy` | `Ignore` | What the API server does when the webhook is unreachable. `Ignore` lets pods through uninstrumented. |
| `webhook.reinvocationPolicy` | `Never` | Admission reinvocation policy. |
| `webhook.namespaceSelectorLabel.key` / `.value` | `aibom.io/enabled` / `"true"` | Namespaces must carry this label to opt in. |
| `webhook.trustedJobControllerIdentity` | `""` | The full username the cluster's Job controller uses when creating a Job's pods, commonly `system:serviceaccount:kube-system:job-controller`. When set, a pod claiming to be a postprocess pod is also checked against it. Verify the value on your cluster first. A wrong value misclassifies real postprocess pods, not only spoofed ones. When empty, only the weaker owner-reference check applies. |
| `networkPolicy.enabled` | `true` | Creates a NetworkPolicy allowing the API server to reach the webhook's port 8443. Set `false` if your cluster manages this itself. See [Troubleshooting](../user-guide/troubleshooting.md). |
| `certManager.issuerName` | `aibom-webhook-selfsigned` | The cert-manager issuer for the webhook's certificate. |
| `certManager.duration` / `.renewBefore` | `8760h` / `720h` | Certificate lifetime and how long before expiry it is renewed. |

### Collection

| Value | Default | Description |
|-------|---------|-------------|
| `monitoringAccess.enabled` | `true` | Binds the `aibom-postprocess` ServiceAccount, in this chart's namespace, to `cluster-monitoring-view` on OpenShift so it can read Prometheus. Needs cluster-scoped permission. When `false`, the install succeeds but telemetry is empty until a cluster-admin applies the binding. |
| `watcher.enabled` | `true` | Runs the Job and pod completion watcher. Without it no AIBOMs are compiled. |
| `prometheus.url` | unset | Where the postprocess Job queries telemetry. Unset, it is `https://thanos-querier.openshift-monitoring.svc:9091` on OpenShift and empty elsewhere. Empty disables telemetry. Authentication uses the Job's ServiceAccount token. |
| `gitProvenance.registryLookup` | `true` | When a workload's image has no `image.openshift.io` `Image` object, which is always the case outside OpenShift, the postprocess Job reads its labels from the image's registry for git provenance. Pulls are anonymous, so private images are skipped. Needs egress from workload namespaces to the registry. Set `false` to stop these requests. |
| `build.enabled` | `false` | Builds the webhook and postprocess images in-cluster from source with an OpenShift BuildConfig, for clusters with no route to Quay. `just deploy-buildconfig` sets it. Rendering fails when it is set on a cluster other than OpenShift. |
| `build.gitRepo` / `build.gitRef` | `https://github.com/gavinsan33/aibom-webhook-service.git` / `main` | What the in-cluster build clones. |
| `debug.keepPostprocessJobs` | `false` | Keeps finished postprocess Jobs and their input ConfigMaps, in the webhook's namespace, for inspection. Leaves one of each per workload indefinitely. Not for production. |
| `debug.telemetryAllPods` | `false` | Queries telemetry for every pod even with no detected GPU, for testing on clusters without GPUs. Not for production. |

## `aibom-workload-namespace` chart

| Value | Default | Description |
|-------|---------|-------------|
| `platform` | `auto` | As in the webhook chart. On OpenShift this chart also creates the image-puller RoleBinding. |
| `aibomSystemNamespace` | `project-aibom` | The namespace the webhook chart was installed into. The Roles that let its watcher and the `aibom-postprocess` ServiceAccount act in this namespace are bound to ServiceAccounts there, and on OpenShift the image-puller RoleBinding is created there. |
| `vllmMetrics.enabled` | `true` | Adds a `PodMonitor` scraping vLLM's `/metrics` on port 8000 for instrumented pods. Needs the Prometheus Operator CRDs, or the install fails. Set `false` without them. Inference metrics also need the webhook chart's `monitoringAccess.enabled`. |
| `scripts.generateSnapshot`, `.runtimeDetector`, `.k8sApi`, `.datasetSidecar` | `""` | Only for installing from a checkout, where `--set-file` reads the scripts from `scripts/aibom-scripts/`. Leave unset for an OCI install, which embeds them. |

## Server flags

The webhook binary takes these flags. The chart sets them from the values above, so don't pass them directly when installing with the chart.

| Flag | Default | Description |
|------|---------|-------------|
| `--tls-cert` | `/certs/tls.crt` | Path to the TLS certificate |
| `--tls-key` | `/certs/tls.key` | Path to the TLS private key |
| `--port` | `8443` | Server port |
| `--discovery-image` | `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime` | Image for the discovery init container. The chart always sets it from `image.discovery`. |
| `--dataset-detection` | `true` | Inject dataset detection hooks into application containers |
| `--enable-watcher` | `true` | Start the Job completion watcher (`watcher.enabled`) |
| `--postprocess-image` | `busybox:latest` | Image for postprocess Jobs. A placeholder; the chart always overrides it with `image.postprocess`. |
| `--dataset-sidecar-image` | `python:3.12-slim` | Image for the dataset-signing sidecar container |
| `--trusted-watcher-identity` | empty | The webhook's own ServiceAccount username, used to check that a Job claiming to be a postprocess Job was created by the watcher. The chart sets it. When empty the check is off and a warning is logged. |
| `--trusted-job-controller-identity` | empty | See `webhook.trustedJobControllerIdentity`. |
| `--prometheus-url` | `https://thanos-querier.openshift-monitoring.svc:9091` | See `prometheus.url`. |
| `--debug-keep-postprocess-jobs`, `--debug-telemetry-all-pods` | `false` | See `debug.*`. |
| `--registry-label-lookup` | `true` | See `gitProvenance.registryLookup`. |
| `--strip-finalizers` | `false` | Removes this project's `aibom.io/log-extraction` and `aibom.io/log-extraction-pod` finalizers from every Job and Pod in the cluster, then exits. The chart's post-delete hook runs it. |
