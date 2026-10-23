# Getting Started

## Prerequisites

- An OpenShift cluster, and `helm` 3.x.
- A cluster-admin account. Both charts install cluster-scoped RBAC, and the workload-namespace chart creates a ClusterRoleBinding (see [Install the webhook](#install-the-webhook)).
- To work from source: Go 1.22+, [`just`](https://github.com/casey/just) (run `make install-just` if you don't have it), and `openssl` for local TLS certificates.

### Cluster requirements

| Dependency | Needed for | Required? | Without it |
|------------|-----------|-----------|------------|
| [cert-manager](https://cert-manager.io/) | Issues and renews the webhook's TLS certificate | Yes | The webhook chart can't issue its certificate, so the API server can't call it. |
| OpenShift AI (the `runtime-pytorch` ImageStream in `redhat-ods-applications`) | The default `aibom-discovery` init container image | Yes, unless you set `image.discovery` | The init container can't pull its image, so instrumented pods fail to start. Fail-open does not cover this. See below. |
| Prometheus Operator CRDs (`monitoring.coreos.com/v1` `PodMonitor`) | Scraping vLLM's `/metrics` for inference performance | Yes, unless vLLM metrics are skipped | The namespace chart install fails outright. Skip it with `--skip-vllm-metrics`, or `--set vllmMetrics.enabled=false` with Helm. |
| OpenShift monitoring (Thanos Querier in `openshift-monitoring`) | All telemetry queries | For telemetry | Telemetry comes back empty. |
| `cluster-monitoring-view` binding for the postprocess ServiceAccount | Letting postprocess read those metrics | For telemetry | Telemetry comes back empty for that namespace. Skip it with `--skip-monitoring-access`, or `--set monitoringAccess.enabled=false` with Helm, and have a cluster-admin apply the binding once. |
| NVIDIA GPU Operator with dcgm-exporter, pod mapping enabled | GPU utilization, memory and power series | For GPU telemetry | GPU metrics are missing. Without pod mapping the `exported_pod` label doesn't exist and the metrics are simply absent. |
| NVIDIA Container Toolkit and driver on GPU nodes | `nvidia-smi` in the `aibom-discovery` init container | For GPU hardware fields | GPU fields read "Not available". |
| OpenShift `image.openshift.io` API | Source commit from image labels | No | That source-detection tier is skipped. |

Telemetry is collected only for pods with a detected GPU.

The discovery image needs only `python3` and `bash`. On a cluster without OpenShift AI, point the chart at any image the cluster can pull:

```bash
helm upgrade --install aibom-webhook oci://quay.io/gsanders/aibom-webhook \
  -n project-aibom --create-namespace --kube-as-user=system:admin \
  --set image.discovery.repository=pytorch/pytorch \
  --set image.discovery.tag=2.2.0-cuda12.1-cudnn8-runtime
```

Supported workload types are Job, JobSet, PyTorchJob and RayJob, plus any pod that requests `nvidia.com/gpu`. The Kubeflow Training Operator (PyTorchJob), JobSet and KubeRay are needed only if you run those kinds.

## Install the webhook

Both charts are published as OCI artifacts, so no checkout of this repository is needed:

```bash
helm upgrade --install aibom-webhook oci://quay.io/gsanders/aibom-webhook \
  -n project-aibom --create-namespace --kube-as-user=system:admin
```

The chart installs cluster-scoped RBAC, which a namespace-scoped admin cannot grant themselves, so run it as a cluster-admin. The commands here use `--kube-as-user=system:admin` to do that by impersonation, which needs impersonation rights and so won't work on managed clusters such as ROSA or ARO. There, log in as a cluster-admin user and drop `--kube-as-user=system:admin` from the `helm` commands and `--as=system:admin` from the `oc` commands.

Without `--version`, Helm resolves to the most recently published mutable tag. To pin an exact commit, pass `--version=<version>-<git-sha>` (for example `--version=0.1.0-abc1234`).

## Enable a workload namespace

Each namespace that runs instrumented workloads needs the `aibom.io/enabled` label, image pull access to the webhook's namespace (`project-aibom`), the `aibom-scripts` ConfigMap, and RBAC that lets workload pods and the postprocess Job write their data through the Kubernetes API. The namespace must already exist.

```bash
oc label namespace <namespace> aibom.io/enabled=true --overwrite
helm upgrade --install aibom-ns-<namespace> oci://quay.io/gsanders/aibom-workload-namespace \
  -n <namespace> --kube-as-user=system:admin
```

From a checkout, `just setup-namespace --namespace=<namespace>` does the same.

!!! note "Upgrading a namespace"
    Re-run the install whenever `scripts/aibom-scripts/*.py` changes. A stale `aibom-scripts` ConfigMap can fail pod startup for every instrumented workload in the namespace.

## Verify

Submit a Job in the namespace, then check the pod:

```bash
oc get pod <pod-name> -n <namespace> -o jsonpath='{.spec.initContainers[*].name}'
# Should output: aibom-discovery
```

When the Job completes, the AIBOM appears as a custom resource:

Preferred: the `oc aibom` plugin from [`oc-aibom`](https://github.com/gavinsan33/oc-aibom) (a plugin subcommand, not built into `oc`), or the console plugin in the web UI:

```bash
# Requires the oc-aibom plugin
oc aibom list -n <namespace>
```

Fallback, with no plugin needed:

```bash
oc get aiboms -n <namespace>
```

## Remove

Uninstall in reverse order, namespace first and then the webhook:

```bash
oc label namespace <namespace> aibom.io/enabled- --as=system:admin
helm uninstall aibom-ns-<namespace> -n <namespace> --kube-as-user=system:admin
helm uninstall aibom-webhook -n project-aibom --kube-as-user=system:admin
```

The `aiboms.aibom.io` CRD and any existing AIBOM resources are left in place.

## Working from source

```bash
just build                       # build
just test                        # all tests (Go + Python)
./scripts/generate-certs.sh      # self-signed TLS certs for local dev
just run                         # run locally
```

`just` with no arguments lists every recipe, including the `deploy*` recipes for installing from a checkout.
