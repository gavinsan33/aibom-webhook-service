# Getting Started

## Prerequisites

- OpenShift 4.16 or later, or Kubernetes 1.29 or later (see [Kubernetes without OpenShift](#kubernetes-without-openshift)), and `helm` 3.x.
- A cluster-admin account. The webhook chart installs cluster-scoped RBAC, including ClusterRoleBindings for the `aibom-postprocess` ServiceAccount (see [Install the webhook](#install-the-webhook)). The workload-namespace chart creates only namespaced RBAC, plus a RoleBinding in the webhook's namespace.
- To work from source: Go 1.22+, [`just`](https://github.com/casey/just) (run `make install-just` if you don't have it), and `openssl` for local TLS certificates.

### Cluster requirements

| Dependency | Needed for | Required? | Without it |
|------------|-----------|-----------|------------|
| [cert-manager](https://cert-manager.io/) | Issues and renews the webhook's TLS certificate outside OpenShift | Only off OpenShift | The webhook chart can't issue its certificate, so the API server can't call it. On OpenShift the built-in service-ca operator issues it instead; set `tls.provider=cert-manager` to use cert-manager there. |
| Pull access to the discovery image | The `aibom-discovery` init container | Yes | Instrumented pods fail to start. Fail-open does not cover this. See below. |
| Prometheus Operator CRDs (`monitoring.coreos.com/v1` `PodMonitor`) | Scraping a serving engine's `/metrics` (vLLM/llm-d, TGI, SGLang, LMDeploy) for inference performance | Yes, unless serving metrics are skipped | The namespace chart install fails outright. Skip it with `--skip-vllm-metrics`, or `--set vllmMetrics.enabled=false` with Helm. |
| A Prometheus API endpoint (`prometheus.url`) | All telemetry queries | For telemetry | Telemetry comes back empty. On OpenShift it defaults to Thanos Querier in `openshift-monitoring`. Elsewhere it is empty until you set it. |
| Read access to Prometheus for the postprocess ServiceAccount | Letting postprocess read those metrics | For telemetry, if Prometheus checks the token | Telemetry comes back empty. On OpenShift the webhook chart binds `cluster-monitoring-view`. If your account can't create it, skip it with `--set monitoringAccess.enabled=false` and have a cluster-admin create it once by hand ([Troubleshooting](troubleshooting.md)). Elsewhere bind it yourself. |
| NVIDIA GPU Operator with dcgm-exporter, pod mapping enabled | GPU utilization, memory and power series | For GPU telemetry | GPU metrics are missing. Without pod mapping the `exported_pod` label doesn't exist and the metrics are simply absent. |
| NVIDIA Container Toolkit and driver on GPU nodes | `nvidia-smi` in the `aibom-discovery` init container | For GPU hardware fields | GPU fields read "Not available". |
| OpenShift `image.openshift.io` API, or egress to the image's registry | Source commit from image labels | No | That source-detection tier is skipped. Without an `Image` object the labels are read from the registry, anonymously, so private images are skipped too. |

Telemetry is collected only for pods with a detected GPU.

Instrumented pods pull the discovery image, `quay.io/project-aibom/aibom-discovery` at the chart's version: a small UBI image (about 230 MB) with `python3`, `bash`, `lscpu` and `lsblk`. If the cluster can't reach Quay, for example because it has no egress, point the chart at a copy it can pull:

```bash
helm upgrade --install aibom-webhook oci://quay.io/project-aibom/aibom-webhook \
  -n project-aibom --create-namespace --kube-as-user=system:admin \
  --set image.discovery.repository=<registry>/<image> \
  --set image.discovery.tag=<tag>
```

Both charts detect OpenShift by its `image.openshift.io` API and choose these defaults from that (`platform: auto`). `helm template` without a cluster can't see the API and renders the non-OpenShift defaults, so when rendering offline, for example with Argo CD, set `platform=openshift` or `platform=kubernetes` explicitly.

`platform` recovers the OpenShift defaults (Thanos Querier for Prometheus, the `Image` RBAC), but **not** the discovery image: that default keys on the OpenShift AI API (`datasciencecluster.opendatahub.io`), which an offline render can't see either, and `platform` has no "OpenShift AI" value to distinguish it from plain OpenShift. So an offline render on OpenShift AI still falls through to the public `pytorch/pytorch` image and, on an egress-restricted cluster, leaves every instrumented pod in `Init:ImagePullBackOff` (see [Troubleshooting](troubleshooting.md#pods-stay-in-initimagepullbackoff)). Set `image.discovery.repository`/`image.discovery.tag` by hand for offline renders — e.g. to the in-cluster `runtime-pytorch` ImageStream (`image-registry.openshift-image-registry.svc:5000/redhat-ods-applications/runtime-pytorch`, tag `pytorch`) on OpenShift AI, or any pullable image with `python3` and `bash`.

Supported workload types are Job, JobSet, PyTorchJob and RayJob, plus any pod that requests `nvidia.com/gpu`. The Kubeflow Training Operator (PyTorchJob), JobSet and KubeRay are needed only if you run those kinds.

## Install the webhook

Both charts are published as OCI artifacts, so no checkout of this repository is needed:

```bash
helm upgrade --install aibom-webhook oci://quay.io/project-aibom/aibom-webhook \
  -n project-aibom --create-namespace --kube-as-user=system:admin
```

The chart installs cluster-scoped RBAC, which a namespace-scoped admin cannot grant themselves, so run it as a cluster-admin. The commands here use `--kube-as-user=system:admin` to do that by impersonation, which needs impersonation rights and so won't work on managed clusters such as ROSA or ARO. There, log in as a cluster-admin user and drop `--kube-as-user=system:admin` from the `helm` commands and `--as=system:admin` from the `oc` commands.

Without `--version`, Helm installs the newest release. Pass `--version=X.Y.Z` to pin one, or `--version=<version>-<git-sha>` for a dev build from `main` (see [Upgrading](../reference/upgrading.md#versions)).

## Install from the OpenShift console

To install from the console's Developer Catalog (Helm Charts) instead of the command line, a cluster-admin adds the project's Helm repository once:

```yaml
apiVersion: helm.openshift.io/v1beta1
kind: HelmChartRepository
metadata:
  name: project-aibom
spec:
  name: Project AIBOM
  connectionConfig:
    url: https://gavinsan33.github.io/aibom-webhook-service
```

Both charts then appear in the catalog, with a form built from their values. Install the webhook chart once, as a cluster-admin, then the namespace chart in each namespace that runs workloads, still after labeling the namespace with `oc label`. Upgrades are not automatic: the Helm release's Upgrade action installs a newer version, and the CRDs still need the separate apply described in [Upgrading](../reference/upgrading.md).

## Kubernetes without OpenShift

The commands on this page use `oc`. `kubectl` takes the same arguments. Other differences:

- **No impersonation.** `system:admin` is an OpenShift identity. Run as a cluster-admin and drop `--kube-as-user=system:admin` and `--as=system:admin`.
- **Install cert-manager.** It issues the webhook's certificate; OpenShift uses its built-in service CA instead.
- **Set `prometheus.url`.** There is no default outside OpenShift.
- **Prometheus access.** Only needed if Prometheus checks the bearer token. The chart binds nothing here, so bind the webhook namespace's `aibom-postprocess` ServiceAccount to a ClusterRole that grants the access your Prometheus requires. Without the Prometheus Operator, also set `vllmMetrics.enabled=false` on the namespace chart.
- **TLS.** There is no service-ca bundle, so an HTTPS Prometheus must have a certificate the system trust store accepts.

```bash
helm upgrade --install aibom-webhook oci://quay.io/project-aibom/aibom-webhook \
  -n project-aibom --create-namespace \
  --set prometheus.url=http://prometheus-operated.monitoring.svc:9090
kubectl label namespace <namespace> aibom.io/enabled=true --overwrite
helm upgrade --install aibom-ns-<namespace> oci://quay.io/project-aibom/aibom-workload-namespace \
  -n <namespace>
```

From a checkout, set `AIBOM_PLATFORM=kubernetes` and the `just` recipes use `kubectl` without impersonation, for example `AIBOM_PLATFORM=kubernetes just setup-namespace --namespace=<namespace>`. `just deploy-buildconfig` and the chart's `build.enabled` need OpenShift. The console plugin needs the OpenShift web console; `kubectl get aiboms` works anywhere.

## Enable a workload namespace

Each namespace that runs instrumented workloads needs the `aibom.io/enabled` label, on OpenShift image pull access to the webhook's namespace (`project-aibom`), the `aibom-scripts` ConfigMap, and RBAC that lets workload pods and the postprocess Job write their data through the Kubernetes API. The namespace must already exist.

```bash
oc label namespace <namespace> aibom.io/enabled=true --overwrite
helm upgrade --install aibom-ns-<namespace> oci://quay.io/project-aibom/aibom-workload-namespace \
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
