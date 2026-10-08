# Getting Started

## Prerequisites

- An OpenShift cluster.
- `helm` 3.x.
- [cert-manager](https://cert-manager.io/) installed in the cluster. It issues and renews the webhook's TLS certificate.
- To work from source: Go 1.22+, [`just`](https://github.com/casey/just) (run `make install-just` if you don't have it), and `openssl` for local TLS certificates.

## Install the webhook

Both charts are published as OCI artifacts, so no checkout of this repository is needed:

```bash
helm upgrade --install aibom-webhook oci://quay.io/gsanders/aibom-webhook \
  -n project-aibom --create-namespace --kube-as-user=system:admin
```

`--kube-as-user=system:admin` is required because the chart installs cluster-scoped RBAC, which a namespace-scoped admin cannot grant themselves.

Without `--version`, Helm resolves to the most recently published mutable tag. To pin an exact commit, pass `--version=<version>-<git-sha>` (for example `--version=0.1.0-abc1234`).

## Enable a workload namespace

Each namespace that runs instrumented workloads needs the `aibom.io/enabled` label, image pull access to `aibom-system`, the `aibom-scripts` ConfigMap, and RBAC that lets workload pods and the postprocess Job write their data through the Kubernetes API. The namespace must already exist.

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
