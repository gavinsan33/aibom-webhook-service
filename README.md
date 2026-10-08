# AIBOM Webhook Service

A Kubernetes mutating admission webhook that automatically instruments AI workloads with AIBOM (AI Bill of Materials) metadata collection. When a pod is created in an opted-in namespace, the webhook injects hardware discovery, dataset detection, and tracking labels. No changes to the user's manifests are required.

When the workload finishes, a postprocess Job compiles everything (hardware, datasets, model and training config, git provenance, Prometheus telemetry) into a signed, immutable `AIBOM` custom resource in the workload's namespace. To browse, filter, and compare AIBOMs, use the companion tools:

- [`oc-aibom`](https://github.com/gavinsan33/oc-aibom): a `kubectl`/`oc` plugin (`oc aibom list|describe|diff|compare`). It also verifies the AIBOM's signature.
- [`aibom-console-plugin`](https://github.com/gavinsan33/aibom-console-plugin): an OpenShift web console plugin with list, detail, and compare views and telemetry charts.

**Full documentation lives in [`docs/`](docs/index.md)** (built with MkDocs). This README only covers what the project does and how to install it.

## How It Works

1. An admin labels a namespace: `oc label namespace my-ns aibom.io/enabled=true`
2. An admin installs the namespace chart, which adds the `aibom-scripts` ConfigMap, RBAC, and signing keys
3. A user submits a Job, JobSet, PyTorchJob, or RayJob (or any pod requesting `nvidia.com/gpu`) in that namespace
4. The API server calls the webhook, which injects an `aibom-discovery` init container (hardware snapshot), dataset detection hooks, a dataset-signing sidecar, and an `aibom.io/instrumented: "true"` label. The user's YAML is untouched
5. When the Job completes (or the pod is deleted, for long-running pods like KServe predictors), the watcher creates a postprocess Job that compiles and creates the `AIBOM`

The webhook always fails open (`failurePolicy: Ignore`): if the service is down or unreachable, pods are created normally, without an AIBOM. See [Troubleshooting](docs/user-guide/troubleshooting.md).

## Prerequisites

- An OpenShift cluster with [cert-manager](https://cert-manager.io/) installed
- `helm` 3.x
- Cluster-admin (the charts install cluster-scoped RBAC)
- To build from source: Go 1.22+ and [`just`](https://github.com/casey/just) (`make install-just`)

## Setup

Both charts are published to Quay as OCI artifacts, so no checkout is needed.

```bash
# 1. Install the webhook
helm upgrade --install aibom-webhook oci://quay.io/gsanders/aibom-webhook \
  -n project-aibom --create-namespace --kube-as-user=system:admin

# 2. Enable a workload namespace (it must already exist)
oc label namespace <namespace> aibom.io/enabled=true --overwrite
helm upgrade --install aibom-ns-<namespace> oci://quay.io/gsanders/aibom-workload-namespace \
  -n <namespace> --kube-as-user=system:admin

# 3. Verify: submit a Job, then check the pod for the init container
oc get pod <pod-name> -n <namespace> -o jsonpath='{.spec.initContainers[*].name}'
# Should output: aibom-discovery
```

Re-run step 2 whenever the scripts change; a stale `aibom-scripts` ConfigMap can fail pod startup in the namespace. After a workload completes:

Preferred: the `oc aibom` plugin (install [`oc-aibom`](https://github.com/gavinsan33/oc-aibom) first; `aibom` is a plugin subcommand, not a built-in `oc` one), or the console plugin in the web UI:

```bash
# Requires the oc-aibom plugin
oc aibom list -n <namespace>
oc aibom describe <name> -n <namespace>
```

Fallback, with no plugin needed (standard `oc`/`kubectl` on the `AIBOM` custom resource):

```bash
oc get aiboms -n <namespace>
oc get aibom <name> -n <namespace> -o yaml
```

Without a published chart, `just deploy`, `just deploy-buildconfig`, and `just setup-namespace` do the same from a checkout; see [Development](docs/reference/development.md).

## Documentation

- [Getting Started](docs/user-guide/getting-started.md): install, enable a namespace, verify, uninstall
- [Annotations](docs/user-guide/annotations.md): optional `aibom.io/*` experiment metadata
- [Troubleshooting](docs/user-guide/troubleshooting.md)
- [Detected Fields](docs/CAPABILITIES.md): every field the pipeline can populate, and how
- [AIBOM Schema](docs/reference/schema.md): how AIBOMs are stored
- [Configuration](docs/reference/configuration.md): webhook flags and chart values
- [Development](docs/reference/development.md): building, `just` recipes, publishing, repo layout
- [`CLAUDE.md`](CLAUDE.md): implementation rationale and trust model
