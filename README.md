# AIBOM Webhook Service

A Kubernetes mutating admission webhook that automatically instruments AI workloads with AIBOM (AI Bill of Materials) metadata collection. When a pod is created in an opted-in namespace, the webhook injects hardware discovery, dataset detection, and tracking labels — no changes to the user's manifests required.

For filtering, inspecting, and comparing the resulting `AIBOM` custom resources, see the [`oc-aibom`](https://github.com/gavinsan33/oc-aibom) `kubectl`/`oc` plugin.

## How It Works

1. An admin labels a namespace: `oc label namespace my-ns aibom.io/enabled=true`
2. An admin creates the `aibom-scripts` ConfigMap in that namespace (see [Setup](#workload-namespace-setup))
3. A user submits a Job, JobSet, PyTorchJob, or RayJob in that namespace
4. The Kubernetes API server calls the webhook before creating the pod
5. The webhook injects an `aibom-discovery` init container (hardware snapshot), dataset detection hooks (`usercustomize.py`), and an `aibom.io/instrumented: "true"` label — see [What Gets Injected](#what-gets-injected)
6. The pod is created with the injections — the user's original YAML is untouched
7. When the Job completes (or is deleted, for long-running pods like KServe predictors), the **watcher** creates a postprocess Job to compile the AIBOM — see [Postprocess Flow](#postprocess-flow)

Pods are matched if they are owned by a Job, JobSet, PyTorchJob, or RayJob, **or** if any container requests `nvidia.com/gpu` resources. The webhook always fails open (`failurePolicy: Ignore`) — if the service is down, pods are created normally.

For the full rules on which Jobs/pods get postprocessed, how JobSet siblings are merged, and how model/dataset config is auto-detected, see `CLAUDE.md`.

## Prerequisites

- Go 1.22+
- An OpenShift cluster (for deployment)
- `helm` 3.x
- [`just`](https://github.com/casey/just) (task runner — everything below runs through it). Don't have it? Run `make install-just` (uses `brew`/`cargo` if available, otherwise the official install script). Then run `just` with no arguments to list all recipes.
- [cert-manager](https://cert-manager.io/) installed in the cluster (for deployment — issues and auto-renews the webhook's TLS certificate; see `charts/aibom-webhook/templates/certificates.yaml`)
- `openssl` (for local-dev TLS cert generation only — see `scripts/generate-certs.sh`)

## Quick Start

```bash
# Build
just build

# Run all tests (Go + Python)
just test

# Or run just one suite
just test go
just test python   # postprocess.py, runtime_detector.py

# Generate self-signed TLS certs for local dev
./scripts/generate-certs.sh

# Run locally
just run
```

## Workload Namespace Setup

Each namespace that runs instrumented workloads needs the `aibom.io/enabled` label, image pull access to `aibom-system`, the `aibom-scripts` ConfigMap, and RBAC letting workload pods and the postprocess Job write their own data directly via the Kubernetes API. The namespace itself must already exist; a single command handles the rest:

```bash
just setup-namespace my-ai-workloads
```

`just setup-namespace`:
1. Labels the namespace `aibom.io/enabled=true` — opts it into webhook instrumentation
2. Runs `helm upgrade --install aibom-ns-<namespace> charts/aibom-workload-namespace -n <namespace>`, which creates the image-puller RoleBinding, the `aibom-scripts` ConfigMap, the `aibom-postprocess` ServiceAccount/RBAC, and the `aibom-workload-data` RBAC (see `charts/aibom-workload-namespace/templates/`)

**Upgrading an existing namespace**: re-run `just setup-namespace <ns>` any time `scripts/aibom-scripts/*.py` changes — `helm upgrade --install` is idempotent. A stale `aibom-scripts` ConfigMap can fail *pod startup* for every instrumented workload in the namespace (not just silently skip dataset detection), since the dataset detector hook mounts `k8s_api.py` via a `subPath` volume mount.

## Cluster Deployment

Deployment is a Helm chart (`charts/aibom-webhook`), covering the `aibom-system` namespace, RBAC, cert-manager `Issuer`/`Certificate`, the webhook `Deployment`/`Service`, the `MutatingWebhookConfiguration`, and the OpenShift `BuildConfig`/`ImageStream` pair used to build both images in-cluster from source. `just deploy` requires cert-manager to already be installed — it issues the webhook's TLS certificate and keeps it renewed automatically.

`just deploy` always installs/upgrades the chart, builds both images in-cluster from source, and rolls out the result. `--version` controls what gets built and deployed, defaulting to the short SHA of the remote tip of `build.gitRepo`/`build.gitRef` (not your local `HEAD`). Each build lands on its own ImageStreamTag, so `just deploy --version=<older-sha>` rebuilds and redeploys that exact historical commit as a rollback. See `CLAUDE.md` for the reasoning behind this versioning scheme.

If your account doesn't have cluster-scoped permission to create/patch CRDs, pass `--skip-crds`; the `aiboms.aibom.io` CRD and `aibom-system` namespace must then already exist (created once via `oc apply -f charts/aibom-webhook/crds/aibom-crd.yaml` and `oc create namespace aibom-system`).

```bash
# Build both images in-cluster from source and roll out the result
just deploy

# Or, overwrite the "latest" tag in place instead of using the resolved commit SHA
just deploy --version=latest

# Roll back: rebuilds and redeploys that exact historical commit
just deploy --version=<older-sha>

# Or, to use externally built/pushed images instead of the in-cluster BuildConfig:
helm upgrade --install aibom-webhook charts/aibom-webhook \
  --set build.enabled=false \
  --set image.webhook.repository=quay.io/<your-org>/aibom-webhook-service \
  --set image.webhook.tag=latest \
  --set image.postprocess.repository=quay.io/<your-org>/aibom-postprocess \
  --set image.postprocess.tag=latest

# Set up a workload namespace (label, image pull access, scripts ConfigMap)
just setup-namespace my-ai-workloads

# Verify: submit a Job, check the pod for the init container
oc get pod <pod-name> -n my-ai-workloads -o jsonpath='{.spec.initContainers[*].name}'
# Should output: aibom-discovery

# Check dataset detector env vars
oc get pod <pod-name> -n my-ai-workloads -o jsonpath='{.spec.containers[0].env[*].name}'
# Should include: AIBOM_DATASET_DETECT AIBOM_DEBUG AIBOM_DATASET_OUTPUT PYTHONPATH
```

To remove the deployment: `just undeploy` (runs `helm uninstall aibom-webhook`, after a confirmation prompt). Helm installs CRDs once but never upgrades or removes them automatically — schema changes to `aiboms.aibom.io` need a manual `oc apply -f charts/aibom-webhook/crds/aibom-crd.yaml`, and `helm uninstall` leaves the CRD (and any AIBOM custom resources) in place.

## Local Testing (without a cluster)

```bash
# Start the server
just run

# In another terminal, send a test admission review
curl -sk -X POST https://localhost:8443/mutate \
  -H "Content-Type: application/json" \
  -d '{
    "apiVersion": "admission.k8s.io/v1",
    "kind": "AdmissionReview",
    "request": {
      "uid": "test",
      "resource": {"group": "", "version": "v1", "resource": "pods"},
      "object": {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
          "name": "test-pod",
          "namespace": "default",
          "ownerReferences": [{"kind": "Job", "name": "my-job", "apiVersion": "batch/v1", "uid": "abc"}]
        },
        "spec": {
          "containers": [{"name": "train", "image": "pytorch:latest"}]
        }
      }
    }
  }'

# Health check
curl -sk https://localhost:8443/healthz
```

## Project Structure

```
cmd/webhook/main.go                # Entrypoint: TLS, HTTP server, watcher, graceful shutdown
internal/
  webhook/
    handler.go                      # AdmissionReview HTTP handler
    mutator.go                      # Pod matching + JSON patch construction
    handler_test.go                 # Unit tests
  watcher/
    watcher.go                      # Job completion watcher + postprocess Job creation
    watcher_test.go                 # Unit tests
  config/config.go                  # Configuration struct
  aibomdata/aibomdata.go            # Shared postprocess Job/ConfigMap naming convention
postprocess/
  postprocess.py                    # AIBOM compiler; creates the AIBOM CR directly (runs in postprocess Job)
  Dockerfile                        # Postprocess container image; also COPYs in scripts/aibom-scripts/k8s_api.py
charts/
  aibom-webhook/                    # Cluster-level install: namespace, CRD, RBAC, certs, Deployment/Service,
    crds/aibom-crd.yaml               # webhook config, OpenShift BuildConfig/ImageStream (just deploy/undeploy)
    templates/
      serviceaccount.yaml
      clusterrole.yaml
      clusterrolebinding.yaml
      certificates.yaml             # cert-manager Issuer + Certificate
      deployment.yaml                # Deployment + Service
      webhook-configuration.yaml
      build.yaml                    # OpenShift BuildConfig + ImageStream
  aibom-workload-namespace/         # Per-namespace install: RBAC + scripts ConfigMap (just setup-namespace)
    templates/
      serviceaccount.yaml
      rbac.yaml
      scripts-configmap.yaml
scripts/
  generate-certs.sh                 # Self-signed TLS cert generation for local dev only (cluster deploy uses cert-manager)
  remote-build-sha.sh                # Resolves the short SHA `just deploy`'s --version defaults to (justfile helper)
  aibom-scripts/
    generate_snapshot.py             # Hardware discovery script (from coldpress)
    runtime_detector.py               # Dataset detection + training runtime hooks (from coldpress)
    k8s_api.py                       # Stdlib-only in-cluster REST client shared by both scripts
examples/
  vllm-inference.yaml               # Example JobSet: vLLM server + guidellm benchmark
  vllm-inference-rhoai.yaml         # Same model via a RHOAI/KServe InferenceService
  granite-lora-finetune.yaml        # Example Job: single-GPU LoRA fine-tuning via trl sft
  granite-lora-finetune-multigpu.yaml  # Same, but 2 GPUs via trl's --num_processes passthrough
  granite-lora-finetune-raw-trainer.yaml  # Same, but via raw transformers.Trainer + peft.LoraConfig (no CLI at all)
tests/
  postprocess/test_postprocess.py   # Unit tests for postprocess.py's CLI-arg detectors and compile_aibom
  aibom_scripts/                    # Unit tests for runtime_detector.py's hooks (fake torch/datasets/transformers/peft modules)
pyproject.toml                      # pytest config (pythonpath into postprocess/ and scripts/aibom-scripts/)
requirements-dev.txt                # Test-only deps (pytest, pyyaml) — production scripts stay dependency-free
Dockerfile                          # Multi-stage build (distroless)
justfile                            # Build, test, deploy recipes (just test, just test go, just test python)
Makefile                            # Bootstrap only: `make install-just` installs the just task runner
```

## Configuration

The webhook server accepts these flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--tls-cert` | `/certs/tls.crt` | Path to TLS certificate |
| `--tls-key` | `/certs/tls.key` | Path to TLS private key |
| `--port` | `8443` | Server port |
| `--discovery-image` | `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime` | Image for the discovery init container |
| `--dataset-detection` | `true` | Inject dataset detection hooks into application containers |
| `--enable-watcher` | `true` | Start the Job completion watcher |
| `--postprocess-image` | `busybox:latest` | Image for AIBOM postprocess Jobs (set to the aibom-postprocess image) |

## What Gets Injected

When the webhook mutates a pod, it adds:

**Init container (`aibom-discovery`):**
- Runs `generate_snapshot.py` from the `aibom-scripts` ConfigMap
- Captures: CPU model/cores/cache, GPU model/count/VRAM/CUDA version, memory, network (RDMA), storage, kernel config, cgroup limits
- Runs benchmarks: CPU compute (MFLOPS), memory bandwidth, disk I/O throughput, context switch latency
- Writes the result directly into the workload's data ConfigMap (key `discovery-<pod-name>.json`) via `k8s_api.py`, an in-cluster REST helper using only the Python stdlib

**Runtime detector (into each application container):**
- Mounts `runtime_detector.py` as `usercustomize.py` on `PYTHONPATH`, plus `k8s_api.py` alongside it
- Python auto-imports it at startup — no code changes needed
- Hooks into PyTorch DataLoader, HuggingFace `datasets.load_dataset`, torchvision datasets, and webdataset, plus `transformers.TrainingArguments`, `transformers.PreTrainedModel.from_pretrained`, and `peft.LoraConfig`
- Captures dataset name, version, split, fingerprint, license, and training args
- Writes the result into the same data ConfigMap (key `dataset-<pod-name>.json`) at process exit

See `CLAUDE.md` for the detection internals (CLI-arg parsing, runtime hooks, KServe storage-path resolution, quantization/parallelization detection).

## Postprocess Flow

When a Job completes or is being deleted (held by the finalizer), the watcher creates an AIBOM postprocess Job that reads/merges the workload's data ConfigMap, removes the holding finalizer, runs `postprocess.py` to compile and create the `AIBOM` custom resource, and cleans up the postprocess Job and ConfigMap on success.

Full rules for which Jobs/pods qualify, how JobSet siblings are merged, model/dataset auto-detection, and Grafana telemetry retries are documented in `CLAUDE.md`.

### AIBOM Annotations

Users can optionally annotate their Jobs with `aibom.io/*` keys to provide experiment metadata:

| Annotation | AIBOM Field |
|------------|-------------|
| `aibom.io/experiment-intent` | `experiment_intent` (training, sft, inference) |
| `aibom.io/experiment-name` | `experiment_name` |
| `aibom.io/model-name` | `model.name` |
| `aibom.io/model-framework` | `model.framework` |
| `aibom.io/dataset-name` | `dataset.declared.name` |
| `aibom.io/dataset-source` | `dataset.declared.source` |
| `aibom.io/dataset-version` | `dataset.declared.version` |
| `aibom.io/dataset-license` | `dataset.declared.license` |
| `aibom.io/optimizer` | `training.optimizer` |
| `aibom.io/batch-size` | `training.batch_size` |
| `aibom.io/epochs` | `training.epochs` |
| `aibom.io/learning-rate` | `training.learning_rate` |
| `aibom.io/top-k` | `inference.top_k` |

Without annotations, the AIBOM is still generated from auto-detected data (hardware discovery, dataset detection, telemetry). Auto-detected values are used as defaults; any corresponding annotation always overrides them.

### Grafana Credentials

To enable telemetry collection, create a secret in each instrumented namespace:

```bash
oc create secret generic aibom-config \
  --from-literal=grafana-url=https://grafana.example.com \
  --from-literal=grafana-api-token=<token> \
  -n my-ai-workloads
```

### AIBOM Storage

Completed AIBOMs are stored as namespaced `AIBOM` custom resources (`aiboms.aibom.io`, `charts/aibom-webhook/crds/aibom-crd.yaml`) — one per completed workload, created in the same namespace the workload ran in. `just deploy` registers the CRD; if your account lacks CRD permissions, use `just deploy --skip-crds` instead and have a cluster-admin apply it once via `oc apply -f charts/aibom-webhook/crds/aibom-crd.yaml`.

Because `AIBOM` is a namespaced resource, it inherits ordinary Kubernetes RBAC: a user granted `get`/`list` on `aiboms` in namespace `team-a` cannot see `team-b`'s AIBOMs.

`spec` is immutable once created — the CRD rejects any `UPDATE` that changes it, even from a user holding `update`/`patch` RBAC on `aiboms.aibom.io`, so a compiled AIBOM can't be silently altered after the fact. This doesn't prevent deletion; that's still governed by ordinary `delete` RBAC on `aiboms.aibom.io`, same as any other namespaced resource.

```bash
# List AIBOMs in a namespace (only visible to users with RBAC on aiboms.aibom.io there)
oc get aiboms -n gavin-test

# Inspect one, including the full compiled AIBOM under spec.data
oc get aibom train-job-abc123 -n gavin-test -o yaml
```

`spec.jobName`, `spec.modelName`, `spec.experimentIntent`, and `spec.collectedAt` are pulled out as printer-friendly summary fields; `spec.data` holds the complete AIBOM JSON exactly as `postprocess.py` produced it.
