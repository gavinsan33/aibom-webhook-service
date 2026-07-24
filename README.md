# AIBOM Webhook Service

A Kubernetes mutating admission webhook that automatically instruments AI workloads with AIBOM (AI Bill of Materials) metadata collection. When a pod is created in an opted-in namespace, the webhook injects hardware discovery, dataset detection, and tracking labels — no changes to the user's manifests required.

## How It Works

1. An admin labels a namespace: `oc label namespace my-ns aibom.io/enabled=true`
2. An admin creates the `aibom-scripts` ConfigMap in that namespace (see [Setup](#workload-namespace-setup))
3. A user submits a Job, JobSet, PyTorchJob, or RayJob in that namespace
4. The Kubernetes API server calls the webhook before creating the pod
5. The webhook injects:
   - An **init container** (`aibom-discovery`) that runs a hardware snapshot script capturing CPU, GPU, memory, network, storage info, and performance benchmarks
   - **Dataset detection hooks** (`usercustomize.py`) that automatically capture which ML datasets are loaded at runtime (PyTorch, HuggingFace, torchvision, webdataset)
   - An **emptyDir volume** (`aibom-data`) for discovery and detection output
   - A label `aibom.io/instrumented: "true"` to prevent double-injection
6. The pod is created with the injections — the user's original YAML is untouched
7. When the Job completes (or is being deleted in a JobSet workflow), the **watcher** detects it and creates a postprocess Job to compile the AIBOM

Pods are matched if they are owned by a Job, JobSet, PyTorchJob, or RayJob, **or** if any container requests `nvidia.com/gpu` resources. When GPU resources are present, the webhook copies the GPU resource request to the discovery init container so `nvidia-smi` can detect the hardware. The webhook always fails open (`failurePolicy: Ignore`) — if the service is down, pods are created normally.

Postprocessing is triggered for Jobs whose pods request GPUs or whose Job has `aibom.io/*` annotations. For JobSet workflows where a server pod is killed rather than completing (e.g., vLLM + client benchmarks), the watcher adds a Kubernetes **finalizer** (`aibom.io/log-extraction`) to hold the Job alive until logs are extracted and the postprocess Job is created.

## Prerequisites

- Go 1.22+
- An OpenShift cluster (for deployment)
- `openssl` (for TLS cert generation)

## Quick Start

```bash
# Build
make build

# Run tests
make test

# Generate self-signed TLS certs (for local dev, adds localhost to SAN)
./scripts/generate-certs.sh --local

# Run locally
make run
```

## Workload Namespace Setup

Each namespace that runs instrumented workloads needs three things: the `aibom.io/enabled` label, image pull access to `aibom-system`, and the `aibom-scripts` ConfigMap. A single command handles all of it:

```bash
make setup-namespace NAMESPACE=my-ai-workloads
```

This runs:
1. `oc label namespace ... aibom.io/enabled=true` — opts the namespace into webhook instrumentation
2. `oc policy add-role-to-group system:image-puller ...` — allows pods to pull the postprocess image from `aibom-system`
3. Creates the `aibom-scripts` ConfigMap with the discovery and dataset detector scripts

## Cluster Deployment

```bash
# Build and push the container image
make docker-build IMG=quay.io/<your-org>/aibom-webhook-service:latest
make docker-push IMG=quay.io/<your-org>/aibom-webhook-service:latest

# Update the image in deploy/deployment.yaml, then deploy
make deploy

# Set up a workload namespace (label, image pull access, scripts ConfigMap)
make setup-namespace NAMESPACE=my-ai-workloads

# Verify: submit a Job, check the pod for the init container
oc get pod <pod-name> -n my-ai-workloads -o jsonpath='{.spec.initContainers[*].name}'
# Should output: aibom-discovery

# Check dataset detector env vars
oc get pod <pod-name> -n my-ai-workloads -o jsonpath='{.spec.containers[0].env[*].name}'
# Should include: AIBOM_DATASET_DETECT AIBOM_DEBUG AIBOM_DATASET_OUTPUT PYTHONPATH
```

## Local Testing (without a cluster)

```bash
# Start the server
make run

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
postprocess/
  postprocess.py                    # AIBOM compiler (runs in postprocess Job)
  Dockerfile                        # Postprocess container image
deploy/
  namespace.yaml                    # aibom-system namespace
  rbac.yaml                         # ServiceAccount, ClusterRole, ClusterRoleBinding
  deployment.yaml                   # Deployment + Service
  build.yaml                        # OpenShift BuildConfig + ImageStream
  webhook-config.yaml               # MutatingWebhookConfiguration
  aibom-scripts-configmap.yaml      # Reference manifest for the scripts ConfigMap
scripts/
  generate-certs.sh                 # Self-signed TLS cert generation
  aibom-scripts/
    generate_snapshot.py             # Hardware discovery script (from coldpress)
    dataset_detector.py              # Dataset detection hooks (from coldpress)
examples/
  vllm-inference.yaml               # Example JobSet: vLLM server + guidellm benchmark
Dockerfile                          # Multi-stage build (distroless)
Makefile                            # Build, test, deploy targets
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
- Writes `discovery.json` to the `aibom-data` volume

**Dataset detector (into each application container):**
- Mounts `dataset_detector.py` as `usercustomize.py` on `PYTHONPATH`
- Python auto-imports it at startup — no code changes needed
- Hooks into PyTorch DataLoader, HuggingFace `datasets.load_dataset`, torchvision datasets, and webdataset
- Captures dataset name, version, split, fingerprint, license, and training args
- Writes `dataset_detected.json` to the `aibom-data` volume at process exit

## Postprocess Flow

When a Job completes or is being deleted (held by the finalizer), the watcher creates an AIBOM postprocess Job:

1. **Data extraction**: The watcher reads pod logs from the Job's pods (and sibling pods in a JobSet), extracting discovery JSON (from the `aibom-discovery` init container) and dataset JSON (from application containers) via delimited markers
2. **ConfigMap creation**: Extracted data plus `aibom.io/*` annotations are stored in a ConfigMap (`{job-name}-aibom-postprocess-data`) in the workload namespace
3. **Finalizer removal**: If the Job has the `aibom.io/log-extraction` finalizer, it is removed after data extraction, allowing Kubernetes to complete the deletion
4. **Postprocess Job**: A Job is created running `postprocess.py`, which:
   - Loads discovery and dataset data from the ConfigMap mount
   - Optionally queries Grafana/Prometheus for telemetry (GPU utilization, memory, power, CPU, network)
   - Compiles everything into an AIBOM JSON document
   - Outputs the AIBOM to stdout (readable via `kubectl logs`)

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
| `aibom.io/optimizer` | `training.optimizer` |
| `aibom.io/batch-size` | `training.batch_size` |
| `aibom.io/epochs` | `training.epochs` |
| `aibom.io/learning-rate` | `training.learning_rate` |

Without annotations, the AIBOM is still generated from auto-detected data (hardware discovery, dataset detection, telemetry).

### Grafana Credentials

To enable telemetry collection, create a secret in each instrumented namespace:

```bash
oc create secret generic aibom-config \
  --from-literal=grafana-url=https://grafana.example.com \
  --from-literal=grafana-api-token=<token> \
  -n my-ai-workloads
```

## Example: vLLM Inference Benchmark

The `examples/vllm-inference.yaml` file shows a JobSet with a vLLM server and a guidellm benchmark client. The server has `aibom.io/*` annotations and GPU resources; the client depends on the server being ready. When the client finishes, the JobSet kills the server — but the finalizer holds it until the watcher extracts discovery logs and creates the postprocess Job.

```bash
# Deploy the example (namespace must be set up first)
oc apply -f examples/vllm-inference.yaml

# Watch progress
oc get pods -n gavin-test -w

# View the AIBOM after postprocessing completes
oc logs -n gavin-test job/aibom-vllm-benchmark-server-0-aibom-postprocess
```

## Roadmap

- **Phase 1** (complete): Webhook with placeholder discovery init container
- **Phase 2** (complete): Real hardware discovery + dataset detector injection
- **Phase 3** (complete): Job watcher + real postprocess container for AIBOM compilation
- **Phase 4**: Production hardening (cert-manager TLS, Helm chart, metrics endpoint, AIBOM storage)