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

Pods are matched if they are owned by a Job, JobSet, PyTorchJob, or RayJob, **or** if any container requests `nvidia.com/gpu` resources. The webhook always fails open (`failurePolicy: Ignore`) — if the service is down, pods are created normally.

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

The `aibom-scripts` ConfigMap must exist in each namespace where instrumented workloads run. It contains the discovery script and dataset detector that get mounted into pods.

```bash
# Create the ConfigMap in a workload namespace
make create-scripts-configmap NAMESPACE=my-ai-workloads

# Or manually:
oc create configmap aibom-scripts \
  --from-file=generate_snapshot.py=scripts/aibom-scripts/generate_snapshot.py \
  --from-file=dataset_detector.py=scripts/aibom-scripts/dataset_detector.py \
  -n my-ai-workloads
```

This ConfigMap is **not** created in `aibom-system` — pods can only mount ConfigMaps from their own namespace.

## Cluster Deployment

```bash
# Build and push the container image
make docker-build IMG=quay.io/<your-org>/aibom-webhook-service:latest
make docker-push IMG=quay.io/<your-org>/aibom-webhook-service:latest

# Update the image in deploy/deployment.yaml, then deploy
make deploy

# Opt in a namespace and create the scripts ConfigMap
oc label namespace my-ai-workloads aibom.io/enabled=true
make create-scripts-configmap NAMESPACE=my-ai-workloads

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
cmd/webhook/main.go                # Entrypoint: TLS, HTTP server, graceful shutdown
internal/
  webhook/
    handler.go                      # AdmissionReview HTTP handler
    mutator.go                      # Pod matching + JSON patch construction
    handler_test.go                 # Unit tests
  config/config.go                  # Configuration struct
deploy/
  namespace.yaml                    # aibom-system namespace
  rbac.yaml                         # ServiceAccount, ClusterRole, ClusterRoleBinding
  deployment.yaml                   # Deployment + Service
  webhook-config.yaml               # MutatingWebhookConfiguration
  aibom-scripts-configmap.yaml      # Reference manifest for the scripts ConfigMap
scripts/
  generate-certs.sh                 # Self-signed TLS cert generation
  aibom-scripts/
    generate_snapshot.py             # Hardware discovery script (from coldpress)
    dataset_detector.py              # Dataset detection hooks (from coldpress)
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

## Roadmap

- **Phase 1**: Webhook with placeholder discovery init container
- **Phase 2** (current): Real hardware discovery + dataset detector injection
- **Phase 3**: Job watcher that detects workload completion and creates postprocess Jobs for AIBOM compilation
- **Phase 4**: Production hardening (cert-manager TLS, Helm chart, metrics endpoint)