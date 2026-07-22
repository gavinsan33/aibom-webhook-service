# AIBOM Webhook Service

A Kubernetes mutating admission webhook that automatically instruments AI workloads with [AIBOM](https://github.com/gavinsan33/coldpress) (AI Bill of Materials) metadata collection. When a pod is created in an opted-in namespace, the webhook injects a hardware discovery init container and stamps tracking labels — no changes to the user's manifests required.

## How It Works

1. An admin labels a namespace: `kubectl label namespace my-ns aibom.io/enabled=true`
2. A user submits a Job, JobSet, PyTorchJob, or RayJob in that namespace
3. The Kubernetes API server calls the webhook before creating the pod
4. The webhook injects:
   - An **init container** (`aibom-discovery`) that captures hardware and pod metadata via Downward API env vars
   - An **emptyDir volume** (`aibom-data`) for discovery output
   - A label `aibom.io/instrumented: "true"` to prevent double-injection
5. The pod is created with the injections — the user's original YAML is untouched

Pods are matched if they are owned by a Job, JobSet, PyTorchJob, or RayJob, **or** if any container requests `nvidia.com/gpu` resources. The webhook always fails open (`failurePolicy: Ignore`) — if the service is down, pods are created normally.

## Prerequisites

- Go 1.22+
- A Kubernetes or OpenShift cluster (for deployment)
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

## Cluster Deployment

```bash
# Build and push the container image
make docker-build IMG=quay.io/<your-org>/aibom-webhook-service:latest
make docker-push IMG=quay.io/<your-org>/aibom-webhook-service:latest

# Update the image in deploy/deployment.yaml, then deploy
make deploy

# Opt in a namespace
kubectl label namespace my-ai-workloads aibom.io/enabled=true

# Verify: submit a Job, check the pod for the init container
kubectl get pod <pod-name> -n my-ai-workloads -o jsonpath='{.spec.initContainers[*].name}'
# Should output: aibom-discovery
```

## Project Structure

```
cmd/webhook/main.go           # Entrypoint: TLS, HTTP server, graceful shutdown
internal/
  webhook/
    handler.go                 # AdmissionReview HTTP handler
    mutator.go                 # Pod matching + JSON patch construction
    handler_test.go            # Unit tests
  config/config.go             # Configuration struct
deploy/
  namespace.yaml               # aibom-system namespace
  rbac.yaml                    # ServiceAccount, ClusterRole, ClusterRoleBinding
  deployment.yaml              # Deployment + Service
  webhook-config.yaml          # MutatingWebhookConfiguration
scripts/generate-certs.sh      # Self-signed TLS cert generation
Dockerfile                     # Multi-stage build (distroless)
Makefile                       # Build, test, deploy targets
```

## Configuration

The webhook server accepts these flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--tls-cert` | `/certs/tls.crt` | Path to TLS certificate |
| `--tls-key` | `/certs/tls.key` | Path to TLS private key |
| `--port` | `8443` | Server port |
| `--discovery-image` | `busybox:latest` | Image for the discovery init container |

## Roadmap

- **Phase 1** (current): Webhook with placeholder discovery init container
- **Phase 2**: Inject real hardware discovery script + dataset detector (`usercustomize.py` monkey-patching)
- **Phase 3**: Job watcher that detects workload completion and creates postprocess Jobs for AIBOM compilation
- **Phase 4**: Production hardening (cert-manager TLS, Helm chart, metrics endpoint)

## Related

- [coldpress](https://github.com/gavinsan33/coldpress) — the CLI tool this service extends
- [Design doc](https://github.com/gavinsan33/coldpress/blob/main/gavin-project-docs/AIBOM_WEBHOOK_SERVICE_DESIGN.md) — full architecture and design decisions
