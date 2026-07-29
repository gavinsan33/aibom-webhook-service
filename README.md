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
7. When the Job completes (or is being deleted in a JobSet workflow), the **watcher** detects it and creates a postprocess Job to compile the AIBOM. For long-running pods with no Job owner (e.g. KServe `InferenceService` predictor pods, which are owned by a ReplicaSet and never "complete"), the watcher instead triggers on the pod's deletion via a separate pod-level finalizer.

Pods are matched if they are owned by a Job, JobSet, PyTorchJob, or RayJob, **or** if any container requests `nvidia.com/gpu` resources. When GPU resources are present, the webhook copies the GPU resource request to the discovery init container so `nvidia-smi` can detect the hardware. The webhook always fails open (`failurePolicy: Ignore`) — if the service is down, pods are created normally.

Postprocessing is triggered for Jobs whose pods request GPUs or whose Job has `aibom.io/*` annotations. For JobSet workflows where a server pod is killed rather than completing (e.g., vLLM + client benchmarks), the watcher adds a Kubernetes **finalizer** (`aibom.io/log-extraction`) to hold the Job alive until logs are extracted and the postprocess Job is created.

Bare/ReplicaSet-owned pods (no `batch.kubernetes.io/job-name` label) that carry `aibom.io/instrumented=true` are postprocessed independently, following the same GPU/annotation qualifying criteria but read directly off the pod (since there's no owning Job/JobSet to read them from — KServe, for instance, propagates `spec.predictor.annotations` onto the pod itself). The watcher adds a distinct **pod-level finalizer** (`aibom.io/log-extraction-pod`) while the pod is running, and since these pods never "complete," postprocessing triggers purely on deletion (`DeletionTimestamp` set). There's no JobSet-style sibling merging for this path — each qualifying pod gets its own postprocess Job independently.

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
  aibom-storage-pvc.yaml            # PVC for collected AIBOM files
  aibom-storage-browser.yaml        # nginx sidecar config + Service for browsing AIBOMs
scripts/
  generate-certs.sh                 # Self-signed TLS cert generation
  aibom-scripts/
    generate_snapshot.py             # Hardware discovery script (from coldpress)
    dataset_detector.py              # Dataset detection hooks (from coldpress)
examples/
  vllm-inference.yaml               # Example JobSet: vLLM server + guidellm benchmark
  vllm-inference-rhoai.yaml         # Same model via a RHOAI/KServe InferenceService
  granite-lora-finetune.yaml        # Example Job: single-GPU LoRA fine-tuning via trl sft
  granite-lora-finetune-multigpu.yaml  # Same, but 2 GPUs via trl's --num_processes passthrough
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
| `--aibom-storage-path` | `/data/aiboms` | Path to store collected AIBOM files (empty to disable) |

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
5. **AIBOM collection**: When the postprocess Job completes, the watcher reads its logs, extracts the AIBOM JSON, and writes it to the storage PVC at `{namespace}/{job-name}_{timestamp}.json`

### Workload Selection and Grouping

**Which jobs get postprocessed?**

A job in an `aibom.io/enabled` namespace is selected for postprocessing if at least one of these is true:

- Any of its pods request `nvidia.com/gpu` resources (limits or requests > 0)
- The job has any `aibom.io/*` annotations (e.g., `aibom.io/model-name`, `aibom.io/experiment-intent`)

Internal labels like `aibom.io/instrumented` and `aibom.io/postprocess-job` are excluded from this check. Jobs that are themselves postprocess jobs (labeled `aibom.io/postprocess-for`) are always skipped.

**When does postprocessing trigger?**

- When the job **completes** (has a `JobComplete` condition), or
- When the job is **being deleted** (`DeletionTimestamp` is set) — this is the finalizer path, used for JobSet server pods that get killed rather than completing naturally

Each job is postprocessed at most once. After the postprocess job is created, the original job is annotated with `aibom.io/postprocess-job` and subsequent events are skipped.

**How are workloads grouped?**

Each qualifying job gets its own postprocess job — there is no cross-job merging. However, if a job belongs to a **JobSet** (has the `jobset.sigs.k8s.io/jobset-name` label), its postprocess job pulls in additional data from siblings:

- **Sibling pod logs**: Discovery and dataset data is extracted from all instrumented pods across the JobSet, not just the triggering job's pods
- **Sibling annotations**: If the triggering job has no `aibom.io/*` annotations, annotations are inherited from other jobs in the same JobSet

In a typical vLLM inference setup (server + client JobSet), only the server job qualifies for postprocessing (it has GPU resources and annotations). The client job has neither, so it's skipped — but the server's postprocess job still includes discovery data from client pods since they share the same JobSet.

**Which pods get postprocessed?**

Bare pods with no Job owner — no `batch.kubernetes.io/job-name` label, e.g. a KServe `InferenceService` predictor pod owned by a ReplicaSet — are postprocessed via a separate path, following the same qualifying criteria as Jobs above but read directly from the pod itself (GPU resources on the pod's containers, or `aibom.io/*` annotations on the pod — KServe propagates `spec.predictor.annotations` down onto the pod). Only pods already carrying `aibom.io/instrumented=true` (i.e., ones the webhook mutated) are candidates.

There's no "complete" state for a long-running pod, so postprocessing triggers purely on deletion (`DeletionTimestamp` set) via a distinct pod-level finalizer (`aibom.io/log-extraction-pod`), and there's no JobSet-style sibling merging — each qualifying pod gets its own postprocess Job independently.

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

Without annotations, the AIBOM is still generated from auto-detected data (hardware discovery, dataset detection, telemetry).

### Model Auto-Detection

`postprocess.py` parses each container's command/args to auto-populate model, fine-tuning, and inference fields without requiring annotations. Detection is command-based, so it also sees into `sh -c "... && trl sft ..."`-style wrapper scripts (common when a job needs to `pip install` before running its training CLI) by shell-splitting the script and scanning the resulting tokens the same way as a plain `command: [...]` list.

Two serving/training tools are currently recognized this way: **vLLM** (serving) and **trl** (fine-tuning). Other serving engines (TGI, SGLang, TensorRT-LLM) and other fine-tuning tools (Axolotl, LLaMA-Factory, raw `transformers.Trainer` scripts) aren't yet supported.

**Quantization from model name**: regex patterns match common quantization markers in the model name/path (e.g. `AWQ`, `GPTQ`, `INT4`/`INT8`, `FP4`/`FP8`, `bitsandbytes`/`NF4`, `Marlin`, `GGUF`, `AQLM`, `EXL2`, and others), extracting both the method and bit width. For example, `drawais/Granite-3.3-8B-Instruct-AWQ-INT4` detects `awq` at 4 bits.

**vLLM CLI arguments**: flags on the server's `vllm serve` / `vllm.entrypoints.openai.api_server` invocation are parsed directly:

| Flag | AIBOM Field |
|------|-------------|
| `--model` | `model.name` |
| `--dtype` | `model.dtype` |
| `--quantization` / `-q` | `model.quantization` |
| `--max-model-len` | `inference.max_model_len` |
| `--tensor-parallel-size` / `-tp` | `inference.tensor_parallel_size` |
| `--gpu-memory-utilization` | `inference.gpu_memory_utilization` |
| `--speculative-model` + `--num-speculative-tokens` (legacy), or `--speculative-config` (modern, JSON or `key=value` list) | `model.speculative_decoding` |
| `--override-generation-config` (JSON or `key=value` list) | `inference.temperature`, `inference.top_p`, `inference.top_k` |

Sampling parameters set per-request by a benchmark client (e.g. temperature passed in an HTTP request body) aren't visible here — this only sees what's baked into the server's own startup command.

**trl CLI arguments**: flags on a `trl sft`/`trl dpo`-style invocation are parsed the same way:

| Flag | AIBOM Field |
|------|-------------|
| `--model_name_or_path` | `model.name` |
| `--use_peft` (combined with presence of `--lora_r`) | `fine_tuning.adaptation_method` (`lora` or `peft`) |
| `--lora_r` | `fine_tuning.lora_rank` |
| `--lora_alpha` | `fine_tuning.lora_alpha` |
| `--learning_rate` | `training.learning_rate` |
| `--per_device_train_batch_size` | `training.batch_size` |
| `--num_train_epochs` | `training.epochs` |
| `--seed` | `training.random_seed` |

**Parallelization strategy** (`training.parallelization_strategy`): detected independently of the training tool being launched, covering three shapes:

- An explicit launcher binary: `accelerate launch --multi_gpu` → `data_parallel`; a bare `deepspeed` launcher or `--deepspeed <config>` flag → `deepspeed`; `torchrun`/`mpirun` → `data_parallel`.
- A bare `--fsdp` flag on the training command itself → `fsdp`.
- Accelerate-launch arguments passed *directly* to a CLI that spawns `accelerate launch` internally — no separate launcher token ever appears in the container's command. `trl`'s CLI supports this: `trl sft ... --num_processes 4` → `data_parallel`, and `trl sft ... --accelerate_config <name>` maps known profile names (`fsdp1`/`fsdp2` → `fsdp`, `zero1`/`zero2`/`zero3` → `deepspeed`, `multi_gpu` → `data_parallel`, `single_gpu` → none). This is the harder, easy-to-miss case — see `examples/granite-lora-finetune-multigpu.yaml`.

This only sees parallelism baked into the command (directly or via one of these launcher-passthrough conventions); in-script sharding (e.g. `device_map="auto"` set directly in Python code, common in QLoRA scripts) isn't visible to command parsing and isn't detected.

**Precedence**: auto-detected values are used as defaults; any corresponding `aibom.io/*` annotation always overrides them.

### Dataset Declaration and Reconciliation

`dataset.declared` is filled in from three sources, in order of precedence:

1. **Annotation** — `aibom.io/dataset-name` (and `dataset-version`/`dataset-source`/`dataset-license`), if set.
2. **CLI arg** — parsed from the training command's `--dataset_name`/`--dataset_config_name`/`--dataset_train_split` flags (e.g. a `trl sft --dataset_name ...` invocation), if no annotation is set.
3. **Inferred from runtime** — copied from the first `dataset.auto_detected` entry, only if neither of the above produced a name.

Whichever source wins is recorded in `dataset.declared.declared_via` (`"annotation"`, `"cli_arg"`, or `"inferred_from_runtime"`), so it's always possible to tell whether a dataset name reflects something the job author actually specified or a best-effort guess from what was observed at runtime.

Every entry in `dataset.auto_detected` also carries a `matches_declared` boolean, comparing its `dataset_name` against the final `dataset.declared.name`. This is the actual reconciliation check: it flags the case where a job declares one dataset (via annotation or CLI arg) but the code loads something different at runtime.

Within `dataset.auto_detected` itself, `dataset_detector.py` correlates hook detections that refer to the same underlying dataset object — e.g. a `datasets.load_dataset(...)` call followed by wrapping the result in a `torch.utils.data.DataLoader(...)` for batching — into a single entry (with a `seen_via` list noting every hook that touched it) rather than recording it twice.

### Grafana Credentials

To enable telemetry collection, create a secret in each instrumented namespace:

```bash
oc create secret generic aibom-config \
  --from-literal=grafana-url=https://grafana.example.com \
  --from-literal=grafana-api-token=<token> \
  -n my-ai-workloads
```

### AIBOM Storage

By default, the watcher collects completed AIBOMs and writes them to a PVC mounted at `/data/aiboms`. Files are organized as `{namespace}/{job-name}_{timestamp}.json`, preserving history across re-runs.

To set up storage:

```bash
# Create the PVC in aibom-system
oc apply -f deploy/aibom-storage-pvc.yaml

# The deployment already mounts it — just redeploy
make redeploy
```

**Browsing AIBOMs**

The `aibom-webhook` pod runs an `aibom-storage-browser` sidecar (nginx) that serves the same PVC read-only over HTTP with directory listing enabled, fronted by a ClusterIP `aibom-storage` Service (`deploy/aibom-storage-browser.yaml`). It's cluster-internal only — no public URL — so access it via port-forward:

```bash
oc port-forward -n aibom-system svc/aibom-storage 8080:80
```

Then open `http://localhost:8080` in a browser to navigate namespaces and download individual AIBOM `.json` files, or fetch one directly:

```bash
curl http://localhost:8080/gavin-test/my-job_20260727T153000Z.json
```

To disable collection entirely, set `--aibom-storage-path=""` in the deployment args.

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

## Example: vLLM via RHOAI/KServe Model Serving

The `examples/vllm-inference-rhoai.yaml` file deploys the same model as a KServe `InferenceService` — the route Red Hat OpenShift AI's Model Serving UI uses — instead of a raw JobSet. It uses a fully custom predictor container (`spec.predictor.containers`) so vLLM pulls the model directly from Hugging Face Hub, avoiding any dependency on a pre-provisioned S3/PVC model store.

KServe predictor pods are owned by a ReplicaSet, not a Job/JobSet/PyTorchJob/RayJob, so the webhook instruments them via the `requestsGPU` fallback match (`internal/webhook/mutator.go`), same as any other GPU pod. Postprocessing is handled by the watcher's pod-level finalizer path (see [Which pods get postprocessed?](#workload-selection-and-grouping)): since the predictor pod never "completes," the watcher holds it open with a distinct finalizer (`aibom.io/log-extraction-pod`) on deletion (e.g. `oc delete pod`, a rollout, or scale-down), extracts its discovery/dataset logs and `aibom.io/*` annotations (propagated onto the pod by KServe from `spec.predictor.annotations`), and creates a postprocess Job directly from that single pod — there's no JobSet to pull sibling data from here. The client Job below still doesn't itself qualify for postprocessing (no GPU resources or `aibom.io/*` annotations, and it isn't part of a JobSet to inherit any) — it only exercises the predictor endpoint.

```bash
# Deploy the example (namespace must be set up first)
oc apply -f examples/vllm-inference-rhoai.yaml
```

## Example: LoRA Fine-Tuning

The `examples/granite-lora-finetune.yaml` file fine-tunes a small Granite base model with a LoRA adapter over the `tatsu-lab/alpaca` dataset, using HuggingFace's `trl sft` CLI — no custom training script needed, same spirit as the vLLM examples invoking a CLI directly. LoRA freezes the base model and only trains a small adapter, so it fits comfortably on a single GPU. The run is capped with `--max_steps 50` to stay a short, testable example rather than a full training pass.

It's a plain `batch/v1` Job (no JobSet needed, since there's no separate client/server split), so it qualifies for postprocessing today via its GPU resources and `aibom.io/*` annotations and triggers normally on `JobComplete`. The dataset load (`datasets.load_dataset("tatsu-lab/alpaca")`) is picked up automatically by the existing HuggingFace hook in `dataset_detector.py`, and since this example sets no `aibom.io/dataset-*` annotation, `dataset.declared` is instead parsed from the `trl sft` command's `--dataset_name` flag (`dataset.declared.declared_via: "cli_arg"`) — see [Dataset Declaration and Reconciliation](#dataset-declaration-and-reconciliation). The `model`, `fine_tuning`, and `training` fields (model name, LoRA rank/alpha, adaptation method, learning rate, batch size, epochs) are all auto-detected from the `trl sft` command itself (see Model Auto-Detection) — no annotations needed for those either.

```bash
# Deploy the example (namespace must be set up first)
oc apply -f examples/granite-lora-finetune.yaml
```

## Example: Multi-GPU LoRA Fine-Tuning

The `examples/granite-lora-finetune-multigpu.yaml` file is identical to the single-GPU example above, except it requests 2 GPUs and adds `--num_processes 2` to the `trl sft` invocation, exercising the parallelization-strategy auto-detection described in [Model Auto-Detection](#model-auto-detection).

It deliberately uses `trl`'s own `accelerate launch` passthrough (`--num_processes`) rather than invoking `accelerate launch` as a separate command — the more common way people actually run multi-GPU `trl` jobs, and the harder case for detection, since no `accelerate`/`torchrun`/`deepspeed` launcher binary ever appears in the container's command for `postprocess.py` to key off of.

```bash
# Deploy the example (namespace must be set up first, and the cluster/node
# must have 2+ GPUs schedulable for this to actually run)
oc apply -f examples/granite-lora-finetune-multigpu.yaml
```

## Roadmap

- **Phase 1** (complete): Webhook with placeholder discovery init container
- **Phase 2** (complete): Real hardware discovery + dataset detector injection
- **Phase 3** (complete): Job watcher + real postprocess container for AIBOM compilation
- **Phase 4** (in progress): Production hardening — AIBOM storage (complete), cert-manager TLS, Helm chart, metrics endpoint