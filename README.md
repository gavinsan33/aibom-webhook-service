# AIBOM Webhook Service

A Kubernetes mutating admission webhook that automatically instruments AI workloads with AIBOM (AI Bill of Materials) metadata collection. When a pod is created in an opted-in namespace, the webhook injects hardware discovery, dataset detection, and tracking labels — no changes to the user's manifests required.

## How It Works

1. An admin labels a namespace: `oc label namespace my-ns aibom.io/enabled=true`
2. An admin creates the `aibom-scripts` ConfigMap in that namespace (see [Setup](#workload-namespace-setup))
3. A user submits a Job, JobSet, PyTorchJob, or RayJob in that namespace
4. The Kubernetes API server calls the webhook before creating the pod
5. The webhook injects:
   - An **init container** (`aibom-discovery`) that runs a hardware snapshot script capturing CPU, GPU, memory, network, storage info, and performance benchmarks, and writes the result directly into a per-workload data ConfigMap via the Kubernetes API
   - **Dataset detection hooks** (`usercustomize.py`) that automatically capture which ML datasets are loaded at runtime (PyTorch, HuggingFace, torchvision, webdataset) and write them into that same ConfigMap
   - A label `aibom.io/instrumented: "true"` to prevent double-injection
6. The pod is created with the injections — the user's original YAML is untouched
7. When the Job completes (or is being deleted in a JobSet workflow), the **watcher** detects it and creates a postprocess Job to compile the AIBOM. For long-running pods with no Job owner (e.g. KServe `InferenceService` predictor pods, which are owned by a ReplicaSet and never "complete"), the watcher instead triggers on the pod's deletion via a separate pod-level finalizer.

Pods are matched if they are owned by a Job, JobSet, PyTorchJob, or RayJob, **or** if any container requests `nvidia.com/gpu` resources. When GPU resources are present, the webhook copies the GPU resource request to the discovery init container so `nvidia-smi` can detect the hardware. The webhook always fails open (`failurePolicy: Ignore`) — if the service is down, pods are created normally.

Postprocessing is triggered for Jobs whose pods request GPUs or whose Job has `aibom.io/*` annotations. For JobSet workflows where a server pod is killed rather than completing (e.g., vLLM + client benchmarks), the watcher adds a Kubernetes **finalizer** (`aibom.io/log-extraction`) to hold the Job alive until the postprocess Job is created — the finalizer's name predates the current data path but is kept as-is to avoid breaking finalizers already held on live objects.

Bare/ReplicaSet-owned pods (no `batch.kubernetes.io/job-name` label) that carry `aibom.io/instrumented=true` are postprocessed independently, following the same GPU/annotation qualifying criteria but read directly off the pod (since there's no owning Job/JobSet to read them from — KServe, for instance, propagates `spec.predictor.annotations` onto the pod itself). The watcher adds a distinct **pod-level finalizer** (`aibom.io/log-extraction-pod`) while the pod is running, and since these pods never "complete," postprocessing triggers purely on deletion (`DeletionTimestamp` set). There's no JobSet-style sibling merging for this path — each qualifying pod gets its own postprocess Job independently.

## Prerequisites

- Go 1.22+
- An OpenShift cluster (for deployment)
- `openssl` (for TLS cert generation)

## Quick Start

```bash
# Build
make build

# Run Go tests
make test

# Run Python tests (postprocess.py, runtime_detector.py)
make test-python

# Generate self-signed TLS certs (for local dev, adds localhost to SAN)
./scripts/generate-certs.sh --local

# Run locally
make run
```

## Workload Namespace Setup

Each namespace that runs instrumented workloads needs the `aibom.io/enabled` label, image pull access to `aibom-system`, the `aibom-scripts` ConfigMap, and RBAC letting workload pods and the postprocess Job write their own data directly via the Kubernetes API. A single command handles all of it:

```bash
make setup-namespace NAMESPACE=my-ai-workloads
```

This runs:
1. `oc label namespace ... aibom.io/enabled=true` — opts the namespace into webhook instrumentation
2. `oc policy add-role-to-group system:image-puller ...` — allows pods to pull the postprocess image from `aibom-system`
3. Creates the `aibom-scripts` ConfigMap with the discovery script, dataset detector, and the shared `k8s_api.py` in-cluster REST helper both of them use
4. Creates the `aibom-postprocess` ServiceAccount + Role/RoleBinding, granting `create`/`get` on `aiboms.aibom.io` scoped to this namespace — the postprocess Job uses this to create the AIBOM custom resource directly
5. Creates the `aibom-workload-data` Role/RoleBinding, granting `create`/`get`/`patch` on `configmaps` (scoped to this namespace) to every ServiceAccount in the namespace — instrumented workload pods use this to write their discovery/dataset data directly into the per-workload data ConfigMap

**Upgrading an existing namespace**: if you set up a namespace before this RBAC/direct-write model existed, re-run `make setup-namespace NAMESPACE=<ns>` — it's idempotent. This matters more than it sounds: the dataset detector hook mounts `k8s_api.py` via a `subPath` volume mount, so a stale `aibom-scripts` ConfigMap missing that key will fail *pod startup* for every instrumented workload in that namespace, not just silently skip dataset detection.

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
  aibomdata/aibomdata.go            # Shared postprocess Job/ConfigMap naming convention
postprocess/
  postprocess.py                    # AIBOM compiler; creates the AIBOM CR directly (runs in postprocess Job)
  Dockerfile                        # Postprocess container image; also COPYs in scripts/aibom-scripts/k8s_api.py
deploy/
  namespace.yaml                    # aibom-system namespace
  rbac.yaml                         # ServiceAccount, ClusterRole, ClusterRoleBinding
  deployment.yaml                   # Deployment + Service
  build.yaml                        # OpenShift BuildConfig + ImageStream
  webhook-config.yaml               # MutatingWebhookConfiguration
  aibom-scripts-configmap.yaml      # Reference manifest for the scripts ConfigMap
  aibom-crd.yaml                    # AIBOM CustomResourceDefinition
scripts/
  generate-certs.sh                 # Self-signed TLS cert generation
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
Makefile                            # Build, test, deploy targets (make test, make test-python)
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
- Writes the result directly into the workload's data ConfigMap (key `discovery-<pod-name>.json`) via `k8s_api.py`, an in-cluster REST helper using only the Python stdlib — no `kubernetes` pip dependency needed even though this runs inside a third-party pinned image

**Dataset detector (into each application container):**
- Mounts `runtime_detector.py` as `usercustomize.py` on `PYTHONPATH`, plus `k8s_api.py` alongside it (also `subPath`-mounted, since `usercustomize.py` needs to `import k8s_api`)
- Python auto-imports it at startup — no code changes needed
- Hooks into PyTorch DataLoader, HuggingFace `datasets.load_dataset`, torchvision datasets, and webdataset
- Also hooks `transformers.TrainingArguments`, `transformers.PreTrainedModel.from_pretrained`, and `peft.LoraConfig` — this catches model/training config for scripts that build these objects directly in Python (e.g. a custom `transformers.Trainer` script), which expose nothing on the command line for `postprocess.py`'s CLI-arg detectors to see
- Captures dataset name, version, split, fingerprint, license, and training args
- Writes the result directly into the same data ConfigMap (key `dataset-<pod-name>.json`) at process exit, via the same `k8s_api.py` helper — since this runs inside the user's own application container, the `k8s_api` import is wrapped in a soft `try/except ImportError` so a missing/stale mount degrades to "no dataset detection" instead of crashing the user's training process at Python startup

## Postprocess Flow

When a Job completes or is being deleted (held by the finalizer), the watcher creates an AIBOM postprocess Job:

1. **Data ConfigMap read/merge**: The Job's pods (and sibling pods in a JobSet) have already written their own discovery/dataset data directly into the per-workload data ConfigMap (`{job-name}-aibom-postprocess-data`) via the Kubernetes API — see [What Gets Injected](#what-gets-injected). The watcher reads that ConfigMap (creating it if the pods never got to write anything) and merges in `annotations.json`/`containers.json`/aggregated `discovery.json`/`dataset.json` keys that `postprocess.py` expects
2. **Finalizer removal**: If the Job has the `aibom.io/log-extraction` finalizer, it is removed after this step, allowing Kubernetes to complete the deletion
3. **Postprocess Job**: A Job is created running `postprocess.py` under a dedicated `aibom-postprocess` ServiceAccount (RBAC scoped to `aiboms.aibom.io` create/get in this namespace only), which:
   - Loads discovery and dataset data from the ConfigMap mount
   - Optionally queries Grafana/Prometheus for telemetry (GPU utilization, memory, power, CPU, network)
   - Compiles everything into an AIBOM JSON document
   - Creates the `AIBOM` custom resource directly via the Kubernetes API (no watcher involvement) — a failed create exits the process non-zero, so Kubernetes' own Job retry/failure handling (`backoffLimit`) becomes the visible signal instead of a silently-dropped log line
4. **Cleanup**: Once the postprocess Job succeeds, the watcher (which only needed to notice the Job's success, not read anything from it) deletes the postprocess Job and its data ConfigMap so a same-named rerun of the workload doesn't collide with leftovers

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

- **Sibling pod data**: Discovery and dataset data is read from all instrumented pods across the JobSet (each already wrote its own ConfigMap keys directly), not just the triggering job's pods
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

Model/training config is auto-populated through two complementary detection layers:

- **CLI-arg parsing** (`postprocess.py`): parses each container's command/args to auto-populate model, fine-tuning, and inference fields without requiring annotations. Detection is command-based, so it also sees into `sh -c "... && trl sft ..."`-style wrapper scripts (common when a job needs to `pip install` before running its training CLI) by shell-splitting the script and scanning the resulting tokens the same way as a plain `command: [...]` list. Two serving/training tools are currently recognized this way: **vLLM** (serving) and **trl** (fine-tuning).
- **Runtime object hooks** (`runtime_detector.py`, see [Dataset detector](#what-gets-injected) above): catches scripts that build their config directly in Python instead of via CLI flags — `transformers.TrainingArguments`, `transformers.PreTrainedModel.from_pretrained` (model name, architecture, dtype, quantization config), and `peft.LoraConfig` (LoRA rank/alpha, and `lora`/`dora`/`rslora`/`qlora` adaptation method). This is what covers plain `transformers.Trainer` scripts, which don't have a recognizable CLI shape at all.

Other serving engines (TGI, SGLang, TensorRT-LLM) and other fine-tuning tools (Axolotl, LLaMA-Factory) aren't yet supported by either layer.

**Quantization from model name**: regex patterns match common quantization markers in the model name/path (e.g. `AWQ`, `GPTQ`, `INT4`/`INT8`, `FP4`/`FP8`, `bitsandbytes`/`NF4`, `Marlin`, `GGUF`, `AQLM`, `EXL2`, and others), extracting both the method and bit width. For example, `drawais/Granite-3.3-8B-Instruct-AWQ-INT4` detects `awq` at 4 bits.

**vLLM CLI arguments**: flags on the server's `vllm serve` / `vllm.entrypoints.openai.api_server` invocation are parsed directly:

| Flag | AIBOM Field |
|------|-------------|
| `--model` | `model.name` |
| `--dtype` | `model.dtype` |
| `--quantization` / `-q` | `model.quantization` |
| `--max-model-len` | `inference.max_model_len` |
| `--tensor-parallel-size` / `-tp` | `inference.tensor_parallel_size` |
| `--pipeline-parallel-size` / `-pp` | `inference.pipeline_parallel_size` |
| `--enable-expert-parallel` | `inference.enable_expert_parallel` |
| `--data-parallel-size` / `-dp` | `inference.data_parallel_size` |
| `--gpu-memory-utilization` | `inference.gpu_memory_utilization` |
| `--speculative-model` + `--num-speculative-tokens` (legacy), or `--speculative-config` (modern, JSON or `key=value` list) | `model.speculative_decoding` |
| `--override-generation-config` (JSON or `key=value` list) | `inference.temperature`, `inference.top_p`, `inference.top_k` |

Sampling parameters set per-request by a benchmark client (e.g. temperature passed in an HTTP request body) aren't visible here — this only sees what's baked into the server's own startup command.

**trl CLI arguments**: flags on a `trl sft`/`trl dpo`-style invocation are parsed the same way:

| Flag | AIBOM Field |
|------|-------------|
| `--model_name_or_path` | `model.name` |
| `--use_peft` (combined with presence of `--lora_r`, `--use_dora`, `--use_rslora`, `--load_in_4bit`/`--load_in_8bit`) | `fine_tuning.adaptation_method` (`lora`, `qlora`, `dora`, `rslora`, or `peft`) |
| `--lora_r` | `fine_tuning.lora_rank` |
| `--lora_alpha` | `fine_tuning.lora_alpha` |
| `--learning_rate` | `training.learning_rate` |
| `--per_device_train_batch_size` | `training.batch_size` |
| `--num_train_epochs` | `training.epochs` |
| `--seed` | `training.random_seed` |

**Parallelization strategy** (`training.parallelization_strategy`): detected independently of the training tool being launched, covering three shapes:

- An explicit launcher binary: `accelerate launch --multi_gpu` → `data_parallel`; a bare `deepspeed` launcher or `--deepspeed <config>` flag → `deepspeed`; `torchrun`/`mpirun` → `data_parallel`.
- A bare `--fsdp` flag on the training command itself → `fsdp`.
- Accelerate-launch arguments passed *directly* to a CLI that spawns `accelerate launch` internally — no separate launcher token ever appears in the container's command. `trl`'s CLI supports this: `trl sft ... --num_processes 4` → `data_parallel`, and `trl sft ... --accelerate_config <path>` is resolved from the file's actual `distributed_type` (`FSDP` → `fsdp`, `DEEPSPEED` → `deepspeed`, `MULTI_GPU`/`MULTI_CPU` → `data_parallel`) — this is the harder, easy-to-miss case — see `examples/granite-lora-finetune-multigpu.yaml`.

The `--accelerate_config` file is read from inside the training container itself, by the same in-container hook (`runtime_detector.py`) that detects datasets — `postprocess.py` has no access to the training container's filesystem after the fact. This requires PyYAML to be present in the training image (a hard dependency of `accelerate` itself, so present whenever `--accelerate_config` is actually used); if it's unavailable, or the config file's `distributed_type` isn't one of the four listed above, detection falls back to guessing from the config filename against a small set of known preset names (`fsdp1`/`fsdp2`/`zero1`/`zero2`/`zero3`/`multi_gpu`/`single_gpu`) — a much weaker heuristic, since it only works if the file happens to be named exactly one of those.

This only sees parallelism baked into the command (directly or via one of these launcher-passthrough conventions) or in the referenced accelerate config's content; in-script sharding (e.g. `device_map="auto"` set directly in Python code, common in QLoRA scripts) isn't visible and isn't detected.

**Precedence**: auto-detected values are used as defaults; any corresponding `aibom.io/*` annotation always overrides them.

### Dataset Declaration and Reconciliation

`dataset.declared` is filled in from three sources, in order of precedence:

1. **Annotation** — `aibom.io/dataset-name` (and `dataset-version`/`dataset-source`/`dataset-license`), if set.
2. **CLI arg** — parsed from the training command's `--dataset_name`/`--dataset_config_name`/`--dataset_train_split` flags (e.g. a `trl sft --dataset_name ...` invocation), if no annotation is set.
3. **Inferred from runtime** — copied from the first `dataset.auto_detected` entry, only if neither of the above produced a name.

Whichever source wins is recorded in `dataset.declared.declared_via` (`"annotation"`, `"cli_arg"`, or `"inferred_from_runtime"`), so it's always possible to tell whether a dataset name reflects something the job author actually specified or a best-effort guess from what was observed at runtime.

Every entry in `dataset.auto_detected` also carries a `matches_declared` boolean, comparing its `dataset_name` against the final `dataset.declared.name`. This is the actual reconciliation check: it flags the case where a job declares one dataset (via annotation or CLI arg) but the code loads something different at runtime.

Within `dataset.auto_detected` itself, `runtime_detector.py` correlates hook detections that refer to the same underlying dataset object into a single entry (with a `seen_via` list noting every hook that touched it) rather than recording it twice. The common case is a `datasets.load_dataset(...)` call followed by wrapping the result in a `torch.utils.data.DataLoader(...)` for batching, matched by object identity. Since scripts commonly transform the dataset first (`.map()`/`.filter()`/`.select()`/`.shuffle()`, or re-fetching a split from a `DatasetDict`) before handing it to `DataLoader` — which returns a *new* object each time — correlation falls back to matching on the dataset's stable `(builder_name, config_name)` identity when object identity doesn't match, so a transformed dataset still merges into its original entry instead of showing up as a second, generically-named (`"Dataset"`) one.

### Grafana Credentials

To enable telemetry collection, create a secret in each instrumented namespace:

```bash
oc create secret generic aibom-config \
  --from-literal=grafana-url=https://grafana.example.com \
  --from-literal=grafana-api-token=<token> \
  -n my-ai-workloads
```

**Ingestion delay and retries**: `postprocess.py` queries Grafana immediately after the workload's pod completes. On some observability backends (e.g. a federated/multi-tenant Prometheus setup) there's a delay between a metric being scraped and it becoming queryable, so a summary query fired this soon can race that delay and come back empty even though the identical query succeeds moments later — this shows up as `resource_utilization` averages being present on some AIBOMs and missing on others for no apparent reason, even though the Grafana Explore link (built from the same time range) always shows the underlying data once it lands. To absorb this, missing summary metrics are retried with a delay (`AIBOM_TELEMETRY_RETRY_ATTEMPTS`, default `3`; `AIBOM_TELEMETRY_RETRY_DELAY_S`, default `45` seconds between attempts) before being recorded as unavailable — only the metrics still missing on a given attempt are re-queried, not the whole batch.

### AIBOM Storage

Completed AIBOMs are stored as namespaced `AIBOM` custom resources (`aiboms.aibom.io`, `deploy/aibom-crd.yaml`) — one per completed workload, created in the same namespace the workload ran in. The CRD is registered by `make deploy`; nothing further needs to be set up per namespace to enable collection.

Because `AIBOM` is a namespaced resource, it inherits ordinary Kubernetes RBAC: a user granted `get`/`list` on `aiboms` in namespace `team-a` cannot see `team-b`'s AIBOMs, without any extra code in this project — the same Role/RoleBinding mechanism admins already use for Pods and Jobs applies here.

```bash
# List AIBOMs in a namespace (only visible to users with RBAC on aiboms.aibom.io there)
oc get aiboms -n gavin-test

# Inspect one, including the full compiled AIBOM under spec.data
oc get aibom train-job-abc123 -n gavin-test -o yaml
```

`spec.jobName`, `spec.modelName`, `spec.experimentIntent`, and `spec.collectedAt` are pulled out as printer-friendly summary fields; `spec.data` holds the complete AIBOM JSON exactly as `postprocess.py` produced it.

## Example: vLLM Inference Benchmark

The `examples/vllm-inference.yaml` file shows a JobSet with a vLLM server and a guidellm benchmark client. The server has `aibom.io/*` annotations and GPU resources; the client depends on the server being ready. When the client finishes, the JobSet kills the server — but the finalizer holds it until the watcher reads the pods' data ConfigMap and creates the postprocess Job.

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

KServe predictor pods are owned by a ReplicaSet, not a Job/JobSet/PyTorchJob/RayJob, so the webhook instruments them via the `requestsGPU` fallback match (`internal/webhook/mutator.go`), same as any other GPU pod. Postprocessing is handled by the watcher's pod-level finalizer path (see [Which pods get postprocessed?](#workload-selection-and-grouping)): since the predictor pod never "completes," the watcher holds it open with a distinct finalizer (`aibom.io/log-extraction-pod`) on deletion (e.g. `oc delete pod`, a rollout, or scale-down), reads its discovery/dataset data ConfigMap and `aibom.io/*` annotations (propagated onto the pod by KServe from `spec.predictor.annotations`), and creates a postprocess Job directly from that single pod — there's no JobSet to pull sibling data from here. The client Job below still doesn't itself qualify for postprocessing (no GPU resources or `aibom.io/*` annotations, and it isn't part of a JobSet to inherit any) — it only exercises the predictor endpoint.

```bash
# Deploy the example (namespace must be set up first)
oc apply -f examples/vllm-inference-rhoai.yaml
```

## Example: LoRA Fine-Tuning

The `examples/granite-lora-finetune.yaml` file fine-tunes a small Granite base model with a LoRA adapter over the `tatsu-lab/alpaca` dataset, using HuggingFace's `trl sft` CLI — no custom training script needed, same spirit as the vLLM examples invoking a CLI directly. LoRA freezes the base model and only trains a small adapter, so it fits comfortably on a single GPU. The run is capped with `--max_steps 50` to stay a short, testable example rather than a full training pass.

It's a plain `batch/v1` Job (no JobSet needed, since there's no separate client/server split), so it qualifies for postprocessing today via its GPU resources and `aibom.io/*` annotations and triggers normally on `JobComplete`. The dataset load (`datasets.load_dataset("tatsu-lab/alpaca")`) is picked up automatically by the existing HuggingFace hook in `runtime_detector.py`, and since this example sets no `aibom.io/dataset-*` annotation, `dataset.declared` is instead parsed from the `trl sft` command's `--dataset_name` flag (`dataset.declared.declared_via: "cli_arg"`) — see [Dataset Declaration and Reconciliation](#dataset-declaration-and-reconciliation). The `model`, `fine_tuning`, and `training` fields (model name, LoRA rank/alpha, adaptation method, learning rate, batch size, epochs) are all auto-detected from the `trl sft` command itself (see Model Auto-Detection) — no annotations needed for those either.

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

## Example: Raw `transformers.Trainer` LoRA Fine-Tuning

The `examples/granite-lora-finetune-raw-trainer.yaml` file runs the same LoRA fine-tune as the examples above, but via a custom Python script that calls `transformers.Trainer`/`peft.LoraConfig` directly instead of the `trl sft` CLI — no CLI flags at all for `postprocess.py`'s command-line detectors to see.

It sets no `aibom.io/model-name` or `aibom.io/dataset-*` annotations, so every field in the resulting AIBOM's `model`, `training`, and `fine_tuning` sections is sourced purely from `runtime_detector.py`'s object-level hooks: `transformers.PreTrainedModel.from_pretrained` (model name/architecture/dtype), `transformers.TrainingArguments` (learning rate, batch size, epochs, seed, dtype), and `peft.LoraConfig` (LoRA rank/alpha, adaptation method) — see [Model Auto-Detection](#model-auto-detection). `dataset.declared.declared_via` should come back as `"inferred_from_runtime"`, since there's no annotation or CLI arg for the dataset name either — the third and last precedence tier described in [Dataset Declaration and Reconciliation](#dataset-declaration-and-reconciliation).

The script also tokenizes the dataset via `.map()` before handing it to `Trainer` (which wraps it in its own `DataLoader` internally) — the same "dataset transformed before being wrapped" shape that `runtime_detector.py`'s dataset dedup relies on its `(builder_name, config_name)` key-based fallback to still collapse into a single `dataset.auto_detected` entry, rather than the transformed copy showing up as a second, generically-named one.

```bash
# Deploy the example (namespace must be set up first)
oc apply -f examples/granite-lora-finetune-raw-trainer.yaml
```

## Roadmap

- **Phase 1** (complete): Webhook with placeholder discovery init container
- **Phase 2** (complete): Real hardware discovery + dataset detector injection
- **Phase 3** (complete): Job watcher + real postprocess container for AIBOM compilation
- **Phase 4** (in progress): Production hardening — AIBOM storage (complete), cert-manager TLS, Helm chart, metrics endpoint