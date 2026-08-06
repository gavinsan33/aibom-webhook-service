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
2. Runs `helm upgrade --install aibom-ns-<namespace> charts/aibom-workload-namespace -n <namespace>`, which:
   - Creates a RoleBinding in `aibom-system` granting `system:image-puller` to `system:serviceaccounts:<namespace>` — allows pods in this namespace to pull the postprocess image
   - Creates the `aibom-scripts` ConfigMap with the discovery script, dataset detector, and the shared `k8s_api.py` in-cluster REST helper both of them use — the contents are read from `scripts/aibom-scripts/*.py` at install time via `--set-file`
   - Creates the `aibom-postprocess` ServiceAccount + Role/RoleBinding, granting `create`/`get` on `aiboms.aibom.io` scoped to this namespace — the postprocess Job uses this to create the AIBOM custom resource directly
   - Creates the `aibom-workload-data` Role/RoleBinding, granting `create`/`get`/`patch` on `configmaps` (scoped to this namespace) to every ServiceAccount in the namespace — instrumented workload pods use this to write their discovery/dataset data directly into the per-workload data ConfigMap

**Upgrading an existing namespace**: if you set up a namespace before this RBAC/direct-write model existed, or `scripts/aibom-scripts/*.py` has changed, re-run `just setup-namespace <ns>` — `helm upgrade --install` is idempotent. This matters more than it sounds: the dataset detector hook mounts `k8s_api.py` via a `subPath` volume mount, so a stale `aibom-scripts` ConfigMap missing that key will fail *pod startup* for every instrumented workload in that namespace, not just silently skip dataset detection.

## Cluster Deployment

Deployment is a Helm chart (`charts/aibom-webhook`), covering the `aibom-system` namespace, RBAC, cert-manager `Issuer`/`Certificate`, the webhook `Deployment`/`Service`, the `MutatingWebhookConfiguration`, and the OpenShift `BuildConfig`/`ImageStream` pair used to build both images in-cluster from source. `just deploy` requires cert-manager to already be installed in the cluster — it issues the webhook's TLS certificate into the `aibom-webhook-certs` Secret and keeps it renewed automatically (no more manually re-running a script before a 365-day cert expires). The `MutatingWebhookConfiguration`'s CA bundle is kept in sync by cert-manager's cainjector via its `cert-manager.io/inject-ca-from` annotation, rather than being pasted in by hand.

`just deploy` always installs/upgrades the chart, builds both images in-cluster from source, and rolls out the result — it doesn't rely on the BuildConfig's `ConfigChange` trigger, which (per OpenShift's own docs) only fires automatically the *first* time a BuildConfig is created, never on later edits like a new output tag. If it only patched the chart and left triggering the build to that trigger, every deploy after the first would silently leave the Deployment pointing at a tag nothing had built yet (`ImagePullBackOff`). On a brand-new namespace, `deploy` detects that this is the first install (`helm status` finds no existing release) and waits on that auto-triggered build instead of also starting its own — starting a second, redundant build on top of the trigger's would just double the build time on that first deploy. Either way, expect a brief `ErrImagePull`/`ImagePullBackOff` on the pod while the build catches up to the Deployment's image reference — it resolves on its own once the image is pushed.

If your account doesn't have cluster-scoped permission to create/patch CustomResourceDefinitions, pass `--skip-crds` — e.g. `just deploy --skip-crds` — to pass `--skip-crds` to Helm and never touch the Namespace object. The `aiboms.aibom.io` CRD (`charts/aibom-webhook/crds/aibom-crd.yaml`) and the namespace must then already exist, created once by a cluster-admin via `oc apply -f charts/aibom-webhook/crds/aibom-crd.yaml` and `oc create namespace aibom-system`.

`--version` is the single source of truth for what `just deploy` builds and how it's labeled: it sets the BuildConfig output ImageStreamTag, the Deployment's image reference, and (unless it's `latest`) `build.gitRef` — so the BuildConfig actually checks out and builds that exact commit, instead of always building whatever `build.gitRef`'s branch currently points to regardless of the tag name. Defaults to the short SHA `scripts/remote-build-sha.sh` resolves (the remote tip of `build.gitRepo`/`build.gitRef`, read straight out of `charts/aibom-webhook/values.yaml` rather than a second hardcoded copy) via `git ls-remote`, not your local `HEAD` — local `HEAD` can be ahead of, behind, or diverged from the remote (e.g. unpushed commits), and pinning `build.gitRef` to the resolved SHA also closes a race where the branch tip could otherwise move between resolving the tag and the build actually cloning it. Each build lands on its own ImageStreamTag instead of overwriting a shared one, so you can roll back with `just deploy --version=<older-sha>` — it rebuilds that exact historical commit rather than relabeling whatever's currently on the branch. Pass `--version=latest` to opt back into the old behavior: a mutable tag that always tracks `build.gitRef`'s branch tip. The resolved value also lands on the Deployment as the `app.kubernetes.io/version` label, so `oc get deployment aibom-webhook -o jsonpath='{.metadata.labels.app\.kubernetes\.io/version}'` tells you exactly what's actually running:

```bash
# Build both images in-cluster from source (default: charts/aibom-webhook/values.yaml build.gitRepo/gitRef),
# tagged with the short SHA of that ref's current remote tip, and roll out the result
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

To remove the deployment: `just undeploy` (runs `helm uninstall aibom-webhook`, after a confirmation prompt). The CRD lives in `charts/aibom-webhook/crds/` — Helm installs CRDs once but never upgrades or removes them automatically, so schema changes to `aiboms.aibom.io` need a manual `oc apply -f charts/aibom-webhook/crds/aibom-crd.yaml`, and `helm uninstall` leaves the CRD (and any AIBOM custom resources) in place.

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

Model/training config is auto-populated through three complementary detection layers:

- **CLI-arg parsing** (`postprocess.py`): parses each container's command/args to auto-populate model, fine-tuning, and inference fields without requiring annotations. Detection is command-based, so it also sees into `sh -c "... && trl sft ..."`-style wrapper scripts (common when a job needs to `pip install` before running its training CLI) by shell-splitting the script and scanning the resulting tokens the same way as a plain `command: [...]` list. Two serving/training tools are currently recognized this way: **vLLM** (serving) and **trl** (fine-tuning).
- **Runtime object hooks** (`runtime_detector.py`, see [Dataset detector](#what-gets-injected) above): catches scripts that build their config directly in Python instead of via CLI flags — `transformers.TrainingArguments`, `transformers.PreTrainedModel.from_pretrained` (model name, architecture, dtype, quantization config), and `peft.LoraConfig` (LoRA rank/alpha, and `lora`/`dora`/`rslora`/`qlora` adaptation method). This is what covers plain `transformers.Trainer` scripts, which don't have a recognizable CLI shape at all.
- **KServe InferenceService storage path** (`detect_model_from_storage` in `postprocess.py`): for a KServe `InferenceService` backed by an S3/MinIO data-connection bucket (`spec.predictor.model.storage.key`/`path`, the ODH/RHOAI convention rather than a custom serving container), the predictor's built-in vLLM container always runs with a fixed `--model=/mnt/models` mount — there's no CLI arg identifying the actual model, since that only exists on the `InferenceService` object. This only applies to predictors already instrumented via the existing GPU-request fallback (see [Workload Selection and Grouping](#workload-selection-and-grouping)) — a GPU-less predictor still isn't instrumented at all, consistent with every other detection layer here. For an instrumented predictor, the watcher (`resolveInferenceServiceStorage` in `watcher.go`) identifies it by its `serving.kserve.io/inferenceservice` label, does a single `Get` on that `InferenceService`, and passes its declared `storage.path`/`storageUri` string through to `postprocess.py` as `model.name`.

  **This is identification, not verification.** `model.name` is just the final path segment of the declared storage location (e.g. `models/tinyllama-1.1b-chat` → `tinyllama-1.1b-chat`) — the watcher never resolves the `storage.key` Secret or reads the bucket, by design (it would need `secrets: get` RBAC it deliberately doesn't have; see `clusterrole.yaml`). A renamed or generically-named bucket path (`models/v2`, `models/final`) will be misreported, and there's no check that what's actually in the bucket matches the name. This also means it's storage-backend-agnostic — MinIO, AWS S3, or anything else KServe's storage-initializer supports — since only the `InferenceService`'s own declared string is read, never the endpoint it points to.

Other serving engines (TGI, SGLang, TensorRT-LLM) and other fine-tuning tools (Axolotl, LLaMA-Factory) aren't yet supported by any of these layers.

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

Completed AIBOMs are stored as namespaced `AIBOM` custom resources (`aiboms.aibom.io`, `charts/aibom-webhook/crds/aibom-crd.yaml`) — one per completed workload, created in the same namespace the workload ran in. `just deploy` registers the CRD; if your account lacks CRD permissions, use `just deploy --skip-crds` instead and have a cluster-admin apply it once via `oc apply -f charts/aibom-webhook/crds/aibom-crd.yaml`. Nothing further needs to be set up per namespace to enable collection.

Because `AIBOM` is a namespaced resource, it inherits ordinary Kubernetes RBAC: a user granted `get`/`list` on `aiboms` in namespace `team-a` cannot see `team-b`'s AIBOMs, without any extra code in this project — the same Role/RoleBinding mechanism admins already use for Pods and Jobs applies here.

```bash
# List AIBOMs in a namespace (only visible to users with RBAC on aiboms.aibom.io there)
oc get aiboms -n gavin-test

# Inspect one, including the full compiled AIBOM under spec.data
oc get aibom train-job-abc123 -n gavin-test -o yaml
```

`spec.jobName`, `spec.modelName`, `spec.experimentIntent`, and `spec.collectedAt` are pulled out as printer-friendly summary fields; `spec.data` holds the complete AIBOM JSON exactly as `postprocess.py` produced it.

## Roadmap

- **Phase 1** (complete): Webhook with placeholder discovery init container
- **Phase 2** (complete): Real hardware discovery + dataset detector injection
- **Phase 3** (complete): Job watcher + real postprocess container for AIBOM compilation
- **Phase 4** (complete): Production hardening — AIBOM storage, cert-manager TLS, securityContext + RBAC least-privilege, Helm chart