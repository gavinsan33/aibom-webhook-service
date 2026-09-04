# Implementation Notes

This file holds the detailed rationale and edge-case behavior behind the mechanics summarized in `README.md`. Read the README first for the high-level picture; this is the "why" and the fine print.

## Pod Matching and Injection

Pods are matched if they are owned by a Job, JobSet, PyTorchJob, or RayJob, **or** if any container requests `nvidia.com/gpu` resources. When GPU resources are present, the webhook copies the GPU resource request to the discovery init container so `nvidia-smi` can detect the hardware. The webhook always fails open (`failurePolicy: Ignore`) — if the service is down, pods are created normally.

For long-running pods with no Job owner (e.g. KServe `InferenceService` predictor pods, owned by a ReplicaSet and never "complete"), the watcher triggers on the pod's deletion via a separate pod-level finalizer instead of Job completion.

Dataset detector's `k8s_api` import is wrapped in a soft `try/except ImportError` — since it runs inside the user's own application container, a missing/stale mount degrades to "no dataset detection" instead of crashing the user's training process at Python startup.

### Discovery Data Signing

`discovery-<pod-name>.json` (hardware/GPU data, written by the `aibom-discovery` init container) is HMAC-SHA256 signed before being written to the data ConfigMap, using a per-namespace key (`aibom-discovery-hmac-key` Secret, created by `charts/aibom-workload-namespace`'s `signing.yaml`) mounted **only** into the discovery init container — never into an application container (see `mutator.go`'s `buildDiscoverySigningKeyVolume`). The watcher reads the same Secret — via a namespaced `Role`/`RoleBinding` (also in `signing.yaml`, scoped by `resourceNames` to that exact name) granted per-namespace to the webhook's ServiceAccount, not a cluster-wide `ClusterRole`, so this identity's read access is limited to namespaces that installed `aibom-workload-namespace`, not a same-named Secret anywhere on the cluster — and verifies the signature before merging a pod's discovery data into the aggregate `discovery.json` a postprocess Job reads; an entry with a missing or invalid signature is dropped and logged, not silently accepted.

This closes both the overwrite race (#29) and first-write forgery specifically for hardware claims: since the app container never has the key, it cannot produce a validly-signed write no matter what RBAC it holds on the ConfigMap itself. Dataset/runtime data from `runtime_detector.py` is deliberately **not** covered by this mechanism — it runs inside the user's own application container, so any key given to it would be reachable by the exact process this is meant to defend against, and signing would add nothing over what RBAC/write-once semantics already provide (see #30's discussion of this ceiling).

A symmetric key (HMAC), not an asymmetric keypair, is used deliberately: the discovery init container's image is external and admin-configurable (`--discovery-image`, defaulting to a public `pytorch/pytorch` image), not guaranteed to have a crypto library installed, so signing had to stay within Python's stdlib (`hmac`/`hashlib`). Since the verifier (the watcher, a project-controlled component) is equally trusted, an asymmetric split (which exists specifically to let an *untrusted* party verify without being able to forge) buys nothing extra here.

A missing Secret (a namespace whose `aibom-workload-namespace` chart install predates `signing.yaml`) is treated as "verification unavailable," not as tampering — discovery data passes through unverified rather than being dropped, so this rolls out gradually across existing namespaces rather than breaking them until they're upgraded.

## Postprocess Flow (detail)

1. **Data ConfigMap read/merge**: The Job's pods (and sibling pods in a JobSet) have already written their own discovery/dataset data directly into the per-workload data ConfigMap (`{job-name}-aibom-postprocess-data`) via the Kubernetes API. The watcher reads that ConfigMap (creating it if the pods never got to write anything) and merges in `annotations.json`/`containers.json`/aggregated `discovery.json`/`dataset.json` keys that `postprocess.py` expects.
2. **Finalizer removal**: If the Job has the `aibom.io/log-extraction` finalizer, it is removed after this step, allowing Kubernetes to complete the deletion. (The finalizer's name predates the current data path but is kept as-is to avoid breaking finalizers already held on live objects.)
3. **Postprocess Job**: Runs `postprocess.py` under a dedicated `aibom-postprocess` ServiceAccount (RBAC scoped to `aiboms.aibom.io` create/get in this namespace only). It loads discovery/dataset data from the ConfigMap mount, optionally queries Prometheus for telemetry, compiles everything into an AIBOM JSON document, and creates the `AIBOM` custom resource directly via the Kubernetes API (no watcher involvement) — a failed create exits the process non-zero, so Kubernetes' own Job retry/failure handling (`backoffLimit`) becomes the visible signal instead of a silently-dropped log line.
4. **Cleanup**: Once the postprocess Job finishes (success or failure — `isJobFinished` treats both as terminal), the watcher deletes the postprocess Job and its data ConfigMap so a same-named rerun of the workload doesn't collide with leftovers. `--debug-keep-postprocess-jobs` (`debug.keepPostprocessJobs` chart value) skips this — the Job is still annotated `aibom.io/aibom-collected` so a resync doesn't try to collect it again, but it and its data ConfigMap are left in place for inspecting postprocess pod logs/exit state or the ConfigMap's merged discovery/dataset data. Leaks one of each per completed workload indefinitely; not for routine production use.

### Workload Selection and Grouping

**Which jobs get postprocessed?** A job in an `aibom.io/enabled` namespace qualifies if any of its pods request `nvidia.com/gpu` resources (limits or requests > 0), or the job has any `aibom.io/*` annotations. Internal labels (`aibom.io/instrumented`, `aibom.io/postprocess-job`) are excluded from this check. Jobs that are themselves postprocess jobs (labeled `aibom.io/postprocess-for`) are always skipped.

**When does postprocessing trigger?** On Job completion (`JobComplete` condition), or on Job deletion (`DeletionTimestamp` set — the finalizer path, used for JobSet server pods killed rather than completing naturally). Each job is postprocessed at most once; after the postprocess job is created, the original job is annotated `aibom.io/postprocess-job` and subsequent events are skipped.

**How are workloads grouped?** Each qualifying job gets its own postprocess job — no cross-job merging. If a job belongs to a **JobSet** (`jobset.sigs.k8s.io/jobset-name` label), its postprocess job pulls in sibling discovery/dataset data across the whole JobSet, and inherits sibling `aibom.io/*` annotations if the triggering job has none. Example: in a vLLM server + client JobSet, only the server job qualifies (GPU + annotations); the client job is skipped but the server's postprocess job still includes discovery data from client pods.

**Which pods get postprocessed?** Bare pods with no Job owner (no `batch.kubernetes.io/job-name` label — e.g. a KServe predictor pod owned by a ReplicaSet) are postprocessed via a separate path: same qualifying criteria as Jobs, but read directly from the pod (GPU resources on its containers, or `aibom.io/*` annotations — KServe propagates `spec.predictor.annotations` down onto the pod). Only pods already carrying `aibom.io/instrumented=true` are candidates. There's no "complete" state for a long-running pod, so postprocessing triggers purely on deletion via a distinct pod-level finalizer (`aibom.io/log-extraction-pod`), and there's no JobSet-style sibling merging here — each qualifying pod gets its own postprocess Job.

## Model Auto-Detection

Model/training config is auto-populated through three complementary detection layers:

- **CLI-arg parsing** (`postprocess.py`): parses each container's command/args to auto-populate model, fine-tuning, and inference fields without requiring annotations. Detection is command-based, so it also sees into `sh -c "... && trl sft ..."`-style wrapper scripts (common when a job needs to `pip install` before running its training CLI) by shell-splitting the script and scanning tokens the same way as a plain `command: [...]` list. Two tools currently recognized: **vLLM** (serving) and **trl** (fine-tuning).
- **Runtime object hooks** (`runtime_detector.py`): catches scripts that build config directly in Python instead of via CLI flags — `transformers.TrainingArguments`, `transformers.PreTrainedModel.from_pretrained` (model name, architecture, dtype, quantization config), and `peft.LoraConfig` (LoRA rank/alpha, adaptation method). This covers plain `transformers.Trainer` scripts with no recognizable CLI shape.
- **KServe InferenceService storage path** (`detect_model_from_storage` in `postprocess.py`): for a KServe `InferenceService` backed by an S3/MinIO data-connection bucket (`spec.predictor.model.storage.key`/`path`, the ODH/RHOAI convention), the predictor's built-in vLLM container always runs with a fixed `--model=/mnt/models` mount — no CLI arg identifies the actual model, since that only exists on the `InferenceService` object. Only applies to predictors already instrumented via the GPU-request fallback; a GPU-less predictor still isn't instrumented at all.

  Resolution happens inside the **`aibom-discovery` init container itself** (`generate_snapshot.py`'s `resolve_inference_service_storage`), at pod startup — not from the watcher, and not lazily at pod deletion. Reasoning: deleting the InferenceService is the normal way one of these pods gets deleted, and Kubernetes' garbage collector removes the InferenceService object from etcd well before the delete cascades to the pod — a lookup at deletion time would 404 almost every time, for exactly the trigger this exists to detect. Resolving at pod startup instead means the InferenceService that caused this pod to exist is essentially guaranteed to still be there.

  The init container knows its InferenceService's name via a downward-API env var (`INFERENCESERVICE_NAME`, only injected for pods already carrying that label — a downward API reference to a nonexistent label fails admission outright, so this can't be added unconditionally for every workload kind), does a single `GET` using the **workload's own namespace-scoped RBAC** (`aibom-workload-inferenceservices` Role, mirroring `aibom-workload-data`), and writes the result into the shared data ConfigMap as `storage-<pod-name>.json`. The always-on webhook/watcher process has no `serving.kserve.io` RBAC at all.

  **This is identification, not verification.** `model.name` is just the final path segment of the declared storage location (e.g. `models/tinyllama-1.1b-chat` → `tinyllama-1.1b-chat`) — nothing resolves the `storage.key` Secret or reads the bucket. A renamed or generically-named bucket path (`models/v2`, `models/final`) will be misreported, and there's no check that the bucket contents match the name. This also makes it storage-backend-agnostic — MinIO, AWS S3, or anything else KServe's storage-initializer supports — since only the `InferenceService`'s own declared string is read, never the endpoint it points to.

Other serving engines (TGI, SGLang, TensorRT-LLM) and other fine-tuning tools (Axolotl, LLaMA-Factory) aren't yet supported by any of these layers.

**Quantization from model name**: regex patterns match common quantization markers (`AWQ`, `GPTQ`, `INT4`/`INT8`, `FP4`/`FP8`, `bitsandbytes`/`NF4`, `Marlin`, `GGUF`, `AQLM`, `EXL2`, and others), extracting both method and bit width. E.g. `drawais/Granite-3.3-8B-Instruct-AWQ-INT4` detects `awq` at 4 bits.

**vLLM CLI arguments** (`vllm serve` / `vllm.entrypoints.openai.api_server`):

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

Sampling parameters set per-request by a benchmark client (e.g. temperature in an HTTP request body) aren't visible here — this only sees what's baked into the server's own startup command. This is an architectural limitation, not a gap slated to close: the request body lives on the wire between the client and an already-running vLLM server, which neither the discovery init container nor the runtime hooks (both scoped to a single container's own process) ever observe. Closing it would require something structurally different — e.g. a sidecar/proxy in front of the server's traffic — which is out of scope here.

**trl CLI arguments** (`trl sft`/`trl dpo`-style):

| Flag | AIBOM Field |
|------|-------------|
| `--model_name_or_path` | `model.name` |
| `--use_peft` (combined with `--lora_r`, `--use_dora`, `--use_rslora`, `--load_in_4bit`/`--load_in_8bit`) | `fine_tuning.adaptation_method` (`lora`, `qlora`, `dora`, `rslora`, or `peft`) |
| `--lora_r` | `fine_tuning.lora_rank` |
| `--lora_alpha` | `fine_tuning.lora_alpha` |
| `--learning_rate` | `training.learning_rate` |
| `--per_device_train_batch_size` | `training.batch_size` |
| `--num_train_epochs` | `training.epochs` |
| `--seed` | `training.random_seed` |

**Parallelization strategy** (`training.parallelization_strategy`), detected independently of the training tool launched:

- Explicit launcher binary: `accelerate launch --multi_gpu` → `data_parallel`; bare `deepspeed` or `--deepspeed <config>` → `deepspeed`; `torchrun`/`mpirun` → `data_parallel`.
- Bare `--fsdp` flag on the training command itself → `fsdp`.
- Accelerate-launch arguments passed *directly* to a CLI that spawns `accelerate launch` internally (no separate launcher token appears in the command). `trl`'s CLI supports this: `trl sft ... --num_processes 4` → `data_parallel`, and `trl sft ... --accelerate_config <path>` is resolved from the file's actual `distributed_type` (`FSDP` → `fsdp`, `DEEPSPEED` → `deepspeed`, `MULTI_GPU`/`MULTI_CPU` → `data_parallel`) — the harder, easy-to-miss case (see `examples/granite-lora-finetune-multigpu.yaml`).

The `--accelerate_config` file is read from inside the training container itself, by `runtime_detector.py` (the same in-container hook that detects datasets) — `postprocess.py` has no access to the training container's filesystem after the fact. This requires PyYAML in the training image (a hard dependency of `accelerate` itself); if unavailable, or the config's `distributed_type` isn't one of the four listed, detection falls back to guessing from the config filename against known preset names (`fsdp1`/`fsdp2`/`zero1`/`zero2`/`zero3`/`multi_gpu`/`single_gpu`) — a much weaker heuristic that only works if the file is named exactly one of those.

In-script sharding with no command/accelerate-config signal at all — e.g. `device_map="auto"` passed directly to `from_pretrained` in Python, common in raw `transformers.Trainer`/QLoRA scripts with no CLI launcher — is covered by a lowest-priority fallback: the `transformers.PreTrainedModel.from_pretrained` runtime hook (`runtime_detector.py`) captures the `device_map` kwarg, and `postprocess.py` maps a multi-device value (anything other than a single `cpu`/`cuda`/`cuda:N` device) to `model_parallel` only if none of the above command/accelerate-config signals produced a strategy first.

## Dataset Declaration and Reconciliation

`dataset.declared` is filled in from three sources, in order of precedence:

1. **Annotation** — `aibom.io/dataset-name` (and `dataset-version`/`dataset-source`/`dataset-license`), if set.
2. **CLI arg** — parsed from `--dataset_name`/`--dataset_config_name`/`--dataset_train_split` flags (e.g. `trl sft --dataset_name ...`), if no annotation is set.
3. **Inferred from runtime** — copied from the first `dataset.auto_detected` entry, only if neither of the above produced a name.

Whichever source wins is recorded in `dataset.declared.declared_via` (`"annotation"`, `"cli_arg"`, or `"inferred_from_runtime"`), so it's always possible to tell whether a dataset name reflects something the job author specified or a best-effort guess from runtime observation.

Every entry in `dataset.auto_detected` carries a `matches_declared` boolean, comparing its `dataset_name` against the final `dataset.declared.name` — the actual reconciliation check, flagging when a job declares one dataset but the code loads something different at runtime.

Within `dataset.auto_detected`, `runtime_detector.py` correlates hook detections referring to the same underlying dataset object into a single entry (with a `seen_via` list noting every hook that touched it) rather than recording it twice. The common case: a `datasets.load_dataset(...)` call followed by wrapping the result in a `torch.utils.data.DataLoader(...)`, matched by object identity. Since scripts commonly transform the dataset first (`.map()`/`.filter()`/`.select()`/`.shuffle()`, or re-fetching a split from a `DatasetDict`) before handing it to `DataLoader` — which returns a *new* object each time — correlation falls back to matching on the dataset's stable `(builder_name, config_name)` identity when object identity doesn't match, so a transformed dataset still merges into its original entry instead of showing up as a second, generically-named (`"Dataset"`) one.

## Git Provenance Detection

`source_code.git_commit`/`git_repository`/`git_branch` covers two distinct scenarios — code baked into the image at build time, and code pulled at runtime by the workload itself (e.g. `sh -c "git clone <url> && python train.py"`) — filled in from five sources, in order of precedence:

1. **Annotation** — `aibom.io/git-repository`/`git-commit`/`git-branch`, if set.
2. **Runtime `.git`-directory read** — `_capture_git_provenance` (`runtime_detector.py`) walks upward from the training process's CWD (mirroring how `git` itself locates the repo root) looking for a `.git` directory, resolves `HEAD` to a commit (following a loose ref file or falling back to `packed-refs`), reads the `origin` remote URL straight out of `.git/config` (no dependency on the `git` binary being installed), and best-effort shells out to `git status --porcelain` for a dirty-tree check if `git` happens to be available (recorded as `source_code.dirty`; omitted entirely, not reported as clean, if it can't be determined). Ranked highest among auto-detected sources because it reflects the actual final checked-out state — after any `git checkout`/`git pull` beyond the initial clone — not just intent.
3. **CLI-parsed `git clone`/`checkout`** — `detect_git_clone_from_containers` (`postprocess.py`) regex-scans the container's own command/args (the same tokens the vLLM/trl CLI detectors already see, including through the `sh -c`/`bash -c` wrapper-script flattening) for a `git clone <url>` invocation and an optional following `git checkout <ref>`/`--branch <ref>`. A bare hex string of plausible SHA length (7-40 chars) is treated as a commit; anything else as a branch/tag. Falls back below the `.git`-directory read (which reflects a script's actual outcome, not just its stated command) but ranks above the image-label tiers below, since it's still specific to *this run's* command rather than a label baked onto the image at build time.
4. **OpenShift BuildConfig commit labels** — if none of the above resolves anything, `detect_git_provenance_from_containers` (`postprocess.py`) looks up the container's image by digest (`containers.json`'s `image_id`, populated in the watcher from `ContainerStatuses[].ImageID` — the resolved digest, not the mutable tag a re-push could spoof) against the cluster-scoped `image.openshift.io` `Image` object, and reads whatever `io.openshift.build.commit.id`/`.ref`/`.source-location` labels OpenShift's build controller baked onto it.
5. **OCI-standard image labels** — if no OpenShift-specific labels are present either, falls back to the vendor-neutral `org.opencontainers.image.revision`/`.source` labels on the same `Image` object's config. These are set by tooling other than an OpenShift `BuildConfig` (GitHub Actions' `docker/metadata-action`, `docker buildx build --label`, Cloud Native Buildpacks, `ko`, Jib, ...), so this extends detection to images built by any CI system that follows the OCI convention — not just in-cluster builds. There's no standard OCI label for the branch, so this tier only ever yields commit + repository, never `git_branch`.

If none of the above resolves anything (a prebuilt image with no CI-stamped provenance, no `.git` directory present, running a command with no visible `git clone`), this falls through to nothing. RBAC for the `Image` lookup (`aibom-postprocess-images` ClusterRole, bound per-namespace) is intentionally narrow — `get` on `images` only, nothing else in `image.openshift.io`.

Whichever source wins is recorded in `source_code.declared_via` (`"annotation"`, `"git_directory"`, `"cli_arg"`, `"openshift_build_label"`, or `"oci_image_label"`), mirroring `dataset.declared.declared_via`. `source_code.dirty`, when present, is surfaced regardless of which source won the identity fields above — it's orthogonal information about whether the executed code matched what's checked into git, only ever known when the `.git`-directory tier resolved.

**This is identification, not verification, and it isn't a security/compliance artifact.** The image-label tiers record what the build controller believes it cloned — not that nothing was injected into the image afterward (e.g. an extra `RUN` step after the `git clone`), and not that the source was vetted. Digest-keying only protects against a re-tagged image claiming someone else's labels; it does nothing about a workload that simply never went through a labeled build in the first place. The CLI-parsed tier is weaker still — it reports what the command *says* it will clone, not what actually landed on disk. Even the `.git`-directory tier, the strongest of the auto-detected sources, only reflects what's on disk at the moment `runtime_detector.py`'s `atexit` hook fires — a script could still delete or rewrite `.git` afterward, or never trigger the hook at all (a non-Python container, or one where `usercustomize.py` never gets imported). Reaching genuine tamper-evident provenance would require a signed attestation (e.g. Tekton Chains/SLSA, verifiable independent of trusting the live cluster) plus admission-time enforcement rejecting unsigned images — both out of scope here; this only records what's cheaply available and labels its source honestly.

`pip install git+...`'s PEP 610 `direct_url.json` isn't implemented.

## Prometheus Telemetry

`postprocess.py` queries Prometheus/Thanos Querier directly (`PROMETHEUS_URL`, defaulting to OpenShift's `https://thanos-querier.openshift-monitoring.svc:9091` — see `charts/aibom-webhook/values.yaml`'s `prometheus.url`) rather than going through Grafana. There's no datasource to discover and no separate credential to provision: auth and TLS trust are both automatic, mirroring gpu-quota-operator's `metrics.Client` (`metrics/prometheus.go`) exactly —

- **Auth**: the postprocess Job's own ServiceAccount token (`/var/run/secrets/kubernetes.io/serviceaccount/token`, auto-mounted and rotated in place by Kubernetes) is read fresh on every request and sent as a Bearer token. Authorization on the Thanos Querier side comes from a `cluster-monitoring-view` ClusterRoleBinding that `charts/aibom-workload-namespace/templates/monitoring.yaml` creates per opted-in namespace (`monitoringAccess.enabled`, default `true` — see `just setup-namespace`'s `skip_monitoring_access` flag for accounts without cluster-scoped RBAC permission).
- **TLS**: the cluster's service-serving CA bundle is injected into the `aibom-service-ca` ConfigMap (`service.beta.openshift.io/inject-cabundle` annotation, same chart) and mounted into the postprocess Job as *optional* — on a cluster with no service-ca operator, or a plain-HTTP dev Prometheus that doesn't need TLS at all (e.g. `mock-openshift-cluster`'s), the mount resolves to an empty directory and the client falls back to the system trust store rather than erroring.

Neither file being present is a hard requirement in either direction — a missing token means requests just go out unauthenticated, and a missing CA bundle means the system trust store is used instead. Both fall back gracefully rather than failing the Job.

**Grafana Explore links** (`resource_utilization.grafana_links`) are preserved as a purely presentational, optional add-on — independent of the actual telemetry queries above, which always go straight to `PROMETHEUS_URL`. Set `grafana.url`/`grafana.datasourceUid` (chart values, `GRAFANA_URL`/`GRAFANA_DATASOURCE_UID` env vars on the postprocess Job) to point at a Grafana instance with a Prometheus datasource for `PROMETHEUS_URL`; `build_grafana_explore_url` in `postprocess.py` builds a shareable deep link into it per pod. Leaving either unset just omits the link for that pod's telemetry — there's no auto-discovery of the datasource UID anymore (that required a Grafana API token, which direct Prometheus querying no longer needs).

`postprocess.py` queries this immediately after the workload's pod completes. On some observability backends (e.g. a federated/multi-tenant Prometheus setup) there's a delay between a metric being scraped and it becoming queryable, so a summary query fired this soon can race that delay and come back empty even though the identical query succeeds moments later — this shows up as `resource_utilization` averages being present on some AIBOMs and missing on others for no apparent reason.

A pod with no detected GPU (`gpu_count` 0 or missing in its discovery data, e.g. `nvidia-smi` found no hardware) is skipped for telemetry entirely — there's no GPU utilization to query, so it's omitted rather than querying anyway. On a mock cluster (e.g. kind) with no real GPU hardware, `nvidia-smi` always reports zero GPUs, so every pod gets skipped and telemetry collection never runs at all. `--debug-telemetry-all-pods` (`debug.telemetryAllPods` chart value, plumbed to the postprocess Job as `AIBOM_DEBUG_TELEMETRY_ALL_PODS`) bypasses this check so telemetry is queried for every pod regardless of detected GPU count — for local testing only, not routine production use.

## Deploy Versioning (`--version`)

`--version` is the single source of truth for what `just deploy` builds and how it's labeled: it sets the BuildConfig output ImageStreamTag, the Deployment's image reference, and (unless it's `latest`) `build.gitRef` — so the BuildConfig actually checks out and builds that exact commit, instead of always building whatever `build.gitRef`'s branch currently points to regardless of the tag name.

Defaults to the short SHA `scripts/remote-build-sha.sh` resolves (the remote tip of `build.gitRepo`/`build.gitRef`, read straight out of `charts/aibom-webhook/values.yaml` rather than a second hardcoded copy) via `git ls-remote`, not local `HEAD` — local `HEAD` can be ahead of, behind, or diverged from the remote (e.g. unpushed commits), and pinning `build.gitRef` to the resolved SHA also closes a race where the branch tip could otherwise move between resolving the tag and the build actually cloning it.

Each build lands on its own ImageStreamTag instead of overwriting a shared one, so `just deploy --version=<older-sha>` rolls back by rebuilding that exact historical commit rather than relabeling whatever's currently on the branch. `--version=latest` opts back into the old behavior: a mutable tag that always tracks `build.gitRef`'s branch tip. The resolved value also lands on the Deployment as the `app.kubernetes.io/version` label: `oc get deployment aibom-webhook -o jsonpath='{.metadata.labels.app\.kubernetes\.io/version}'` tells you exactly what's running.

`just deploy` doesn't rely on the BuildConfig's `ConfigChange` trigger, which (per OpenShift's own docs) only fires automatically the *first* time a BuildConfig is created, never on later edits like a new output tag — relying on it would silently leave the Deployment pointing at a tag nothing had built yet (`ImagePullBackOff`) on every deploy after the first. On a brand-new namespace, `deploy` detects the first install (`helm status` finds no existing release) and waits on that auto-triggered build instead of also starting its own, to avoid doubling the build time on that first deploy. Either way, expect a brief `ErrImagePull`/`ImagePullBackOff` on the pod while the build catches up to the Deployment's image reference — it resolves on its own once the image is pushed.
