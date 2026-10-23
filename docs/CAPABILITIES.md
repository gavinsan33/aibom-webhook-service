# Detected Fields Reference

Every field the AIBOM pipeline can fill in, how each one is detected, and where that detection runs. Where a field has more than one possible source, they are listed in the order they win.

Three components do detection, in this order in the pipeline:

1. **`aibom-discovery` init container** (`generate_snapshot.py`) — hardware/software facts, captured once at pod startup.
2. **Runtime hooks** (`runtime_detector.py`, mounted as `sitecustomize.py`) — dataset/model/training objects observed live inside the app container's own Python process.
3. **Postprocess Job** (`postprocess.py`) — parses container CLI args, reconciles/merges everything above, resolves git provenance, and queries Prometheus for telemetry.

Any field can also be set directly via an `aibom.io/*` annotation, which overrides auto-detected values for nearly every field — see the [annotation table](https://github.com/gavinsan33/aibom-webhook-service/blob/main/README.md#aibom-annotations) in the README. Setting `aibom.io/model-name` to a name other than the detected one also drops the storage-derived `model.revision`/`base_model`/`size_bytes`, since those describe the detected model.

---

## Hardware & System (`generate_snapshot.py`)

Captured once per pod by the discovery init container into `discovery-<pod>.json`, HMAC-signed with a per-namespace key (see [Signing Keys](reference/signing.md)).

**Surfacing note**: `postprocess.py` copies only the rows marked ✅ *surfaced* into `environment.*`. Rows marked ⚠️ *captured only* are still written to `discovery-<pod>.json` but never reach the final `AIBOM`: that file lives in the per-workload data ConfigMap, which is deleted once the postprocess Job succeeds. They're left out because they are node-level tuning, point-in-time readings, or wrong from inside a container. Values of `N/A`/empty are omitted from the AIBOM.

**CPU / memory / kernel**

| Field | Detection | Surfaced? |
|---|---|---|
| CPU model | `/proc/cpuinfo` (`model name`) | ✅ `environment.cpu_model` |
| CPU count | `/proc/cpuinfo` processor count | ✅ `environment.cpu_cores` |
| Cores per socket / threads per core | `lscpu` | ✅ `environment.cpu` |
| Architecture | `uname -m` | ✅ `environment.cpu` |
| L3 cache size | `lscpu` | ✅ `environment.cpu` |
| L1d / L1i / L2 cache size | `lscpu` | ⚠️ captured only |
| Current / max / min clock frequency | `/sys/devices/system/cpu/cpu0/cpufreq/*` | ⚠️ captured only — `cpu0`'s idle floor and boost ceiling, `N/A` on many VMs |
| Total memory | `/proc/meminfo` (`MemTotal`) | ✅ `environment.memory_gb` |
| Available / free memory | `/proc/meminfo` (`MemAvailable`/`MemFree`) | ⚠️ captured only — point-in-time reading |
| NUMA node count | `/sys/devices/system/node/node*` listing | ✅ `environment.numa_nodes` |
| Kernel version | `uname -r` | ✅ `environment.kernel_version` |
| Uptime | `/proc/uptime` | ⚠️ captured only — point-in-time reading |

**GPU**

| Field | Detection | Surfaced? |
|---|---|---|
| GPU count | `nvidia-smi --query-gpu=name` (line count) | ✅ `environment.gpu_count` |
| GPU model(s) | `nvidia-smi --query-gpu=name` | ✅ `environment.gpu_type` — only the **first line** of a multi-GPU-model listing; a mixed-model node's other GPU types are dropped |
| GPU memory per device (VRAM) | `nvidia-smi --query-gpu=memory.total` | ✅ `environment.gpu_memory_mb` (list, one per GPU) |
| GPU driver version | `nvidia-smi --query-gpu=driver_version` | ✅ `environment.driver_version` |
| CUDA version | `nvidia-smi` (`CUDA Version` line) | ✅ `environment.cuda_version` |

The discovery init container's own `nvidia.com/gpu` claim — what makes `nvidia-smi` work in it at all, since the NVIDIA Container Toolkit injects the binary and NVML driver library from the node's driver only when the claim is present — is not detected independently but copied from the pod's **total** `nvidia.com/gpu` allocation (the sum of each container's effective claim), so `nvidia-smi` sees every GPU the pod is scheduled with, not just the first container's.

**Network**

| Field | Detection | Surfaced? |
|---|---|---|
| RDMA devices / count | `/sys/class/infiniband` listing | ✅ `environment.network` |
| Primary interface MTU | `/sys/class/net/eth0/mtu` | ✅ `environment.network` |
| Interface names | `/sys/class/net` listing | ⚠️ captured only |
| TCP read/write memory buffers | `/proc/sys/net/ipv4/tcp_{rmem,wmem}` | ⚠️ captured only — node defaults |
| TCP congestion control algorithm | `/proc/sys/net/ipv4/tcp_congestion_control` | ⚠️ captured only — node default |

**Storage (hardware)** (Don't confuse this with the KServe `storage-<pod>.json` file described below, which *is* surfaced — this is disk/block-device hardware info.)

| Field | Detection | Surfaced? |
|---|---|---|
| Block devices + size | `lsblk -nd -o NAME,SIZE` (loop devices excluded) | ✅ `environment.storage` |
| NVMe device count | `ls /dev/nvme*` piped through `grep -c nvme0n` — counts only controller 0's namespaces and partitions, and `/dev` inside an unprivileged container usually has no host NVMe devices at all (#115) | ⚠️ captured only — unreliable |
| `/tmp` size / available space | `df -h /tmp` | ⚠️ captured only — overlay size, point-in-time free space |
| Active I/O scheduler | `/sys/block/sda/queue/scheduler` (hardcoded `sda`; empty on NVMe-only nodes) | ⚠️ captured only — unreliable |

**Kernel / cgroup performance config**

| Field | Detection | Surfaced? |
|---|---|---|
| CPU governor | `/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor` | ✅ `environment.kernel_config` |
| NUMA balancing | `/proc/sys/kernel/numa_balancing` | ✅ `environment.kernel_config` |
| Transparent hugepages | `/sys/kernel/mm/transparent_hugepage/enabled` | ✅ `environment.kernel_config` |
| Max map count | `/proc/sys/vm/max_map_count` | ✅ `environment.kernel_config` |
| Swappiness, dirty ratio / dirty background ratio, max open file handles (system-wide) | `/proc/sys/vm/*`, `/proc/sys/fs/file-max` | ⚠️ captured only — node tuning with no bearing on the run |
| Max user processes / open files / stack size / memory size | `ulimit -u/-n/-s/-m` | ⚠️ captured only — these are the discovery init container's limits, not the workload's |
| cgroup CPU quota / period | `/sys/fs/cgroup/cpu/cpu.cfs_{quota,period}_us` — cgroup v1 path only; always `N/A` on cgroup v2 nodes (OpenShift 4.14+) | ⚠️ captured only — unreliable |
| cgroup memory limit | `/sys/fs/cgroup/memory/memory.limit_in_bytes` — cgroup v1 path only; always `N/A` on cgroup v2 nodes | ⚠️ captured only — unreliable |

**Pod metadata** — ✅ surfaced into `execution_metadata.pods[]`: `pod_name`/`pod_uid`/`pod_namespace`/`pod_ip`/`node_name`/`start_time` (from downward-API env vars + a capture timestamp).

**InferenceService storage resolution** (KServe only) — ✅ surfaced. Reads `spec.predictor.model.storage.{path,key}` / `storageUri` off the pod's own `InferenceService` object (via its `INFERENCESERVICE_NAME` downward-API env var) and writes `storage-<pod>.json`; consumed by `postprocess.py`'s `detect_model_from_storage` to derive `model.name` (see below). Identification only — doesn't touch the actual bucket.

---

## Dataset Detection (`runtime_detector.py`, reconciled in `postprocess.py`)

**Runtime hooks** (installed into the app container's own Python process):

| Hook | Fires on | Captures |
|---|---|---|
| `torch.utils.data.DataLoader.__init__` | DataLoader construction | dataset class, root/path, split, transform name, dataset name, URLs, content fingerprint (hash of file names/sizes/mtimes), batch size |
| `datasets.load_dataset` (HuggingFace) | call time | dataset name (for file builders like `json`/`csv`/`parquet`, the `data_files` or `data_dir`, with the builder in `builder`), config name, split(s), revision, `data_dir`/`data_files`, fingerprint, version, license, description |
| torchvision datasets (19 known classes: MNIST family, CIFAR, ImageNet, ImageFolder, CelebA, LSUN, STL10, SVHN, VOC*, Coco*, Flickr*, Places365) | construction | dataset name (the most-derived known class: CIFAR100, not also CIFAR10), root, split, download flag, fingerprint |
| `webdataset.WebDataset.__init__` | construction | dataset name, URL list |

Dataset entries referring to the same underlying object are correlated (not double-recorded) first by **object identity**, then by **`(builder_name, config_name, split)`** for the common case of a transformed dataset (`.map`/`.filter`/`.select`/`.shuffle`, or a `DatasetDict` split pull) wrapped in a fresh `DataLoader`. If two loads share that key, a loader over a transformed copy gets its own entry rather than a guessed one. Each merged entry tracks a `seen_via` list of every hook that touched it. Entries from different processes and pods are deduped on name, source, config, split, revision, `data_files`, `data_dir` and root/path, so two local files or a train and test split stay separate.

**CLI-arg detection** (`postprocess.py`, from container command/args): `--dataset_name`, `--dataset_config_name` and `--dataset_train_split` (→ `dataset.declared.config_name`/`split`). A trl `--config` YAML isn't read; the runtime hook sees the load it causes.

**Reconciliation** (`postprocess.py`, `dataset.declared`) — precedence order:

1. `aibom.io/dataset-*` annotations → `declared_via: annotation`
2. CLI args above → `declared_via: cli_arg`
3. First runtime-detected entry → `declared_via: inferred_from_runtime`

Every `dataset.auto_detected[]` entry gets `matches_declared` — whether its name matches whichever source won above, ignoring case and a leading org or directory (`imdb` matches `stanfordnlp/imdb`). It's `null` when the declared name was inferred from runtime, since there's no declaration to check against.

---

## Model / Training Config Detection

**Runtime hooks** (`runtime_detector.py`):

| Hook | Captures |
|---|---|
| `transformers.TrainingArguments.__post_init__` | learning rate, batch size, epochs, optimizer, random seed, `training.mixed_precision` (`bf16`/`fp16` flags; not the weights' dtype) |
| `transformers.PreTrainedModel.from_pretrained` | model name, `device_map` and how many accelerators it resolved to, architecture (`config.architectures[0]`), dtype, quantization method (bitsandbytes/GPTQ/AWQ/HQQ/AQLM class-name or flag detection), quantization bit width. Only the process's **first** model load is recorded, so a DPO reference, reward or teacher model loaded afterwards doesn't replace the trained model; one loaded *before* it still would |
| `peft.LoraConfig.__init__` | LoRA rank, LoRA alpha, adaptation method (`dora`/`rslora`/`lora`; `lora` becomes `qlora` when the runtime hooks also saw a quantized base model, in either order) |

**CLI-arg detection** (`postprocess.py`) — parses each container's command/args, including through `sh -c "... && cmd"` wrapper-script flattening:

*vLLM* (`vllm serve` / `vllm.entrypoints.openai.api_server`):

| Flag | Field |
|---|---|
| `--model` | `model.name` (only the `--model` flag — the positional `vllm serve <model>` form isn't detected, #108) |
| `--dtype` | `model.dtype` |
| `--quantization` / `-q` | `model.quantization` |
| `--max-model-len` | `inference.max_model_len` |
| `--tensor-parallel-size` / `-tp` | `inference.tensor_parallel_size` |
| `--pipeline-parallel-size` / `-pp` | `inference.pipeline_parallel_size` |
| `--enable-expert-parallel` | `inference.enable_expert_parallel` |
| `--data-parallel-size` / `-dp` | `inference.data_parallel_size` |
| `--gpu-memory-utilization` | `inference.gpu_memory_utilization` |
| `--speculative-model`+`--num-speculative-tokens` (legacy), or `--speculative-config` (JSON/`key=value`) | `model.speculative_decoding` |
| `--override-generation-config` (JSON/`key=value`) | `inference.temperature`, `.top_p`, `.top_k` |

Only the kebab-case spellings above are recognized; vLLM also accepts snake_case (`--max_model_len`), which isn't detected (#108).

Also surfaced as `inference.served_model_name`, `max_num_seqs`, `seed`, `trust_remote_code`, `enforce_eager`, `enable_prefix_caching` (no annotation override). `--port` is parsed but not surfaced — a deployment detail.

*trl* (`trl sft`/`trl dpo`-style):

| Flag | Field |
|---|---|
| `--model_name_or_path` | `model.name` |
| `--use_peft` (+ `--lora_r`/`--use_dora`/`--use_rslora`/`--load_in_4bit`/`--load_in_8bit`) | `fine_tuning.adaptation_method` (`lora`/`qlora`/`dora`/`rslora`/`peft`) — these boolean flags are only recognized with an explicit value (`--use_peft true`); a bare `--use_peft` followed by another flag is ignored (#108) |
| `--lora_r` | `fine_tuning.lora_rank` |
| `--lora_alpha` | `fine_tuning.lora_alpha` |
| `--learning_rate` | `training.learning_rate` |
| `--per_device_train_batch_size` | `training.batch_size` |
| `--num_train_epochs` | `training.epochs` |
| `--seed` | `training.random_seed` |

*Other serving engines* — each identified from its own server command, setting `inference.serving_engine` and `model.name` (plus tensor-parallel size, max length, dtype, quantization where given). `model.name` also feeds quantization-from-name and the inference `experiment_intent`.

| Engine | Invocation | Model flag |
|---|---|---|
| TGI | `text-generation-launcher ...` (also the image's ENTRYPOINT, args-only) | `--model-id` |
| SGLang | `python -m sglang.launch_server ...` | `--model-path` |
| LMDeploy | `lmdeploy serve api_server <model> ...` | positional |

**Quantization from model name** (regex, applied as a fallback wherever no explicit flag/config gave a method) — recognizes `AWQ`, `GPTQ` (+ int4/int8 variants), `INT4`/`INT8`, `FP4`/`FP8` (incl. `NVFP4`/`MXFP4`), `bitsandbytes`/`NF4`, `Marlin`, `GGUF`/`GGML`, `AQLM`, `EXL2`, `SqueezeLLM`, `HQQ`, `QuIP`, `EETQ`, `AutoRound` — extracting both method and bit width where the name encodes one.

**KServe InferenceService storage model detection** — for predictors with no CLI model flag, `model.name` comes from the `InferenceService`'s declared storage, by scheme (then run through the same quantization-from-name regex). Recorded in `model.name_declared_via`:

| Declared storage | `model.name` | `model.revision` | `name_declared_via` |
|---|---|---|---|
| `pvc://` (pre-pulled model) | `repo_id` from the model dir's README, else the folder name | commit from `.cache/huggingface/download/*.metadata` | `model_files_readme` / `storage_path` |
| `hf://org/model[:rev]` | `org/model` | `rev`, if pinned | `hf_uri` |
| `oci://registry/repo[:tag]` (ModelCar) | `registry/repo` (tag dropped) | pulled image digest from `containers.json`, if the model container is in the pod spec | `oci_uri` |
| S3/MinIO `storage.path`, `s3://`, `gs://`, `https://`, `azure://` | last path segment (e.g. `models/tinyllama-1.1b-chat` → `tinyllama-1.1b-chat`) | — | `storage_path` |

Identification only — nothing verifies the weights. A renamed/generic path is misreported, an unpinned `hf://` is whatever `main` was at pull time, and bucket contents are never read.

**Parallelization strategy** (`training.parallelization_strategy`), independent of training tool:

- `accelerate launch --multi_gpu`, bare `torchrun`/`mpirun` → `data_parallel`
- bare `deepspeed`, or `--deepspeed <config>` → `deepspeed`
- bare `--fsdp` flag → `fsdp`
- `trl ... --num_processes N>1` (accelerate-launch args passed straight to a CLI that spawns `accelerate launch` internally) → `data_parallel`
- `trl ... --accelerate_config <path>` → read from the file's own `distributed_type` (`FSDP`→`fsdp`, `DEEPSPEED`→`deepspeed`, `MULTI_GPU`/`MULTI_CPU`→`data_parallel`); falls back to guessing from the filename against known presets (`fsdp1`/`fsdp2`/`zero1`/`zero2`/`zero3`/`multi_gpu`/`single_gpu`) if PyYAML is unavailable or `distributed_type` isn't recognized
- Lowest-priority fallback: `device_map` captured by the `from_pretrained` runtime hook → `model_parallel` when the model's resolved placement (`model.hf_device_map`) spans more than one accelerator, only if nothing above already produced a strategy. Single-device maps (`{"": 0}`, `0`, `cuda:1`, `auto` on one GPU) and cpu/disk offload don't count

---

## Git Provenance Detection

Covers both code baked into the image and code cloned at runtime, in precedence order:

| Source | What it reads | `declared_via` |
|---|---|---|
| `aibom.io/git-*` annotations | user-supplied | `annotation` |
| `.git`-directory read (`runtime_detector.py`) | walks up from the running script's directory, then the CWD, for `.git` (a directory, or a worktree/submodule `gitdir:` file), resolves `HEAD` (loose ref or `packed-refs`, must end in a full SHA), reads the `origin` URL (else the first remote) from `.git/config`, best-effort `git status --porcelain -uno` for a dirty-tree flag (untracked files don't count) | `git_directory` |
| CLI-parsed `git clone`/`checkout` (`postprocess.py`) | container command/args (through `sh -c` flattening): `git clone <url>` and any later `git checkout`/`switch` on that clone | `cli_arg` |
| OpenShift BuildConfig image labels | `image.openshift.io` `Image` object (looked up by image **digest**, not tag) — `io.openshift.build.commit.id`/`.ref`/`.source-location`; ignored if it disagrees with the OCI revision label | `openshift_build_label` |
| OCI-standard image labels | same `Image` object — `org.opencontainers.image.revision`/`.source` (no branch equivalent). With no `Image` object (any non-OpenShift cluster, or an image OpenShift never imported), both label tiers read the image config from the registry by digest instead: anonymous pulls only, every response checked against its digest, off with `gitProvenance.registryLookup=false` | `oci_image_label` |

A source missing a field (a `.git` with no remote) is filled from the next one down, and annotations are filled from detection, only when the two don't name a different commit or repository; any `aibom.io/git-*` annotation makes `declared_via` `annotation`. `source_code.dirty` is surfaced when the `.git`-directory tier's checkout is the one reported. This is identification, not tamper-evident provenance: the image-label tiers record what the build controller believed it cloned, the command tier what the command says it will clone, and the `.git` read only the checkout at the moment the process exits.

---

## Performance / Telemetry Metrics (Prometheus)

Queried directly against Prometheus/Thanos Querier (`PROMETHEUS_URL`) once the workload's pods complete, only for pods with a detected GPU (unless `AIBOM_DEBUG_TELEMETRY_ALL_PODS=true`). Missing metrics are retried (`AIBOM_TELEMETRY_RETRY_ATTEMPTS`, default `3`; `AIBOM_TELEMETRY_RETRY_DELAY_S`, default `45`s) to absorb scrape-to-queryable delay.

| Metric | Source | Unit |
|---|---|---|
| GPU utilization | `DCGM_FI_DEV_GPU_UTIL` | percent |
| GPU memory used | `DCGM_FI_DEV_FB_USED` | MiB |
| GPU power draw | `DCGM_FI_DEV_POWER_USAGE` | watts |
| CPU usage | `container_cpu_usage_seconds_total` (rate) | cores |
| Memory usage | `container_memory_working_set_bytes` | GiB (binary, like a Kubernetes `8Gi` limit) |
| Network receive throughput | `container_network_receive_bytes_total` (rate) | Mbps (decimal, 10⁶ bits/s) |
| Network transmit throughput | `container_network_transmit_bytes_total` (rate) | Mbps (decimal) |
| Storage read throughput | `container_fs_reads_bytes_total` (rate) | MBps (decimal, 10⁶ bytes/s) |
| Storage write throughput | `container_fs_writes_bytes_total` (rate) | MBps (decimal) |

Each query aggregates to one series per pod — summed across containers, GPUs and network interfaces, averaged for GPU utilization — and is scoped to the workload's namespace. CPU, memory and storage exclude the injected `aibom-dataset-sidecar` container.

In the AIBOM itself, each metric is recorded as summary statistics (`resource_utilization.metrics.<name>`: `min`/`max`/`avg`/`p95`, plus first/middle/last-third segment averages). A downsampled time series of each metric, about 200 points, is stored separately in an `AIBOMTelemetry` object referenced from `spec.data.telemetry_series_ref` (see [Telemetry](reference/schema.md#telemetry)). The series covers each pod's full run with no cold-start trim; the rest of this paragraph applies to the summary statistics. Each pod's window runs from its start time (parsed as UTC) to when its containers finished (`finished_at` in `containers.json`), or to collection time if it hasn't finished. The first scrape interval (30 s, `AIBOM_SERIES_SCRAPE_INTERVAL_S`) is excluded as cold start, capped at half the run length; a run too short for the full exclusion is flagged via `summary_includes_cold_start`. Stats are rounded once, after display scaling: 2 decimals at magnitude ≥ 1, 3 significant figures below that.

### Inference serving metrics (per engine)

For a pod whose `serving_engine` is detected (see *Other serving engines* above), the engine's own `/metrics` endpoint is scraped — via the `aibom-vllm-metrics` `PodMonitor` (`vllmMetrics.enabled`, default off) — and recorded under `inference.performance.metrics.<name>` with the same `min`/`max`/`avg`/`p95`/segment shape as above. All engines map to the same seven canonical fields; each engine just exposes a different subset under different upstream metric names:

| `inference.performance` field | vLLM / llm-d | TGI | SGLang | LMDeploy |
|---|:---:|:---:|:---:|:---:|
| `time_to_first_token_seconds` (TTFT) | ✅ | — | ✅ | ✅ |
| `inter_token_latency_seconds` (ITL) | ✅ | ✅ | ✅ | ✅ |
| `num_requests_running` | ✅ | ✅ | ✅ | ✅ |
| `num_requests_waiting` | ✅ | ✅ | ✅ | ✅ |
| `kv_cache_usage` (%) | ✅ | — | ✅ | ✅ |
| `prompt_throughput` (tok/s) | ✅ | — | ✅ | ✅ |
| `generation_throughput` (tok/s) | ✅ | ✅ | ✅ | ✅ |

**vLLM and llm-d are identical** (llm-d runs vLLM and is detected as `vllm`). **vLLM/llm-d, SGLang and LMDeploy record the same full set**; only the source metric names differ (e.g. `vllm:kv_cache_usage_perc` vs `sglang:token_usage`). **TGI is the outlier** — it exposes no TTFT, KV-cache or prompt-throughput metric, so those are simply absent. A metric an engine doesn't export is skipped, not reported null. SGLang and LMDeploy also need `--enable-metrics`; a workload that moves the engine off its default port (vLLM/llm-d 8000, TGI 80, SGLang 30000, LMDeploy 23333) isn't scraped (#86).

---

## What's Not Detected

These are explicit non-goals, not gaps waiting to close:

- Per-request sampling params (e.g. a benchmark client's temperature) set after a vLLM server is already running.
- TensorRT-LLM/Triton serving, or fine-tuning tools beyond trl (Axolotl, LLaMA-Factory). vLLM, TGI, SGLang, and LMDeploy *are* detected.
- `pip install git+...`'s PEP 610 `direct_url.json`.
- Verification that a declared git commit/dataset/model actually matches what's on disk or in a bucket — every "detection" above is identification, not an attestation.
