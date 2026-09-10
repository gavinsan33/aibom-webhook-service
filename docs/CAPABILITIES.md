# Detected Fields Reference

Every field the AIBOM pipeline can populate, how it's detected, and where the detection happens. This is the field-level companion to `CLAUDE.md` (which covers precedence rules and edge cases in more depth) — read that first for the "why," this is the "what."

Three components do detection, in this order in the pipeline:

1. **`aibom-discovery` init container** (`generate_snapshot.py`) — hardware/software facts + benchmarks, captured once at pod startup.
2. **Runtime hooks** (`runtime_detector.py`, mounted as `usercustomize.py`) — dataset/model/training objects observed live inside the app container's own Python process.
3. **Postprocess Job** (`postprocess.py`) — parses container CLI args, reconciles/merges everything above, resolves git provenance, and queries Prometheus for telemetry.

Any field can also be set directly via an `aibom.io/*` annotation, which always overrides auto-detected values — see the [annotation table](../README.md#aibom-annotations) in the README.

---

## Hardware & System (`generate_snapshot.py`)

Captured once per pod by the discovery init container into `discovery-<pod>.json`, HMAC-signed (see `CLAUDE.md`).

**⚠️ Surfacing note**: `postprocess.py` only ever reads a small, fixed subset of this file — `gpu.gpu_models`/`gpu_count`/`cuda_version`/`gpu_driver_version` and `system.cpu_model`/`cpu_count`/`memory_total_gb`/`numa_node_count`/`kernel_version` — into `environment.*`. Everything else below is genuinely captured (it's in `discovery-<pod>.json`) but is **never read by any downstream code**, so it exists only in the per-workload data ConfigMap, which is deleted once the postprocess Job succeeds (see `CLAUDE.md`'s Postprocess Flow) — it never reaches the final `AIBOM` custom resource. Rows are marked ✅ *surfaced* (→ `environment.<field>`) or ⚠️ *captured only* accordingly.

**CPU / memory / kernel**

| Field | Detection | Surfaced? |
|---|---|---|
| CPU model | `/proc/cpuinfo` (`model name`) | ✅ `environment.cpu_model` |
| CPU count | `/proc/cpuinfo` processor count | ✅ `environment.cpu_cores` |
| Cores per socket / threads per core | `lscpu` | ⚠️ captured only |
| Architecture | `uname -m` | ⚠️ captured only |
| Current / max / min clock frequency | `/sys/devices/system/cpu/cpu0/cpufreq/*` | ⚠️ captured only |
| L1d / L1i / L2 / L3 cache size | `lscpu` | ⚠️ captured only |
| Total memory | `/proc/meminfo` (`MemTotal`) | ✅ `environment.memory_gb` |
| Available / free memory | `/proc/meminfo` (`MemAvailable`/`MemFree`) | ⚠️ captured only |
| NUMA node count | `/sys/devices/system/node/node*` listing | ✅ `environment.numa_nodes` |
| Kernel version | `uname -r` | ✅ `environment.kernel_version` |
| Uptime | `/proc/uptime` | ⚠️ captured only |

**GPU**

| Field | Detection | Surfaced? |
|---|---|---|
| GPU count | `nvidia-smi --query-gpu=name` (line count) | ✅ `environment.gpu_count` |
| GPU model(s) | `nvidia-smi --query-gpu=name` | ✅ `environment.gpu_type` — only the **first line** of a multi-GPU-model listing; a mixed-model node's other GPU types are dropped |
| GPU memory per device (VRAM) | `nvidia-smi --query-gpu=memory.total` | ⚠️ captured only |
| GPU driver version | `nvidia-smi --query-gpu=driver_version` | ✅ `environment.driver_version` |
| CUDA version | `nvidia-smi` (`CUDA Version` line) | ✅ `environment.cuda_version` |

The GPU resource request itself (used to decide whether to run this detection at all) is copied from the pod's own `nvidia.com/gpu` container resource request, not detected independently.

**Network** — ⚠️ all fields below are captured but never surfaced into the compiled AIBOM:

| Field | Detection |
|---|---|
| Interface names | `/sys/class/net` listing |
| RDMA devices / count | `/sys/class/infiniband` listing |
| Primary interface MTU | `/sys/class/net/eth0/mtu` |
| TCP read/write memory buffers | `/proc/sys/net/ipv4/tcp_{rmem,wmem}` |
| TCP congestion control algorithm | `/proc/sys/net/ipv4/tcp_congestion_control` |

**Storage (hardware)** — ⚠️ all fields below are captured but never surfaced into the compiled AIBOM. (Don't confuse this with the KServe `storage-<pod>.json` file described below, which *is* surfaced — this is disk/block-device hardware info.)

| Field | Detection |
|---|---|
| Block devices + size | `lsblk -nd -o NAME,SIZE` (loop devices excluded) |
| NVMe device count | `/dev/nvme*n` listing |
| `/tmp` size / available space | `df -h /tmp` |
| Active I/O scheduler | `/sys/block/sda/queue/scheduler` |

**Kernel / cgroup performance config** — ⚠️ all fields below are captured but never surfaced into the compiled AIBOM:

| Field | Detection |
|---|---|
| CPU governor | `/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor` |
| NUMA balancing | `/proc/sys/kernel/numa_balancing` |
| Transparent hugepages | `/sys/kernel/mm/transparent_hugepage/enabled` |
| Swappiness | `/proc/sys/vm/swappiness` |
| Dirty ratio / dirty background ratio | `/proc/sys/vm/dirty_ratio`, `dirty_background_ratio` |
| Max map count | `/proc/sys/vm/max_map_count` |
| Max open file handles (system-wide) | `/proc/sys/fs/file-max` |
| Max user processes / open files / stack size / memory size (this container) | `ulimit -u/-n/-s/-m` |
| cgroup CPU quota / period | `/sys/fs/cgroup/cpu/cpu.cfs_{quota,period}_us` |
| cgroup memory limit | `/sys/fs/cgroup/memory/memory.limit_in_bytes` |

**Pod metadata** — ✅ surfaced into `execution_metadata.pods[]`: `pod_name`/`pod_uid`/`pod_namespace`/`pod_ip`/`node_name`/`start_time` (from downward-API env vars + a capture timestamp).

**Benchmarks** — actually executed, not just read from `/proc`/`/sys` — ⚠️ **all four are captured but never surfaced into the compiled AIBOM** (no `aibom["benchmarks"]` or similar section exists in `postprocess.py`):

| Benchmark | Measures | Method |
|---|---|---|
| CPU compute | MFLOPS | tight loop of 10M floating-point ops, timed |
| Memory bandwidth | write/read MB/s | write then read a 100MB buffer in 8-byte strides, timed |
| Disk I/O | write/read throughput MB/s | write/read 50×1MB blocks to a temp file with `fsync`, timed |
| Context switch overhead | avg ms per process spawn | spawn a trivial subprocess 100×, timed |

**InferenceService storage resolution** (KServe only) — ✅ surfaced. Reads `spec.predictor.model.storage.{path,key}` / `storageUri` off the pod's own `InferenceService` object (via its `INFERENCESERVICE_NAME` downward-API env var) and writes `storage-<pod>.json`; consumed by `postprocess.py`'s `detect_model_from_storage` to derive `model.name` (see below). Identification only — doesn't touch the actual bucket.

---

## Dataset Detection (`runtime_detector.py`, reconciled in `postprocess.py`)

**Runtime hooks** (installed into the app container's own Python process):

| Hook | Fires on | Captures |
|---|---|---|
| `torch.utils.data.DataLoader.__init__` | DataLoader construction | dataset class, root/path, split, transform name, dataset name, URLs, content fingerprint (hash of file names/sizes/mtimes), batch size |
| `datasets.load_dataset` (HuggingFace) | call time | dataset name, config name, split(s), revision, `data_dir`/`data_files`, fingerprint, version, license, description |
| torchvision datasets (19 known classes: MNIST family, CIFAR, ImageNet, ImageFolder, CelebA, LSUN, STL10, SVHN, VOC*, Coco*, Flickr*, Places365) | construction | dataset name, root, split, download flag, fingerprint |
| `webdataset.WebDataset.__init__` | construction | dataset name, URL list |

Dataset entries referring to the same underlying object are correlated (not double-recorded) first by **object identity**, then by **`(builder_name, config_name)`** for the common case of a transformed dataset (`.map`/`.filter`/`.select`/`.shuffle`, or a `DatasetDict` split pull) wrapped in a fresh `DataLoader`. Each merged entry tracks a `seen_via` list of every hook that touched it.

**CLI-arg detection** (`postprocess.py`, from container command/args): `--dataset_name`, `--dataset_config_name`, `--dataset_train_split`.

**Reconciliation** (`postprocess.py`, `dataset.declared`) — precedence order:

1. `aibom.io/dataset-*` annotations → `declared_via: annotation`
2. CLI args above → `declared_via: cli_arg`
3. First runtime-detected entry → `declared_via: inferred_from_runtime`

Every `dataset.auto_detected[]` entry gets `matches_declared` — whether its name matches whichever source won above.

---

## Model / Training Config Detection

**Runtime hooks** (`runtime_detector.py`):

| Hook | Captures |
|---|---|
| `transformers.TrainingArguments.__init__` | learning rate, batch size, epochs, optimizer, random seed, dtype (`bf16`/`fp16` flags) |
| `transformers.PreTrainedModel.from_pretrained` | model name, `device_map`, architecture (`config.architectures[0]`), dtype, quantization method (bitsandbytes/GPTQ/AWQ/HQQ/AQLM class-name or flag detection), quantization bit width |
| `peft.LoraConfig.__init__` | LoRA rank, LoRA alpha, adaptation method (`dora`/`rslora`/`qlora`/`lora`, from `use_dora`/`use_rslora`/quantization presence) |

**CLI-arg detection** (`postprocess.py`) — parses each container's command/args, including through `sh -c "... && cmd"` wrapper-script flattening:

*vLLM* (`vllm serve` / `vllm.entrypoints.openai.api_server`):

| Flag | Field |
|---|---|
| `--model` | `model.name` |
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

⚠️ Also parsed but **never surfaced** into the compiled AIBOM (dropped after the intermediate detection dict): `--served-model-name`, `--max-num-seqs`, `--seed`, `--trust-remote-code`, `--enforce-eager`, `--enable-prefix-caching`, `--port`.

*trl* (`trl sft`/`trl dpo`-style):

| Flag | Field |
|---|---|
| `--model_name_or_path` | `model.name` |
| `--use_peft` (+ `--lora_r`/`--use_dora`/`--use_rslora`/`--load_in_4bit`/`--load_in_8bit`) | `fine_tuning.adaptation_method` (`lora`/`qlora`/`dora`/`rslora`/`peft`) |
| `--lora_r` | `fine_tuning.lora_rank` |
| `--lora_alpha` | `fine_tuning.lora_alpha` |
| `--learning_rate` | `training.learning_rate` |
| `--per_device_train_batch_size` | `training.batch_size` |
| `--num_train_epochs` | `training.epochs` |
| `--seed` | `training.random_seed` |

**Quantization from model name** (regex, applied as a fallback wherever no explicit flag/config gave a method) — recognizes `AWQ`, `GPTQ` (+ int4/int8 variants), `INT4`/`INT8`, `FP4`/`FP8` (incl. `NVFP4`/`MXFP4`), `bitsandbytes`/`NF4`, `Marlin`, `GGUF`/`GGML`, `AQLM`, `EXL2`, `SqueezeLLM`, `HQQ`, `QuIP`, `EETQ`, `AutoRound` — extracting both method and bit width where the name encodes one.

**KServe InferenceService storage-path model detection** — for predictors with no CLI model flag, derives `model.name` as the last path segment of the `InferenceService`'s declared S3/MinIO storage location (e.g. `models/tinyllama-1.1b-chat` → `tinyllama-1.1b-chat`), then runs it through the same quantization-from-name regex. Identification only — a renamed/generic bucket path is misreported, and bucket contents are never read.

**Parallelization strategy** (`training.parallelization_strategy`), independent of training tool:

- `accelerate launch --multi_gpu`, bare `torchrun`/`mpirun` → `data_parallel`
- bare `deepspeed`, or `--deepspeed <config>` → `deepspeed`
- bare `--fsdp` flag → `fsdp`
- `trl ... --num_processes N>1` (accelerate-launch args passed straight to a CLI that spawns `accelerate launch` internally) → `data_parallel`
- `trl ... --accelerate_config <path>` → read from the file's own `distributed_type` (`FSDP`→`fsdp`, `DEEPSPEED`→`deepspeed`, `MULTI_GPU`/`MULTI_CPU`→`data_parallel`); falls back to guessing from the filename against known presets (`fsdp1`/`fsdp2`/`zero1`/`zero2`/`zero3`/`multi_gpu`/`single_gpu`) if PyYAML is unavailable or `distributed_type` isn't recognized
- Lowest-priority fallback: `device_map` captured by the `from_pretrained` runtime hook — any multi-device value → `model_parallel`, only if nothing above already produced a strategy

---

## Git Provenance Detection

Covers both code baked into the image and code cloned at runtime, in precedence order:

| Source | What it reads | `declared_via` |
|---|---|---|
| `aibom.io/git-*` annotations | user-supplied | `annotation` |
| `.git`-directory read (`runtime_detector.py`) | walks up from CWD for `.git`, resolves `HEAD` (loose ref or `packed-refs`), reads `origin` URL from `.git/config`, best-effort `git status --porcelain` for a dirty-tree flag | `git_directory` |
| CLI-parsed `git clone`/`checkout` (`postprocess.py`) | regex-scans container command/args (through `sh -c` flattening) for `git clone <url>` + optional `git checkout <ref>`/`--branch <ref>`; a 7–40 char hex token is treated as a commit, else a branch | `cli_arg` |
| OpenShift BuildConfig image labels | `image.openshift.io` `Image` object (looked up by image **digest**, not tag) — `io.openshift.build.commit.id`/`.ref`/`.source-location` | `openshift_build_label` |
| OCI-standard image labels | same `Image` object — `org.opencontainers.image.revision`/`.source` (no branch equivalent) | `oci_image_label` |

`source_code.dirty` is surfaced independently whenever the `.git`-directory tier resolved it, regardless of which tier won identity. This is identification, not tamper-evident provenance — see `CLAUDE.md` for the full caveats.

---

## Performance / Telemetry Metrics (Prometheus)

Queried directly against Prometheus/Thanos Querier (`PROMETHEUS_URL`) once the workload's pods complete, only for pods with a detected GPU (unless `AIBOM_DEBUG_TELEMETRY_ALL_PODS=true`). Missing metrics are retried (`AIBOM_TELEMETRY_RETRY_ATTEMPTS`, default `3`; `AIBOM_TELEMETRY_RETRY_DELAY_S`, default `45`s) to absorb scrape-to-queryable delay — see `CLAUDE.md`.

| Metric | Source | Unit |
|---|---|---|
| GPU utilization | `dcgm_gpu_util` | percent |
| GPU memory used | `dcgm_fb_used` | MiB |
| GPU power draw | `dcgm_power_usage` | watts |
| CPU usage | `container_cpu_usage_seconds_total` (rate) | cores |
| Memory usage | `container_memory_working_set_bytes` | GB |
| Network receive throughput | `container_network_receive_bytes_total` (rate) | Mbps |
| Network transmit throughput | `container_network_transmit_bytes_total` (rate) | Mbps |
| Storage read throughput | `container_fs_reads_bytes_total` (rate) | MB/s |
| Storage write throughput | `container_fs_writes_bytes_total` (rate) | MB/s |

Each metric is recorded as summary statistics only (`resource_utilization.metrics.<name>`: `min`/`max`/`avg`/`p95`, plus first/middle/last-third segment averages) — not a raw time series. A short-lived run that couldn't fully exclude the first-scrape-interval cold-start window is flagged via `summary_includes_cold_start`.

---

## What's Not Detected

Per `CLAUDE.md`, these are explicit non-goals, not gaps waiting to close:

- Per-request sampling params (e.g. a benchmark client's temperature) set after a vLLM server is already running.
- Other serving engines (TGI, SGLang, TensorRT-LLM) or fine-tuning tools (Axolotl, LLaMA-Factory).
- `pip install git+...`'s PEP 610 `direct_url.json`.
- Verification that a declared git commit/dataset/model actually matches what's on disk or in a bucket — every "detection" above is identification, not an attestation.
