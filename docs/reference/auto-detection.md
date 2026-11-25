# Auto-Detection Flow

Each diagram shows how one subsystem of the pipeline decides what to record. [Detected Fields](../CAPABILITIES.md) lists the fields that come out, and [AIBOM Schema](schema.md) gives their shapes. The reasons and edge cases are in `IMPLEMENTATION_NOTES.md`, in the sections listed under each heading.

In the precedence chains (`A → B → C`), the first source that yields a value wins. An `aibom.io/*` annotation wins over detection for nearly every field.

## Overview

```mermaid
flowchart LR
    adm["1 · Admission"] --> disc["2 · Discovery"]
    adm --> rt["3 · Runtime hooks"]
    disc --> w["4 · Watcher"]
    rt --> w
    w --> md["5 · Model"]
    md --> int["6 · Intent"]
    int --> ds["7 · Dataset"]
    int --> sec["8 · Training, fine-tuning,<br/>inference"]
    w --> git["9 · Git"]
    w --> env["10 · Environment"]
    w --> tel["11 · Telemetry"]
    w --> sts["12 · Status"]
    ds --> out["13 · Output"]
    sec --> out
    git --> out
    env --> out
    tel --> out
    sts --> out
```

## 1 · Admission

The webhook decides which pods to instrument and what to inject into them. *Notes: Pod Matching and Injection; `aibom.io/postprocess-for` Trust Boundary.*

```mermaid
flowchart TD
    JA["Job CREATE"] --> JB{"requester == --trusted-watcher-identity?"}
    JB -- no --> JC["Strip aibom.io/postprocess-for<br/>from the Job and its pod template"]
    JB -- yes --> JD["Keep the label: a real postprocess Job"]

    A0["Pod CREATE in a namespace<br/>labeled aibom.io/enabled=true"] --> A1{"postprocess-for label<br/>+ batch/v1 Job owner<br/>+ job-controller identity, if configured?"}
    A1 -- yes --> A1X["Skip: postprocess pod"]
    A1 -- no --> A2{"Owned by a Job or PyTorchJob?<br/>(JobSet via its child Jobs;<br/>Ray pods only via a GPU)"}
    A2 -- yes --> AI
    A2 -- no --> A3{"Any container, init container or native<br/>sidecar claims a GPU resource?<br/>nvidia.com/gpu, gpu.shared, mig-*, amd.com/gpu<br/>(DRA resourceClaims not recognized)"}
    A3 -- yes --> AI
    A3 -- no --> A3X["Not instrumented: strip any workload-supplied<br/>aibom.io/instrumented(-by)"]

    AI["Label aibom.io/instrumented=true"] --> AJ["Inject the aibom-discovery init container<br/>GPU claim = pod peak total, shared replicas capped at 1;<br/>placed first if a GPU-holding native sidecar exists"]
    AI --> AK["Inject aibom-dataset-sidecar (native sidecar)"]
    AI --> AL["Mount /aibom-hooks, PYTHONPATH=/aibom-hooks<br/>(+ the aibom.io/python-path annotation,<br/>or prepended to an existing PYTHONPATH)"]
    AI --> AM{"KServe pod with pvc:// in<br/>storage-initializer-sourceuri?"}
    AM -- yes --> AM1["Mount the PVC subPath read-only at /mnt/aibom-model<br/>on discovery only (AIBOM_MODEL_DIR)"]
    AI --> AN{"Has batch.kubernetes.io/job-name?"}
    AN -- yes --> AN1["Add finalizer aibom.io/log-extraction-pod<br/>(held until postprocessed)"]
```

## 2 · Discovery init container

The discovery init container (`generate_snapshot.py`) records the hardware and the model's storage once, at pod startup. Each stage runs under a 60 s deadline, and a stage that fails or times out logs a warning instead of failing the pod. *Notes: Discovery Init Container Fails Open; AMD; Model Auto-Detection (KServe storage); Discovery Data Signing.*

```mermaid
flowchart TD
    D0["lscpu, /proc, sysfs, lsblk, ulimit, cgroup →<br/>system, storage, network, performance config"]
    D1{"nvidia-smi sees GPUs?"}
    D1 -- yes --> D1Y["gpu_count, gpu_models, VRAM per device,<br/>driver, cuda_version"]
    D1 -- no --> D2{"Allocated /dev/dri/renderD* nodes<br/>with sysfs vendor 0x1002?"}
    D2 -- yes --> D2Y["AMD, into the same fields: product_name or PCI id,<br/>mem_info_vram_total, amdgpu module version;<br/>XCP partitions = 'AMD GPU', VRAM N/A"]
    D2 -- no --> D2N["gpu_count 0"]
    D3{"INFERENCESERVICE_NAME set?<br/>(KServe-labeled pod)"}
    D3 -- yes --> D3Y["GET the InferenceService (workload RBAC):<br/>storage.key + path, or storageUri"]
    D4{"AIBOM_MODEL_DIR mounted?"}
    D4 -- yes --> D4Y["read_model_source_files (small regular files only):<br/>README front matter + H1 → repo_id, base_model<br/>(license_link only if its repo matches the H1)<br/>.cache/huggingface/download/*.metadata → revision<br/>config.json → architecture, dtype<br/>model.safetensors.index.json → size_bytes"]
    D5["HMAC-sign discovery-POD.json and storage-POD.json<br/>(aibom-discovery-hmac-key) → patch the data ConfigMap"]
    D0 --> D5
    D1Y --> D5
    D2Y --> D5
    D2N --> D5
    D3Y --> D5
    D4Y --> D5
```

## 3 · Runtime hooks

`runtime_detector.py` runs as `sitecustomize` in every Python process in the app container. *Notes: How the hooks get installed; Runtime Detection Output Durability; Determinism; DataLoader Sampling; Optimizer and Scheduler; Numerical Precision; Distributed Topology; Software Versions; Dataset Declaration and Reconciliation; Git Provenance Detection.*

```mermaid
flowchart TD
    R0{"Interpreter started with -I, -E or -S?"}
    R0 -- yes --> R0X["No hooks: the AIBOM uses CLI, discovery,<br/>watcher and Prometheus data only"]
    R0 -- no --> R1["Run the image's shadowed sitecustomize;<br/>install _PostImportHookFinder on sys.meta_path"]
    R1 --> R2{"Which trigger module just finished importing?<br/>(hooks read sys.modules, never import)"}
    R2 -- torch --> RH1["DataLoader + resolved sampler,<br/>torch.manual_seed, Optimizer.__init__ (first only),<br/>torch.autocast dtypes"]
    R2 -- "datasets / torchvision / webdataset" --> RH2["load_dataset, dataset constructors<br/>(most-derived torchvision class records)"]
    R2 -- transformers.training_args --> RH3["TrainingArguments.__post_init__:<br/>lr, batch, epochs, seed, bf16/fp16, adam_*,<br/>weight_decay, data_seed, fsdp, deepspeed"]
    R2 -- transformers.modeling_utils --> RH4["from_pretrained, first load wins:<br/>name, architecture, dtype, quantization config,<br/>device_map, model_device_count from hf_device_map"]
    R2 -- peft --> RH5["LoraConfig: lora / dora / rslora, r, alpha"]
    R2 -- "random / numpy.random" --> RH6["Seed hooks"]
    R2 -- "accelerate / deepspeed / fsdp /<br/>transformer_engine / torchao.float8" --> RH7["Accelerator, DeepSpeedEngine, FSDP (Mixed)Precision,<br/>fp8_autocast, convert_to_float8_training:<br/>precision + distributed, first per process"]
    R2 -- "megatron.core.parallel_state /<br/>device_mesh" --> RH8["TP / PP / CP / EP sizes, named mesh dims"]

    RH1 --> RS{"Seed set inside a DataLoader worker,<br/>or on RANK != 0?"}
    RH6 --> RS
    RS -- yes --> RSX["Dropped"]
    RS -- no --> RSY["seeds: latest value per generator<br/>+ history if reseeded"]

    RH1 --> RC1["Match to an existing dataset entry:<br/>1. same object<br/>2. builder + config + split + cache dir<br/>3. the sole load_dataset entry (save_to_disk reload)<br/>else a new entry; seen_via lists every hook"]
    RH2 --> RC1

    RF["_flush: debounced 2 s, atexit, SIGTERM<br/>(skipped if this process detected nothing)"]
    RH3 --> RF
    RH4 --> RF
    RH5 --> RF
    RH7 --> RF
    RH8 --> RF
    RSY --> RF
    RC1 --> RF
    RF --> RF1{"Training activity in this process?<br/>(seeded, DataLoader or TrainingArguments)"}
    RF1 -- yes --> RF2["Read now: cudnn / deterministic-algorithm flags,<br/>CUBLAS_WORKSPACE_CONFIG, TF32, float32_matmul_precision,<br/>WORLD_SIZE / GROUP_WORLD_SIZE / LOCAL_WORLD_SIZE"]
    RF --> RF7["argv: --accelerate_config / --config_file →<br/>distributed_type via PyYAML,<br/>else the preset file name (fsdp1, zero3, multi_gpu, ...)"]
    RF --> RF3["Git: walk up from the script dir, then CWD, for .git<br/>(following gitdir files); HEAD → commit, branch;<br/>origin (else first) remote URL, redacted;<br/>git status -uno → dirty, if git exists<br/>(only in a process that detected something)"]
    RF --> RF4["Software: tracked packages present in sys.modules,<br/>python, torch CUDA / HIP, cuDNN (no CUDA init), NCCL"]
    RF2 --> RF5
    RF7 --> RF5
    RF3 --> RF5
    RF4 --> RF5
    RF5["flock + merge into $AIBOM_DATASET_OUTPUT<br/>(git_* replaced as one group, packages unioned,<br/>datasets deduped on identity fields)"]
    RF5 --> RF6["Dataset sidecar polls the file → HMAC-sign<br/>(aibom-dataset-hmac-key) → patch the data ConfigMap"]
```

## 4 · Watcher

The watcher decides when to postprocess a workload and what to hand the postprocess Job. *Notes: Workload Selection and Grouping; Stale Finalizers; Job Pods Held Until Postprocessed; Postprocess Flow.*

```mermaid
flowchart TD
    W0{"Namespace opted out or being deleted?"}
    W0 -- yes --> W0X["Release our finalizers, no AIBOM"]
    W0 -- no --> W1{"Pod has batch.kubernetes.io/job-name?"}
    W1 -- "yes: Job path" --> WJ1{"Job Complete or Failed,<br/>or being deleted?"}
    WJ1 -- yes --> WJ2{"Itself a postprocess Job, or already<br/>annotated aibom.io/postprocess-job?"}
    WJ2 -- yes --> WSkip["Skip"]
    WJ2 -- no --> WQ
    W1 -- "no: bare pod path<br/>(KServe predictor, PyTorchJob worker)" --> WP1{"instrumented=true and the pod<br/>Succeeded / Failed or being deleted?"}
    WP1 -- yes --> WQ
    WQ{"Qualifies? Any pod claims a GPU,<br/>or any aibom.io/* annotation<br/>(internal labels excluded)"}
    WQ -- no --> WN["Release pod finalizers, no AIBOM"]
    WQ -- yes --> WG["Gather pods by Job name + UID;<br/>JobSet: add sibling pods, inherit sibling<br/>annotations if the trigger has none"]
    WG --> WV["Verify discovery / storage / dataset HMAC per pod<br/>(bad signature: drop; key unavailable: pass unverified)"]
    WV --> WM["mergeDatasets: dedupe datasets on identity;<br/>git_* group from the first pod that has it;<br/>other runtime_info keys first-pod-wins<br/>(completion index, then newest attempt)"]
    WG --> WC["containers.json from the live Pod API:<br/>image, image_id, gpu_limit, pod_gpu_count, cpu/mem limits,<br/>terminated reason + exit, restarts, finished_at,<br/>job_result, job_failure_reason, llm-d.ai/role"]
    WM --> WO
    WC --> WO
    WO["Inputs ConfigMap → postprocess Job in the webhook namespace"]
```

## 5 · Model

*Notes: Model Auto-Detection.*

### Container commands

```mermaid
flowchart TD
    M0["Per container: flatten command + args;<br/>expand sh/bash -c, drop comments + heredocs,<br/>split on && || ; | & newline; skip pip / uv / conda segments"]
    M0 --> M1{"vllm serve, or -m vllm.entrypoints...api_server?<br/>(vllm bench ignored)"}
    M1 -- no --> M2{"text-generation-launcher?"}
    M2 -- no --> M3{"python -m sglang.launch_server?"}
    M3 -- no --> M4{"lmdeploy serve api_server?"}
    M4 -- no --> M5{"trl SUBCOMMAND or -m trl?"}
    M5 -- no --> M6{"axolotl train or -m axolotl.cli.train?"}
    M6 -- no --> M7{"llamafactory-cli / lmf train?"}
    M7 -- no --> M8{"No command, args only, image is vllm-openai<br/>or llm-d xpu / cpu / rocm?"}
    M8 -- no --> M9{"No command, args only, TGI image?"}
    M9 -- no --> M10["Nothing from this container; try the next one"]
    M1 -- yes --> MS
    M2 -- yes --> MS
    M3 -- yes --> MS
    M4 -- yes --> MS
    M8 -- yes --> MS
    M9 -- yes --> MS
    M5 -- yes --> MT
    M6 -- yes --> MT
    M7 -- yes --> MT
    MS["serving_engine + model (--model wins over positional),<br/>dtype, quantization, max-model-len, TP / PP / DP / EP,<br/>gpu-memory-utilization, speculative, generation overrides"]
    MT["training_framework + model, LoRA / adapter, lr, batch,<br/>epochs, seed, optimizer, bf16/fp16, --dataset_name<br/>(config YAMLs are not read)"]
    MQ["Quantization from the model name when no flag set it:<br/>AWQ, GPTQ, INT4/8, FP4/8, NF4, GGUF Q4_K_M, w4a16, ...<br/>Unexpanded $VARS and malformed values dropped"]
    MS --> MQ
    MT --> MQ
    M0 --> MP["Every container, first value per key:<br/>--fsdp → fsdp; deepspeed / --deepspeed → deepspeed;<br/>accelerate preset config name; --multi_gpu, torchrun / mpirun<br/>with more than 1 process, --num_processes above 1 → data_parallel;<br/>--mixed_precision"]
```

### KServe storage

```mermaid
flowchart TD
    MST0{"storage-POD.json present?"}
    MST0 -- no --> MSTn["CLI result only"]
    MST0 -- yes --> MST1{"model_files.repo_id?"}
    MST1 -- yes --> MSTa["name = repo_id (model_files_readme)"]
    MST1 -- no --> MST2{"storageUri hf://org/model[:rev]?"}
    MST2 -- yes --> MSTb["name = org/model, revision = rev (hf_uri)"]
    MST2 -- no --> MST3{"storageUri oci://?"}
    MST3 -- yes --> MSTc["name = ref without tag, revision = digest of the<br/>container running that image (oci_uri)"]
    MST3 -- no --> MSTd["name = last path segment (storage_path)"]
    MSTa --> MSTM
    MSTb --> MSTM
    MSTc --> MSTM
    MSTd --> MSTM
    MSTM["Merge: the storage name overrides the CLI's /mnt/models;<br/>CLI dtype / architecture kept over config.json"]
```

### Field precedence

`detected` is the merged result of the two diagrams above, and `hook` is the `from_pretrained` or TrainingArguments runtime hook. If the `aibom.io/model-name` annotation names a different model, postprocess drops `revision`, `base_model`, and `size_bytes`.

```mermaid
flowchart LR
    n0[/"model.name"/] --> n1["annotation"] --> n2["detected"] --> n3["hook"]
    a0[/"architecture"/] --> a1["annotation"] --> a2["hook"] --> a3["detected"]
    d0[/"dtype"/] --> d1["annotation"] --> d2["detected<br/>(CLI over config.json)"] --> d3["hook"]
    q0[/"quantization"/] --> q1["annotation"] --> q2["CLI flag"] --> q3["name regex"] --> q4["hook"]
    f0[/"framework"/] --> f1["annotation"] --> f2["serving_engine"] --> f3["training_framework"] --> f4["hook"]
```

## 6 · Experiment intent

Intent decides which sections the AIBOM gets, so a wrong intent drops whole sections. *Notes: Experiment Intent Detection.*

```mermaid
flowchart TD
    I0{"aibom.io/experiment-intent set?"}
    I0 -- yes --> I0Y["Use it (annotation)"]
    I0 -- no --> I1{"CLI resolved a serving_engine?"}
    I1 -- yes --> I1Y["inference"]
    I1 -- no --> I2{"CLI resolved a training_framework?"}
    I2 -- yes --> I3{"CLI resolved an adaptation_method?"}
    I3 -- yes --> I3Y["sft"]
    I3 -- no --> I3N["training"]
    I2 -- no --> I2N["unknown, declared_via null<br/>(runtime hooks alone never set intent)"]
    IG["training / sft → training section<br/>sft → fine_tuning section<br/>inference → inference section<br/>dataset section if declared, detected, or training / sft"]
    I0Y --> IG
    I1Y --> IG
    I3Y --> IG
    I3N --> IG
    I2N --> IG
```

## 7 · Dataset

*Notes: Dataset Declaration and Reconciliation.*

```mermaid
flowchart TD
    S0{"aibom.io/dataset-name?"}
    S0 -- yes --> S0Y["declared via annotation"]
    S0 -- no --> S1{"--dataset_name on the first<br/>container that has it?"}
    S1 -- yes --> S1Y["declared via cli_arg<br/>(+ config_name, split)"]
    S1 -- no --> S2{"Any auto_detected entry?"}
    S2 -- yes --> S2Y["Copy the first entry<br/>(inferred_from_runtime)"]
    S2 -- no --> S2N["No declared name"]
    S3["Each auto_detected entry gets matches_declared:<br/>name match ignoring case and a leading org / dir;<br/>null when declared was inferred from runtime"]
    S0Y --> S3
    S1Y --> S3
    S2Y --> S3
```

## 8 · Training, fine-tuning, and inference

The AIBOM has each of these sections only for the intents named in its heading. *Notes: Determinism; Optimizer and Scheduler; Numerical Precision; Distributed Topology; Model Auto-Detection (parallelization, and the vLLM and trl flag tables).*

### Training (intent `training` or `sft`)

`hook` is the TrainingArguments runtime hook unless named otherwise.

```mermaid
flowchart LR
    s0[/"learning_rate, batch_size,<br/>epochs, random_seed"/] --> s1["annotation"] --> s2["hook"] --> s3["CLI"] --> s4["lr: constructed optimizer's lr<br/>seed: the seed every generator agrees on"]
    o0[/"optimizer"/] --> o1["annotation"] --> o2["hook"] --> o3["CLI"] --> o4["constructed optimizer class"]
    c0[/"optimizer_config<br/>(one source as a whole)"/] --> c1["constructed Optimizer"] --> c2["TrainingArguments<br/>adam_* / weight_decay"] --> c3["CLI"]
    m0[/"mixed_precision"/] --> m1["Accelerator"] --> m2["DeepSpeedEngine"] --> m3["FSDP param_dtype"] --> m4["TrainingArguments<br/>bf16 / fp16"] --> m5["FP8 hook"] --> m6["single-dtype autocast"] --> m7["ACCELERATE_MIXED_PRECISION"] --> m8["CLI"]
    p0[/"parallelization_strategy"/] --> p1["annotation"] --> p2["accelerate config<br/>distributed_type"] --> p3["CLI launcher / flags"] --> p4["device_map over 2+ devices<br/>= model_parallel"] --> p5["first non-data_parallel<br/>strategy in distributed"]
    w0[/"distributed world size"/] --> w1["env at flush"] --> w2["Accelerator"] --> w3["accelerate config"] --> w4["CLI"]
    b0[/"distributed FSDP /<br/>DeepSpeed blocks"/] --> b1["applied object"] --> b2["plugin"] --> b3["config"]
```

`determinism` (seeds and flags) and `precision` (TF32, autocast, FP8, FSDP, DeepSpeed) come only from the runtime hooks.

### Fine-tuning (intent `sft`)

```mermaid
flowchart LR
    a0[/"adaptation_method"/] --> a1["annotation"] --> a2["CLI: trl --use_peft + dora / rslora / 4-bit,<br/>axolotl --adapter,<br/>LLaMA-Factory --finetuning_type"] --> a3["peft hook<br/>(lora + quantization seen<br/>at runtime = qlora)"]
    r0[/"lora_rank, lora_alpha"/] --> r1["annotation"] --> r2["CLI"] --> r3["peft hook"]
```

### Inference (intent `inference`)

```mermaid
flowchart LR
    e0[/"serving_engine, max_model_len,<br/>TP / PP / DP, gpu_memory_utilization"/] --> e1["annotation"] --> e2["CLI"]
    t0[/"temperature, top_p, top_k"/] --> t1["annotation"] --> t2["CLI --override-generation-config"]
    sr0[/"serving_role"/] --> sr1["llm-d.ai/role, when every pod with a role agrees"]
```

Postprocess reads only the server's startup command, so it never sees sampling parameters that a client sends with each request.

## 9 · Git provenance

*Notes: Git Provenance Detection.*

```mermaid
flowchart TD
    G0["Runtime .git read (git_directory)"]
    G1["CLI git clone / checkout / switch, following cd and -C;<br/>several clones: the last one cd'd into, else the first (cli_arg)"]
    G2["merge_provenance: the lower tier fills missing fields<br/>only when commit and repository agree"]
    G0 --> G2
    G1 --> G2
    G2 --> G3{"Commit and repository both known?"}
    G3 -- no --> G4{"OpenShift Image object for the container's<br/>image_id has labels? (sidecars skipped)"}
    G4 -- no --> G5{"Registry lookup enabled? Anonymous token,<br/>manifest + config verified against the digest"}
    G5 -- yes --> G6
    G4 -- yes --> G6{"io.openshift.build.commit.* labels?<br/>(both label tiers ignored if they disagree)"}
    G6 -- yes --> G7["openshift_build_label: commit, ref, source"]
    G6 -- no --> G8{"org.opencontainers.image.revision / source?"}
    G8 -- yes --> G9["oci_image_label: commit, repository, no branch"]
    G7 --> G10["merge_provenance into the detected result"]
    G9 --> G10
    G3 -- yes --> G11
    G10 --> G11{"Any aibom.io/git-* annotation?"}
    G11 -- yes --> G12["Annotations win; detection fills their missing fields<br/>if it is the same checkout; declared_via = annotation"]
    G11 -- no --> G13["declared_via = the winning detection tier"]
    G12 --> G14
    G13 --> G14
    G14["redact_git_url on the winner;<br/>dirty kept only while the .git tier's checkout is reported"]
```

## 10 · Environment

*Notes: Discovery Data Signing (multi-pod `environment`, typed values); Which resources count as GPUs; Software Versions.*

```mermaid
flowchart TD
    E0["Pick the GPU pods running together at the busiest moment<br/>(discovery start_time → last container finished_at)"]
    E0 --> E1{"Pod's containers claim a GPU?"}
    E1 -- yes --> E1Y["Pod gpu_count = claim<br/>(pod_gpu_count, else the sum of gpu_limit)"]
    E1 -- no --> E1N["Pod gpu_count = discovery count"]
    E1Y --> E2["environment.gpu_count = sum over those pods"]
    E1N --> E2
    E0 --> E3{"DCGM_FI_DEV_FB_TOTAL for exported_pod,<br/>or AMD gpu_total_vram, in Prometheus?"}
    E3 -- yes --> E3Y["gpu_type, gpu_memory_mb from Prometheus"]
    E3 -- no --> E3N["From discovery nvidia-smi / AMD sysfs<br/>(VRAM list all or nothing)"]
    E0 --> E4["cpu, memory, numa, cuda, driver, kernel, rdma, block devices:<br/>the first GPU pod's discovery, converted to typed values"]
    E5["python, libraries, torch CUDA / HIP, cuDNN, NCCL:<br/>runtime hooks, even without discovery data"]
```

## 11 · Telemetry

*Notes: Prometheus Telemetry; Telemetry Retries; Segmented Performance Stats; Telemetry Time Series; Inference Performance Telemetry.*

```mermaid
flowchart TD
    P0{"PROMETHEUS_URL set?"}
    P0 -- no --> P0X["No resource_utilization, no series"]
    P0 -- yes --> P1{"Pod has a GPU? (claim, else discovery;<br/>--debug-telemetry-all-pods overrides)"}
    P1 -- no --> P1X["Skip this pod"]
    P1 -- yes --> P2["Range query per metric, start_time + cold start → last finished_at:<br/>GPU (DCGM or AMD exporter), CPU, memory, network, storage<br/>→ min / max / avg / p95 / thirds, + limit for CPU and memory"]
    P2 --> P3{"Metric came back empty?"}
    P3 -- "yes, attempts left" --> P3R["Wait 45 s, re-query only the missing metrics<br/>(3 attempts)"]
    P3R --> P2
    P2 --> P4{"serving_engine is vllm, tgi, sglang or lmdeploy?"}
    P4 -- yes --> P5["Engine /metrics via PodMonitor ports:<br/>TTFT, ITL, running / waiting, KV cache, throughput<br/>→ inference.performance (TGI: no TTFT or KV cache)"]
    P2 --> P6
    P5 --> P6
    P6["Series: one query per metric over all pods, about 200 points;<br/>fit under 900 KB (coarsen first, then drop per-pod series)<br/>→ AIBOMTelemetry + sha256 ref in spec.data"]
```

## 12 · Execution status

*Notes: Pod Termination Status.*

```mermaid
flowchart TD
    X0["Per pod, from containers.json: OOMKilled → other non-zero exit<br/>→ Completed; null when no status was captured"]
    X0 --> X1{"Its Job is Complete but the pod isn't Completed?"}
    X1 -- yes --> X1Y["retried_attempt, left out of the rollup"]
    X1 -- no --> X2["Workload: OOMKilled → first other non-Completed → Completed<br/>→ Job Complete = Completed → Job Failed = Failed<br/>→ DeadlineExceeded unless OOMKilled"]
```

## 13 · Output

*Notes: Compiled AIBOM Signing; Telemetry Time Series.*

```mermaid
flowchart LR
    O1["compile_aibom → spec.data"] --> O2["Ed25519 over JCS(spec.data),<br/>if the key is mounted"] --> O3["Create the AIBOM CR"] --> O4["Set its ownerReference<br/>on the AIBOMTelemetry"]
```
