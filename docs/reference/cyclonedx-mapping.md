# CycloneDX mapping

How an `AIBOM` record (`aiboms.aibom.io`, `spec.data` as `postprocess.py` writes it) is expressed as a CycloneDX 1.6 document, so a run record can be read by the tools that already read software and machine-learning BOMs (Dependency-Track, GUAC, Lightwell) and linked to the runtime inventory that [k8s-aibom](https://github.com/GoogleCloudPlatform/k8s-aibom) writes for the same workload. This is a serialization of the record, not a replacement for it: `spec.data` stays the signed source and the CycloneDX form is derived from it.

The field list follows [Detected Fields](../CAPABILITIES.md) and the record structure follows `compile_aibom` in `postprocess.py`, checked against two real records (a DeepSpeed ZeRO-3 LoRA fine-tune and a vLLM serving run, both aibom-webhook 0.1.0). Rows marked *not seen in the records* come from Detected Fields alone and are kept so the table stays complete; their rule is the same as the neighbouring rows and has not been exercised against a real value. Every `spec.data` path is listed once with its rule: **direct** (copied into a native CycloneDX field), **transformed** (native field after a conversion stated in the row), **property** (a `properties[]` entry under the `aibomrun:` namespace, see [Property namespace](#property-namespace)), or **dropped**, with the reason.

Two principles carry over from the record itself. Every value is identification, not attestation, so nothing in the mapping claims more than the record does; the `*_declared_via` fields travel with the facts they qualify. And the mapping never invents a value: a field the record does not have produces no CycloneDX element.

## Document identity and lifecycle

| `spec.data` path | CycloneDX 1.6 | Rule |
|---|---|---|
| (the record) | `bomFormat: "CycloneDX"`, `specVersion: "1.6"`, `version: 1` | fixed |
| (the record) | `serialNumber: urn:uuid:<v5 UUID>` | transformed: UUID v5 of `<namespace>/<AIBOM object name>` in a fixed per-project namespace, so the same record always gets the same serial and a BOM-Link to it is stable (see [Linking](#linking-to-k8s-aibom)) |
| `_metadata.generated_at` | `metadata.timestamp` | direct (RFC 3339) |
| `experiment_intent` | `metadata.lifecycles[0].phase`: `build` for `training` and `sft`, `operations` for `inference` | transformed. CycloneDX separates as-trained from as-deployed by lifecycle, not by document kind; a training run is a build of the model, a serving run is operations. `unknown` emits no lifecycle |
| `experiment_intent`, `experiment_intent_declared_via` | property `aibomrun:experiment:intent`, `aibomrun:experiment:intent:declaredVia` on `metadata.component` | property: the raw value is kept beside the lifecycle it was mapped to |
| `experiment_name` | `metadata.component.name` | direct |
| `experiment_description` | `metadata.component.description` | direct |
| `_metadata.generator`, `_metadata.aibom_version` | `metadata.tools.components[0]`: `type: application`, `name: aibom-webhook`, `version` from `aibom_version` | direct |
| `_metadata.schema_compliance`, `_metadata.dataset_detection` | dropped | notes about the record format, not about the run |
| `execution_metadata.job_id`, `.namespace` | `metadata.component.group` = namespace, property `aibomrun:job:name` | the run is the root component (below); the Job name is kept as a property because `name` carries `experiment_name`, which defaults to the Job name but can differ |
| `execution_metadata.status`, `.duration_seconds` | properties `aibomrun:job:status`, `aibomrun:job:durationSeconds` on `metadata.component` | property. The serving record's status is `DeadlineExceeded` while its one pod reports `Completed`: the Job's `activeDeadlineSeconds` ended a benchmark server that would otherwise run forever, which is the record's own distinction between pod status and workload status, and the mapping keeps both |
| `execution_metadata.pods[]` | per pod: properties `aibomrun:pod:<i>:name`, `:uid`, `:namespace`, `:node`, `:startTime`, `:status`, `:exitCode` on `metadata.component`; nodes become `device` components, see [Hardware](#hardware-as-device-components) | property, indexed in record order. `start_time` is when the discovery init container ran, in that container's local time with no offset (`2026-11-03T16:29:57.639439`); postprocess treats it as UTC, which holds for the default container time zone, so the mapping appends `Z`. `pod_ip` is dropped: ephemeral and not useful outside the cluster |
| `execution_metadata.pods[].restart_count`, `.last_termination_reason`, `.last_exit_code`, `.serving_role`, `.retried_attempt` | properties `aibomrun:pod:<i>:restartCount`, `:lastTerminationReason`, `:lastExitCode`, `:servingRole`, `:retriedAttempt` | property; *not seen in the records* |

## The run as model component and modelCard

The root of the document, `metadata.component`, is the run. Its type depends on intent: a training or fine-tuning run is a `machine-learning-model` component with a `modelCard`, because the run produced a model (an adapter, or new weights). A serving run is an `application` component whose served model is a `machine-learning-model` in `components[]`; the served weights were not produced by this run, so the modelCard sits on the served model, not on the run. In both cases the model facts below land on that `machine-learning-model` component.

| `spec.data` path | CycloneDX 1.6 | Rule |
|---|---|---|
| `model.name` | model component `name`; `purl: pkg:generic/<name>` when the name is a Hugging Face `org/model` id or a plain name, `pkg:oci/<repo>@sha256:<revision>` when `name_declared_via` is `oci_uri` and `revision` is a digest | transformed. `pkg:huggingface` exists in the purl types list but has no stable version rule across tools; `pkg:generic` with `?download_url=https://huggingface.co/<name>` as a qualifier is readable everywhere. Open question 1 |
| `model.name_declared_via` | property `aibomrun:model:name:declaredVia` | property; *not seen in the records* |
| `model.version` | model component `version` | direct |
| `model.revision` | property `aibomrun:model:revision`; also the purl version when it is a commit or digest | transformed; *not seen in the records* |
| `model.base_model` | `pedigree.ancestors[0]`: a `machine-learning-model` component with that name | transformed: the base model is what this one was derived from, which is exactly what pedigree ancestors mean; *not seen in the records* |
| `model.size_bytes` | property `aibomrun:model:sizeBytes` | property; *not seen in the records* |
| `model.architecture` | `modelCard.modelParameters.modelArchitecture` | direct |
| `model.framework` | property `aibomrun:model:framework` | property. CycloneDX has no model-framework field; for serving runs the same value is the runtime, see [Software](#software) |
| `model.dtype` | property `aibomrun:model:dtype` | property |
| `model.quantization`, `model.quantization_bits` | properties `aibomrun:model:quantization`, `aibomrun:model:quantization:bits` | property |
| `model.speculative_decoding` (`method`, `num_speculative_tokens`, `prompt_lookup_max` in the serving record) | property `aibomrun:model:speculativeDecoding` (JSON string) | property; structured value kept whole rather than flattened, since its keys follow the engine's own config (`ngram` with `prompt_lookup_max` in the record; a draft-model config would carry different keys) |
| `experiment_intent` (training, sft) | `modelCard.modelParameters.approach.type: supervised` for `sft`; for `training` no approach is emitted | transformed. `sft` is supervised by definition; a plain `training` intent does not say |
| `experiment_intent` (training, sft) | `modelCard.modelParameters.task: "fine-tuning"` for `sft`, `"training"` for `training` | transformed |
| `dataset.*` | `modelCard.modelParameters.datasets[]` as `{ "ref": "<bom-ref of the data component>" }` | transformed; the data components themselves are in [Datasets](#datasets) |
| `resource_utilization.metrics.*`, `inference.performance.metrics.*` | `modelCard.quantitativeAnalysis.performanceMetrics[]` | transformed, see [Telemetry](#telemetry) |

## Hardware as device components

The record describes the first discovered pod's environment and treats it as the run's. The mapping keeps that: one GPU `device` component per GPU of the first pod, one `device` per distinct node name across pods, and the CPU, memory and kernel facts as properties on the node device. A `device` component is the CycloneDX type for hardware; `device-driver` is used for the GPU driver.

| `spec.data` path | CycloneDX 1.6 | Rule |
|---|---|---|
| `environment.gpu_type`, `environment.gpu_count`, `environment.gpu_memory_mb[i]` | `components[]`: `gpu_count` entries of `type: device`, `name` = `gpu_type`, `bom-ref: gpu:<i>`, property `aibomrun:gpu:memoryMb` from `gpu_memory_mb[i]` when present | transformed. The record keeps only the first model name of a mixed node; the mapping inherits that limitation and says so in a comment property `aibomrun:gpu:modelsTruncated: "true"` only when the record carries more memory entries than names, which is the one case where it can be detected |
| `environment.driver_version` | `components[]`: `type: device-driver`, `name: nvidia-driver`, `version` | direct |
| `environment.cuda_version` | property `aibomrun:gpu:cudaDriverVersion` on the driver component | property; this is the driver's CUDA ceiling, not the toolkit (that is `torch_cuda_version`, under Software) |
| `execution_metadata.pods[].node_name` | `components[]`: one `type: device` per distinct node, `name` = node name, `bom-ref: node:<name>` | transformed |
| `environment.cpu_model`, `.cpu_cores`, `.cpu.cpu_architecture`, `.cpu.cpu_cores_per_socket`, `.cpu.cpu_threads_per_core`, `.cpu.cache_l3`, `.memory_gb`, `.numa_nodes`, `.kernel_version` | properties on the first node device: `aibomrun:node:cpu:model`, `:cpu:cores`, `:cpu:architecture`, `:cpu:coresPerSocket`, `:cpu:threadsPerCore`, `:cpu:cacheL3`, `:memoryGb`, `:numaNodes`, `:kernel:version` | property. They are node facts read from inside the first pod, so they are attached to that pod's node only. In the records `cpu_cores`, `numa_nodes`, `cpu_cores_per_socket` and `cpu_threads_per_core` are strings (`"512"`); CycloneDX property values are strings anyway, so nothing is lost, but see open question 7 |
| `environment.network.rdma_devices`, `.rdma_device_count`, `.primary_mtu` | properties `aibomrun:node:network:rdmaDevices` (JSON array string), `:rdmaDeviceCount`, `:primaryMtu` on the first node device | property. The record carries `rdma_devices` as one comma-separated string (`mlx5_0,mlx5_1,...`); the mapping splits it into a JSON array |
| `environment.storage.block_devices` | property `aibomrun:node:storage:blockDevices` (JSON array string) on the first node device | property. The record carries it as `lsblk` output lines in one string (`nvme0n1 894.3G\nnvme1n1     7T`); the mapping splits on newlines and collapses whitespace |
| `environment.kernel_config.cpu_governor`, `.numa_balancing`, `.transparent_hugepages`, `.max_map_count` | properties `aibomrun:node:kernel:cpuGovernor`, `:numaBalancing`, `:transparentHugepages`, `:maxMapCount` | property |
| `environment.framework_version` | dropped as a separate field; the framework and version are already the `library` component for torch (below) | dropped, duplicate |

`dependencies[]` carries one entry: the root component depends on every top-level component, including the devices. CycloneDX has no "ran on" relation; a dependency edge from the run to the hardware is the closest native statement and is what graph consumers expect.

## Datasets

Each `dataset.auto_detected[]` entry becomes a `data` component with `type: dataset` in its `data[]` block. The declared dataset is not a separate component: it is the same dataset seen from a different source, so it is recorded as properties on the root component and reconciled against the detected entries through `matches_declared`, exactly as the record does.

| `spec.data` path | CycloneDX 1.6 | Rule |
|---|---|---|
| `dataset.auto_detected[i].dataset_name` | `components[]`: `type: data`, `name`, `bom-ref: dataset:<i>`; `data[0].type: dataset`, `data[0].name` = same, `data[0].classification: "dataset"` | direct |
| `.version` | component `version` | direct |
| `.source` | `data[0].contents.url` when it is a URL; otherwise property `aibomrun:dataset:source` | transformed. In the record `source` is the detecting call (`datasets.load_dataset`), not a location, so it lands in the property |
| `.license` | component `licenses[0].license.id` when it is an SPDX id, else `licenses[0].license.name` | transformed |
| `.description` | component `description` | direct |
| `.fingerprint` | component `hashes[0]`: `alg: SHA-256`, `content` = the fingerprint, only if the fingerprint is a hex SHA-256; else property `aibomrun:dataset:fingerprint` | transformed. In the training record the fingerprint is 16 hex characters (`a5048bbbf302198f`), so it goes to the property, never to `hashes`. The record's fingerprint is a hash of file names, sizes and mtimes, not of contents; the property `aibomrun:dataset:fingerprint:kind: "metadata"` says so either way, so no reader mistakes it for a content hash (#185) |
| `.split`, `.splits[]` | property `aibomrun:dataset:split`: `split` as is, `splits` comma-joined | property. Both are current: `split` is a string, set when the load named one (`load_dataset(split=...)`, a torchvision train/test flag); `splits` is a list, set when `load_dataset` returned a `DatasetDict` (`["train"]` in the training record) |
| `.config_name`, `.revision`, `.builder` | properties `aibomrun:dataset:configName`, `:revision`, `:builder` | property; *not seen in the records* |
| `.root`, `.path`, `.data_dir`, `.data_files` | properties `aibomrun:dataset:root`, `:path`, `:dataDir`, `:dataFiles` (JSON string) | property; these are paths inside the container, kept because they are what makes a local dataset identifiable; *not seen in the records* |
| `.url`, `.urls` | one `externalReferences[]` of `type: distribution` per URL; `urls` is a list or a dict of name to URL, and a dict's values are used | transformed; *not seen in the records* |
| `.dataset_class`, `.transform`, `.download`, `.description`, `.license` | properties `aibomrun:dataset:class`, `:transform`, `:download`; `description` and `license` as in the rows above | property; *not seen in the records* (the real entry carries `dataset_name`, `source`, `splits`, `version`, `fingerprint`, `batch_size`, `seen_via`, `matches_declared`) |
| `.batch_size` | property `aibomrun:dataset:batchSize` | property |
| `.sampling.*` (`sampler`, `shuffle`, `drop_last`, `num_workers`, `generator_seed`, `batch_sampler`, `sampler_seed`, `sampler_drop_last`) | properties `aibomrun:dataset:sampling:<camelCased key>` | property; *not seen in the records* |
| `.seen_via[]`, `.detected_via` | properties `aibomrun:dataset:seenVia` (comma-joined), `aibomrun:dataset:detectedVia` | property; `detected_via` *not seen in the records* |
| `.matches_declared` | property `aibomrun:dataset:matchesDeclared` (`true`, `false`, or absent for `null`) | property |
| `dataset.declared.name`, `.version`, `.source`, `.license`, `.config_name`, `.split`, `.declared_via` | properties `aibomrun:dataset:declared:name`, `:version`, `:source`, `:license`, `:configName`, `:split`, `:declaredVia` on `metadata.component` | property |

Every dataset component is also referenced from `modelCard.modelParameters.datasets[]` by `bom-ref`.

## Software

| `spec.data` path | CycloneDX 1.6 | Rule |
|---|---|---|
| (container image of the workload) | not in `spec.data` | dropped: the record does not carry the image reference or digest. Open question 2 |
| `environment.python_version` | `components[]`: `type: platform`, `name: python`, `version`, `purl: pkg:generic/python@<version>` | transformed |
| `environment.libraries{dist: version}` | one `components[]` entry per library: `type: library`, `name`, `version`, `purl: pkg:pypi/<normalized name>@<version>` | transformed. Distribution names are normalised per PEP 503 for the purl. The records carry NVIDIA container builds with local version labels (`2.11.0a0+a6c236b9fd.nv26.3.46836102`); the purl keeps the full string as the version, which is what `pkg:pypi` allows, and such a version will not resolve on PyPI, which is correct: it is not a PyPI release |
| `environment.torch_cuda_version`, `.torch_hip_version`, `.cudnn_version`, `.nccl_version` | properties on the `torch` library component: `aibomrun:torch:cudaVersion`, `:hipVersion`, `:cudnnVersion`, `:ncclVersion` | property; they describe the torch build, not separate installed packages. `torch_hip_version` *not seen in the records* |
| `model.framework` (serving: `vllm`, `tgi`, `sglang`, `lmdeploy`) | `components[]`: `type: application`, `name` = engine, property `aibomrun:runtime:name` | transformed. This is the component k8s-aibom also emits for the same workload, as `application` with `k8s-aibom:runtime:name`, which is what makes the two documents comparable |
| `model.framework` (training: `trl`, `axolotl`, `llamafactory`) | property `aibomrun:model:framework`, as in the model table; the tool is also in `environment.libraries` as a `library` component (`trl` 1.17.1 in the record) | property. Detected Fields mentions a `training_framework` field; the records carry the tool only in `model.framework`, so no separate row |

## Training configuration

Training facts are modelCard material: hyperparameters, precision, determinism and topology describe how the model was produced. CycloneDX's modelCard has no hyperparameter schema, so they go in `modelCard.properties` rather than `modelParameters`, under the same namespace.

| `spec.data` path | CycloneDX 1.6 (`modelCard.properties` unless stated) | Rule |
|---|---|---|
| `training.optimizer` | `aibomrun:training:optimizer` | property |
| `training.optimizer_config.class`, `.declared_via`, `.params.*`, `.param_groups[]`, `.optim_args` | `aibomrun:training:optimizerConfig:class`, `:declaredVia`, `:params:<camelCased key>` one property per key (`lr`, `betas` as JSON, `eps`, `weight_decay`, `fused`, `amsgrad`, `capturable`, `decoupled_weight_decay`, `differentiable`, `maximize` in the record), `:paramGroups` (JSON string), `:optimArgs` | property. Per-key properties for `params` because the record's keys are plain scalars; `param_groups` stays a JSON string. `param_groups` and `optim_args` *not seen in the records* |
| `training.lr_scheduler.type`, `.warmup_ratio`, `.warmup_steps`, `.declared_via` | `aibomrun:training:lrScheduler:type`, `:warmupRatio`, `:warmupSteps`, `:declaredVia` | property. `declared_via` is in the record (`training_arguments`) though not in Detected Fields; `warmup_ratio` *not seen in the records* |
| `training.learning_rate`, `.batch_size`, `.epochs`, `.random_seed` | `aibomrun:training:learningRate`, `:batchSize`, `:epochs`, `:randomSeed` | property |
| `training.determinism.*` (`seeds.{python,numpy,torch}`, `cudnn_deterministic`, `cudnn_benchmark`, `deterministic_algorithms` in the record; `seed_history`, `deterministic_algorithms_warn_only`, `cublas_workspace_config`, `python_hash_seed`, `data_seed`, `full_determinism` per Detected Fields) | `aibomrun:training:determinism:<camelCased path>`; `seed_history` as JSON string | property; the second group *not seen in the records* |
| `training.mixed_precision`, `.mixed_precision_declared_via` | `aibomrun:training:mixedPrecision`, `aibomrun:training:mixedPrecisionDeclaredVia` | property |
| `training.precision.*` (`tf32_matmul`, `tf32_cudnn`, `float32_matmul_precision`, `deepspeed` in the record; `autocast[]`, `fp8.{backend,recipe}`, `fsdp_mixed_precision.*` per Detected Fields) | `aibomrun:training:precision:<camelCased path>`; arrays and objects as JSON strings | property; the second group *not seen in the records* |
| `training.parallelization_strategy` | `aibomrun:training:parallelizationStrategy` | property |
| `training.distributed.*` (`world_size`, `nodes`, `processes_per_node`, `world_declared_via`, `strategies[]`, `deepspeed.{zero_stage,offload_optimizer,offload_param,declared_via}` in the record; `tensor_parallel_size`, `pipeline_parallel_size`, `context_parallel_size`, `expert_parallel_size`, `data_parallel_size`, `parallel_sizes_declared_via`, `device_mesh`, `fsdp.*` per Detected Fields) | `aibomrun:training:distributed:<camelCased path>`; `strategies` as JSON array string, `device_mesh` JSON string | property; the second group *not seen in the records* |
| `fine_tuning.adaptation_method`, `.lora_rank`, `.lora_alpha` | `aibomrun:fineTuning:adaptationMethod`, `:loraRank`, `:loraAlpha` | property |
| `inference.*` (serving runs: `serving_engine`, `max_model_len`, `tensor_parallel_size`, `pipeline_parallel_size`, `enable_expert_parallel`, `data_parallel_size`, `gpu_memory_utilization`, `temperature`, `top_p`, `top_k`, `max_tokens`, `served_model_name`, `max_num_seqs`, `seed`, `trust_remote_code`, `enforce_eager`, `enable_prefix_caching`, `serving_role`) | properties `aibomrun:inference:<camelCased key>` on the runtime `application` component | property. Serving configuration describes the server, so it sits on the runtime component, where k8s-aibom's readers already look for runtime facts. Null values emit nothing: the serving record sets `max_model_len`, `gpu_memory_utilization`, `temperature`, `top_p`, `top_k` and leaves the rest null. `serving_role` *not seen in the records* |

## Provenance

| `spec.data` path | CycloneDX 1.6 | Rule |
|---|---|---|
| `source_code.git_repository` | `metadata.component.externalReferences[]`: `type: vcs`, `url` = repository; also `pedigree.commits[0].url` on the root | direct. The record has already redacted credentials from the URL |
| `source_code.git_commit` | `metadata.component.pedigree.commits[0].uid` | direct |
| `source_code.git_branch` | property `aibomrun:source:gitBranch` on the root | property; `commit` has no branch field |
| `source_code.declared_via` | property `aibomrun:source:declaredVia` | property |
| `source_code.dirty` | `pedigree.notes`: `"working tree had uncommitted changes to tracked files"`; also property `aibomrun:source:dirty: "true"` | transformed; the note is for humans, the property for machines. Absent when the record has no `dirty`; present only in the third sample |

`pedigree` lives on the root component. For a serving run the root is an `application`, and pedigree is allowed on any component type, so the placement is the same. Both real records have every `source_code` field null (no annotation, no `.git`, no image label resolved), so the sample built from the training record has no pedigree; the third sample beside it, [`sample-training-run-with-git.cdx.json`](../examples/cyclonedx/sample-training-run-with-git.cdx.json), is the same record with `source_code` populated in the shape the project described (`git_repository`, `git_commit`, `git_branch`, `dirty`, `declared_via`) and shows the pedigree, the `vcs` reference and the dirty-tree note.

## Telemetry

Summary statistics map to `modelCard.quantitativeAnalysis.performanceMetrics[]`, one entry per statistic. The metric type is the record's metric name qualified by the statistic, the value is the number as a string, and the unit is the `slice`, which CycloneDX defines as free text describing the subset the metric applies to and which is the only native place for a unit.

| `spec.data` path | CycloneDX 1.6 | Rule |
|---|---|---|
| `resource_utilization.metrics.<m>.{min,max,avg,p95}` | `performanceMetrics[]`: `type: "<m>:<stat>"`, `value: "<number>"`, `slice: "<unit>"` | transformed |
| `resource_utilization.metrics.<m>.segments.{first_third,middle_third,last_third}` | `performanceMetrics[]`: `type: "<m>:avg:<segment>"`, same unit | transformed |
| `resource_utilization.metrics.<m>.limit` | `performanceMetrics[]`: `type: "<m>:limit"` | transformed |
| `resource_utilization.metrics.<m>.unit` | the `slice` of each entry above | transformed |
| `resource_utilization.collected_at`, `.summary_includes_cold_start` | properties `aibomrun:telemetry:collectedAt`, `aibomrun:telemetry:summaryIncludesColdStart` on `modelCard.properties` | property |
| `resource_utilization.note` | dropped; an absent `quantitativeAnalysis` says the same | dropped |
| `inference.performance.metrics.<m>.*`, `.collected_at`, `.summary_includes_cold_start` | same shapes, metric type prefixed `inference:` (`inference:time_to_first_token_seconds:p95`), properties `aibomrun:inference:telemetry:collectedAt`, `:summaryIncludesColdStart` | transformed. For serving runs the modelCard sits on the served model, so its performance metrics describe serving that model on this configuration, which is what they are |
| `telemetry_series_ref.{kind,name,sha256,size_bytes,schema_version,window.{start,end,step_seconds}}` | `metadata.component.externalReferences[]`: `type: quality-metrics`, `url: urn:aibomrun:<namespace>/<kind>/<name>`, `hashes[0]: {alg: SHA-256, content: <sha256>}`, `comment: "<size_bytes> bytes, schema <schema_version>, window <start>..<end> step <step_seconds>s"` | transformed. The record spells the step `step_seconds`; the `AIBOMTelemetry` CRD spells the same value `stepSeconds`. The series is not in the document; the reference carries the hash the signed record already holds, so a reader with cluster access can fetch `aibomtel/<name>` and verify it the way `oc aibomrun describe` does. Open question 3 covers the URN scheme |
| legacy `telemetry_series_ref.{configmap,key}` | same reference with `url: urn:aibomrun:<namespace>/ConfigMap/<configmap>#<key>` | transformed; *not seen in the records* |

`resource_utilization` is per run, not per pod, in the record; the mapping does not try to recover per-pod figures.

## Signature

`spec.signature` is a base64 Ed25519 signature over the RFC 8785 canonical form of `spec.data`, with the raw public key in `spec.signaturePublicKey`. CycloneDX carries an enveloped signature in JSON Signature Format at the document root, and JSF defines Ed25519 (`algorithm: "Ed25519"`, public key as a JWK of `kty: OKP`, `crv: Ed25519`).

The existing signature cannot simply be copied into the CycloneDX document, because it signs the record's bytes, not the document's. Two honest options, and the mapping uses the first:

1. **Carry the record's signature as evidence of the source.** `metadata.component.externalReferences[]` of `type: attestation` with `url: urn:aibomrun:<namespace>/AIBOM/<name>` and `hashes[0]: {alg: SHA-256, content: <sha256 of the RFC 8785 bytes>}`, plus properties `aibomrun:signature:algorithm: "Ed25519"`, `aibomrun:signature:value: <spec.signature>`, `aibomrun:signature:publicKey: <spec.signaturePublicKey>` on `metadata.component`. A verifier fetches the record, canonicalises `spec.data`, and checks the Ed25519 signature against the embedded key, which is exactly what `oc aibomrun describe` does today. The CycloneDX document itself is unsigned and says so.
2. **Sign the CycloneDX document too.** If the converter runs in the postprocess Job it holds the per-namespace key and can add a root `signature` in JSF with the same key. That makes the CycloneDX document self-verifying, at the cost of two signatures over two serialisations of one record. Worth doing only if a downstream consumer verifies JSF, and none of the three named ones do today.

Properties under `aibomrun:signature:*` are facts about the record's signature, not a claim that the CycloneDX bytes are signed.

## Linking to k8s-aibom

A serving workload on an enabled namespace will have both an `AIBOM` record from this project and an `AIBOM` document from k8s-aibom. CycloneDX links documents with a BOM-Link: `urn:cdx:<serialNumber UUID>/<version>` for a whole document, `urn:cdx:<uuid>/<version>#<bom-ref>` for one element in it.

From this project's document to k8s-aibom's: `metadata.component.externalReferences[]` of `type: bom` with `url: urn:cdx:<k8s-aibom serialNumber>/<version>` and `comment: "runtime inventory for the same workload"`. The workload identity that makes the match is the namespace and the controller name: k8s-aibom's root component is `workload:<namespace>/<Kind>/<name>` with properties `k8s-aibom:workload:uid`; this project's record has `execution_metadata.job_id` and `.namespace` and the pods' owner is the Job. The converter resolves the link by reading k8s-aibom's `AIBOM` object for the Job (or for the workload that owns the pods) from the cluster when it runs; without cluster access the link is omitted, never guessed.

From k8s-aibom's document to this project's: the same `type: bom` reference pointing at `urn:cdx:<this document's serialNumber>/1`, which is why the serial number above is derived deterministically from the record's name: k8s-aibom can compute it without reading the record. That half is a k8s-aibom change and is tracked there. Open question 4.

## Property namespace

Every `property` above uses the top-level namespace `aibomrun`, with colon-separated sub-namespaces and camelCase leaf names, following the grammar of the [CycloneDX Property Taxonomy](https://github.com/CycloneDX/cyclonedx-property-taxonomy). Registering it is one row in that repository's README, pointing at a taxonomy page in this repository, as k8s-aibom did for `k8s-aibom`. `aibomrun` is suggested because it is already the CRD short name, the plugin name and the thing users type; `project-aibom` is the alternative if the project name should be the namespace. Until the row is merged the names are the same, so nothing in the mapping changes at registration.

Sub-namespaces used: `aibomrun:experiment`, `aibomrun:job`, `aibomrun:pod`, `aibomrun:model`, `aibomrun:gpu`, `aibomrun:node`, `aibomrun:dataset`, `aibomrun:torch`, `aibomrun:runtime`, `aibomrun:training`, `aibomrun:fineTuning`, `aibomrun:inference`, `aibomrun:source`, `aibomrun:telemetry`, `aibomrun:signature`.

## Deliberately not mapped

- **Captured-only discovery fields** (L1 and L2 cache, clock frequencies, free memory, uptime, interface names, TCP buffers, NVMe count, `/tmp` size, I/O scheduler, swappiness, ulimits, cgroup v1 limits). They never reach `spec.data`; Detected Fields explains why.
- **`execution_metadata.pods[].pod_ip`.** Ephemeral, and a document that leaves the cluster should not carry pod addresses.
- **`_metadata.schema_compliance` and `_metadata.dataset_detection`.** Statements about the record format.
- **`resource_utilization.note`.** Its absence of metrics is already visible.
- **The time series itself.** Too large for a BOM and already addressed by hash through the reference.
- **`environment.framework_version` as its own field.** Redundant with the torch library component.
- **Anything not in `spec.data`**: the container image reference and digest, the pod spec, resource requests. The record does not have them, so the document cannot claim them; see open question 2.

## Open questions

1. **Model purl.** `pkg:generic/<name>?download_url=https://huggingface.co/<name>` versus `pkg:huggingface/<org>/<model>@<revision>`. Both records name Hugging Face repositories (`ibm-granite/granite-3.3-2b-instruct`, `drawais/Granite-3.3-8B-Instruct-AWQ-INT4`) and neither has a revision, so the two forms differ only in type. The mapping uses `pkg:generic` and switches if Lightwell or Dependency-Track prefer `pkg:huggingface`.
2. **Container image.** Neither record carries the image reference or digest, although `containers.json` has it during postprocess. Adding `execution_metadata.pods[].containers[]` with `image` and `imageID` to `spec.data` would give the document a `container` component with a `pkg:oci` purl, and that is the single field that makes the k8s-aibom link checkable without cluster access (same digest in both documents). A record change, not a mapping change.
3. **URN scheme for cluster objects.** `urn:aibomrun:<namespace>/<Kind>/<name>` is a placeholder for "an object in the cluster this record came from". If the project prefers `https://` URLs to the console or an `oc` plugin scheme, the mapping follows.
4. **The k8s-aibom half of the link.** k8s-aibom documents do not emit a `serialNumber` today, and the reverse link needs one. The forward link works regardless; the reverse is tracked on the k8s-aibom side.
5. **Metric unit placement.** Units ride in `performanceMetrics[].slice`, which is free text by design. If a consumer treats `slice` as a data-subset label, the alternative is folding the unit into `type` (`gpu_utilization_percent:p95`). The mapping keeps `slice` until a consumer objects.
6. **Per-pod telemetry.** The records aggregate across pods (both have one pod, so it did not matter here). If per-pod figures matter for multi-node runs, that is a record change first.
7. **Numbers as strings in `environment`.** `cpu_cores`, `numa_nodes`, `cpu_cores_per_socket`, `cpu_threads_per_core`, `rdma_device_count` and `primary_mtu` are strings in the records (`"512"`, `"1400"`) while `gpu_count`, `memory_gb` and `gpu_memory_mb[]` are numbers. CycloneDX properties are strings so the document is unaffected, but a record that typed them consistently would spare every consumer a cast.
8. **`rdma_devices` and `block_devices` as flat strings.** The record carries `mlx5_0,mlx5_1,...` and raw `lsblk` lines in one string each; the mapping splits them. Emitting arrays in the record would remove the parsing from every reader.
9. **`step_seconds` versus `stepSeconds`.** The record's `telemetry_series_ref.window` uses `step_seconds`; the `AIBOMTelemetry` CRD's `spec.window` uses `stepSeconds` for the same value. Harmless, but a reader matching the two by hand will trip on it once.
10. **Pod `start_time` without a timezone.** `2026-11-03T16:29:57.639439` versus the `Z`-suffixed timestamps everywhere else in the record. The mapping appends `Z`, the same UTC assumption postprocess makes when it computes durations.

## Validation

The three sample documents beside this page were built by applying every rule above, through the reference converter `convert-samples.py`, to two real records the project supplied: a DeepSpeed ZeRO-3 LoRA fine-tune of `ibm-granite/granite-3.3-2b-instruct` on two H100s ([`sample-training-run.cdx.json`](../examples/cyclonedx/sample-training-run.cdx.json)), and a vLLM benchmark server running an AWQ-quantized Granite 8B with ngram speculative decoding ([`sample-serving-run.cdx.json`](../examples/cyclonedx/sample-serving-run.cdx.json)). The third, [`sample-training-run-with-git.cdx.json`](../examples/cyclonedx/sample-training-run-with-git.cdx.json), is the training record with `source_code` populated, since both real records have it null, so the pedigree rules are exercised too. All three validate against the CycloneDX 1.6 JSON schema. Before publication the namespace was renamed, the signature and public key truncated, and the records' own placeholder node name, pod IP and UIDs kept as supplied; nothing else was changed.
