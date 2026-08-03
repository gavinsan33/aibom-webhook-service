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
