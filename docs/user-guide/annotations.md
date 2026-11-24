# Annotations

You can annotate your Jobs with `aibom.io/*` keys to provide experiment metadata. Annotations are optional: without them, the AIBOM is still generated from auto-detected data (hardware discovery, dataset detection, telemetry).

| Annotation | AIBOM field |
|------------|-------------|
| `aibom.io/experiment-intent` | `experiment_intent` (`training`, `sft`, `inference`) |
| `aibom.io/experiment-name` | `experiment_name` |
| `aibom.io/experiment-description` | `experiment_description` |
| `aibom.io/model-name` | `model.name` |
| `aibom.io/model-version` | `model.version` |
| `aibom.io/model-architecture` | `model.architecture` |
| `aibom.io/model-framework` | `model.framework` |
| `aibom.io/dtype` | `model.dtype` |
| `aibom.io/quantization` | `model.quantization` |
| `aibom.io/quantization-bits` | `model.quantization_bits` |
| `aibom.io/git-repository` | `source_code.git_repository` |
| `aibom.io/git-commit` | `source_code.git_commit` |
| `aibom.io/git-branch` | `source_code.git_branch` |
| `aibom.io/dataset-name` | `dataset.declared.name` |
| `aibom.io/dataset-source` | `dataset.declared.source` |
| `aibom.io/dataset-version` | `dataset.declared.version` |
| `aibom.io/dataset-license` | `dataset.declared.license` |
| `aibom.io/optimizer` | `training.optimizer` |
| `aibom.io/batch-size` | `training.batch_size` |
| `aibom.io/epochs` | `training.epochs` |
| `aibom.io/learning-rate` | `training.learning_rate` |
| `aibom.io/random-seed` | `training.random_seed` |
| `aibom.io/parallelization-strategy` | `training.parallelization_strategy` |
| `aibom.io/adaptation-method` | `fine_tuning.adaptation_method` |
| `aibom.io/lora-rank` | `fine_tuning.lora_rank` |
| `aibom.io/lora-alpha` | `fine_tuning.lora_alpha` |
| `aibom.io/serving-engine` | `inference.serving_engine` |
| `aibom.io/max-model-len` | `inference.max_model_len` |
| `aibom.io/tensor-parallel-size` | `inference.tensor_parallel_size` |
| `aibom.io/pipeline-parallel-size` | `inference.pipeline_parallel_size` |
| `aibom.io/data-parallel-size` | `inference.data_parallel_size` |
| `aibom.io/gpu-memory-utilization` | `inference.gpu_memory_utilization` |
| `aibom.io/temperature` | `inference.temperature` |
| `aibom.io/top-p` | `inference.top_p` |
| `aibom.io/top-k` | `inference.top_k` |
| `aibom.io/max-tokens` | `inference.max_tokens` |

`training.*` fields are written only for `training` and `sft` runs, `fine_tuning.*` only for `sft`, and `inference.*` only for `inference`.

One pod annotation is read by the webhook rather than the AIBOM: `aibom.io/python-path`. The webhook sets `PYTHONPATH` on instrumented containers and can't see an `ENV PYTHONPATH` baked into the image, so declare it here (for example `/app:/app/lib`) and it's appended after the hook directory.

## Precedence

Auto-detected values are used as defaults, and a corresponding annotation overrides them.

For every field the pipeline can populate, see [Detected Fields](../CAPABILITIES.md).
