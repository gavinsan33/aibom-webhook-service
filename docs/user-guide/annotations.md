# Annotations

You can annotate your Jobs with `aibom.io/*` keys to provide experiment metadata. Annotations are optional: without them, the AIBOM is still generated from auto-detected data (hardware discovery, dataset detection, telemetry).

| Annotation | AIBOM field |
|------------|-------------|
| `aibom.io/experiment-intent` | `experiment_intent` (`training`, `sft`, `inference`) |
| `aibom.io/experiment-name` | `experiment_name` |
| `aibom.io/model-name` | `model.name` |
| `aibom.io/model-framework` | `model.framework` |
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
| `aibom.io/top-k` | `inference.top_k` |

## Precedence

Auto-detected values are used as defaults, and a corresponding annotation overrides them.

The exceptions are `aibom.io/learning-rate`, `aibom.io/batch-size`, `aibom.io/epochs` and `aibom.io/random-seed`. For these, the runtime hooks win, then the command-line arguments, and the annotation is used only when neither produced a value.

For every field the pipeline can populate, see [Detected Fields](../CAPABILITIES.md).
