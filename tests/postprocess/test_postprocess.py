import postprocess as pp


# ---------------------------------------------------------------------------
# Quantization detection
# ---------------------------------------------------------------------------


def test_detect_quantization_awq():
    assert pp.detect_quantization_from_name("Meta-Llama-3-8B-Instruct-AWQ") == {
        "quantization_method": "awq",
        "quantization_bits": 4,
    }


def test_detect_quantization_gptq_int4():
    result = pp.detect_quantization_from_name("Mixtral-8x7B-GPTQ-Int4")
    assert result == {"quantization_method": "gptq", "quantization_bits": 4}


def test_detect_quantization_captures_dynamic_bit_count():
    result = pp.detect_quantization_from_name("model-HQQ-3bit")
    assert result == {"quantization_method": "hqq", "quantization_bits": 3}


def test_detect_quantization_no_match_returns_none():
    assert pp.detect_quantization_from_name("ibm-granite/granite-3.3-2b-instruct") is None


def test_detect_quantization_empty_name_returns_none():
    assert pp.detect_quantization_from_name(None) is None
    assert pp.detect_quantization_from_name("") is None


# ---------------------------------------------------------------------------
# vLLM CLI detection
# ---------------------------------------------------------------------------


def test_detect_vllm_from_command_basic_flags():
    command = [
        "vllm", "serve", "--model", "meta-llama/Llama-3-8B",
        "--tensor-parallel-size", "2",
        "--gpu-memory-utilization", "0.9",
        "--trust-remote-code",
    ]
    result = pp.detect_vllm_from_command(command)
    assert result["serving_engine"] == "vllm"
    assert result["model_name"] == "meta-llama/Llama-3-8B"
    assert result["tensor_parallel_size"] == 2
    assert result["gpu_memory_utilization"] == 0.9
    assert result["trust_remote_code"] is True


def test_detect_vllm_infers_quantization_from_model_name():
    command = ["vllm", "serve", "--model", "some-model-AWQ"]
    result = pp.detect_vllm_from_command(command)
    assert result["quantization_method"] == "awq"
    assert result["quantization_bits"] == 4


def test_detect_vllm_explicit_quantization_flag_wins_over_name():
    command = ["vllm", "serve", "--model", "some-model-AWQ", "--quantization", "gptq"]
    result = pp.detect_vllm_from_command(command)
    assert result["quantization"] == "gptq"
    assert "quantization_method" not in result


def test_detect_vllm_normalizes_legacy_speculative_flags():
    command = [
        "vllm", "serve", "--model", "meta-llama/Llama-3-8B",
        "--speculative-model", "meta-llama/Llama-3-1B",
        "--num-speculative-tokens", "5",
    ]
    result = pp.detect_vllm_from_command(command)
    assert result["speculative_config"] == {
        "model": "meta-llama/Llama-3-1B",
        "num_speculative_tokens": 5,
    }
    assert "speculative_model" not in result


def test_detect_vllm_returns_none_for_non_vllm_command():
    assert pp.detect_vllm_from_command(["python", "train.py"]) is None


def test_detect_vllm_returns_none_for_empty_command():
    assert pp.detect_vllm_from_command([]) is None
    assert pp.detect_vllm_from_command(None) is None


# ---------------------------------------------------------------------------
# KServe InferenceService storage-path detection (S3/MinIO data-connection)
# ---------------------------------------------------------------------------


def test_detect_model_from_storage_uses_last_path_segment():
    result = pp.detect_model_from_storage({"storage_path": "models/tinyllama-1.1b-chat"})
    assert result == {"model_name": "tinyllama-1.1b-chat"}


def test_detect_model_from_storage_falls_back_to_storage_uri():
    result = pp.detect_model_from_storage({"storage_uri": "s3://bucket/models/granite-3.0-8b-instruct/"})
    assert result == {"model_name": "granite-3.0-8b-instruct"}


def test_detect_model_from_storage_infers_quantization_from_name():
    result = pp.detect_model_from_storage({"storage_path": "models/some-model-AWQ"})
    assert result["model_name"] == "some-model-AWQ"
    assert result["quantization_method"] == "awq"


def test_detect_model_from_storage_returns_none_when_empty():
    assert pp.detect_model_from_storage({}) is None
    assert pp.detect_model_from_storage(None) is None
    assert pp.detect_model_from_storage({"storage_key": "minio-data-connection"}) is None


# ---------------------------------------------------------------------------
# trl CLI detection
# ---------------------------------------------------------------------------


def test_detect_trl_from_command_plain_lora():
    command = [
        "trl", "sft", "--model_name_or_path", "ibm-granite/granite-3.3-2b-instruct",
        "--use_peft", "true", "--lora_r", "16", "--lora_alpha", "32",
    ]
    result = pp.detect_trl_from_command(command)
    assert result["training_framework"] == "trl"
    assert result["model_name"] == "ibm-granite/granite-3.3-2b-instruct"
    assert result["lora_rank"] == 16
    assert result["lora_alpha"] == 32
    assert result["adaptation_method"] == "lora"


def test_detect_trl_from_command_qlora_from_quantized_load():
    command = [
        "trl", "sft", "--model_name_or_path", "some-model",
        "--use_peft", "true", "--lora_r", "16", "--load_in_4bit", "true",
    ]
    result = pp.detect_trl_from_command(command)
    assert result["adaptation_method"] == "qlora"


def test_detect_trl_from_command_dora():
    command = [
        "trl", "sft", "--model_name_or_path", "some-model",
        "--use_peft", "true", "--lora_r", "16", "--use_dora", "true",
    ]
    assert pp.detect_trl_from_command(command)["adaptation_method"] == "dora"


def test_detect_trl_from_command_peft_without_lora_rank():
    command = ["trl", "sft", "--model_name_or_path", "some-model", "--use_peft", "true"]
    result = pp.detect_trl_from_command(command)
    assert result["adaptation_method"] == "peft"


def test_detect_trl_returns_none_for_non_trl_command():
    assert pp.detect_trl_from_command(["python", "serve.py"]) is None


# ---------------------------------------------------------------------------
# Shell-wrapped command flattening
# ---------------------------------------------------------------------------


def test_flatten_container_command_expands_sh_c_wrapper():
    container = {
        "command": ["sh", "-c"],
        "args": ["pip install trl && trl sft --model_name_or_path foo --use_peft true"],
    }
    tokens = pp._flatten_container_command(container)
    assert tokens == [
        "pip", "install", "trl", "&&", "trl", "sft",
        "--model_name_or_path", "foo", "--use_peft", "true",
    ]


def test_flatten_container_command_joins_line_continuations():
    container = {
        "command": ["bash", "-c"],
        "args": ["trl sft --use_peft \\\n--lora_r 16"],
    }
    tokens = pp._flatten_container_command(container)
    # Without joining the backslash-newline, "--lora_r" would be swallowed as
    # the (spurious) value of the preceding bare boolean flag "--use_peft".
    assert tokens == ["trl", "sft", "--use_peft", "--lora_r", "16"]


def test_flatten_container_command_passthrough_when_not_wrapped():
    container = {"command": ["trl", "sft"], "args": ["--use_peft", "true"]}
    assert pp._flatten_container_command(container) == ["trl", "sft", "--use_peft", "true"]


# ---------------------------------------------------------------------------
# Parallelization strategy detection
# ---------------------------------------------------------------------------


def test_detect_parallelization_bare_fsdp_flag():
    assert pp.detect_parallelization_from_command(["--fsdp", "full_shard"]) == {
        "parallelization_strategy": "fsdp"
    }


def test_detect_parallelization_deepspeed_launcher():
    assert pp.detect_parallelization_from_command(["deepspeed", "train.py"]) == {
        "parallelization_strategy": "deepspeed"
    }


def test_detect_parallelization_accelerate_config_name():
    tokens = ["trl", "sft", "--accelerate_config", "/configs/zero3.yaml"]
    assert pp.detect_parallelization_from_command(tokens) == {
        "parallelization_strategy": "deepspeed"
    }


def test_detect_parallelization_multi_gpu_flag():
    assert pp.detect_parallelization_from_command(["accelerate", "launch", "--multi_gpu"]) == {
        "parallelization_strategy": "data_parallel"
    }


def test_detect_parallelization_single_gpu_returns_none():
    tokens = ["trl", "sft", "--accelerate_config", "/configs/single_gpu.yaml"]
    assert pp.detect_parallelization_from_command(tokens) is None


def test_parallelization_strategy_from_device_map_auto():
    assert pp._parallelization_strategy_from_device_map("auto") == "model_parallel"


def test_parallelization_strategy_from_device_map_single_device_returns_none():
    assert pp._parallelization_strategy_from_device_map("cuda:0") is None


def test_parallelization_strategy_from_device_map_none_returns_none():
    assert pp._parallelization_strategy_from_device_map(None) is None


def test_detect_parallelization_no_signal_returns_none():
    assert pp.detect_parallelization_from_command(["python", "train.py"]) is None
    assert pp.detect_parallelization_from_command([]) is None


# ---------------------------------------------------------------------------
# Dataset CLI detection
# ---------------------------------------------------------------------------


def test_detect_dataset_from_command_requires_dataset_name():
    tokens = ["trl", "sft", "--dataset_config_name", "en"]
    assert pp.detect_dataset_from_command(tokens) is None


def test_detect_dataset_from_command_basic():
    tokens = ["trl", "sft", "--dataset_name", "tatsu-lab/alpaca", "--dataset_train_split", "train"]
    result = pp.detect_dataset_from_command(tokens)
    assert result == {"dataset_name": "tatsu-lab/alpaca", "dataset_split": "train"}


def test_detect_dataset_from_containers_uses_first_match():
    containers = [
        {"command": ["python", "sidecar.py"]},
        {"command": ["trl", "sft", "--dataset_name", "tatsu-lab/alpaca"]},
    ]
    assert pp.detect_dataset_from_containers(containers) == {"dataset_name": "tatsu-lab/alpaca"}


def test_detect_dataset_from_containers_no_match_returns_none():
    containers = [{"command": ["python", "sidecar.py"]}]
    assert pp.detect_dataset_from_containers(containers) is None


# ---------------------------------------------------------------------------
# Git provenance detection
# ---------------------------------------------------------------------------


def test_image_digest_extracts_sha256():
    image_id = "image-registry.openshift-image-registry.svc:5000/ns/train@sha256:abc123"
    assert pp._image_digest(image_id) == "sha256:abc123"


def test_image_digest_none_when_not_digest_pinned():
    assert pp._image_digest("quay.io/org/train:latest") is None
    assert pp._image_digest(None) is None
    assert pp._image_digest("") is None


def test_detect_git_provenance_from_containers_reads_build_labels(monkeypatch):
    containers = [{"image_id": "quay.io/org/train@sha256:abc123"}]

    def fake_get_cluster_object(group, version, plural, name):
        assert (group, version, plural, name) == ("image.openshift.io", "v1", "images", "sha256:abc123")
        return {
            "dockerImageMetadata": {
                "Config": {
                    "Labels": {
                        "io.openshift.build.commit.id": "deadbeef",
                        "io.openshift.build.commit.ref": "main",
                        "io.openshift.build.source-location": "https://github.com/org/train",
                    }
                }
            }
        }

    monkeypatch.setattr(pp.k8s_api, "get_cluster_object", fake_get_cluster_object)
    result = pp.detect_git_provenance_from_containers(containers)
    assert result == {
        "git_commit": "deadbeef",
        "git_repository": "https://github.com/org/train",
        "git_branch": "main",
        "detected_via": "openshift_build_label",
    }


def test_detect_git_provenance_falls_back_to_oci_labels(monkeypatch):
    containers = [{"image_id": "quay.io/org/train@sha256:abc123"}]
    monkeypatch.setattr(
        pp.k8s_api,
        "get_cluster_object",
        lambda *a, **k: {
            "dockerImageMetadata": {
                "Config": {
                    "Labels": {
                        "org.opencontainers.image.revision": "cafef00d",
                        "org.opencontainers.image.source": "https://github.com/org/train",
                    }
                }
            }
        },
    )
    result = pp.detect_git_provenance_from_containers(containers)
    assert result == {
        "git_commit": "cafef00d",
        "git_repository": "https://github.com/org/train",
        "git_branch": None,
        "detected_via": "oci_image_label",
    }


def test_detect_git_provenance_prefers_openshift_label_over_oci(monkeypatch):
    containers = [{"image_id": "quay.io/org/train@sha256:abc123"}]
    monkeypatch.setattr(
        pp.k8s_api,
        "get_cluster_object",
        lambda *a, **k: {
            "dockerImageMetadata": {
                "Config": {
                    "Labels": {
                        "io.openshift.build.commit.id": "deadbeef",
                        "org.opencontainers.image.revision": "cafef00d",
                    }
                }
            }
        },
    )
    result = pp.detect_git_provenance_from_containers(containers)
    assert result["git_commit"] == "deadbeef"
    assert result["detected_via"] == "openshift_build_label"


def test_detect_git_provenance_skips_containers_without_digest(monkeypatch):
    containers = [{"image_id": "quay.io/org/train:latest"}]
    monkeypatch.setattr(
        pp.k8s_api, "get_cluster_object", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not be called"))
    )
    assert pp.detect_git_provenance_from_containers(containers) is None


def test_detect_git_provenance_none_when_image_lacks_commit_label(monkeypatch):
    containers = [{"image_id": "quay.io/org/train@sha256:abc123"}]
    monkeypatch.setattr(
        pp.k8s_api, "get_cluster_object", lambda *a, **k: {"dockerImageMetadata": {"Config": {"Labels": {}}}}
    )
    assert pp.detect_git_provenance_from_containers(containers) is None


def test_detect_git_provenance_none_when_image_not_found(monkeypatch):
    containers = [{"image_id": "quay.io/org/train@sha256:abc123"}]
    monkeypatch.setattr(pp.k8s_api, "get_cluster_object", lambda *a, **k: None)
    assert pp.detect_git_provenance_from_containers(containers) is None


def test_detect_git_provenance_degrades_quietly_on_error(monkeypatch):
    containers = [{"image_id": "quay.io/org/train@sha256:abc123"}]

    def raising(*a, **k):
        raise RuntimeError("no RBAC")

    monkeypatch.setattr(pp.k8s_api, "get_cluster_object", raising)
    assert pp.detect_git_provenance_from_containers(containers) is None


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def test_safe_get_nested_present():
    data = {"model": {"name": "foo"}}
    assert pp.safe_get(data, "model", "name") == "foo"


def test_safe_get_missing_returns_default():
    assert pp.safe_get({}, "model", "name", default="bar") == "bar"
    assert pp.safe_get({"model": "not-a-dict"}, "model", "name", default="bar") == "bar"


def test_try_int_and_float():
    assert pp._try_int("4") == 4
    assert pp._try_int("not-a-number") is None
    assert pp._try_int(None) is None
    assert pp._try_float("0.5") == 0.5
    assert pp._try_float("nope") is None


def test_first_not_none():
    assert pp._first_not_none(None, None, 3, 4) == 3
    assert pp._first_not_none(None, None) is None


def test_ms_to_promql_duration_minutes_and_hours():
    assert pp.ms_to_promql_duration(90_000) == "1m"
    assert pp.ms_to_promql_duration(2 * 3600 * 1000) == "2h"
    # Clamped to a 60s floor so a very short run still yields a valid range.
    assert pp.ms_to_promql_duration(500) == "1m"


# ---------------------------------------------------------------------------
# compile_aibom reconciliation
# ---------------------------------------------------------------------------


def test_compile_aibom_merges_cli_detected_model_and_dataset():
    detected_model = {
        "serving_engine": "vllm",
        "model_name": "meta-llama/Llama-3-8B",
        "quantization_method": "awq",
        "quantization_bits": 4,
    }
    cli_dataset = {"dataset_name": "tatsu-lab/alpaca"}
    annotations = {"experiment-intent": "inference"}

    aibom = pp.compile_aibom(
        discoveries=[], detected_datasets=[], runtime_info={}, annotations=annotations,
        telemetry=None, detected_model=detected_model, cli_dataset=cli_dataset,
    )

    assert aibom["model"]["name"] == "meta-llama/Llama-3-8B"
    assert aibom["model"]["quantization"] == "awq"
    assert aibom["model"]["quantization_bits"] == 4
    assert aibom["model"]["framework"] == "vllm"


def test_compile_aibom_annotation_overrides_detected_model():
    detected_model = {"model_name": "detected-model"}
    annotations = {"experiment-intent": "inference", "model-name": "annotated-model"}

    aibom = pp.compile_aibom(
        discoveries=[], detected_datasets=[], runtime_info={}, annotations=annotations,
        telemetry=None, detected_model=detected_model, cli_dataset=None,
    )

    assert aibom["model"]["name"] == "annotated-model"


def test_compile_aibom_infers_declared_dataset_from_auto_detected():
    detected_datasets = [{"dataset_name": "tatsu-lab/alpaca", "source": "datasets.load_dataset"}]
    annotations = {"experiment-intent": "sft"}

    aibom = pp.compile_aibom(
        discoveries=[], detected_datasets=detected_datasets, runtime_info={},
        annotations=annotations, telemetry=None, detected_model=None, cli_dataset=None,
    )

    assert aibom["dataset"]["declared"]["name"] == "tatsu-lab/alpaca"
    assert aibom["dataset"]["declared"]["declared_via"] == "inferred_from_runtime"
    assert aibom["dataset"]["auto_detected"][0]["matches_declared"] is True


def test_compile_aibom_flags_dataset_mismatch():
    detected_datasets = [{"dataset_name": "some-other-dataset", "source": "datasets.load_dataset"}]
    annotations = {"experiment-intent": "sft", "dataset-name": "tatsu-lab/alpaca"}

    aibom = pp.compile_aibom(
        discoveries=[], detected_datasets=detected_datasets, runtime_info={},
        annotations=annotations, telemetry=None, detected_model=None, cli_dataset=None,
    )

    assert aibom["dataset"]["declared"]["name"] == "tatsu-lab/alpaca"
    assert aibom["dataset"]["auto_detected"][0]["matches_declared"] is False


def test_compile_aibom_uses_detected_provenance_when_no_annotation():
    detected_provenance = {
        "git_commit": "deadbeef",
        "git_repository": "https://github.com/org/train",
        "git_branch": "main",
        "detected_via": "openshift_build_label",
    }
    aibom = pp.compile_aibom(
        discoveries=[], detected_datasets=[], runtime_info={},
        annotations={"experiment-intent": "training"}, telemetry=None,
        detected_provenance=detected_provenance,
    )
    assert aibom["source_code"] == {
        "git_repository": "https://github.com/org/train",
        "git_commit": "deadbeef",
        "git_branch": "main",
        "declared_via": "openshift_build_label",
    }


def test_compile_aibom_uses_oci_label_declared_via():
    detected_provenance = {
        "git_commit": "cafef00d",
        "git_repository": "https://github.com/org/train",
        "git_branch": None,
        "detected_via": "oci_image_label",
    }
    aibom = pp.compile_aibom(
        discoveries=[], detected_datasets=[], runtime_info={},
        annotations={"experiment-intent": "training"}, telemetry=None,
        detected_provenance=detected_provenance,
    )
    assert aibom["source_code"]["declared_via"] == "oci_image_label"


def test_compile_aibom_annotation_overrides_detected_provenance():
    detected_provenance = {"git_commit": "detected-sha", "git_repository": "detected-repo"}
    annotations = {
        "experiment-intent": "training",
        "git-commit": "annotated-sha",
        "git-repository": "annotated-repo",
    }
    aibom = pp.compile_aibom(
        discoveries=[], detected_datasets=[], runtime_info={}, annotations=annotations,
        telemetry=None, detected_provenance=detected_provenance,
    )
    assert aibom["source_code"]["git_commit"] == "annotated-sha"
    assert aibom["source_code"]["git_repository"] == "annotated-repo"
    assert aibom["source_code"]["declared_via"] == "annotation"


def test_compile_aibom_no_provenance_source_leaves_declared_via_none():
    aibom = pp.compile_aibom(
        discoveries=[], detected_datasets=[], runtime_info={},
        annotations={"experiment-intent": "training"}, telemetry=None,
    )
    assert aibom["source_code"]["declared_via"] is None
    assert aibom["source_code"]["git_commit"] is None


def test_compile_aibom_no_telemetry_notes_unavailable():
    aibom = pp.compile_aibom(
        discoveries=[], detected_datasets=[], runtime_info={},
        annotations={"experiment-intent": "unknown"}, telemetry=None,
    )
    assert aibom["resource_utilization"] == {"note": "No telemetry data available."}


# ---------------------------------------------------------------------------
# compile_aibom: runtime_info fallbacks (transformers/peft runtime hooks,
# for scripts with no CLI flags for detect_trl_from_command to see)
# ---------------------------------------------------------------------------


def test_compile_aibom_uses_runtime_info_for_model_identity_when_no_cli_detection():
    runtime_info = {
        "model_name": "ibm-granite/granite-3.3-2b-instruct",
        "model_architecture": "GraniteForCausalLM",
        "training_framework": "transformers.Trainer",
        "quantization_method": "bitsandbytes",
        "quantization_bits": 4,
        "dtype": "bfloat16",
    }
    aibom = pp.compile_aibom(
        discoveries=[], detected_datasets=[], runtime_info=runtime_info,
        annotations={"experiment-intent": "sft"}, telemetry=None,
    )
    assert aibom["model"]["name"] == "ibm-granite/granite-3.3-2b-instruct"
    assert aibom["model"]["architecture"] == "GraniteForCausalLM"
    assert aibom["model"]["framework"] == "transformers.Trainer"
    assert aibom["model"]["quantization"] == "bitsandbytes"
    assert aibom["model"]["quantization_bits"] == 4
    assert aibom["model"]["dtype"] == "bfloat16"


def test_compile_aibom_annotation_still_overrides_runtime_info_model_identity():
    runtime_info = {"model_name": "detected-via-hook"}
    annotations = {"experiment-intent": "sft", "model-name": "annotated-model"}
    aibom = pp.compile_aibom(
        discoveries=[], detected_datasets=[], runtime_info=runtime_info,
        annotations=annotations, telemetry=None,
    )
    assert aibom["model"]["name"] == "annotated-model"


def test_compile_aibom_uses_runtime_info_for_training_and_fine_tuning():
    runtime_info = {
        "optimizer": "adamw_bnb_8bit",
        "random_seed": 1234,
        "adaptation_method": "qlora",
        "lora_rank": 16,
        "lora_alpha": 32,
    }
    aibom = pp.compile_aibom(
        discoveries=[], detected_datasets=[], runtime_info=runtime_info,
        annotations={"experiment-intent": "sft"}, telemetry=None,
    )
    assert aibom["training"]["optimizer"] == "adamw_bnb_8bit"
    assert aibom["training"]["random_seed"] == 1234
    assert aibom["fine_tuning"]["adaptation_method"] == "qlora"
    assert aibom["fine_tuning"]["lora_rank"] == 16
    assert aibom["fine_tuning"]["lora_alpha"] == 32


def test_compile_aibom_falls_back_to_device_map_for_parallelization_strategy():
    runtime_info = {"model_device_map": "auto"}
    aibom = pp.compile_aibom(
        discoveries=[], detected_datasets=[], runtime_info=runtime_info,
        annotations={"experiment-intent": "training"}, telemetry=None,
    )
    assert aibom["training"]["parallelization_strategy"] == "model_parallel"


def test_compile_aibom_cli_detected_strategy_overrides_device_map_fallback():
    runtime_info = {"model_device_map": "auto"}
    detected_model = {"parallelization_strategy": "data_parallel"}
    aibom = pp.compile_aibom(
        discoveries=[], detected_datasets=[], runtime_info=runtime_info,
        annotations={"experiment-intent": "training"}, telemetry=None,
        detected_model=detected_model,
    )
    assert aibom["training"]["parallelization_strategy"] == "data_parallel"
