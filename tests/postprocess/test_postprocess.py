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
# Git provenance detection: CLI-parsed `git clone`/`checkout`
# ---------------------------------------------------------------------------


def test_detect_git_clone_from_command_basic():
    tokens = ["git", "clone", "https://github.com/org/repo", "&&", "python", "train.py"]
    assert pp.detect_git_clone_from_command(tokens) == {
        "git_repository": "https://github.com/org/repo",
    }


def test_detect_git_clone_from_command_with_checkout_sha():
    tokens = [
        "git", "clone", "https://github.com/org/repo", "&&",
        "cd", "repo", "&&",
        "git", "checkout", "deadbeefcafe0123", "&&",
        "python", "train.py",
    ]
    result = pp.detect_git_clone_from_command(tokens)
    assert result["git_repository"] == "https://github.com/org/repo"
    assert result["git_commit"] == "deadbeefcafe0123"


def test_detect_git_clone_from_command_with_checkout_branch():
    tokens = [
        "git", "clone", "https://github.com/org/repo", "&&",
        "git", "checkout", "feature/my-branch",
    ]
    result = pp.detect_git_clone_from_command(tokens)
    assert result["git_branch"] == "feature/my-branch"
    assert "git_commit" not in result


def test_detect_git_clone_from_command_branch_flag():
    tokens = ["git", "clone", "--branch", "main", "--depth", "1", "https://github.com/org/repo"]
    result = pp.detect_git_clone_from_command(tokens)
    assert result == {"git_repository": "https://github.com/org/repo", "git_branch": "main"}


def test_detect_git_clone_from_command_no_clone_returns_none():
    assert pp.detect_git_clone_from_command(["python", "train.py"]) is None
    assert pp.detect_git_clone_from_command([]) is None
    assert pp.detect_git_clone_from_command(None) is None


def test_detect_git_clone_from_containers_uses_first_match():
    containers = [
        {"command": ["python", "sidecar.py"]},
        {"command": ["sh", "-c"], "args": ["git clone https://github.com/org/repo && python train.py"]},
    ]
    result = pp.detect_git_clone_from_containers(containers)
    assert result["git_repository"] == "https://github.com/org/repo"
    assert result["detected_via"] == "cli_arg"


def test_detect_git_clone_from_containers_no_match_returns_none():
    assert pp.detect_git_clone_from_containers([{"command": ["python", "train.py"]}]) is None


# ---------------------------------------------------------------------------
# Git provenance detection: runtime .git-directory read (runtime_info)
# ---------------------------------------------------------------------------


def test_detect_git_provenance_from_runtime_info_basic():
    runtime_info = {
        "git_commit": "deadbeef",
        "git_repository": "https://github.com/org/repo.git",
        "git_branch": "main",
        "git_dirty": False,
    }
    result = pp.detect_git_provenance_from_runtime_info(runtime_info)
    assert result == {
        "git_commit": "deadbeef",
        "git_repository": "https://github.com/org/repo.git",
        "git_branch": "main",
        "detected_via": "git_directory",
        "git_dirty": False,
    }


def test_detect_git_provenance_from_runtime_info_omits_dirty_when_absent():
    runtime_info = {"git_commit": "deadbeef"}
    result = pp.detect_git_provenance_from_runtime_info(runtime_info)
    assert "git_dirty" not in result


def test_detect_git_provenance_from_runtime_info_none_when_no_signal():
    assert pp.detect_git_provenance_from_runtime_info({}) is None
    assert pp.detect_git_provenance_from_runtime_info({"learning_rate": 0.1}) is None


# ---------------------------------------------------------------------------
# Git provenance detection: OpenShift/OCI image labels
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


# ---------------------------------------------------------------------------
# collect_telemetry GPU-skip / debug override
# ---------------------------------------------------------------------------


def _no_gpu_discovery():
    return {
        "pod_metadata": {
            "uid": "pod-uid-1",
            "name": "web-pod",
            "start_time": "2026-01-01T00:00:00Z",
        },
        "gpu": {"gpu_count": 0},
    }


def test_collect_telemetry_skips_pod_with_no_gpu(monkeypatch):
    monkeypatch.setattr(pp, "DEBUG_TELEMETRY_ALL_PODS", False)
    summary = pp.collect_telemetry([_no_gpu_discovery()])
    assert summary["pods"] == []


def test_collect_telemetry_debug_flag_includes_pod_with_no_gpu(monkeypatch):
    monkeypatch.setattr(pp, "DEBUG_TELEMETRY_ALL_PODS", True)
    monkeypatch.setattr(pp, "query_prometheus_range", lambda *a, **k: None)
    monkeypatch.setattr(pp, "query_prometheus_instant", lambda *a, **k: None)
    # Avoid the real ingestion-delay retry/sleep loop when summary queries
    # keep returning no data (there's no live Prometheus in this test).
    monkeypatch.setattr(pp, "TELEMETRY_RETRY_ATTEMPTS", 1)
    summary = pp.collect_telemetry([_no_gpu_discovery()])
    assert [p["pod_name"] for p in summary["pods"]] == ["web-pod"]


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


def test_compile_aibom_uses_cli_detected_provenance_with_repository_only():
    # A plain `git clone <url>` with no `checkout` never yields a commit --
    # declared_via should still resolve off git_repository alone.
    detected_provenance = {"git_repository": "https://github.com/org/repo", "detected_via": "cli_arg"}
    aibom = pp.compile_aibom(
        discoveries=[], detected_datasets=[], runtime_info={},
        annotations={"experiment-intent": "training"}, telemetry=None,
        detected_provenance=detected_provenance,
    )
    assert aibom["source_code"]["git_repository"] == "https://github.com/org/repo"
    assert aibom["source_code"]["git_commit"] is None
    assert aibom["source_code"]["declared_via"] == "cli_arg"


def test_compile_aibom_surfaces_dirty_flag_from_runtime_tier():
    detected_provenance = {
        "git_commit": "deadbeef",
        "git_repository": "https://github.com/org/repo",
        "git_branch": "main",
        "detected_via": "git_directory",
        "git_dirty": True,
    }
    aibom = pp.compile_aibom(
        discoveries=[], detected_datasets=[], runtime_info={},
        annotations={"experiment-intent": "training"}, telemetry=None,
        detected_provenance=detected_provenance,
    )
    assert aibom["source_code"]["declared_via"] == "git_directory"
    assert aibom["source_code"]["dirty"] is True


def test_compile_aibom_no_dirty_key_when_tier_does_not_report_it():
    detected_provenance = {"git_commit": "deadbeef", "detected_via": "openshift_build_label"}
    aibom = pp.compile_aibom(
        discoveries=[], detected_datasets=[], runtime_info={},
        annotations={"experiment-intent": "training"}, telemetry=None,
        detected_provenance=detected_provenance,
    )
    assert "dirty" not in aibom["source_code"]


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
# compute_metric_stats
# ---------------------------------------------------------------------------


def _points(*values):
    return [{"timestamp": f"2024-01-01T00:{i:02d}:00", "value": v} for i, v in enumerate(values)]


def test_compute_metric_stats_empty_returns_none():
    assert pp.compute_metric_stats([]) is None


def test_compute_metric_stats_min_max_avg_p95():
    stats = pp.compute_metric_stats(_points(10, 20, 30, 40, 50, 60, 70, 80, 90, 100))
    assert stats["min"] == 10
    assert stats["max"] == 100
    assert stats["avg"] == 55
    assert stats["p95"] == 100


def test_compute_metric_stats_segments_reflect_run_shape():
    # A run that starts hot and cools off -- the average alone hides this.
    stats = pp.compute_metric_stats(_points(90, 90, 90, 50, 50, 50, 10, 10, 10))
    assert stats["segments"]["first_third"] == 90
    assert stats["segments"]["middle_third"] == 50
    assert stats["segments"]["last_third"] == 10


def test_compute_metric_stats_uses_timestamp_order_not_input_order():
    points = [
        {"timestamp": "2024-01-01T00:02:00", "value": 10},
        {"timestamp": "2024-01-01T00:00:00", "value": 90},
        {"timestamp": "2024-01-01T00:01:00", "value": 50},
    ]
    stats = pp.compute_metric_stats(points)
    assert stats["segments"]["first_third"] == 90
    assert stats["segments"]["last_third"] == 10


def test_compute_metric_stats_too_few_points_for_thirds_omits_empty_segments():
    # With fewer than 3 points, first/middle_third have no whole slice to
    # average -- only last_third (the remainder) gets a value.
    stats = pp.compute_metric_stats(_points(10, 20))
    assert stats["segments"] == {"first_third": None, "middle_third": None, "last_third": 15}


# ---------------------------------------------------------------------------
# compile_aibom: resource_utilization from segmented telemetry
# ---------------------------------------------------------------------------


def _pod_metrics(avg, min_, max_, p95, unit="percent"):
    return {
        "data_point_count": 10,
        "unit": unit,
        "min": min_,
        "max": max_,
        "avg": avg,
        "p95": p95,
        "segments": {"first_third": max_, "middle_third": avg, "last_third": min_},
    }


def test_compile_aibom_utilization_reports_segmented_metrics():
    telemetry = {
        "collected_at": "2024-01-01T00:00:00Z",
        "pods": [
            {
                "pod_name": "job-abc",
                "metrics": {"gpu_utilization": _pod_metrics(avg=60, min_=10, max_=95, p95=94)},
            }
        ],
    }
    aibom = pp.compile_aibom(
        discoveries=[], detected_datasets=[], runtime_info={},
        annotations={}, telemetry=telemetry,
    )
    utilization = aibom["resource_utilization"]
    assert utilization["avg_gpu_utilization_pct"] == 60
    detail = utilization["metrics"]["gpu_utilization"]
    assert detail == {
        "unit": "percent",
        "min": 10,
        "max": 95,
        "avg": 60,
        "p95": 94,
        "segments": {"first_third": 95, "middle_third": 60, "last_third": 10},
    }


def test_compile_aibom_utilization_merges_jobset_sibling_pods():
    telemetry = {
        "collected_at": "2024-01-01T00:00:00Z",
        "pods": [
            {
                "pod_name": "server-0",
                "metrics": {"gpu_utilization": _pod_metrics(avg=40, min_=5, max_=80, p95=75)},
            },
            {
                "pod_name": "server-1",
                "metrics": {"gpu_utilization": _pod_metrics(avg=60, min_=20, max_=99, p95=95)},
            },
        ],
    }
    aibom = pp.compile_aibom(
        discoveries=[], detected_datasets=[], runtime_info={},
        annotations={}, telemetry=telemetry,
    )
    detail = aibom["resource_utilization"]["metrics"]["gpu_utilization"]
    # True min/max across sibling pods; avg/p95 averaged across them.
    assert detail["min"] == 5
    assert detail["max"] == 99
    assert detail["avg"] == 50
    assert detail["p95"] == 85


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
