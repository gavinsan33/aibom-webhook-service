import base64
import json
import re
import time
from datetime import datetime, timezone

import pytest

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
# Other serving engines: TGI, SGLang, LMDeploy (all share the vLLM
# detection shape; here we check the engine is identified, the model name is
# pulled out, and non-matching commands return None)
# ---------------------------------------------------------------------------


def test_detect_tgi_from_command():
    result = pp.detect_tgi_from_command(
        ["text-generation-launcher", "--model-id", "bigscience/bloom-560m", "--num-shard", "2", "--quantize", "bitsandbytes"]
    )
    assert result["serving_engine"] == "tgi"
    assert result["model_name"] == "bigscience/bloom-560m"
    assert result["tensor_parallel_size"] == 2
    assert result["quantization"] == "bitsandbytes"


def test_detect_tgi_through_shell_wrapper():
    # `detect_model_from_containers` flattens the sh -c wrapper before detection.
    result = pp.detect_model_from_containers(
        [{"command": ["sh", "-c", "pip install x && text-generation-launcher --model-id org/m --max-total-tokens 4096"]}]
    )
    assert result["serving_engine"] == "tgi"
    assert result["model_name"] == "org/m"
    assert result["max_model_len"] == 4096


def test_detect_tgi_entrypoint_only_image():
    container = {"image": "ghcr.io/huggingface/text-generation-inference:3.0", "args": ["--model-id", "org/m"]}
    result = pp._detect_tgi_from_entrypoint_args(container)
    assert result == {"serving_engine": "tgi", "model_name": "org/m"}


def test_detect_sglang_from_command():
    result = pp.detect_sglang_from_command(
        ["python", "-m", "sglang.launch_server", "--model-path", "Qwen/Qwen2.5-7B", "--tp-size", "4", "--context-length", "8192"]
    )
    assert result["serving_engine"] == "sglang"
    assert result["model_name"] == "Qwen/Qwen2.5-7B"
    assert result["tensor_parallel_size"] == 4
    assert result["max_model_len"] == 8192


def test_detect_lmdeploy_positional_model():
    result = pp.detect_lmdeploy_from_command(
        ["lmdeploy", "serve", "api_server", "internlm/internlm2-chat-7b", "--tp", "2", "--session-len", "4096"]
    )
    assert result["serving_engine"] == "lmdeploy"
    assert result["model_name"] == "internlm/internlm2-chat-7b"
    assert result["tensor_parallel_size"] == 2
    assert result["max_model_len"] == 4096


def test_detect_lmdeploy_quantization_from_name():
    result = pp.detect_lmdeploy_from_command(["lmdeploy", "serve", "api_server", "org/Llama-3-8B-AWQ"])
    assert result["quantization_method"] == "awq"


def test_detect_lmdeploy_quantization_from_flag():
    result = pp.detect_lmdeploy_from_command(
        ["lmdeploy", "serve", "api_server", "org/m", "--model-format", "awq", "--cache-max-entry-count", "0.8"]
    )
    assert result["quantization"] == "awq"
    assert result["gpu_memory_utilization"] == 0.8


def test_detect_sglang_mem_fraction():
    result = pp.detect_sglang_from_command(
        ["python", "-m", "sglang.launch_server", "--model-path", "org/m", "--mem-fraction-static", "0.85"]
    )
    assert result["gpu_memory_utilization"] == 0.85


def test_new_engines_return_none_for_unrelated_commands():
    for fn in (pp.detect_tgi_from_command, pp.detect_sglang_from_command, pp.detect_lmdeploy_from_command):
        assert fn(["python", "train.py"]) is None
        assert fn([]) is None
        assert fn(None) is None
    # `lmdeploy chat` / `lmdeploy serve gradio` aren't the api_server.
    assert pp.detect_lmdeploy_from_command(["lmdeploy", "chat", "org/m"]) is None


def test_detect_model_from_containers_identifies_each_engine():
    cases = {
        "tgi": [{"command": ["text-generation-launcher", "--model-id", "org/m"]}],
        "sglang": [{"command": ["python", "-m", "sglang.launch_server", "--model-path", "org/m"]}],
        "lmdeploy": [{"command": ["lmdeploy", "serve", "api_server", "org/m"]}],
    }
    for engine, containers in cases.items():
        assert pp.detect_model_from_containers(containers)["serving_engine"] == engine


@pytest.mark.parametrize("image", [
    "ghcr.io/llm-d/llm-d-xpu:v1.0.0",
    "ghcr.io/llm-d/llm-d-rocm@sha256:" + "a" * 64,
    "quay.io/mirror/llm-d-cpu:v1",
])
def test_llm_d_model_server_image_detected_as_vllm(image):
    result = pp._detect_vllm_from_entrypoint_args({"image": image, "args": ["--model", "org/m", "--port", "8200"]})
    assert result["serving_engine"] == "vllm"
    assert result["model_name"] == "org/m"


@pytest.mark.parametrize("image", [
    "ghcr.io/llm-d/llm-d-router-disagg-sidecar:v1.0.0",
    "ghcr.io/llm-d/llm-d-routing-sidecar:v0.2.0",
    "ghcr.io/llm-d/llm-d-inference-scheduler:v0.2.0",
    "ghcr.io/llm-d/llm-d-router-endpoint-picker:v1.0.0",
    "ghcr.io/llm-d/llm-d-inference-sim:latest",
    "ghcr.io/llm-d/llm-d-xpu-sglang:v0.10.0",
    # ENTRYPOINT is NVIDIA's wrapper, not vLLM: args alone don't start a server.
    "ghcr.io/llm-d/llm-d-cuda:v0.9.0",
])
def test_llm_d_non_vllm_images_not_detected(image):
    assert pp._detect_vllm_from_entrypoint_args({"image": image, "args": ["--port=8000"]}) is None


def test_llm_d_routing_sidecar_listed_first_does_not_hide_vllm():
    # A routing sidecar run as a regular container ahead of vLLM (llm-d's own
    # charts make it a native sidecar, which containers.json never includes).
    containers = [
        {"name": "routing-proxy", "image": "ghcr.io/llm-d/llm-d-router-disagg-sidecar:v1.0.0",
         "args": ["--port=8000", "--model-server-port=8200", "--kv-connector=nixlv2"]},
        {"name": "vllm", "image": "ghcr.io/llm-d/llm-d-xpu:v1.0.0",
         "args": ["--model", "Qwen/Qwen3-0.6B", "--port", "8200"]},
    ]
    result = pp.detect_model_from_containers(containers)
    assert result["serving_engine"] == "vllm"
    assert result["model_name"] == "Qwen/Qwen3-0.6B"


# ---------------------------------------------------------------------------
# Per-engine serving telemetry query generation
# ---------------------------------------------------------------------------


def test_build_serving_queries_reproduces_vllm():
    """The generator must reproduce the hand-verified vLLM queries byte-for-byte
    -- this is what lets every other engine inherit their confirmed correctness.
    If this breaks, the generated PromQL for all four engines is suspect."""
    stats = pp.SERVING_TELEMETRY_QUERIES["vllm"]
    assert stats["time_to_first_token_seconds"]["query"] == (
        'sum(rate(vllm:time_to_first_token_seconds_sum{namespace="{namespace}", pod="{pod_name}"}[5m]))'
        ' / sum(rate(vllm:time_to_first_token_seconds_count{namespace="{namespace}", pod="{pod_name}"}[5m]))'
    )
    assert stats["inter_token_latency_seconds"]["query"] == (
        '(sum(rate(vllm:inter_token_latency_seconds_sum{namespace="{namespace}", pod="{pod_name}"}[5m]))'
        ' / sum(rate(vllm:inter_token_latency_seconds_count{namespace="{namespace}", pod="{pod_name}"}[5m])))'
        ' or (sum(rate(vllm:time_per_output_token_seconds_sum{namespace="{namespace}", pod="{pod_name}"}[5m]))'
        ' / sum(rate(vllm:time_per_output_token_seconds_count{namespace="{namespace}", pod="{pod_name}"}[5m])))'
    )
    assert stats["num_requests_running"]["query"] == 'sum(vllm:num_requests_running{namespace="{namespace}", pod="{pod_name}"})'
    assert stats["kv_cache_usage"]["query"] == (
        '100 * (avg(vllm:kv_cache_usage_perc{namespace="{namespace}", pod="{pod_name}"})'
        ' or avg(vllm:gpu_cache_usage_perc{namespace="{namespace}", pod="{pod_name}"}))'
    )
    assert stats["prompt_throughput"]["query"] == 'sum(rate(vllm:prompt_tokens_total{namespace="{namespace}", pod="{pod_name}"}[5m]))'

    series = pp.SERVING_SERIES_QUERIES["vllm"]
    assert series["time_to_first_token_seconds"]["query"] == (
        'sum by (pod) (rate(vllm:time_to_first_token_seconds_sum{namespace="{namespace}", pod=~"{pod_regex}"}[5m]))'
        ' / sum by (pod) (rate(vllm:time_to_first_token_seconds_count{namespace="{namespace}", pod=~"{pod_regex}"}[5m]))'
    )
    assert series["num_requests_running"]["query"] == (
        '{fn} by (pod) ({fn}_over_time(vllm:num_requests_running{namespace="{namespace}", pod=~"{pod_regex}"}[{win}s]))'
    )
    assert series["num_requests_running"]["gauge"] is True
    assert series["num_requests_running"]["aggregation"] == "sum"
    assert series["kv_cache_usage"]["query"] == (
        '100 * ({fn} by (pod) ({fn}_over_time(vllm:kv_cache_usage_perc{namespace="{namespace}", pod=~"{pod_regex}"}[{win}s]))'
        ' or {fn} by (pod) ({fn}_over_time(vllm:gpu_cache_usage_perc{namespace="{namespace}", pod=~"{pod_regex}"}[{win}s])))'
    )


def test_serving_query_sets_consistent_across_engines():
    # Same canonical metric keys everywhere (TGI is a documented subset), and
    # stats/series sets always match within an engine.
    for engine in pp.SERVING_ENGINE_METRICS:
        assert set(pp.SERVING_TELEMETRY_QUERIES[engine]) == set(pp.SERVING_SERIES_QUERIES[engine])
        assert set(pp.SERVING_TELEMETRY_QUERIES[engine]) <= set(pp.SERVING_TELEMETRY_QUERIES["vllm"])
    assert pp.SERVING_TELEMETRY_QUERIES["lmdeploy"]["kv_cache_usage"]["query"].startswith("100 * ")
    assert "tgi_request_mean_time_per_token_duration_sum" in pp.SERVING_TELEMETRY_QUERIES["tgi"]["inter_token_latency_seconds"]["query"]
    # TGI's prompt tokens come from the input-length histogram's _sum, and it
    # has no TTFT or KV-cache metric to query.
    assert pp.SERVING_TELEMETRY_QUERIES["tgi"]["prompt_throughput"]["query"] == (
        'sum(rate(tgi_request_input_length_sum{namespace="{namespace}", pod="{pod_name}"}[5m]))'
    )
    assert set(pp.SERVING_TELEMETRY_QUERIES["vllm"]) - set(pp.SERVING_TELEMETRY_QUERIES["tgi"]) == {
        "time_to_first_token_seconds",
        "kv_cache_usage",
    }


def test_collect_vllm_telemetry_uses_selected_engine(monkeypatch):
    seen = []
    monkeypatch.setattr(pp, "query_prometheus_range", lambda q, *a, **k: seen.append(q) or None)
    pp.collect_vllm_telemetry([_serving_discovery()], engine="sglang")
    assert any("sglang:" in q for q in seen)
    assert not any("vllm:" in q for q in seen)


# ---------------------------------------------------------------------------
# KServe InferenceService storage-path detection (S3/MinIO data-connection)
# ---------------------------------------------------------------------------


def test_detect_model_from_storage_uses_last_path_segment():
    result = pp.detect_model_from_storage({"storage_path": "models/tinyllama-1.1b-chat"})
    assert result == {"model_name": "tinyllama-1.1b-chat", "model_name_declared_via": "storage_path"}


def test_detect_model_from_storage_falls_back_to_storage_uri():
    result = pp.detect_model_from_storage({"storage_uri": "s3://bucket/models/granite-3.0-8b-instruct/"})
    assert result == {"model_name": "granite-3.0-8b-instruct", "model_name_declared_via": "storage_path"}


def test_detect_model_from_storage_infers_quantization_from_name():
    result = pp.detect_model_from_storage({"storage_path": "models/some-model-AWQ"})
    assert result["model_name"] == "some-model-AWQ"
    assert result["quantization_method"] == "awq"


def test_detect_model_from_storage_hf_uri_keeps_org():
    result = pp.detect_model_from_storage({"storage_uri": "hf://ibm-granite/granite-3.3-2b-instruct"})
    assert result == {"model_name": "ibm-granite/granite-3.3-2b-instruct", "model_name_declared_via": "hf_uri"}


def test_detect_model_from_storage_hf_uri_splits_revision():
    result = pp.detect_model_from_storage({"storage_uri": "hf://TheBloke/Llama-2-7B-AWQ:abc123def"})
    assert result["model_name"] == "TheBloke/Llama-2-7B-AWQ"
    assert result["model_revision"] == "abc123def"
    assert result["quantization_method"] == "awq"


def test_detect_model_from_storage_oci_uri_uses_pulled_digest():
    containers = [{"image": "quay.io/acme/modelcar-granite:1.0", "image_id": "quay.io/acme/modelcar-granite@sha256:" + "a" * 64}]
    result = pp.detect_model_from_storage({"storage_uri": "oci://quay.io/acme/modelcar-granite:1.0"}, containers)
    assert result == {
        "model_name": "quay.io/acme/modelcar-granite",
        "model_name_declared_via": "oci_uri",
        "model_revision": "sha256:" + "a" * 64,
    }


def test_detect_model_from_storage_oci_uri_without_matching_container():
    result = pp.detect_model_from_storage({"storage_uri": "oci://localhost:5000/acme/m:1.0"}, [])
    assert result == {"model_name": "localhost:5000/acme/m", "model_name_declared_via": "oci_uri"}


def test_detect_model_from_storage_prefers_model_files_repo_id():
    result = pp.detect_model_from_storage({
        "storage_uri": "pvc://llm-serving-storage/qwen-32b",
        "model_files": {
            "repo_id": "Qwen/Qwen2.5-32B-Instruct",
            "base_model": "Qwen/Qwen2.5-32B",
            "revision": "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd",
            "architectures": ["Qwen2ForCausalLM"],
            "dtype": "bfloat16",
            "total_size_bytes": 65527752704,
        },
    })
    assert result == {
        "model_name": "Qwen/Qwen2.5-32B-Instruct",
        "model_name_declared_via": "model_files_readme",
        "dtype": "bfloat16",
        "architecture": "Qwen2ForCausalLM",
        "model_revision": "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd",
        "base_model": "Qwen/Qwen2.5-32B",
        "model_size_bytes": 65527752704,
    }


def test_detect_model_from_storage_model_files_without_repo_id_keeps_folder_name():
    result = pp.detect_model_from_storage({
        "storage_uri": "pvc://llm-serving-storage/qwen-32b",
        "model_files": {"dtype": "bfloat16"},
    })
    assert result["model_name"] == "qwen-32b"
    assert result["model_name_declared_via"] == "storage_path"
    assert result["dtype"] == "bfloat16"


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
# Axolotl / LLaMA-Factory fine-tuning detection
# ---------------------------------------------------------------------------


def test_detect_axolotl_kebab_overrides():
    # Modern Click CLI: kebab-case overrides after the config path.
    result = pp.detect_axolotl_from_command(
        ["axolotl", "train", "config.yml", "--base-model", "org/m",
         "--lora-r", "16", "--lora-alpha", "32", "--adapter", "qlora",
         "--learning-rate", "2e-4", "--num-epochs", "3"]
    )
    assert result["training_framework"] == "axolotl"
    assert result["model_name"] == "org/m"
    assert result["lora_rank"] == 16
    assert result["adaptation_method"] == "qlora"
    assert result["learning_rate"] == 2e-4
    assert result["epochs"] == 3


def test_detect_axolotl_lora_plus_4bit_is_qlora():
    result = pp.detect_axolotl_from_command(
        ["axolotl", "train", "c.yml", "--adapter", "lora", "--load-in-4bit"]
    )
    assert result["adaptation_method"] == "qlora"


def test_detect_axolotl_legacy_module_form_config_only():
    # Legacy form, config-only: framework detected, nothing from the YAML.
    result = pp.detect_axolotl_from_command(
        ["python", "-m", "axolotl.cli.train", "qlora.yml"]
    )
    assert result == {"training_framework": "axolotl"}


def test_detect_llamafactory_flag_style_qlora():
    result = pp.detect_llamafactory_from_command(
        ["llamafactory-cli", "train", "--model_name_or_path", "org/m",
         "--finetuning_type", "lora", "--lora_rank", "8", "--quantization_bit", "4",
         "--num_train_epochs", "2.0", "--per_device_train_batch_size", "8"]
    )
    assert result["training_framework"] == "llama-factory"
    assert result["model_name"] == "org/m"
    assert result["lora_rank"] == 8
    assert result["adaptation_method"] == "qlora"
    assert result["epochs"] == 2.0
    assert result["batch_size"] == 8


def test_detect_llamafactory_full_finetune_has_no_adapter():
    result = pp.detect_llamafactory_from_command(
        ["llamafactory-cli", "train", "--model_name_or_path", "org/m", "--finetuning_type", "full"]
    )
    assert "adaptation_method" not in result


def test_detect_llamafactory_config_only():
    result = pp.detect_llamafactory_from_command(["llamafactory-cli", "train", "sft.yaml"])
    assert result == {"training_framework": "llama-factory"}


def test_axolotl_llamafactory_return_none_for_unrelated():
    for fn in (pp.detect_axolotl_from_command, pp.detect_llamafactory_from_command):
        assert fn(["python", "train.py"]) is None
        assert fn([]) is None
        assert fn(None) is None
    # Non-train subcommands aren't training runs.
    assert pp.detect_axolotl_from_command(["axolotl", "preprocess", "c.yml"]) is None
    assert pp.detect_llamafactory_from_command(["llamafactory-cli", "export", "c.yaml"]) is None


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


REDACTION_CASES = [
    ("https://user:ghp_SECRET@github.com/org/repo.git", "https://github.com/org/repo.git"),
    ("https://ghp_SECRET@github.com/org/repo", "https://github.com/org/repo"),
    ("https://x-access-token:ghp_SECRET@github.com:8443/org/repo", "https://github.com:8443/org/repo"),
    ("https://github.com/org/repo.git?private_token=SECRET", "https://github.com/org/repo.git"),
    ("https://github.com/org/repo.git?token=SECRET#frag", "https://github.com/org/repo.git"),
    ("ssh://git:SECRET@host/org/repo.git", "ssh://git@host/org/repo.git"),
    ("ssh://git@host/org/repo.git", "ssh://git@host/org/repo.git"),
    ("git@github.com:org/repo.git", "git@github.com:org/repo.git"),
    ("https://github.com/org/repo", "https://github.com/org/repo"),
    ("/local/path/repo", "/local/path/repo"),
    (None, None),
    ("", ""),
]


@pytest.mark.parametrize("url,expected", REDACTION_CASES)
def test_redact_git_url(url, expected):
    assert pp.redact_git_url(url) == expected


def test_redact_git_url_unparseable_still_drops_userinfo():
    out = pp.redact_git_url("https://user:SECRET@[::1/org/repo?token=SECRET")
    assert "SECRET" not in out


def test_detect_git_clone_from_command_strips_credentials():
    tokens = ["git", "clone", "https://x-access-token:ghp_SECRET@github.com/org/repo", "&&", "python", "train.py"]
    assert pp.detect_git_clone_from_command(tokens) == {"git_repository": "https://github.com/org/repo"}


def test_detect_git_provenance_from_runtime_info_strips_credentials():
    result = pp.detect_git_provenance_from_runtime_info(
        {"git_commit": "abc1234", "git_repository": "https://u:SECRET@github.com/org/repo"}
    )
    assert result["git_repository"] == "https://github.com/org/repo"


def test_compile_aibom_redacts_annotation_and_detected_repository():
    for annotations, provenance in (
        ({"git-repository": "https://u:SECRET@github.com/org/repo"}, None),
        ({}, {"git_repository": "https://u:SECRET@github.com/org/repo", "detected_via": "future_tier"}),
    ):
        aibom = pp.compile_aibom(
            discoveries=[], detected_datasets=[], runtime_info={}, annotations=annotations,
            telemetry=None, detected_model=None, cli_dataset=None,
            detected_provenance=provenance,
        )
        assert aibom["source_code"]["git_repository"] == "https://github.com/org/repo"
        assert "SECRET" not in json.dumps(aibom)


def test_pod_gpu_count_sums_claims():
    containers = [
        {"pod_name": "p1", "gpu_limit": 2},
        {"pod_name": "p1", "gpu_limit": 2},
        {"pod_name": "p1"},            # no GPU claim -> ignored
        {"pod_name": "other", "gpu_limit": 8},
    ]
    assert pp.pod_gpu_count("p1", containers) == 4
    assert pp.pod_gpu_count("none", containers) is None
    assert pp.pod_gpu_count("p1", None) is None


def test_compile_aibom_prefers_pod_spec_gpu_count_over_nvidia_smi(monkeypatch):
    # Discovery's in-pod nvidia-smi miscounts (reports 1); the pod-spec claim (4)
    # is authoritative. DCGM off so gpu_type still comes from discovery.
    monkeypatch.setattr(pp, "_dcgm_gpu_hardware", lambda pod: {})
    discoveries = [{
        "pod_metadata": {"name": "p1", "uid": "u1"},
        "gpu": {"gpu_count": "1", "gpu_models": "NVIDIA A100"},
        "system": {},
    }]
    containers = [
        {"pod_name": "p1", "gpu_limit": 2},
        {"pod_name": "p1", "gpu_limit": 2},
    ]
    aibom = pp.compile_aibom(
        discoveries=discoveries, detected_datasets=[], runtime_info={}, annotations={},
        telemetry=None, containers=containers,
    )
    assert aibom["environment"]["gpu_count"] == 4
    assert aibom["environment"]["gpu_type"] == "NVIDIA A100"


def test_compile_aibom_falls_back_to_discovery_gpu_count_without_claim(monkeypatch):
    monkeypatch.setattr(pp, "_dcgm_gpu_hardware", lambda pod: {})
    discoveries = [{
        "pod_metadata": {"name": "p1"},
        "gpu": {"gpu_count": "8"},
        "system": {},
    }]
    aibom = pp.compile_aibom(
        discoveries=discoveries, detected_datasets=[], runtime_info={}, annotations={},
        telemetry=None, containers=[],
    )
    assert aibom["environment"]["gpu_count"] == "8"


def test_dcgm_gpu_hardware_parses_models_and_memory(monkeypatch):
    monkeypatch.setattr(pp, "JOB_NAMESPACE", "ns")
    response = {"status": "success", "data": {"result": [
        {"metric": {"gpu": "1", "modelName": "NVIDIA A100-SXM4-40GB"}, "value": [1.7e9, "40960"]},
        {"metric": {"gpu": "0", "modelName": "NVIDIA A100-SXM4-40GB"}, "value": [1.7e9, "40960"]},
    ]}}
    monkeypatch.setattr(pp, "query_prometheus_instant", lambda promql, t: response)
    hw = pp._dcgm_gpu_hardware("p1")
    assert hw["gpu_models"] == "NVIDIA A100-SXM4-40GB\nNVIDIA A100-SXM4-40GB"
    assert hw["gpu_memory_mb"] == [40960, 40960]  # ordered by gpu index (0, then 1)


def test_dcgm_gpu_hardware_orders_gpu_indices_numerically(monkeypatch):
    monkeypatch.setattr(pp, "JOB_NAMESPACE", "ns")
    result = [
        {"metric": {"gpu": str(i), "modelName": "H100"}, "value": [1.7e9, str(1000 + i)]}
        for i in (10, 2, 11, 0, 9, 1)
    ]
    monkeypatch.setattr(pp, "query_prometheus_instant",
                        lambda promql, t: {"status": "success", "data": {"result": result}})
    hw = pp._dcgm_gpu_hardware("p1")
    assert hw["gpu_memory_mb"] == [1000, 1001, 1002, 1009, 1010, 1011]


def test_dcgm_gpu_hardware_empty_when_no_series(monkeypatch):
    monkeypatch.setattr(pp, "JOB_NAMESPACE", "ns")
    monkeypatch.setattr(pp, "query_prometheus_instant",
                        lambda promql, t: {"status": "success", "data": {"result": []}})
    assert pp._dcgm_gpu_hardware("p1") == {}


def test_dcgm_gpu_hardware_degrades_quietly_on_error(monkeypatch):
    monkeypatch.setattr(pp, "JOB_NAMESPACE", "ns")

    def raising(promql, t):
        raise RuntimeError("thanos down")

    monkeypatch.setattr(pp, "query_prometheus_instant", raising)
    assert pp._dcgm_gpu_hardware("p1") == {}


def test_load_datasets_redacts_runtime_info_repository(tmp_path, monkeypatch):
    (tmp_path / "dataset.json").write_text(json.dumps(
        {"datasets": [], "runtime_info": {"git_repository": "https://u:SECRET@github.com/org/repo"}}
    ))
    monkeypatch.setattr(pp, "INPUT_DIR", str(tmp_path))
    _, runtime_info = pp.load_datasets()
    assert runtime_info["git_repository"] == "https://github.com/org/repo"


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


@pytest.mark.parametrize("script, expected", [
    # Option values aren't the repository (#113).
    ("git clone --reference /cache/mirror.git https://h/a.git", {"git_repository": "https://h/a.git"}),
    ("git clone https://h/a.git && cd a && git checkout -b exp origin/exp",
     {"git_repository": "https://h/a.git", "git_branch": "exp"}),
    ("git clone https://h/a.git && cd a && git checkout --detach 0123abc",
     {"git_repository": "https://h/a.git", "git_commit": "0123abc"}),
    # FETCH_HEAD says nothing, and clears the clone's -b.
    ("git clone -b dev https://h/a.git && cd a && git fetch origin pull/1/head && git checkout FETCH_HEAD",
     {"git_repository": "https://h/a.git"}),
    ("git clone https://h/a.git && git -C a checkout 0123abcd",
     {"git_repository": "https://h/a.git", "git_commit": "0123abcd"}),
    ("git clone https://h/a.git && cd a\npip install -e .\ngit checkout -b foo",
     {"git_repository": "https://h/a.git", "git_branch": "foo"}),
    # Several clones: the one the script cd's into, with its own checkout.
    ("git clone https://h/a.git && git clone https://huggingface.co/datasets/x/y && cd a && git checkout 1a2b3c4d",
     {"git_repository": "https://h/a.git", "git_commit": "1a2b3c4d"}),
    ("git clone https://h/a.git && git clone https://h/b.git && cd b && git checkout dev && python train.py",
     {"git_repository": "https://h/b.git", "git_branch": "dev"}),
    ("git clone https://h/a.git code && cd /w/code && git checkout v1",
     {"git_repository": "https://h/a.git", "git_branch": "v1"}),
    # All-digit refs are tags; a 64-hex ref is a SHA-256 commit.
    ("git clone https://h/a.git && cd a && git checkout 20240101",
     {"git_repository": "https://h/a.git", "git_branch": "20240101"}),
    ("git clone https://h/a.git && cd a && git checkout " + "a" * 64,
     {"git_repository": "https://h/a.git", "git_commit": "a" * 64}),
    ("git clone $REPO && cd x", None),
])
def test_detect_git_clone_from_command_cases(script, expected):
    tokens = pp._flatten_container_command({"command": ["sh", "-c", script]})
    assert pp.detect_git_clone_from_command(tokens) == expected


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


def test_detect_git_provenance_prefers_openshift_label_when_labels_agree(monkeypatch):
    containers = [{"image_id": "quay.io/org/train@sha256:abc123"}]
    image = {"dockerImageMetadata": {"Config": {"Labels": {
        "io.openshift.build.commit.id": "deadbeef",
        "io.openshift.build.commit.ref": "main",
        "org.opencontainers.image.revision": "deadbeef",
    }}}}
    monkeypatch.setattr(pp.k8s_api, "get_cluster_object", lambda *a, **k: image)
    result = pp.detect_git_provenance_from_containers(containers)
    assert result["git_commit"] == "deadbeef"
    assert result["detected_via"] == "openshift_build_label"


def test_detect_git_provenance_ignores_conflicting_labels(monkeypatch):
    # One of them was inherited from a base image via FROM (#113).
    containers = [{"image_id": "quay.io/org/train@sha256:abc123"}]
    image = {"dockerImageMetadata": {"Config": {"Labels": {
        "io.openshift.build.commit.id": "deadbeef",
        "org.opencontainers.image.revision": "cafef00d",
    }}}}
    monkeypatch.setattr(pp.k8s_api, "get_cluster_object", lambda *a, **k: image)
    assert pp.detect_git_provenance_from_containers(containers) is None


def test_detect_git_provenance_skips_sidecars_and_uses_spec_digest(monkeypatch):
    # istio-proxy is skipped; a bare sha256: image_id falls back to the
    # digest-pinned spec image (#113).
    containers = [
        {"name": "istio-proxy", "image_id": "quay.io/istio/proxy@sha256:mesh"},
        {"name": "trainer", "image_id": "sha256:config", "image": "quay.io/org/train@sha256:app"},
    ]
    looked_up = []

    def fake_get(group, version, plural, name):
        looked_up.append(name)
        return {"dockerImageMetadata": {"Config": {"Labels": {"org.opencontainers.image.revision": "cafef00d"}}}}

    monkeypatch.setattr(pp.k8s_api, "get_cluster_object", fake_get)
    assert pp.detect_git_provenance_from_containers(containers)["git_commit"] == "cafef00d"
    assert looked_up == ["sha256:app"]


def test_detect_git_provenance_follows_manifest_list(monkeypatch):
    images = {
        "sha256:list": {"dockerImageManifests": [{"digest": "sha256:amd64"}, {"digest": "sha256:arm64"}]},
        "sha256:amd64": {"dockerImageMetadata": {"Config": {"Labels": {
            "org.opencontainers.image.revision": "cafef00d",
        }}}},
    }
    monkeypatch.setattr(pp.k8s_api, "get_cluster_object", lambda g, v, p, name: images.get(name))
    containers = [{"image_id": "quay.io/org/train@sha256:list"}]
    assert pp.detect_git_provenance_from_containers(containers)["git_commit"] == "cafef00d"


@pytest.mark.parametrize("primary, fallback, expected", [
    # A .git without a remote is completed from the clone command.
    ({"git_commit": "abc1234", "detected_via": "git_directory"},
     {"git_repository": "https://h/a.git", "detected_via": "cli_arg"},
     {"git_commit": "abc1234", "git_repository": "https://h/a.git", "detected_via": "git_directory"}),
    # Abbreviated vs full SHA and scp vs https URL are the same checkout.
    ({"git_commit": "abc1234", "git_repository": "git@h:org/a.git"},
     {"git_commit": "abc1234" + "0" * 33, "git_repository": "https://h/org/a", "git_branch": "main"},
     {"git_commit": "abc1234", "git_repository": "git@h:org/a.git", "git_branch": "main"}),
    # A different commit or repo is a different checkout: nothing is mixed in.
    ({"git_commit": "abc1234"}, {"git_commit": "fff9999", "git_repository": "https://h/b"}, {"git_commit": "abc1234"}),
    ({"git_repository": "https://h/a"}, {"git_repository": "https://h/b", "git_dirty": True}, {"git_repository": "https://h/a"}),
    (None, {"git_commit": "abc1234"}, {"git_commit": "abc1234"}),
])
def test_merge_provenance(primary, fallback, expected):
    assert pp.merge_provenance(primary, fallback) == expected


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


_REAL_REGISTRY_GET = pp._registry_get


@pytest.fixture(autouse=True)
def _no_registry_network(monkeypatch):
    def offline(*a, **k):
        raise RuntimeError("registry access disabled in tests")

    monkeypatch.setattr(pp, "_registry_get", offline)


@pytest.mark.parametrize("ref, expected", [
    ("python@sha256:ab", ("registry-1.docker.io", "library/python")),
    ("docker.io/library/python@sha256:ab", ("registry-1.docker.io", "library/python")),
    ("docker-pullable://org/train@sha256:ab", ("registry-1.docker.io", "org/train")),
    ("quay.io/org/train:1.0@sha256:ab", ("quay.io", "org/train")),
    ("localhost:5000/train@sha256:ab", ("localhost:5000", "train")),
    ("ghcr.io/org/sub/train@sha256:ab", ("ghcr.io", "org/sub/train")),
    ("sha256:ab", None),
])
def test_parse_image_ref(ref, expected):
    assert pp._parse_image_ref(ref) == expected


def _digest(obj):
    body = json.dumps(obj).encode()
    return "sha256:" + pp.hashlib.sha256(body).hexdigest(), body


class _FakeRegistry:
    """Serves blobs by URL and demands a bearer token, like Docker Hub."""

    def __init__(self, blobs, token="tok"):
        self.blobs, self.token, self.calls = blobs, token, []

    def __call__(self, url, accept=None, token=None, hops=0):
        self.calls.append((url, token))
        if url.startswith("https://auth.example/token"):
            assert "scope=repository%3Aorg%2Ftrain%3Apull" in url
            return json.dumps({"token": self.token}).encode()
        if token != self.token:
            headers = {"WWW-Authenticate": 'Bearer realm="https://auth.example/token",service="reg"'}
            raise pp.urllib.error.HTTPError(url, 401, "unauthorized", headers, None)
        return self.blobs[url]


def _registry_image(labels):
    config_digest, config = _digest({"config": {"Labels": labels}})
    manifest_digest, manifest = _digest({"config": {"digest": config_digest}})
    index_digest, index = _digest({"manifests": [
        {"digest": "sha256:attestation", "platform": {"os": "unknown"}},
        {"digest": manifest_digest, "platform": {"os": "linux", "architecture": "amd64"}},
    ]})
    base = "https://quay.io/v2/org/train"
    blobs = {
        f"{base}/manifests/{index_digest}": index,
        f"{base}/manifests/{manifest_digest}": manifest,
        f"{base}/blobs/{config_digest}": config,
    }
    return index_digest, blobs


def test_detect_git_provenance_reads_labels_from_registry_without_image_object(monkeypatch):
    digest, blobs = _registry_image({
        "org.opencontainers.image.revision": "cafef00d",
        "org.opencontainers.image.source": "https://github.com/org/train",
    })
    registry = _FakeRegistry(blobs)
    monkeypatch.setattr(pp, "_registry_get", registry)
    monkeypatch.setattr(pp.k8s_api, "get_cluster_object", lambda *a, **k: None)
    containers = [{"image_id": f"quay.io/org/train@{digest}"}]
    assert pp.detect_git_provenance_from_containers(containers) == {
        "git_commit": "cafef00d",
        "git_repository": "https://github.com/org/train",
        "git_branch": None,
        "detected_via": "oci_image_label",
    }
    # One token request, reused for the child manifest and the config blob.
    assert sum("auth.example" in url for url, _ in registry.calls) == 1


def test_registry_labels_rejected_on_digest_mismatch(monkeypatch):
    digest, blobs = _registry_image({"org.opencontainers.image.revision": "cafef00d"})
    blobs[f"https://quay.io/v2/org/train/manifests/{digest}"] = b'{"manifests": []}'
    monkeypatch.setattr(pp, "_registry_get", _FakeRegistry(blobs))
    monkeypatch.setattr(pp.k8s_api, "get_cluster_object", lambda *a, **k: None)
    assert pp.detect_git_provenance_from_containers([{"image_id": f"quay.io/org/train@{digest}"}]) is None


def test_registry_lookup_skipped_when_image_object_has_labels_or_disabled(monkeypatch):
    def fail(*a, **k):
        raise AssertionError("registry should not be queried")

    monkeypatch.setattr(pp, "_registry_get", fail)
    image = {"dockerImageMetadata": {"Config": {"Labels": {"org.opencontainers.image.revision": "cafef00d"}}}}
    monkeypatch.setattr(pp.k8s_api, "get_cluster_object", lambda *a, **k: image)
    containers = [{"image_id": "quay.io/org/train@sha256:abc123"}]
    assert pp.detect_git_provenance_from_containers(containers)["git_commit"] == "cafef00d"

    monkeypatch.setattr(pp.k8s_api, "get_cluster_object", lambda *a, **k: None)
    monkeypatch.setattr(pp, "REGISTRY_LABEL_LOOKUP", False)
    assert pp.detect_git_provenance_from_containers(containers) is None


class _FakeResponse:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n):
        return self.body[:n]


def test_registry_get_drops_token_on_cross_host_redirect(monkeypatch):
    seen = []

    def fake_open(req, timeout):
        seen.append((req.full_url, req.get_header("Authorization")))
        if req.full_url.startswith("https://quay.io/"):
            raise pp.urllib.error.HTTPError(req.full_url, 307, "redirect", {"Location": "https://cdn.example/blob"}, None)
        return _FakeResponse(b"{}")

    monkeypatch.setattr(pp, "_registry_get", _REAL_REGISTRY_GET)
    monkeypatch.setattr(pp._registry_opener, "open", fake_open)
    assert pp._registry_get("https://quay.io/v2/org/train/blobs/sha256:ab", token="tok") == b"{}"
    assert seen == [
        ("https://quay.io/v2/org/train/blobs/sha256:ab", "Bearer tok"),
        ("https://cdn.example/blob", None),
    ]
    with pytest.raises(ValueError):
        pp._registry_get("http://quay.io/v2/")


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
# parse_range_response NaN handling
# ---------------------------------------------------------------------------


def test_parse_range_response_drops_nan_samples():
    # A ratio-of-rates query (e.g. VLLM_TELEMETRY_QUERIES' TTFT sum-rate /
    # count-rate) divides 0/0 into "NaN" whenever a window had zero completed
    # requests -- Prometheus's own JSON encoding for it.
    response = {
        "status": "success",
        "data": {
            "result": [
                {"values": [[1700000000, "1.5"], [1700000030, "NaN"], [1700000060, "2.5"]]}
            ]
        },
    }
    points = pp.parse_range_response(response)
    assert [p["value"] for p in points] == [1.5, 2.5]


# ---------------------------------------------------------------------------
# collect_vllm_telemetry (no GPU-count gate, unlike collect_telemetry)
# ---------------------------------------------------------------------------


def _serving_discovery():
    return {
        "pod_metadata": {
            "uid": "pod-uid-2",
            "name": "vllm-pod",
            "start_time": "2026-01-01T00:00:00Z",
        },
        "gpu": {"gpu_count": 1},
    }


def test_collect_vllm_telemetry_queries_every_pod_regardless_of_gpu(monkeypatch):
    # Unlike collect_telemetry, there's no GPU-count skip -- a pod with no
    # GPU discovery data at all should still be queried.
    monkeypatch.setattr(pp, "query_prometheus_range", lambda *a, **k: None)
    monkeypatch.setattr(pp, "TELEMETRY_RETRY_ATTEMPTS", 1)
    summary = pp.collect_vllm_telemetry([_no_gpu_discovery()])
    assert [p["pod_name"] for p in summary["pods"]] == ["web-pod"]


def test_collect_vllm_telemetry_collects_configured_metrics(monkeypatch):
    def fake_query_range(promql, start_ms, end_ms):
        return {
            "status": "success",
            "data": {"result": [{"values": [[1700000000, "0.2"], [1700000030, "0.4"]]}]},
        }

    monkeypatch.setattr(pp, "query_prometheus_range", fake_query_range)
    summary = pp.collect_vllm_telemetry([_serving_discovery()])
    assert len(summary["pods"]) == 1
    metrics = summary["pods"][0]["metrics"]
    assert set(metrics.keys()) == set(pp.VLLM_TELEMETRY_QUERIES.keys())
    assert metrics["time_to_first_token_seconds"]["avg"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# compile_aibom reconciliation
# ---------------------------------------------------------------------------


def test_compile_aibom_annotation_intent_overrides_detected_model():
    detected_model = {"serving_engine": "vllm"}
    annotations = {"experiment-intent": "training"}

    aibom = pp.compile_aibom(
        discoveries=[], detected_datasets=[], runtime_info={}, annotations=annotations,
        telemetry=None, detected_model=detected_model, cli_dataset=None,
    )

    assert aibom["experiment_intent"] == "training"
    assert aibom["experiment_intent_declared_via"] == "annotation"


def test_compile_aibom_surfaces_extra_discovery_and_vllm_fields():
    discovery = {
        "system": {"cpu_architecture": "x86_64", "cache_l3": "N/A", "cache_l2": "10 MiB", "uptime_seconds": "5"},
        "gpu": {"gpu_memory_per_device_mb": "81920\n81920"},
        "network": {"primary_mtu": "1400", "tcp_rmem": "4096", "rdma_devices": "None"},
        "storage": {"block_devices": "nvme0n1 894G", "tmpfs_size": "10G", "nvme_count": "0"},
        "performance_config": {"cpu_governor": "performance", "swappiness": "10"},
        "process_limits": {"max_open_files": "1024"},
        "benchmarks": {"cpu_compute": {"mflops": "100"}},
    }
    detected_model = {"serving_engine": "vllm", "seed": 7, "max_num_seqs": 64, "port": 8000}

    aibom = pp.compile_aibom(
        discoveries=[discovery], detected_datasets=[], runtime_info={}, annotations={},
        telemetry=None, detected_model=detected_model, cli_dataset=None,
    )

    env = aibom["environment"]
    assert env["cpu"] == {"cpu_architecture": "x86_64"}  # N/A dropped, L2 not surfaced
    assert env["gpu_memory_mb"] == [81920, 81920]
    assert env["network"] == {"primary_mtu": "1400"}  # "None" dropped, TCP tuning not surfaced
    assert env["storage"] == {"block_devices": "nvme0n1 894G"}
    assert env["kernel_config"] == {"cpu_governor": "performance"}  # swappiness not surfaced
    assert "process_limits" not in env and "benchmarks" not in env
    inf = aibom["inference"]
    assert (inf["seed"], inf["max_num_seqs"]) == (7, 64)
    assert inf["enforce_eager"] is None and "port" not in inf


def test_compile_aibom_infers_inference_intent_from_vllm_detection():
    detected_model = {"serving_engine": "vllm"}

    aibom = pp.compile_aibom(
        discoveries=[], detected_datasets=[], runtime_info={}, annotations={},
        telemetry=None, detected_model=detected_model, cli_dataset=None,
    )

    assert aibom["experiment_intent"] == "inference"
    assert aibom["experiment_intent_declared_via"] == "inferred_from_model_detection"
    assert aibom["inference"]["serving_engine"] == "vllm"


def test_compile_aibom_infers_sft_intent_from_trl_peft_detection():
    detected_model = {"training_framework": "trl", "adaptation_method": "lora"}

    aibom = pp.compile_aibom(
        discoveries=[], detected_datasets=[], runtime_info={}, annotations={},
        telemetry=None, detected_model=detected_model, cli_dataset=None,
    )

    assert aibom["experiment_intent"] == "sft"
    assert aibom["experiment_intent_declared_via"] == "inferred_from_model_detection"
    assert aibom["fine_tuning"]["adaptation_method"] == "lora"


def test_compile_aibom_infers_training_intent_from_trl_without_peft():
    detected_model = {"training_framework": "trl"}

    aibom = pp.compile_aibom(
        discoveries=[], detected_datasets=[], runtime_info={}, annotations={},
        telemetry=None, detected_model=detected_model, cli_dataset=None,
    )

    assert aibom["experiment_intent"] == "training"
    assert aibom["experiment_intent_declared_via"] == "inferred_from_model_detection"
    assert "fine_tuning" not in aibom


def test_compile_aibom_intent_unknown_when_nothing_resolves():
    aibom = pp.compile_aibom(
        discoveries=[], detected_datasets=[], runtime_info={}, annotations={},
        telemetry=None, detected_model=None, cli_dataset=None,
    )

    assert aibom["experiment_intent"] == "unknown"
    assert aibom["experiment_intent_declared_via"] is None


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
    # Copied from the entry itself, so there's nothing to reconcile against.
    assert aibom["dataset"]["auto_detected"][0]["matches_declared"] is None


@pytest.mark.parametrize("detected,declared,expected", [
    ("stanfordnlp/imdb", "imdb", True),
    ("IMDB", "imdb", True),
    ("/data/train.json", "train.json", True),
    ("org1/imdb", "org2/imdb", False),
    ("imdb", "imdb-extended", False),
    (None, "imdb", False),
])
def test_dataset_names_match(detected, declared, expected):
    assert pp._dataset_names_match(detected, declared) is expected


def test_compile_aibom_declares_cli_config_and_split():
    aibom = pp.compile_aibom(
        discoveries=[], detected_datasets=[{"dataset_name": "glue", "source": "datasets.load_dataset"}],
        runtime_info={}, annotations={"experiment-intent": "sft"}, telemetry=None, detected_model=None,
        cli_dataset={"dataset_name": "glue", "dataset_config": "mrpc", "dataset_split": "train"},
    )

    declared = aibom["dataset"]["declared"]
    assert (declared["declared_via"], declared["config_name"], declared["split"]) == ("cli_arg", "mrpc", "train")
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


def test_compile_aibom_annotation_fills_from_matching_detection_only():
    # Annotated repo + detected commit/dirty from the same repo are combined,
    # labelled annotation; a detection from another repo is dropped (#113).
    detected = {
        "git_commit": "abc1234", "git_repository": "https://github.com/org/repo.git",
        "git_dirty": True, "detected_via": "git_directory",
    }
    for repo, same in (("https://github.com/org/repo", True), ("https://github.com/org/other", False)):
        aibom = pp.compile_aibom(
            discoveries=[], detected_datasets=[], runtime_info={},
            annotations={"experiment-intent": "training", "git-repository": repo},
            telemetry=None, detected_provenance=detected,
        )
        sc = aibom["source_code"]
        assert sc["declared_via"] == "annotation"
        assert sc["git_repository"] == repo
        assert sc["git_commit"] == ("abc1234" if same else None)
        assert ("dirty" in sc) is same


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
# compile_aibom: execution_metadata.duration_seconds
# ---------------------------------------------------------------------------


def test_compile_aibom_computes_duration_from_earliest_pod_start():
    discoveries = [
        {"pod_metadata": {"name": "job-abc", "start_time": "2024-01-01T00:05:00"}},
    ]
    aibom = pp.compile_aibom(
        discoveries=discoveries, detected_datasets=[], runtime_info={},
        annotations={}, telemetry=None,
    )
    assert aibom["execution_metadata"]["duration_seconds"] >= 0


def test_compile_aibom_duration_uses_earliest_of_jobset_sibling_pods():
    discoveries = [
        {"pod_metadata": {"name": "server-0", "start_time": "2024-01-01T00:10:00"}},
        {"pod_metadata": {"name": "server-1", "start_time": "2024-01-01T00:02:00"}},
    ]
    aibom_late_start_only = pp.compile_aibom(
        discoveries=discoveries[:1], detected_datasets=[], runtime_info={},
        annotations={}, telemetry=None,
    )
    aibom_with_earlier_sibling = pp.compile_aibom(
        discoveries=discoveries, detected_datasets=[], runtime_info={},
        annotations={}, telemetry=None,
    )
    # Including the earlier-starting sibling pod should only ever lengthen
    # the computed duration, never shorten it.
    assert (
        aibom_with_earlier_sibling["execution_metadata"]["duration_seconds"]
        >= aibom_late_start_only["execution_metadata"]["duration_seconds"]
    )


def test_compile_aibom_duration_omitted_when_no_pod_start_time():
    aibom = pp.compile_aibom(
        discoveries=[], detected_datasets=[], runtime_info={},
        annotations={}, telemetry=None,
    )
    assert aibom["execution_metadata"]["duration_seconds"] is None


# ---------------------------------------------------------------------------
# pod_status_from_containers / execution_metadata status
# ---------------------------------------------------------------------------


def test_pod_status_from_containers_oomkilled():
    containers = [
        {"pod_name": "job-abc", "name": "training", "terminated_reason": "OOMKilled", "exit_code": 137},
    ]
    status, exit_code = pp.pod_status_from_containers("job-abc", containers)
    assert status == "OOMKilled"
    assert exit_code == 137


def test_pod_status_from_containers_oomkilled_wins_over_other_container():
    # A sidecar exiting cleanly shouldn't hide the main container's OOM kill.
    containers = [
        {"pod_name": "job-abc", "name": "sidecar", "terminated_reason": "Completed", "exit_code": 0},
        {"pod_name": "job-abc", "name": "training", "terminated_reason": "OOMKilled", "exit_code": 137},
    ]
    status, exit_code = pp.pod_status_from_containers("job-abc", containers)
    assert status == "OOMKilled"
    assert exit_code == 137


def test_pod_status_from_containers_completed():
    containers = [
        {"pod_name": "job-abc", "name": "training", "terminated_reason": "Completed", "exit_code": 0},
    ]
    status, exit_code = pp.pod_status_from_containers("job-abc", containers)
    assert status == "Completed"
    assert exit_code == 0


def test_pod_status_from_containers_no_status_reported():
    status, exit_code = pp.pod_status_from_containers("job-abc", [])
    assert status is None
    assert exit_code is None


def test_pod_status_from_containers_ignores_other_pods():
    containers = [
        {"pod_name": "other-pod", "name": "training", "terminated_reason": "OOMKilled", "exit_code": 137},
    ]
    status, exit_code = pp.pod_status_from_containers("job-abc", containers)
    assert status is None
    assert exit_code is None


def test_compile_aibom_pod_status_oomkilled():
    discoveries = [{"pod_metadata": {"name": "job-abc", "start_time": "2024-01-01T00:05:00"}}]
    containers = [
        {"pod_name": "job-abc", "name": "training", "terminated_reason": "OOMKilled", "exit_code": 137},
    ]
    aibom = pp.compile_aibom(
        discoveries=discoveries, detected_datasets=[], runtime_info={},
        annotations={}, telemetry=None, containers=containers,
    )
    pod = aibom["execution_metadata"]["pods"][0]
    assert pod["status"] == "OOMKilled"
    assert pod["exit_code"] == 137
    assert "retried_attempt" not in pod  # no job_result, so not treated as a retry
    assert aibom["execution_metadata"]["status"] == "OOMKilled"


def test_compile_aibom_status_rolls_up_oomkilled_across_jobset_siblings():
    discoveries = [
        {"pod_metadata": {"name": "server-0"}},
        {"pod_metadata": {"name": "server-1"}},
    ]
    containers = [
        {"pod_name": "server-0", "name": "server", "terminated_reason": "Completed", "exit_code": 0},
        {"pod_name": "server-1", "name": "server", "terminated_reason": "OOMKilled", "exit_code": 137},
    ]
    aibom = pp.compile_aibom(
        discoveries=discoveries, detected_datasets=[], runtime_info={},
        annotations={}, telemetry=None, containers=containers,
    )
    # One sibling OOMing should still surface at the job level even though
    # the other completed cleanly.
    assert aibom["execution_metadata"]["status"] == "OOMKilled"


def test_compile_aibom_status_none_when_no_container_status_reported():
    discoveries = [{"pod_metadata": {"name": "job-abc"}}]
    aibom = pp.compile_aibom(
        discoveries=discoveries, detected_datasets=[], runtime_info={},
        annotations={}, telemetry=None,
    )
    pod = aibom["execution_metadata"]["pods"][0]
    assert pod["status"] is None
    assert pod["exit_code"] is None
    assert aibom["execution_metadata"]["status"] is None


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


def test_compile_aibom_utilization_scales_storage_throughput_to_mbps():
    # Decimal MB/s (10**6 bytes), matching the "MBps" label.
    telemetry = {
        "collected_at": "2024-01-01T00:00:00Z",
        "pods": [
            {
                "pod_name": "job-abc",
                "metrics": {
                    "storage_read_throughput": _pod_metrics(
                        avg=10 * 10**6, min_=10**6, max_=20 * 10**6,
                        p95=19 * 10**6, unit="bytes_per_sec",
                    ),
                    "storage_write_throughput": _pod_metrics(
                        avg=5 * 10**6, min_=512 * 1000, max_=8 * 10**6,
                        p95=7 * 10**6, unit="bytes_per_sec",
                    ),
                },
            }
        ],
    }
    aibom = pp.compile_aibom(
        discoveries=[], detected_datasets=[], runtime_info={},
        annotations={}, telemetry=telemetry,
    )
    metrics = aibom["resource_utilization"]["metrics"]
    assert metrics["storage_read_throughput"]["unit"] == "MBps"
    assert metrics["storage_read_throughput"]["avg"] == 10
    assert metrics["storage_read_throughput"]["max"] == 20
    assert metrics["storage_write_throughput"]["unit"] == "MBps"
    assert metrics["storage_write_throughput"]["avg"] == 5


# ---------------------------------------------------------------------------
# compile_aibom: inference.performance from vLLM telemetry
# ---------------------------------------------------------------------------


def test_compile_aibom_populates_inference_performance_for_vllm():
    vllm_telemetry = {
        "collected_at": "2024-01-01T00:00:00Z",
        "pods": [
            {
                "pod_name": "vllm-pod",
                "includes_cold_start": False,
                "metrics": {
                    "time_to_first_token_seconds": _pod_metrics(avg=0.3, min_=0.1, max_=0.5, p95=0.45, unit="seconds"),
                    "kv_cache_usage": _pod_metrics(avg=40, min_=10, max_=70, p95=65, unit="percent"),
                },
            }
        ],
    }
    aibom = pp.compile_aibom(
        discoveries=[], detected_datasets=[], runtime_info={},
        annotations={"experiment-intent": "inference"}, telemetry=None,
        detected_model={"serving_engine": "vllm"}, vllm_telemetry=vllm_telemetry,
    )
    performance = aibom["inference"]["performance"]
    assert performance["metrics"]["time_to_first_token_seconds"]["avg"] == 0.3
    assert performance["metrics"]["kv_cache_usage"]["avg"] == 40
    assert performance["summary_includes_cold_start"] is False


def test_compile_aibom_omits_inference_performance_without_vllm_telemetry():
    aibom = pp.compile_aibom(
        discoveries=[], detected_datasets=[], runtime_info={},
        annotations={"experiment-intent": "inference"}, telemetry=None,
        detected_model={"serving_engine": "vllm"}, vllm_telemetry=None,
    )
    assert "performance" not in aibom["inference"]


def test_compile_aibom_ignores_vllm_telemetry_for_non_inference_intent():
    vllm_telemetry = {
        "collected_at": "2024-01-01T00:00:00Z",
        "pods": [{"pod_name": "vllm-pod", "metrics": {
            "time_to_first_token_seconds": _pod_metrics(avg=0.3, min_=0.1, max_=0.5, p95=0.45, unit="seconds"),
        }}],
    }
    aibom = pp.compile_aibom(
        discoveries=[], detected_datasets=[], runtime_info={},
        annotations={"experiment-intent": "training"}, telemetry=None,
        vllm_telemetry=vllm_telemetry,
    )
    assert "inference" not in aibom


# ---------------------------------------------------------------------------
# compile_aibom: resource_utilization.metrics.<name>.limit
# ---------------------------------------------------------------------------


def test_compile_aibom_memory_usage_reports_limit_in_gb():
    telemetry = {
        "collected_at": "2024-01-01T00:00:00Z",
        "pods": [
            {
                "pod_name": "job-abc",
                "metrics": {"memory_usage": _pod_metrics(avg=6 * 1024**3, min_=2 * 1024**3, max_=8 * 1024**3, p95=7.8 * 1024**3, unit="bytes")},
            }
        ],
    }
    containers = [
        {"pod_name": "job-abc", "name": "training", "memory_limit_bytes": 8 * 1024**3},
    ]
    aibom = pp.compile_aibom(
        discoveries=[], detected_datasets=[], runtime_info={},
        annotations={}, telemetry=telemetry, containers=containers,
    )
    detail = aibom["resource_utilization"]["metrics"]["memory_usage"]
    assert detail["max"] == 8.0
    assert detail["limit"] == 8.0


def test_compile_aibom_cpu_usage_reports_limit_in_cores():
    telemetry = {
        "collected_at": "2024-01-01T00:00:00Z",
        "pods": [
            {
                "pod_name": "job-abc",
                "metrics": {"cpu_usage": _pod_metrics(avg=1.5, min_=0.5, max_=1.9, p95=1.8, unit="cores")},
            }
        ],
    }
    containers = [
        {"pod_name": "job-abc", "name": "training", "cpu_limit_millis": 2000},
    ]
    aibom = pp.compile_aibom(
        discoveries=[], detected_datasets=[], runtime_info={},
        annotations={}, telemetry=telemetry, containers=containers,
    )
    detail = aibom["resource_utilization"]["metrics"]["cpu_usage"]
    assert detail["limit"] == 2.0


def test_compile_aibom_memory_usage_no_limit_key_when_no_container_limit_set():
    telemetry = {
        "collected_at": "2024-01-01T00:00:00Z",
        "pods": [
            {
                "pod_name": "job-abc",
                "metrics": {"memory_usage": _pod_metrics(avg=6 * 1024**3, min_=2 * 1024**3, max_=8 * 1024**3, p95=7.8 * 1024**3, unit="bytes")},
            }
        ],
    }
    aibom = pp.compile_aibom(
        discoveries=[], detected_datasets=[], runtime_info={},
        annotations={}, telemetry=telemetry, containers=None,
    )
    assert "limit" not in aibom["resource_utilization"]["metrics"]["memory_usage"]


def test_compile_aibom_gpu_utilization_never_reports_limit():
    # GPU/network/storage have no Kubernetes resource-limit concept.
    telemetry = {
        "collected_at": "2024-01-01T00:00:00Z",
        "pods": [{"pod_name": "job-abc", "metrics": {"gpu_utilization": _pod_metrics(avg=60, min_=10, max_=95, p95=94)}}],
    }
    containers = [{"pod_name": "job-abc", "name": "training", "memory_limit_bytes": 8 * 1024**3}]
    aibom = pp.compile_aibom(
        discoveries=[], detected_datasets=[], runtime_info={},
        annotations={}, telemetry=telemetry, containers=containers,
    )
    assert "limit" not in aibom["resource_utilization"]["metrics"]["gpu_utilization"]


def test_compile_aibom_memory_usage_limit_is_tightest_across_jobset_siblings():
    telemetry = {
        "collected_at": "2024-01-01T00:00:00Z",
        "pods": [
            {"pod_name": "server-0", "metrics": {"memory_usage": _pod_metrics(avg=4 * 1024**3, min_=2 * 1024**3, max_=6 * 1024**3, p95=5.5 * 1024**3, unit="bytes")}},
            {"pod_name": "client-0", "metrics": {"memory_usage": _pod_metrics(avg=1 * 1024**3, min_=0.5 * 1024**3, max_=1.5 * 1024**3, p95=1.4 * 1024**3, unit="bytes")}},
        ],
    }
    containers = [
        {"pod_name": "server-0", "name": "server", "memory_limit_bytes": 16 * 1024**3},
        {"pod_name": "client-0", "name": "client", "memory_limit_bytes": 2 * 1024**3},
    ]
    aibom = pp.compile_aibom(
        discoveries=[], detected_datasets=[], runtime_info={},
        annotations={}, telemetry=telemetry, containers=containers,
    )
    # The client's tighter 2GB limit wins even though the server used more memory.
    assert aibom["resource_utilization"]["metrics"]["memory_usage"]["limit"] == 2.0


def test_pod_resource_limit_sums_multiple_containers_in_one_pod():
    containers = [
        {"pod_name": "job-abc", "name": "main", "memory_limit_bytes": 4 * 1024**3},
        {"pod_name": "job-abc", "name": "sidecar", "memory_limit_bytes": 1 * 1024**3},
    ]
    assert pp.pod_resource_limit("job-abc", containers, "memory_limit_bytes") == 5 * 1024**3


def test_pod_resource_limit_none_when_no_container_reports_it():
    containers = [{"pod_name": "job-abc", "name": "main"}]
    assert pp.pod_resource_limit("job-abc", containers, "memory_limit_bytes") is None


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


def test_compile_aibom_annotations_override_training_fields():
    # README: annotations always override (#109).
    runtime_info = {"learning_rate": 1e-4, "batch_size": 4, "epochs": 3, "random_seed": 1}
    detected_model = {"learning_rate": 2e-4, "batch_size": 8, "epochs": 2, "random_seed": 2}
    aibom = pp.compile_aibom(
        discoveries=[], detected_datasets=[], runtime_info=runtime_info,
        annotations={
            "experiment-intent": "training", "learning-rate": "5e-5", "batch-size": "16",
            "epochs": "1", "random-seed": "99",
        },
        telemetry=None, detected_model=detected_model,
    )
    t = aibom["training"]
    assert (t["learning_rate"], t["batch_size"], t["epochs"], t["random_seed"]) == (5e-5, 16, 1, 99)


def test_compile_aibom_reports_mixed_precision_separately_from_dtype():
    runtime_info = {"dtype": "float32", "mixed_precision": "bf16"}
    aibom = pp.compile_aibom(
        discoveries=[], detected_datasets=[], runtime_info=runtime_info,
        annotations={"experiment-intent": "training"}, telemetry=None,
    )
    assert aibom["model"]["dtype"] == "float32"
    assert aibom["training"]["mixed_precision"] == "bf16"


def test_compile_aibom_runtime_lora_on_quantized_base_is_qlora():
    # Independent of whether LoraConfig or the quantized load came first (#109).
    runtime_info = {"adaptation_method": "lora", "quantization_method": "bitsandbytes"}
    aibom = pp.compile_aibom(
        discoveries=[], detected_datasets=[], runtime_info=runtime_info,
        annotations={"experiment-intent": "sft"}, telemetry=None,
    )
    assert aibom["fine_tuning"]["adaptation_method"] == "qlora"


def test_compile_aibom_model_name_annotation_drops_storage_details():
    # Storage-derived revision/base_model/size describe the detected model,
    # not whichever one the annotation names (#109).
    detected_model = {
        "model_name": "Qwen/Qwen2.5-32B-Instruct",
        "model_name_declared_via": "model_files_readme",
        "model_revision": "5ede1c9", "base_model": "Qwen/Qwen2.5-32B", "model_size_bytes": 1,
    }
    for name, kept in (("other/model", False), ("Qwen/Qwen2.5-32B-Instruct", True)):
        aibom = pp.compile_aibom(
            discoveries=[], detected_datasets=[], runtime_info={},
            annotations={"experiment-intent": "inference", "model-name": name},
            telemetry=None, detected_model=detected_model,
        )
        assert ("revision" in aibom["model"]) is kept
        assert ("base_model" in aibom["model"]) is kept
        assert ("size_bytes" in aibom["model"]) is kept


def test_compile_aibom_falls_back_to_device_map_for_parallelization_strategy():
    runtime_info = {"model_device_map": "auto", "model_device_count": 2}
    aibom = pp.compile_aibom(
        discoveries=[], detected_datasets=[], runtime_info=runtime_info,
        annotations={"experiment-intent": "training"}, telemetry=None,
    )
    assert aibom["training"]["parallelization_strategy"] == "model_parallel"


def test_compile_aibom_single_device_map_is_not_model_parallel():
    # {"": 0}, "cuda:1", "auto" on one GPU all resolve to one device (#109).
    runtime_info = {"model_device_map": "{'': 0}", "model_device_count": 1}
    aibom = pp.compile_aibom(
        discoveries=[], detected_datasets=[], runtime_info=runtime_info,
        annotations={"experiment-intent": "training"}, telemetry=None,
    )
    assert aibom["training"]["parallelization_strategy"] is None


def test_compile_aibom_cli_detected_strategy_overrides_device_map_fallback():
    runtime_info = {"model_device_map": "auto", "model_device_count": 2}
    detected_model = {"parallelization_strategy": "data_parallel"}
    aibom = pp.compile_aibom(
        discoveries=[], detected_datasets=[], runtime_info=runtime_info,
        annotations={"experiment-intent": "training"}, telemetry=None,
        detected_model=detected_model,
    )
    assert aibom["training"]["parallelization_strategy"] == "data_parallel"


def _generate_ed25519_pem():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key = Ed25519PrivateKey.generate()
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def test_sign_aibom_returns_none_when_no_signing_key_path(monkeypatch):
    monkeypatch.setattr(pp, "SIGNING_KEY_PATH", "")
    assert pp.sign_aibom({"a": 1}) == (None, None)


def test_sign_aibom_returns_none_when_key_file_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(pp, "SIGNING_KEY_PATH", str(tmp_path / "does-not-exist"))
    assert pp.sign_aibom({"a": 1}) == (None, None)


def test_sign_aibom_produces_a_verifiable_signature(monkeypatch, tmp_path):
    import rfc8785
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    key_path = tmp_path / "ed25519-key"
    key_path.write_bytes(_generate_ed25519_pem())
    monkeypatch.setattr(pp, "SIGNING_KEY_PATH", str(key_path))

    aibom = {"b": 2, "a": 1}
    signature_b64, public_key_b64 = pp.sign_aibom(aibom)
    assert signature_b64 and public_key_b64

    public_key = Ed25519PublicKey.from_public_bytes(base64.b64decode(public_key_b64))
    canonical = rfc8785.dumps(aibom)
    # Raises if invalid -- no exception here is the assertion.
    public_key.verify(base64.b64decode(signature_b64), canonical)


def test_sign_aibom_signature_changes_if_payload_differs(monkeypatch, tmp_path):
    import rfc8785
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    from cryptography.exceptions import InvalidSignature

    key_path = tmp_path / "ed25519-key"
    key_path.write_bytes(_generate_ed25519_pem())
    monkeypatch.setattr(pp, "SIGNING_KEY_PATH", str(key_path))

    signature_b64, public_key_b64 = pp.sign_aibom({"a": 1})
    public_key = Ed25519PublicKey.from_public_bytes(base64.b64decode(public_key_b64))
    tampered = rfc8785.dumps({"a": 2})
    with pytest.raises(InvalidSignature):
        public_key.verify(base64.b64decode(signature_b64), tampered)


def test_sign_aibom_signature_survives_json_round_trip(monkeypatch, tmp_path):
    """sign_aibom signs the in-memory Python dict, but a real verifier never
    sees that object -- it only sees whatever comes back out of the
    Kubernetes API after the AIBOM's `data` went through a JSON encode (the
    POST body k8s_api.create_custom_object sends) and a JSON decode (a GET
    or `kubectl get -o json` later). If re-canonicalizing *that*
    round-tripped object didn't reproduce the exact bytes that were signed,
    verification would spuriously fail for every AIBOM, not just tampered
    ones. See CLAUDE.md's Compiled AIBOM Signing section.
    """
    import rfc8785
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    key_path = tmp_path / "ed25519-key"
    key_path.write_bytes(_generate_ed25519_pem())
    monkeypatch.setattr(pp, "SIGNING_KEY_PATH", str(key_path))

    aibom = {
        "model": {"name": "tinyllama-1.1b-chat", "quantization": None},
        "training": {"learning_rate": 2e-5, "epochs": 3, "random_seed": 42},
        "fine_tuning": {"lora_rank": 16},
        "tags": ["sft", "lora"],
        "dirty": False,
    }
    signature_b64, public_key_b64 = pp.sign_aibom(aibom)
    public_key = Ed25519PublicKey.from_public_bytes(base64.b64decode(public_key_b64))

    # Simulates the round trip through the Kubernetes API: the CR's `data`
    # field is sent as JSON and later read back as JSON, never as the same
    # Python object sign_aibom saw.
    round_tripped = json.loads(json.dumps(aibom))
    canonical = rfc8785.dumps(round_tripped)

    # Raises if invalid -- no exception here is the assertion.
    public_key.verify(base64.b64decode(signature_b64), canonical)


def test_sign_aibom_matches_go_jcs_reference_output():
    """RFC 8785 is only useful here because two independent implementations
    (this repo's Python `rfc8785`, and oc-aibom's Go `gowebpki/jcs`) agree
    on the canonical bytes for the same logical JSON value -- which a
    hand-rolled sort_keys/separators convention never guaranteed across
    languages (different float formatting, different non-ASCII escaping
    defaults). This locks in a known-good byte string, produced by actually
    running the Go reference implementation against this exact fixture, so
    a future rfc8785 upgrade that drifted from the spec would be caught
    here rather than only showing up as oc-aibom verify failures.
    """
    import rfc8785

    aibom = {
        "model": {"name": "tinyllama-1.1b-chat", "quantization": None},
        "training": {"learning_rate": 2e-5, "epochs": 3, "random_seed": 42},
        "tags": ["sft", "lora"],
        "dirty": False,
    }
    assert rfc8785.dumps(aibom) == (
        b'{"dirty":false,"model":{"name":"tinyllama-1.1b-chat","quantization":null},'
        b'"tags":["sft","lora"],"training":{"epochs":3,"learning_rate":0.00002,"random_seed":42}}'
    )


# ---------------------------------------------------------------------------
# Persisted telemetry time series
# ---------------------------------------------------------------------------


def _range_response(*series):
    """series: (labels, [(ts, value), ...]) tuples -> Prometheus matrix response."""
    return {
        "status": "success",
        "data": {"result": [{"metric": labels, "values": [[ts, str(v)] for ts, v in pts]} for labels, pts in series]},
    }


def test_parse_range_series_keeps_series_separate_with_normalized_labels():
    response = _range_response(
        ({"exported_pod": "p0", "gpu": "1", "instance": "x"}, [(100, 5), (130, 6)]),
        ({"exported_pod": "p0", "gpu": "0"}, [(100, 1), (130, 2)]),
    )
    series = pp.parse_range_series(response)
    assert [s["labels"] for s in series] == [{"pod": "p0", "gpu": "0"}, {"pod": "p0", "gpu": "1"}]
    assert series[0]["points"] == [[100, 1], [130, 2]]


def test_parse_range_series_drops_nan_and_empty_series():
    response = _range_response(
        ({"pod": "a"}, [(100, "NaN"), (130, 2.5)]),
        ({"pod": "b"}, [(100, "NaN")]),
    )
    series = pp.parse_range_series(response)
    assert series == [{"labels": {"pod": "a"}, "points": [[130, 2.5]]}]


def test_parse_range_series_failed_response_is_empty():
    assert pp.parse_range_series(None) == []
    assert pp.parse_range_series({"status": "error"}) == []


def test_aggregate_points_sum_avg_max():
    series = [
        {"labels": {}, "points": [[0, 1], [30, 2]]},
        {"labels": {}, "points": [[0, 3], [30, 6]]},
    ]
    assert pp._aggregate_points(series, "sum") == [[0, 4], [30, 8]]
    assert pp._aggregate_points(series, "avg") == [[0, 2], [30, 4]]
    assert pp._aggregate_points(series, "max") == [[0, 3], [30, 6]]


def test_parse_start_utc_treats_suffixless_time_as_utc():
    assert (
        pp._parse_start_utc("2026-01-01T00:00:00").timestamp()
        == pp._parse_start_utc("2026-01-01T00:00:00Z").timestamp()
    )


def test_pod_regex_skips_invalid_names_and_escapes_dots():
    assert pp._pod_regex(["a-1", "b.c", 'bad"name', None]) == "a-1|b[.]c"


def test_series_queries_exclude_dataset_sidecar_from_per_container_metrics():
    for name in ("cpu_usage", "memory_usage", "storage_read_throughput", "storage_write_throughput"):
        assert 'container!="aibom-dataset-sidecar"' in pp.SERIES_QUERIES[name]["query"]


def _series_pods(*names):
    return {"pods": [{"pod_name": n, "start_time": "2026-01-01T00:00:00Z"} for n in names]}


def test_collect_telemetry_series_builds_schema_with_aggregates_and_detail(monkeypatch):
    monkeypatch.setattr(pp, "JOB_NAMESPACE", "ns")
    seen = []

    def fake_query_range(promql, start_ms, end_ms, step_seconds=None):
        seen.append((promql, step_seconds))
        if "container_cpu_usage_seconds_total" in promql:
            return _range_response(
                ({"pod": "p0", "container": "trainer"}, [(1767225600, 1.0), (1767225630, 2.0)]),
                ({"pod": "p1", "container": "trainer"}, [(1767225600, 3.0), (1767225630, 4.0)]),
            )
        if "DCGM_FI_DEV_GPU_UTIL" in promql:
            value = 90 if promql.startswith("max by") else 50
            return _range_response(
                ({"exported_pod": "p0", "gpu": "0"}, [(1767225600, value)]),
                ({"exported_pod": "p1", "gpu": "0"}, [(1767225600, value - 20)]),
            )
        return None

    monkeypatch.setattr(pp, "query_prometheus_range", fake_query_range)
    doc = json.loads(pp.collect_telemetry_series(_series_pods("p0", "p1"), None))

    assert doc["schema_version"] == 1
    assert doc["window"]["end"] > doc["window"]["start"]
    assert doc["window"]["step_seconds"] >= 15
    assert doc["pods"] == ["p0", "p1"]
    # Only metrics that returned data are present.
    assert set(doc["metrics"]) == {"cpu_usage", "gpu_utilization"}

    cpu = doc["metrics"]["cpu_usage"]
    assert cpu["unit"] == "cores" and cpu["aggregation"] == "sum"
    assert cpu["aggregate"] == [[1767225600, 4], [1767225630, 6]]
    assert "aggregate_max" not in cpu  # not a gauge
    assert [s["labels"] for s in cpu["series"]] == [
        {"pod": "p0", "container": "trainer"},
        {"pod": "p1", "container": "trainer"},
    ]

    gpu = doc["metrics"]["gpu_utilization"]
    assert gpu["aggregation"] == "avg"
    assert gpu["aggregate"] == [[1767225600, 40]]  # avg of 50 and 30
    assert gpu["aggregate_max"] == [[1767225600, 90]]  # max of 90 and 70
    assert gpu["series"][0]["labels"] == {"pod": "p0", "gpu": "0"}

    # One query per metric for all pods, scoped to the namespace.
    cpu_queries = [q for q, _ in seen if "container_cpu_usage_seconds_total" in q]
    assert len(cpu_queries) == 1
    assert 'namespace="ns"' in cpu_queries[0] and 'pod=~"p0|p1"' in cpu_queries[0]
    assert all(step == doc["window"]["step_seconds"] for _, step in seen)


def test_collect_telemetry_series_step_and_gauge_window_floored_at_scrape_interval(monkeypatch):
    from datetime import datetime, timedelta, timezone

    monkeypatch.setattr(pp, "JOB_NAMESPACE", "ns")
    start = (datetime.now(timezone.utc) - timedelta(minutes=20)).strftime("%Y-%m-%dT%H:%M:%SZ")
    seen = []

    def fake_query_range(promql, start_ms, end_ms, step_seconds=None):
        seen.append((promql, step_seconds))
        if "DCGM_FI_DEV_GPU_UTIL" in promql:
            return _range_response(({"exported_pod": "p0", "gpu": "0"}, [(1767225600, 50)]))
        return None

    monkeypatch.setattr(pp, "query_prometheus_range", fake_query_range)
    doc = json.loads(pp.collect_telemetry_series({"pods": [{"pod_name": "p0", "start_time": start}]}, None))

    # 20 min / 200 points = 6s, which would be narrower than a 30s scrape interval.
    assert doc["window"]["step_seconds"] == pp.SERIES_SCRAPE_INTERVAL_S == 30
    gauge_queries = [q for q, _ in seen if "DCGM_FI_DEV_GPU_UTIL" in q]
    assert gauge_queries and all("[30s]" in q for q in gauge_queries)


def test_collect_telemetry_series_targets_roughly_the_configured_point_count(monkeypatch):
    monkeypatch.setattr(pp, "SERIES_TARGET_POINTS", 200)
    calls = []
    monkeypatch.setattr(
        pp, "query_prometheus_range", lambda q, s, e, step_seconds=None: calls.append((s, e, step_seconds)) or None
    )
    pp.collect_telemetry_series(_series_pods("p0"), None)
    start_ms, end_ms, step = calls[0]
    assert 150 <= (end_ms - start_ms) / 1000 / step <= 200


def test_collect_telemetry_series_none_when_prometheus_returns_nothing(monkeypatch):
    monkeypatch.setattr(pp, "query_prometheus_range", lambda *a, **k: None)
    assert pp.collect_telemetry_series(_series_pods("p0"), None) is None


def test_collect_telemetry_series_none_without_pods_or_valid_start(monkeypatch):
    monkeypatch.setattr(pp, "query_prometheus_range", lambda *a, **k: pytest.fail("should not query"))
    assert pp.collect_telemetry_series(None, None) is None
    assert pp.collect_telemetry_series({"pods": [{"pod_name": "p0", "start_time": "garbage"}]}, None) is None


def test_collect_telemetry_series_includes_vllm_metrics_for_vllm_pods(monkeypatch):
    def fake_query_range(promql, start_ms, end_ms, step_seconds=None):
        if "vllm:generation_tokens_total" in promql:
            return _range_response(({"pod": "v0"}, [(1767225600, 12)]))
        return None

    monkeypatch.setattr(pp, "query_prometheus_range", fake_query_range)
    doc = json.loads(pp.collect_telemetry_series(None, _series_pods("v0")))
    assert doc["metrics"]["generation_throughput"]["unit"] == "tokens_per_sec"
    assert doc["metrics"]["generation_throughput"]["aggregate"] == [[1767225600, 12]]


def test_collect_telemetry_series_omits_detail_beyond_series_cap(monkeypatch):
    monkeypatch.setattr(pp, "SERIES_MAX_SERIES_PER_METRIC", 2)

    def fake_query_range(promql, start_ms, end_ms, step_seconds=None):
        if "container_network_receive_bytes_total" in promql:
            return _range_response(*[({"pod": f"p{i}", "interface": "eth0"}, [(100, 1)]) for i in range(3)])
        return None

    monkeypatch.setattr(pp, "query_prometheus_range", fake_query_range)
    entry = json.loads(pp.collect_telemetry_series(_series_pods("p0", "p1", "p2"), None))["metrics"]["network_receive"]
    assert entry["aggregate"] == [[100, 3]]
    assert entry["series_omitted"] is True and "series" not in entry


def test_fit_series_doc_drops_largest_detail_first_and_keeps_aggregates():
    def metric(n_points):
        pts = [[i, i * 1.5] for i in range(n_points)]
        return {"unit": "x", "aggregation": "sum", "aggregate": pts[:3], "series": [{"labels": {}, "points": pts}]}

    doc = {"metrics": {"small": metric(5), "big": metric(500)}}
    cap = len(pp._encode_series_doc(doc)) - 100
    result = json.loads(pp._fit_series_doc(doc, cap))
    assert result["metrics"]["big"]["series_omitted"] is True and "series" not in result["metrics"]["big"]
    assert "series" in result["metrics"]["small"]
    assert len(result["metrics"]["big"]["aggregate"]) == 3


def test_fit_series_doc_none_when_aggregates_alone_exceed_cap():
    doc = {
        "window": {"start": 0, "end": 999, "step_seconds": 1},
        "metrics": {"m": {"unit": "x", "aggregation": "sum", "aggregate": [[i, i] for i in range(1000)]}},
    }
    assert pp._fit_series_doc(doc, 100) is None


def _gridded_doc(n_points, n_series, step=30, start=1000):
    """A doc with one gauge metric: an aggregate line, a peak line and n_series
    per-GPU series, all on the same n_points-long grid."""
    ts = [start + i * step for i in range(n_points)]
    return {
        "schema_version": 1,
        "window": {"start": start, "end": ts[-1], "step_seconds": step},
        "metrics": {
            "gpu_utilization": {
                "unit": "percent",
                "aggregation": "avg",
                "aggregate": [[t, 50.0] for t in ts],
                "aggregate_max": [[t, 60.0 + i] for i, t in enumerate(ts)],
                "series": [
                    {"labels": {"pod": "p0", "gpu": str(g)}, "points": [[t, float(g)] for t in ts]}
                    for g in range(n_series)
                ],
            }
        },
    }


def test_rebucket_points_averages_and_labels_buckets_on_the_original_grid():
    points = [[1000, 10], [1030, 20], [1060, 30], [1090, 50], [1120, 100]]
    assert pp._rebucket_points(points, 1000, 60, "avg") == [[1000, 15], [1060, 40], [1120, 100]]
    assert pp._rebucket_points(points, 1000, 60, "max") == [[1000, 20], [1060, 50], [1120, 100]]


def test_rebucket_series_doc_doubles_step_and_keeps_peaks_as_maxes():
    doc = _gridded_doc(200, 2)
    coarse = pp._rebucket_series_doc(doc, 2)
    metric = coarse["metrics"]["gpu_utilization"]

    assert coarse["window"]["step_seconds"] == 60
    assert coarse["window"]["start"] == doc["window"]["start"]
    assert len(metric["aggregate"]) == 100
    assert metric["aggregate"][0] == [1000, 50]
    # aggregate_max is 60, 61, 62, ... -- each merged bucket keeps the larger of its two.
    assert metric["aggregate_max"][0] == [1000, 61]
    assert metric["aggregate_max"][1] == [1060, 63]
    assert [s["labels"] for s in metric["series"]] == [{"pod": "p0", "gpu": "0"}, {"pod": "p0", "gpu": "1"}]
    assert len(metric["series"][0]["points"]) == 100
    # The input is untouched.
    assert len(doc["metrics"]["gpu_utilization"]["aggregate"]) == 200


def test_fit_series_doc_reduces_resolution_before_dropping_detail(monkeypatch):
    monkeypatch.setattr(pp, "SERIES_MIN_POINTS", 100)
    doc = _gridded_doc(200, 20)
    full = len(pp._encode_series_doc(doc).encode())
    half = len(pp._encode_series_doc(pp._rebucket_series_doc(doc, 2)).encode())
    assert half < full
    encoded = pp._fit_series_doc(doc, (full + half) // 2)  # too big at 200 points, fits at 100
    result = json.loads(encoded)
    metric = result["metrics"]["gpu_utilization"]

    assert result["window"]["step_seconds"] == 60
    assert len(metric["aggregate"]) == 100
    # Per-GPU detail survived: resolution was given up instead.
    assert "series_omitted" not in metric and len(metric["series"]) == 20
    assert len(metric["aggregate_max"]) == 100


def test_fit_series_doc_leaves_full_resolution_when_it_already_fits():
    doc = _gridded_doc(200, 2)
    result = json.loads(pp._fit_series_doc(doc, 10_000_000))
    assert result["window"]["step_seconds"] == 30
    assert len(result["metrics"]["gpu_utilization"]["aggregate"]) == 200


def test_fit_series_doc_never_goes_below_the_minimum_points(monkeypatch):
    monkeypatch.setattr(pp, "SERIES_MIN_POINTS", 100)
    # 150 points can't be halved (75 < 100), so detail is dropped at full resolution.
    doc = _gridded_doc(150, 20)
    aggregates_only = len(pp._encode_series_doc(_without_series(doc)).encode())
    result = json.loads(pp._fit_series_doc(doc, aggregates_only + 50))
    metric = result["metrics"]["gpu_utilization"]
    assert result["window"]["step_seconds"] == 30
    assert len(metric["aggregate"]) == 150
    assert metric["series_omitted"] is True and "series" not in metric


def _without_series(doc):
    return {**doc, "metrics": {n: {k: v for k, v in m.items() if k != "series"} for n, m in doc["metrics"].items()}}


def test_fit_series_doc_reduces_resolution_then_drops_detail_when_still_too_big(monkeypatch):
    monkeypatch.setattr(pp, "SERIES_MIN_POINTS", 100)
    doc = _gridded_doc(200, 40)
    coarse_aggregates_only = len(
        pp._encode_series_doc(_without_series(pp._rebucket_series_doc(doc, 2))).encode()
    )
    result = json.loads(pp._fit_series_doc(doc, coarse_aggregates_only + 50))
    metric = result["metrics"]["gpu_utilization"]
    # Both steps were needed: coarsest allowed resolution, and no detail.
    assert result["window"]["step_seconds"] == 60 and len(metric["aggregate"]) == 100
    assert metric["series_omitted"] is True and "series" not in metric
    assert len(metric["aggregate_max"]) == 100


def test_collect_telemetry_series_halves_resolution_to_keep_gpu_detail(monkeypatch):
    pods = [f"p{i}" for i in range(4)]

    def fake_query_range(promql, start_ms, end_ms, step_seconds=None):
        if "DCGM_FI_DEV_GPU_UTIL" not in promql:
            return None
        ts = list(range(int(start_ms / 1000), int(end_ms / 1000) + 1, step_seconds))
        return _range_response(
            *[({"exported_pod": p, "gpu": str(g)}, [(t, 50 + g) for t in ts]) for p in pods for g in range(8)]
        )

    monkeypatch.setattr(pp, "query_prometheus_range", fake_query_range)
    monkeypatch.setattr(pp, "SERIES_TARGET_POINTS", 200)
    telemetry = {"pods": [{"pod_name": p, "start_time": "2026-01-01T00:00:00Z"} for p in pods]}

    full_size = len(pp.collect_telemetry_series(telemetry, None).encode())
    monkeypatch.setattr(pp, "SERIES_MAX_BYTES", int(full_size * 0.7))
    doc = json.loads(pp.collect_telemetry_series(telemetry, None))
    metric = doc["metrics"]["gpu_utilization"]

    assert len(metric["aggregate"]) <= 101
    assert len(metric["series"]) == 32  # every GPU of every pod is still there
    assert "series_omitted" not in metric


def test_publish_series_object_creates_aibomtelemetry_and_returns_ref(monkeypatch):
    import hashlib

    monkeypatch.setattr(pp, "JOB_NAME", "train-job")
    monkeypatch.setattr(pp, "JOB_NAMESPACE", "ns")
    created = {}
    monkeypatch.setattr(
        pp.k8s_api, "create_custom_object",
        lambda ns, group, version, plural, body: created.update(
            ns=ns, group=group, version=version, plural=plural, body=body
        ),
    )
    encoded = json.dumps({"window": {"start": 1, "end": 2, "step_seconds": 15}, "metrics": {}})
    ref = pp.publish_series_object(encoded)

    assert (created["ns"], created["group"], created["version"], created["plural"]) == (
        "ns", "aibom.io", "v1alpha1", "aibomtelemetries",
    )
    body = created["body"]
    assert body["apiVersion"] == "aibom.io/v1alpha1" and body["kind"] == "AIBOMTelemetry"
    assert body["metadata"]["name"].startswith("train-job-telemetry-")
    assert body["metadata"]["namespace"] == "ns"
    assert body["metadata"]["labels"] == {"aibom.io/job-name": "train-job"}
    assert "ownerReferences" not in body["metadata"]  # set after the AIBOM exists
    assert "status" not in body
    # The payload is stored as the exact string, so a reader can hash it.
    assert body["spec"]["seriesJson"] == encoded
    assert body["spec"]["schemaVersion"] == 1
    assert body["spec"]["sizeBytes"] == len(encoded.encode())
    assert body["spec"]["window"] == {"start": 1, "end": 2, "stepSeconds": 15}

    assert ref == {
        "schema_version": 1,
        "kind": "AIBOMTelemetry",
        "name": body["metadata"]["name"],
        "sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        "size_bytes": len(encoded.encode()),
        "window": {"start": 1, "end": 2, "step_seconds": 15},
    }


def test_publish_series_object_none_when_create_fails(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("forbidden")

    monkeypatch.setattr(pp.k8s_api, "create_custom_object", boom)
    encoded = json.dumps({"window": {"start": 1, "end": 2, "step_seconds": 15}, "metrics": {}})
    assert pp.publish_series_object(encoded) is None


# main(): ordering, ownerReference and cleanup around the AIBOM create.


@pytest.fixture
def main_env(monkeypatch):
    """Runs main() with everything but the Kubernetes calls stubbed, recording
    the calls made against k8s_api in order."""
    calls = []
    monkeypatch.setattr(pp, "JOB_NAME", "train-job")
    monkeypatch.setattr(pp, "JOB_NAMESPACE", "ns")
    monkeypatch.setattr(pp, "PROMETHEUS_URL", "http://prometheus")
    monkeypatch.setattr(pp, "load_discovery", lambda: [])
    monkeypatch.setattr(pp, "load_datasets", lambda: ([], {}))
    monkeypatch.setattr(pp, "load_annotations", lambda: {})
    monkeypatch.setattr(pp, "load_containers", lambda: [])
    monkeypatch.setattr(pp, "load_storage", lambda: {})
    monkeypatch.setattr(pp, "collect_telemetry", lambda discoveries, containers=None: {"pods": []})
    encoded = json.dumps({"window": {"start": 1, "end": 2, "step_seconds": 15}, "metrics": {}})
    monkeypatch.setattr(pp, "collect_telemetry_series", lambda t, v, e="vllm": encoded)
    monkeypatch.setattr(pp, "sign_aibom", lambda aibom: (None, None))

    def fake_create(ns, group, version, plural, body):
        calls.append(("create", plural, body))
        if plural == "aiboms":
            if fake_create.aibom_error:
                raise RuntimeError("admission denied")
            return {"metadata": {"name": "train-job-abc12", "uid": "aibom-uid"}}
        return {}

    fake_create.aibom_error = False
    monkeypatch.setattr(pp.k8s_api, "create_custom_object", fake_create)
    monkeypatch.setattr(
        pp.k8s_api, "set_custom_object_owner",
        lambda ns, group, version, plural, name, owner: calls.append(("owner", plural, name, owner)),
    )
    return calls, fake_create, encoded


def test_main_stores_series_object_signs_its_digest_and_owns_it_to_the_aibom(main_env):
    import hashlib

    calls, _, encoded = main_env
    pp.main()

    kinds = [c[:2] for c in calls]
    # Telemetry object first (its digest must be inside the AIBOM's signed data),
    # then the AIBOM, then the ownerReference.
    assert kinds == [("create", "aibomtelemetries"), ("create", "aiboms"), ("owner", "aibomtelemetries")]

    telemetry_name = calls[0][2]["metadata"]["name"]
    aibom_data = calls[1][2]["spec"]["data"]
    ref = aibom_data["telemetry_series_ref"]
    assert ref["kind"] == "AIBOMTelemetry" and ref["name"] == telemetry_name
    assert calls[0][2]["spec"]["seriesJson"] == encoded
    assert ref["sha256"] == hashlib.sha256(encoded.encode()).hexdigest()

    _, _, owned_name, owner = calls[2]
    assert owned_name == telemetry_name
    assert owner == {
        "apiVersion": "aibom.io/v1alpha1", "kind": "AIBOM", "name": "train-job-abc12",
        "uid": "aibom-uid", "blockOwnerDeletion": True,
    }


def test_main_warns_about_orphaned_series_object_when_aibom_create_fails(main_env, capsys):
    calls, fake_create, _ = main_env
    fake_create.aibom_error = True
    with pytest.raises(SystemExit) as exc:
        pp.main()

    assert exc.value.code == 1
    telemetry_name = calls[0][2]["metadata"]["name"]
    # No delete is attempted (the Role withholds it) and no owner is set; the
    # orphan is reported with the command to remove it instead.
    assert [c[:2] for c in calls] == [("create", "aibomtelemetries"), ("create", "aiboms")]
    err = capsys.readouterr().err
    assert f"AIBOMTelemetry/{telemetry_name} is now orphaned" in err
    assert f"oc delete aibomtel {telemetry_name} -n ns" in err


def test_main_creates_aibom_without_reference_when_series_object_create_fails(main_env, monkeypatch):
    calls, fake_create, _ = main_env
    real_create = pp.k8s_api.create_custom_object

    def create(ns, group, version, plural, body):
        if plural == "aibomtelemetries":
            raise RuntimeError("quota exceeded")
        return real_create(ns, group, version, plural, body)

    monkeypatch.setattr(pp.k8s_api, "create_custom_object", create)
    pp.main()

    aibom_call = [c for c in calls if c[:2] == ("create", "aiboms")][0]
    assert "telemetry_series_ref" not in aibom_call[2]["spec"]["data"]
    assert not any(c[0] == "owner" for c in calls)


def test_main_skips_series_object_when_no_series(main_env, monkeypatch):
    calls, _, _ = main_env
    monkeypatch.setattr(pp, "collect_telemetry_series", lambda t, v, e="vllm": None)
    pp.main()
    assert [c[:2] for c in calls] == [("create", "aiboms")]


# ---------------------------------------------------------------------------
# Telemetry stats correctness (#111)
# ---------------------------------------------------------------------------


def _substituted(query_defs, monkeypatch):
    monkeypatch.setattr(pp, "JOB_NAMESPACE", "team-ns")
    return {name: pp._substitute_pod_query(q["query"], "pod-a") for name, q in query_defs.items()}


def test_stats_queries_aggregate_to_one_series_per_pod(monkeypatch):
    # Pooling every per-container/per-GPU/per-interface series into one
    # sample list averaged them together (e.g. an 8 GiB app container plus a
    # 50 MiB sidecar reported ~4 GiB). Every stats query must aggregate.
    for name, promql in _substituted({**pp.TELEMETRY_QUERIES, **pp.VLLM_TELEMETRY_QUERIES}, monkeypatch).items():
        assert re.match(r"^(\(|100 \* \()?(sum|avg)( by \(pod\))? ?\(", promql), (name, promql)


def test_stats_queries_are_namespace_scoped(monkeypatch):
    for name, promql in _substituted({**pp.TELEMETRY_QUERIES, **pp.VLLM_TELEMETRY_QUERIES}, monkeypatch).items():
        assert "{namespace}" not in promql and "{pod_name}" not in promql, name
        # Every selector in the query carries the namespace filter.
        assert promql.count('pod="pod-a"') == promql.count('namespace="team-ns"'), (name, promql)


def test_stats_queries_exclude_dataset_sidecar(monkeypatch):
    queries = _substituted(pp.TELEMETRY_QUERIES, monkeypatch)
    for name in ("cpu_usage", "memory_usage", "storage_read_throughput", "storage_write_throughput"):
        assert 'container!="aibom-dataset-sidecar"' in queries[name], name


def test_kv_cache_usage_scaled_to_percent_with_pre_rename_fallback():
    for defs in (pp.VLLM_TELEMETRY_QUERIES, pp.VLLM_SERIES_QUERIES):
        q = defs["kv_cache_usage"]["query"]
        assert q.startswith("100 * (")
        assert "vllm:kv_cache_usage_perc" in q and "vllm:gpu_cache_usage_perc" in q
        assert defs["kv_cache_usage"]["unit"] == "percent"


def test_inter_token_latency_falls_back_to_pre_rename_metric():
    for defs in (pp.VLLM_TELEMETRY_QUERIES, pp.VLLM_SERIES_QUERIES):
        q = defs["inter_token_latency_seconds"]["query"]
        assert "vllm:inter_token_latency_seconds_sum" in q
        assert " or " in q and "vllm:time_per_output_token_seconds_sum" in q


def test_round_stat_keeps_small_values_significant():
    assert pp._round_stat(0.0144) == 0.0144
    assert pp._round_stat(0.000634) == 0.000634
    assert pp._round_stat(0.123456) == 0.123
    assert pp._round_stat(12.3456) == 12.35
    assert pp._round_stat(0) == 0
    assert pp._round_stat(None) is None


def test_aggregate_pod_metrics_does_not_round_small_latencies_to_zero():
    pods = [{"pod_name": "p", "metrics": {"inter_token_latency_seconds": _pod_metrics(
        avg=0.0144, min_=0.0101, max_=0.0213, p95=0.0198, unit="seconds")}}]
    m = pp.aggregate_pod_metrics(pods, {"inter_token_latency_seconds": (None, "seconds")})
    assert m["inter_token_latency_seconds"]["avg"] == 0.0144
    assert m["inter_token_latency_seconds"]["min"] == 0.0101


def test_compute_metric_stats_does_not_round_before_scaling():
    stats = pp.compute_metric_stats(_points(0.0006, 0.0007, 0.0008))
    assert stats["avg"] == pytest.approx(0.0007)


def test_pod_stats_window_ends_at_container_finish_time():
    containers = [
        {"pod_name": "p", "name": "a", "finished_at": "2026-01-01T00:10:00Z"},
        {"pod_name": "p", "name": "b", "finished_at": "2026-01-01T00:20:00Z"},
        {"pod_name": "other", "name": "a", "finished_at": "2026-01-01T05:00:00Z"},
    ]
    start_ms, end_ms, stats_start_ms, includes_cold_start = pp._pod_stats_window(
        "p", "2026-01-01T00:00:00Z", containers
    )
    assert end_ms - start_ms == 20 * 60 * 1000
    # Cold start is one scrape interval, not the 5-minute rate window.
    assert stats_start_ms - start_ms == pp.SCRAPE_INTERVAL_MS
    assert includes_cold_start is False


def test_pod_stats_window_falls_back_to_now_when_pod_still_running():
    start_dt = datetime.fromtimestamp(time.time() - 600, timezone.utc)
    containers = [
        {"pod_name": "p", "name": "a", "finished_at": "2026-01-01T00:10:00Z"},
        {"pod_name": "p", "name": "b"},  # still running
    ]
    _, end_ms, _, _ = pp._pod_stats_window("p", start_dt.isoformat(), containers)
    assert abs(end_ms - time.time() * 1000) < 5000


def test_pod_stats_window_treats_suffixless_start_as_utc():
    containers = [{"pod_name": "p", "name": "a", "finished_at": "2026-01-01T01:00:00Z"}]
    with_z = pp._pod_stats_window("p", "2026-01-01T00:00:00Z", containers)
    naive = pp._pod_stats_window("p", "2026-01-01T00:00:00", containers)
    assert with_z == naive


def test_pod_stats_window_short_run_flags_cold_start():
    containers = [{"pod_name": "p", "name": "a", "finished_at": "2026-01-01T00:00:40Z"}]
    start_ms, _, stats_start_ms, includes_cold_start = pp._pod_stats_window(
        "p", "2026-01-01T00:00:00Z", containers
    )
    assert stats_start_ms - start_ms == 20 * 1000  # half the 40 s run
    assert includes_cold_start is True


def test_collect_telemetry_queries_through_pod_finish_time(monkeypatch):
    ends = []

    def fake_query_range(promql, start_ms, end_ms):
        ends.append(end_ms)
        return {"status": "success", "data": {"result": [{"values": [[1767225700, "1"]]}]}}

    monkeypatch.setattr(pp, "query_prometheus_range", fake_query_range)
    discovery = {
        "pod_metadata": {"uid": "u", "name": "p", "start_time": "2026-01-01T00:00:00Z"},
        "gpu": {"gpu_count": 1},
    }
    containers = [{"pod_name": "p", "name": "a", "finished_at": "2026-01-01T00:30:00Z"}]
    summary = pp.collect_telemetry([discovery], containers)
    finished_ms = int(datetime(2026, 1, 1, 0, 30, tzinfo=timezone.utc).timestamp() * 1000)
    assert ends and set(ends) == {finished_ms}
    assert summary["pods"][0]["end_ms"] == finished_ms


def test_collect_telemetry_series_window_ends_at_last_pod_finish(monkeypatch):
    windows = []
    monkeypatch.setattr(
        pp, "_collect_series_metrics",
        lambda defs, ns, pods, start_ms, end_ms, step: windows.append((start_ms, end_ms)) or {"m": {}},
    )
    monkeypatch.setattr(pp, "_fit_series_doc", lambda doc, cap: json.dumps(doc))
    telemetry = {"pods": [
        {"pod_name": "a", "start_time": "2026-01-01T00:00:00Z", "end_ms": 1767226200000},
        {"pod_name": "b", "start_time": "2026-01-01T00:05:00Z", "end_ms": 1767226800000},
    ]}
    pp.collect_telemetry_series(telemetry, None)
    assert windows[0] == (1767225600000, 1767226800000)


def test_compile_aibom_duration_ends_at_last_pod_finish():
    discoveries = [{"pod_metadata": {"name": "p", "start_time": "2026-01-01T00:00:00"}}]
    containers = [{"pod_name": "p", "name": "a", "finished_at": "2026-01-01T00:15:00Z"}]
    aibom = pp.compile_aibom(
        discoveries=discoveries, detected_datasets=[], runtime_info={}, annotations={},
        telemetry=None, containers=containers,
    )
    assert aibom["execution_metadata"]["duration_seconds"] == 15 * 60


def test_compile_aibom_memory_is_gib_and_network_is_decimal_mbps():
    telemetry = {"pods": [{"pod_name": "p", "metrics": {
        "memory_usage": _pod_metrics(avg=2 * 1024**3, min_=1024**3, max_=3 * 1024**3, p95=3 * 1024**3, unit="bytes"),
        "network_receive": _pod_metrics(avg=125_000, min_=125_000, max_=125_000, p95=125_000, unit="bytes_per_sec"),
    }}]}
    aibom = pp.compile_aibom(
        discoveries=[], detected_datasets=[], runtime_info={}, annotations={}, telemetry=telemetry,
    )
    metrics = aibom["resource_utilization"]["metrics"]
    assert metrics["memory_usage"]["unit"] == "GiB"
    assert metrics["memory_usage"]["avg"] == 2
    assert metrics["network_receive"]["unit"] == "Mbps"
    assert metrics["network_receive"]["avg"] == 1  # 125 kB/s * 8 = 1 Mbit/s


# ---------------------------------------------------------------------------
# CLI parsing: forms that used to be missed (#108)
# ---------------------------------------------------------------------------


def _sh(script, shell=("sh", "-c")):
    return {"command": list(shell), "args": [script]}


def test_detect_vllm_positional_model():
    result = pp.detect_vllm_from_command(["vllm", "serve", "meta-llama/Llama-3.1-8B", "--port", "8000"])
    assert result["model_name"] == "meta-llama/Llama-3.1-8B"
    assert result["port"] == 8000


def test_detect_vllm_explicit_model_flag_wins_over_positional():
    result = pp.detect_vllm_from_command(["vllm", "serve", "a/one", "--model", "b/two"])
    assert result["model_name"] == "b/two"


def test_detect_vllm_positional_not_confused_with_unknown_flag_value():
    result = pp.detect_vllm_from_command(["vllm", "serve", "--host", "0.0.0.0", "--port", "8000"])
    assert "model_name" not in result
    assert result["serving_engine"] == "vllm"


def test_detect_vllm_bare_serve_still_identifies_engine():
    assert pp.detect_vllm_from_command(["vllm", "serve", "$MODEL"]) == {"serving_engine": "vllm"}


def test_detect_vllm_snake_case_flags():
    result = pp.detect_vllm_from_command(
        ["vllm", "serve", "--max_model_len", "4096", "--tensor_parallel_size=2", "--model", "x/y"]
    )
    assert result["max_model_len"] == 4096
    assert result["tensor_parallel_size"] == 2


def test_detect_vllm_api_server_module():
    result = pp.detect_vllm_from_command(
        ["python3", "-m", "vllm.entrypoints.openai.api_server", "--model", "x/y"]
    )
    assert result["model_name"] == "x/y"


def test_detect_vllm_entrypoint_only_image():
    container = {
        "image": "docker.io/vllm/vllm-openai:v0.6.0",
        "args": ["--model", "x/y", "--max_model_len", "4096"],
    }
    result = pp.detect_model_from_containers([container])
    assert result["serving_engine"] == "vllm"
    assert result["model_name"] == "x/y"
    assert result["max_model_len"] == 4096


def test_detect_vllm_entrypoint_only_image_with_serve_and_positional():
    container = {"image": "registry:5000/vllm/vllm-openai@sha256:abc", "args": ["serve", "x/y"]}
    assert pp.detect_model_from_containers([container])["model_name"] == "x/y"


def test_entrypoint_fallback_needs_vllm_image_and_no_command():
    assert pp.detect_model_from_containers([{"image": "python:3.12", "args": ["--model", "x"]}]) == {}
    overridden = {"image": "vllm/vllm-openai:v0", "command": ["python", "app.py"], "args": ["--model", "x"]}
    assert pp.detect_model_from_containers([overridden]) == {}


def test_detect_trl_bare_boolean_flags():
    result = pp.detect_trl_from_command(
        ["trl", "sft", "--use_peft", "--lora_r", "16", "--load_in_4bit", "--model_name_or_path", "m"]
    )
    assert result["adaptation_method"] == "qlora"
    assert result["lora_rank"] == 16
    assert result["model_name"] == "m"


def test_detect_trl_explicit_false_and_negated_flags():
    assert "adaptation_method" not in pp.detect_trl_from_command(["trl", "sft", "--use_peft", "false"])
    assert "adaptation_method" not in pp.detect_trl_from_command(["trl", "sft", "--use_peft=false"])
    assert "adaptation_method" not in pp.detect_trl_from_command(["trl", "sft", "--no_use_peft"])


def test_detect_trl_hyphenated_flags():
    result = pp.detect_trl_from_command(["trl", "sft", "--use-peft", "--lora-r", "8"])
    assert result["adaptation_method"] == "lora"
    assert result["lora_rank"] == 8


def test_detect_trl_config_file_still_identifies_framework():
    assert pp.detect_trl_from_command(["trl", "sft", "--config", "cfg.yaml"]) == {
        "training_framework": "trl"
    }


def test_detect_trl_fractional_epochs():
    result = pp.detect_trl_from_command(["trl", "sft", "--num_train_epochs", "0.5"])
    assert result["epochs"] == 0.5
    assert pp.detect_trl_from_command(["trl", "sft", "--num_train_epochs", "3"])["epochs"] == 3


def test_detect_trl_load_in_8bit_does_not_leak_when_4bit_set():
    result = pp.detect_trl_from_command(
        ["trl", "sft", "--use_peft", "--lora_r", "4", "--load_in_4bit", "--load_in_8bit", "false"]
    )
    assert "load_in_8bit" not in result and "load_in_4bit" not in result


def test_detect_trl_python_module_form():
    result = pp.detect_trl_from_command(["python", "-m", "trl", "sft", "--seed", "7"])
    assert result["random_seed"] == 7


@pytest.mark.parametrize("shell", [("bash", "-l", "-c"), ("bash", "-euc"), ("bash", "-xc"),
                                    ("sh", "-c"), ("bash", "-o", "pipefail", "-c")])
def test_flatten_handles_shell_flag_variants(shell):
    tokens = pp._flatten_container_command(_sh("vllm serve a/b", shell))
    assert tokens == ["vllm", "serve", "a/b"]


def test_flatten_splits_separators_without_spaces():
    tokens = pp._flatten_container_command(_sh("pip install x;vllm serve a/b&&echo done"))
    assert tokens == ["pip", "install", "x", ";", "vllm", "serve", "a/b", "&&", "echo", "done"]


def test_flatten_treats_unquoted_newline_as_separator():
    tokens = pp._flatten_container_command(_sh("pip install trl\ntrl sft --seed 1"))
    assert pp.detect_trl_from_command(tokens) == {"training_framework": "trl", "random_seed": 1}


def test_flatten_keeps_quoted_newline_in_one_token():
    tokens = pp._flatten_container_command(_sh("python -c 'a\nb'"))
    assert tokens == ["python", "-c", "a\nb"]


def test_accelerate_use_fsdp_and_use_deepspeed():
    assert pp.detect_parallelization_from_command(["accelerate", "launch", "--use_fsdp", "t.py"]) == {
        "parallelization_strategy": "fsdp"
    }
    assert pp.detect_parallelization_from_command(["accelerate", "launch", "--use_deepspeed", "t.py"]) == {
        "parallelization_strategy": "deepspeed"
    }


def test_accelerate_config_file_flag_resolves_preset_name():
    tokens = ["accelerate", "launch", "--config_file", "/cfg/fsdp2.yaml", "train.py"]
    assert pp.detect_parallelization_from_command(tokens) == {"parallelization_strategy": "fsdp"}


def test_hyphenated_num_processes():
    tokens = ["trl", "sft", "--num-processes", "4"]
    assert pp.detect_parallelization_from_command(tokens) == {
        "parallelization_strategy": "data_parallel"
    }


# ---------------------------------------------------------------------------
# CLI parsing: false positives (#108)
# ---------------------------------------------------------------------------


def test_trl_use_vllm_flag_is_not_a_vllm_server():
    result = pp.detect_model_from_containers([_sh("trl grpo --use_vllm --seed 42")])
    assert "serving_engine" not in result
    assert result["training_framework"] == "trl"
    assert result["random_seed"] == 42


def test_pip_install_trl_vllm_extra_is_not_a_vllm_server():
    result = pp.detect_model_from_containers([_sh("pip install trl[vllm] && trl sft --seed 1")])
    assert "serving_engine" not in result
    assert result["training_framework"] == "trl"


def test_pip_install_trl_peft_is_not_a_trl_invocation():
    assert pp.detect_trl_from_command(["pip", "install", "trl", "peft"]) is None
    assert pp.detect_model_from_containers([_sh("pip install trl peft && python train.py")]) == {}


def test_pip_dash_q_is_not_vllm_quantization():
    result = pp.detect_model_from_containers(
        [_sh("pip install -q vllm==0.6 && vllm serve --model org/m-AWQ")]
    )
    assert "quantization" not in result
    assert result["quantization_method"] == "awq"


def test_vllm_bench_client_does_not_override_server():
    script = "vllm serve x/server & sleep 5; vllm bench serve --model y/client --seed 3"
    result = pp.detect_model_from_containers([_sh(script)])
    assert result["model_name"] == "x/server"
    assert "seed" not in result


def test_vllm_bench_alone_is_not_a_server():
    assert pp.detect_vllm_from_command(["vllm", "bench", "serve", "--model", "y"]) is None


def test_torchrun_single_process_is_not_data_parallel():
    assert pp.detect_parallelization_from_command(["torchrun", "--nproc_per_node=1", "t.py"]) is None
    assert pp.detect_parallelization_from_command(["torchrun", "--nproc-per-node", "1", "t.py"]) is None
    assert pp.detect_parallelization_from_command(["mpirun", "-np", "1", "python", "t.py"]) is None


def test_torchrun_multi_process_or_node_is_data_parallel():
    for tokens in (
        ["torchrun", "--nproc_per_node=8", "t.py"],
        ["torchrun", "--nproc_per_node=1", "--nnodes=2", "t.py"],
        ["torchrun", "t.py"],
    ):
        assert pp.detect_parallelization_from_command(tokens) == {
            "parallelization_strategy": "data_parallel"
        }


def test_pip_install_deepspeed_is_not_a_deepspeed_launch():
    assert pp.detect_model_from_containers([_sh("pip install deepspeed && python train.py")]) == {}


def test_launcher_must_be_the_command_word():
    assert pp.detect_parallelization_from_command(["python", "train.py", "torchrun"]) is None


def test_python_dash_m_torch_distributed_run_is_a_launcher():
    tokens = ["python", "-m", "torch.distributed.run", "--nproc_per_node=4", "t.py"]
    assert pp.detect_parallelization_from_command(tokens) == {
        "parallelization_strategy": "data_parallel"
    }


def test_unexpanded_shell_variables_are_not_emitted_as_values():
    for script in ("vllm serve --model $MODEL", "vllm serve --model ${MODEL}",
                   "vllm serve --model $(MODEL_ID)", "vllm serve $MODEL"):
        assert "model_name" not in pp.detect_model_from_containers([_sh(script)]), script
    assert "epochs" not in pp.detect_trl_from_command(["trl", "sft", "--num_train_epochs", "$E"])


def test_vllm_boolean_flag_with_explicit_value():
    assert pp.detect_vllm_from_command(
        ["vllm", "serve", "m", "--enable-expert-parallel=false"]
    )["enable_expert_parallel"] is False
    assert pp.detect_vllm_from_command(
        ["vllm", "serve", "m", "--no-enable-prefix-caching"]
    )["enable_prefix_caching"] is False
    assert pp.detect_vllm_from_command(
        ["vllm", "serve", "m", "--enable-expert-parallel"]
    )["enable_expert_parallel"] is True


def _intent_for(script):
    aibom = pp.compile_aibom(
        discoveries=[], detected_datasets=[], runtime_info={}, annotations={}, telemetry=None,
        detected_model=pp.detect_model_from_containers([_sh(script)]),
    )
    return aibom["experiment_intent"]


def test_use_vllm_no_longer_flips_intent_to_inference():
    """The end-to-end failure from the issue: `--use_vllm` made trl look like
    a vLLM server, so intent became `inference` and the training sections
    were dropped."""
    assert _intent_for("trl sft --use_peft --lora_r 16 --use_vllm --model_name_or_path m") == "sft"


def test_trl_config_file_yields_training_intent_instead_of_unknown():
    assert _intent_for("trl sft --config cfg.yaml") == "training"


# ---------------------------------------------------------------------------
# CLI parsing: malformed and unconventional user input
# ---------------------------------------------------------------------------


_VLLM_LONG_FLAGS = [k for k in pp._VLLM_ARG_MAP if k.startswith("--")]


@pytest.mark.parametrize("flag", _VLLM_LONG_FLAGS)
def test_every_vllm_flag_accepts_snake_case(flag):
    """vLLM's own parser accepts `--max_model_len` for `--max-model-len`."""
    name, conv = pp._VLLM_ARG_MAP[flag]
    snake = "--" + flag[2:].replace("-", "_")
    value = {int: "2", pp._human_readable_int: "2", pp._finite_float: "0.5", bool: None}.get(conv, "a=1")
    tail = [snake] if value is None else [snake, value]
    kebab = [flag] if value is None else [flag, value]
    assert pp.detect_vllm_from_command(["vllm", "serve", "m"] + tail) == pp.detect_vllm_from_command(
        ["vllm", "serve", "m"] + kebab
    )


def test_vllm_mixed_separators_and_negated_snake_case():
    result = pp.detect_vllm_from_command(
        ["vllm", "serve", "m", "--max_model-len", "1024", "--no_enable_prefix_caching",
         "--enable_expert_parallel", "--gpu_memory_utilization=0.85"]
    )
    assert result["max_model_len"] == 1024
    assert result["enable_prefix_caching"] is False
    assert result["enable_expert_parallel"] is True
    assert result["gpu_memory_utilization"] == 0.85


@pytest.mark.parametrize("value,expected", [
    ("4096", 4096), ("32k", 32000), ("32K", 32768), ("25.6k", 25600), ("1m", 1_000_000),
])
def test_vllm_max_model_len_human_readable(value, expected):
    result = pp.detect_vllm_from_command(["vllm", "serve", "m", "--max-model-len", value])
    assert result["max_model_len"] == expected


@pytest.mark.parametrize("args", [
    ["--max-model-len", "abc"], ["--max-model-len", "0.5K"], ["--max-model-len", "4096.5"],
    ["--tensor-parallel-size", "two"], ["--gpu-memory-utilization", "90%"],
    ["--gpu-memory-utilization", "nan"], ["--port", "8000.0"],
])
def test_vllm_malformed_typed_values_are_dropped_not_stored_as_strings(args):
    result = pp.detect_vllm_from_command(["vllm", "serve", "m"] + args)
    name = pp._VLLM_ARG_MAP[args[0]][0]
    assert name not in result
    assert result["model_name"] == "m"


@pytest.mark.parametrize("args", [
    ["--learning_rate", "nan"], ["--learning_rate", "inf"], ["--num_train_epochs", "inf"],
    ["--num_train_epochs", "-inf"], ["--lora_r", "sixteen"], ["--seed", "1e3x"],
    ["--per_device_train_batch_size", "8.5"],
])
def test_trl_malformed_or_non_finite_values_are_dropped(args):
    """NaN/inf would make the AIBOM invalid JSON and break JCS signing."""
    result = pp.detect_trl_from_command(["trl", "sft"] + args)
    name = pp._TRL_ARG_MAP[args[0]][0]
    assert name not in result
    assert result["training_framework"] == "trl"


def test_empty_flag_values_are_ignored():
    result = pp.detect_vllm_from_command(["vllm", "serve", "--model=", "--port=", "--dtype", ""])
    assert result == {"serving_engine": "vllm"}
    assert "model_name" not in pp.detect_trl_from_command(["trl", "sft", "--model_name_or_path="])


def test_vllm_flag_missing_its_value_does_not_swallow_the_next_flag():
    result = pp.detect_vllm_from_command(["vllm", "serve", "--model", "--port", "8000"])
    assert "model_name" not in result
    assert result["port"] == 8000


def test_vllm_trailing_flag_with_no_value():
    assert pp.detect_vllm_from_command(["vllm", "serve", "m", "--max-model-len"]) == {
        "serving_engine": "vllm", "model_name": "m"
    }


def test_vllm_repeated_flag_last_one_wins():
    """argparse semantics: a later `--model` overrides an earlier one."""
    result = pp.detect_vllm_from_command(["vllm", "serve", "--model", "a/one", "--model", "b/two"])
    assert result["model_name"] == "b/two"


def test_vllm_short_flags_with_equals_and_negative_numbers():
    result = pp.detect_vllm_from_command(["vllm", "serve", "m", "-tp=4", "-pp", "2", "--seed", "-1"])
    assert result["tensor_parallel_size"] == 4
    assert result["pipeline_parallel_size"] == 2
    assert result["seed"] == -1


def test_explicit_quantization_flag_suppresses_name_guess():
    result = pp.detect_vllm_from_command(["vllm", "serve", "org/m-AWQ", "--quantization", "fp8"])
    assert result["quantization"] == "fp8"
    assert "quantization_method" not in result


@pytest.mark.parametrize("model,method", [
    ("/mnt/models/Qwen2-7B-Instruct-AWQ", "awq"),
    ("./llama-7b.Q4_K_M.gguf", "gguf"),
    ("org/m-awq", "awq"),
    ("org/模型-GPTQ-Int8", "gptq"),
])
def test_quantization_detected_from_unconventional_model_names(model, method):
    assert pp.detect_vllm_from_command(["vllm", "serve", model])["quantization_method"] == method


@pytest.mark.parametrize("model", ["org/Awqward-7b", "org/print4-model", "org/fp8ish-chat"])
def test_quantization_markers_inside_words_are_not_matched(model):
    result = pp.detect_vllm_from_command(["vllm", "serve", model])
    assert result["model_name"] == model
    assert "quantization_method" not in result


def test_model_name_with_spaces_survives_quoting():
    result = pp.detect_model_from_containers([_sh("vllm serve 'org/my model' --max-model-len 2048")])
    assert result["model_name"] == "org/my model"
    assert result["max_model_len"] == 2048


def test_model_name_containing_hash_is_not_a_comment():
    assert pp.detect_model_from_containers([_sh("vllm serve org/m#v2")])["model_name"] == "org/m#v2"


def test_shell_comments_are_ignored():
    script = "# vllm serve old/model --max-model-len 1\nvllm serve new/model  # --max-model-len 99\n"
    result = pp.detect_model_from_containers([_sh(script)])
    assert result["model_name"] == "new/model"
    assert "max_model_len" not in result


def test_apostrophes_in_comments_and_heredoc_dont_break_tokenizing():
    """An unbalanced `'` in a comment or heredoc body made shlex raise, so
    nothing in the script was detected (examples/axolotl-finetune.yaml)."""
    script = (
        "cat > /tmp/c.yml <<'EOF'\n"
        "# axolotl's cache\n"
        "base_model: x\n"
        "EOF\n"
        "# can't write /\n"
        "axolotl train /tmp/c.yml \\\n  --base-model org/m --adapter lora --lora-r 16\n"
    )
    result = pp.detect_model_from_containers([_sh(script)])
    assert result["training_framework"] == "axolotl"
    assert result["model_name"] == "org/m"
    assert result["lora_rank"] == 16


def test_heredoc_body_is_not_a_command():
    result = pp.detect_model_from_containers([_sh("cat > s.sh <<EOF\nvllm serve a/b\nEOF\ntrl sft --seed 3\n")])
    assert "serving_engine" not in result
    assert result["random_seed"] == 3


def test_trailing_comment_does_not_merge_the_next_line():
    """shlex's own comment handling ate the newline, so `pip install` and the
    server ended up in one (package-install) command."""
    result = pp.detect_model_from_containers([_sh("pip install vllm  # pin later\nvllm serve a/b")])
    assert result["model_name"] == "a/b"


def test_blank_lines_and_crlf_line_endings():
    result = pp.detect_model_from_containers(
        [_sh("pip install trl\r\n\r\n\ntrl sft --seed 5 \\\r\n  --lora_r 8\r\n")]
    )
    assert result["random_seed"] == 5


def test_unbalanced_quote_fails_closed():
    assert pp.detect_model_from_containers([_sh('vllm serve "org/m --port 1')]) == {}


def test_env_assignments_and_exec_prefix_before_vllm():
    result = pp.detect_model_from_containers(
        [_sh("CUDA_VISIBLE_DEVICES=0,1 HF_HOME=/cache exec vllm serve a/b -tp 2")]
    )
    assert result["model_name"] == "a/b"
    assert result["tensor_parallel_size"] == 2


def test_absolute_path_to_vllm_binary():
    assert pp.detect_vllm_from_command(["/opt/venv/bin/vllm", "serve", "m"])["model_name"] == "m"


@pytest.mark.parametrize("value,expected", [
    ("True", "lora"), ("TRUE", "lora"), ("yes", "lora"), ("1", "lora"),
    ("False", None), ("no", None), ("0", None),
])
def test_trl_use_peft_boolean_spellings(value, expected):
    result = pp.detect_trl_from_command(["trl", "sft", "--use_peft", value, "--lora_r", "8"])
    assert result.get("adaptation_method") == expected


def test_trl_bool_flag_does_not_consume_a_following_non_bool_value():
    result = pp.detect_trl_from_command(["trl", "sft", "--use_peft", "org/model"])
    assert result["adaptation_method"] == "peft"
    assert "model_name" not in result


def test_trl_dora_and_rslora_together_fall_back_to_generic_peft():
    result = pp.detect_trl_from_command(["trl", "sft", "--use_peft", "--use_dora", "--use_rslora"])
    assert result["adaptation_method"] == "peft"


def test_trl_lora_flags_without_use_peft_do_not_claim_adaptation():
    result = pp.detect_trl_from_command(["trl", "sft", "--lora_r", "16", "--lora_alpha", "32"])
    assert "adaptation_method" not in result
    assert result["lora_rank"] == 16


def test_trl_scientific_and_hyphenated_flags():
    result = pp.detect_trl_from_command(
        ["trl", "sft", "--learning-rate", "2e-5", "--per-device-train-batch-size", "8",
         "--model-name-or-path", "org/m"]
    )
    assert result["learning_rate"] == 2e-5
    assert result["batch_size"] == 8
    assert result["model_name"] == "org/m"


def test_trl_flag_case_is_significant():
    """argparse is case sensitive, so trl would reject these too."""
    assert "learning_rate" not in pp.detect_trl_from_command(["trl", "sft", "--Learning_Rate", "1e-4"])


def test_trl_without_subcommand_is_not_an_invocation():
    assert pp.detect_trl_from_command(["trl", "--help"]) is None
    assert pp.detect_trl_from_command(["python", "trl.py", "--seed", "1"]) is None


def test_trl_module_path_through_accelerate():
    result = pp.detect_model_from_containers(
        [_sh("accelerate launch --num_processes 4 -m trl.scripts.sft --use_peft --lora_r 8")]
    )
    assert result["training_framework"] == "trl"
    assert result["adaptation_method"] == "lora"
    assert result["parallelization_strategy"] == "data_parallel"


# ---------------------------------------------------------------------------
# Quantization names, hf:// URIs (#108)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, method, bits",
    [
        ("Qwen2-7B-Instruct-gptq-int4", "gptq", 4),
        ("Qwen2-7B-Instruct-GPTQ-Int8", "gptq", 8),
        ("llama-3-8b.Q4_K_M.gguf", "gguf", 4),
        ("llama-3-8b-Q8_0.gguf", "gguf", 8),
        ("llama-3-8b-IQ3_XS.gguf", "gguf", 3),
        ("model.gguf", "gguf", None),
        ("Llama-3.1-8B-fp8_e4m3", "fp8", 8),
        ("Meta-Llama-3.1-8B-Instruct-quantized.w4a16", "compressed-tensors", 4),
        ("Meta-Llama-3.1-8B-Instruct-quantized.w8a8", "compressed-tensors", 8),
        ("Llama-3.1-8B-Instruct-FP8-dynamic", "fp8", 8),
        ("Llama-3-8B-AWQ", "awq", 4),
        ("Llama-3-8B-AWQ-INT4", "awq", 4),
    ],
)
def test_quantization_name_variants(name, method, bits):
    result = pp.detect_quantization_from_name(name)
    assert result["quantization_method"] == method
    assert result.get("quantization_bits") == bits


@pytest.mark.parametrize("name", ["Llama-3-8B", "Qwen2.5-7B-Instruct", "Equipment-Model", "Q-learning-8B"])
def test_quantization_name_no_false_positives(name):
    assert pp.detect_quantization_from_name(name) is None


def test_hf_uri_subpath_and_revision_forms():
    assert pp._parse_storage_uri("hf://org/model/sub/dir") == ("org/model", None, "hf_uri")
    assert pp._parse_storage_uri("hf://org/model@main") == ("org/model", "main", "hf_uri")
    assert pp._parse_storage_uri("hf://org/model:abc123") == ("org/model", "abc123", "hf_uri")
    assert pp._parse_storage_uri("hf://org/model@v1/sub") == ("org/model", "v1", "hf_uri")


# ---------------------------------------------------------------------------
# Termination status: retries, restarts, Job outcome (#112)
# ---------------------------------------------------------------------------


def _status_aibom(pod_names, containers):
    return pp.compile_aibom(
        discoveries=[{"pod_metadata": {"name": n}} for n in pod_names],
        detected_datasets=[], runtime_info={}, annotations={}, telemetry=None,
        containers=containers,
    )


def _container(pod, reason, exit_code, **extra):
    return {"pod_name": pod, "name": "training", "terminated_reason": reason,
            "exit_code": exit_code, **extra}


def test_retried_job_that_completed_is_reported_completed():
    containers = [
        _container("job-a1", "OOMKilled", 137, job_result="Complete"),
        _container("job-a2", "Error", 1, job_result="Complete"),
        _container("job-a3", "Completed", 0, job_result="Complete"),
    ]
    em = _status_aibom(["job-a1", "job-a2", "job-a3"], containers)["execution_metadata"]
    assert em["status"] == "Completed"


def test_failed_attempts_keep_their_own_detail_and_are_flagged():
    containers = [
        _container("job-a1", "OOMKilled", 137, job_result="Complete"),
        _container("job-a2", "Completed", 0, job_result="Complete"),
    ]
    pods = {p["pod_name"]: p for p in _status_aibom(["job-a1", "job-a2"], containers)["execution_metadata"]["pods"]}
    assert pods["job-a1"]["status"] == "OOMKilled"
    assert pods["job-a1"]["exit_code"] == 137
    assert pods["job-a1"]["retried_attempt"] is True
    assert "retried_attempt" not in pods["job-a2"]


def test_failed_job_still_reports_the_failure():
    containers = [
        _container("job-a1", "OOMKilled", 137, job_result="Failed"),
        _container("job-a2", "Error", 1, job_result="Failed"),
    ]
    em = _status_aibom(["job-a1", "job-a2"], containers)["execution_metadata"]
    assert em["status"] == "OOMKilled"
    assert not any(p.get("retried_attempt") for p in em["pods"])


def test_failed_job_with_no_reported_container_failure_is_failed():
    containers = [_container("job-a1", "Completed", 0, job_result="Failed")]
    assert _status_aibom(["job-a1"], containers)["execution_metadata"]["status"] == "Failed"


def test_only_the_completed_jobs_failed_pods_are_discounted_in_a_jobset():
    containers = [
        _container("server-a1", "OOMKilled", 137, job_result="Complete"),
        _container("server-a2", "Completed", 0, job_result="Complete"),
        _container("client-a1", "Error", 1, job_result="Failed"),
    ]
    em = _status_aibom(["server-a1", "server-a2", "client-a1"], containers)["execution_metadata"]
    assert em["status"] == "Error"


def test_completed_job_with_no_terminated_state_is_completed():
    containers = [{"pod_name": "job-a1", "name": "training", "job_result": "Complete"}]
    assert _status_aibom(["job-a1"], containers)["execution_metadata"]["status"] == "Completed"


def test_restart_history_surfaces_an_earlier_oom_kill():
    containers = [_container(
        "job-a1", "Completed", 0,
        restart_count=1, last_terminated_reason="OOMKilled", last_exit_code=137,
    )]
    em = _status_aibom(["job-a1"], containers)["execution_metadata"]
    pod = em["pods"][0]
    assert pod["status"] == "Completed"  # the final outcome is unchanged
    assert pod["restart_count"] == 1
    assert pod["last_termination_reason"] == "OOMKilled"
    assert pod["last_exit_code"] == 137
    assert em["status"] == "Completed"


def test_restart_info_prefers_oomkilled_and_sums_restarts():
    containers = [
        {"pod_name": "p", "name": "a", "restart_count": 2, "last_terminated_reason": "Error", "last_exit_code": 1},
        {"pod_name": "p", "name": "b", "restart_count": 1, "last_terminated_reason": "OOMKilled", "last_exit_code": 137},
    ]
    assert pp.pod_restart_info("p", containers) == {
        "restart_count": 3, "last_termination_reason": "OOMKilled", "last_exit_code": 137,
    }


def test_restart_info_empty_when_nothing_restarted():
    assert pp.pod_restart_info("p", [_container("p", "Completed", 0)]) == {}
    pod = _status_aibom(["p"], [_container("p", "Completed", 0)])["execution_metadata"]["pods"][0]
    assert "restart_count" not in pod


# ---------------------------------------------------------------------------
# llm-d prefill/decode role (#167)
# ---------------------------------------------------------------------------


def _role_aibom(roles):
    return pp.compile_aibom(
        discoveries=[{"pod_metadata": {"name": n}} for n in roles],
        detected_datasets=[], runtime_info={}, annotations={}, telemetry=None,
        detected_model={"serving_engine": "vllm", "model_name": "org/m"},
        containers=[{"pod_name": n, "name": "vllm", **({"serving_role": r} if r else {})} for n, r in roles.items()],
    )


def test_llm_d_role_recorded_on_pod_and_inference():
    aibom = _role_aibom({"ms-decode-0": "decode"})
    assert aibom["execution_metadata"]["pods"][0]["serving_role"] == "decode"
    assert aibom["inference"]["serving_role"] == "decode"


def test_llm_d_role_omitted_when_pods_disagree_or_have_none():
    mixed = _role_aibom({"ms-prefill-0": "prefill", "ms-decode-0": "decode"})
    assert "serving_role" not in mixed["inference"]
    assert [p["serving_role"] for p in mixed["execution_metadata"]["pods"]] == ["prefill", "decode"]

    plain = _role_aibom({"vllm-0": None})
    assert "serving_role" not in plain["inference"]
    assert "serving_role" not in plain["execution_metadata"]["pods"][0]
