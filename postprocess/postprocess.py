#!/usr/bin/env python3
"""AIBOM postprocess -- compile an AI Bill of Materials.

Runs as a Kubernetes Job after an instrumented workload completes.
Reads discovery and dataset data from a ConfigMap mount, optionally
queries Prometheus for telemetry, and produces an AIBOM JSON document.
"""

import json
import os
import re
import shlex
import ssl
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import urllib.request
import urllib.parse
import urllib.error

import k8s_api

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_DIR = os.environ.get("AIBOM_INPUT_DIR", "/data/input")
JOB_NAME = os.environ.get("AIBOM_JOB_NAME", "")
JOB_NAMESPACE = os.environ.get("AIBOM_JOB_NAMESPACE", "")
PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", "")

# Used only to build a clickable Grafana Explore deep link into
# resource_utilization.grafana_links — actual telemetry queries always go straight to
# PROMETHEUS_URL above, never through Grafana. Either being empty just omits the link.
GRAFANA_URL = os.environ.get("GRAFANA_URL", "")
GRAFANA_DATASOURCE_UID = os.environ.get("GRAFANA_DATASOURCE_UID", "")

# Auth is always automatic, never configured per-query: the postprocess Job's own
# ServiceAccount token (Kubernetes auto-mounts and rotates this in place, roughly
# hourly) is read fresh on every request and sent as a Bearer token; the cluster's
# service-serving CA bundle (injected into a ConfigMap by the service-ca operator,
# see watcher.go's serviceCAConfigMapName) is trusted for TLS if present. Either
# file being absent (a plain-HTTP dev Prometheus, or running outside a pod) falls
# back to no Authorization header / the system trust store rather than erroring —
# mirrors gpu-quota-operator's metrics.Client (metrics/prometheus.go).
SERVICE_ACCOUNT_TOKEN_FILE = "/var/run/secrets/kubernetes.io/serviceaccount/token"
SERVICE_CA_CERT_FILE = "/etc/aibom-postprocess/service-ca/service-ca.crt"

# The observability backend can lag behind real time before freshly-scraped
# samples become queryable, so a summary query fired immediately after the
# workload's pod completes can race that ingestion delay and come back empty
# even though the same query succeeds moments later. Retry missing summary
# metrics with a delay rather than accepting the first empty result.
TELEMETRY_RETRY_ATTEMPTS = int(os.environ.get("AIBOM_TELEMETRY_RETRY_ATTEMPTS", "3"))
TELEMETRY_RETRY_DELAY_S = int(os.environ.get("AIBOM_TELEMETRY_RETRY_DELAY_S", "45"))

# Normally a pod with no detected GPU (gpu_count 0/missing, e.g. nvidia-smi found
# nothing) is skipped for telemetry entirely -- there's no GPU utilization to query.
# On a mock cluster (e.g. kind) with no real GPU hardware at all, that means every
# pod gets skipped and telemetry never gets exercised. Debug-only escape hatch to
# query telemetry for every pod regardless of detected GPU count.
DEBUG_TELEMETRY_ALL_PODS = os.environ.get("AIBOM_DEBUG_TELEMETRY_ALL_PODS", "").lower() == "true"

TELEMETRY_QUERIES = {
    "gpu_utilization": 'nerc:dcgm_gpu_util:avg5m{exported_pod="{pod_name}"}',
    "gpu_memory_used": 'nerc:dcgm_fb_used:avg5m{exported_pod="{pod_name}"}',
    "gpu_power": 'nerc:dcgm_power_usage:avg5m{exported_pod="{pod_name}"}',
    "cpu_usage": 'rate(container_cpu_usage_seconds_total{pod="{pod_name}", container!="POD", container!=""}[10m])',
    "memory_usage": 'container_memory_working_set_bytes{pod="{pod_name}", container!="POD", container!=""}',
    "network_receive": 'rate(container_network_receive_bytes_total{pod="{pod_name}"}[10m])',
    "network_transmit": 'rate(container_network_transmit_bytes_total{pod="{pod_name}"}[10m])',
}

SCRAPE_INTERVAL_MS = 5 * 60 * 1000

SUMMARY_QUERIES = {
    "avg_gpu_utilization": {
        "query": 'avg_over_time(nerc:dcgm_gpu_util:avg5m{exported_pod="{pod_name}"}[{duration}])',
        "unit": "percent",
    },
    "avg_gpu_memory_used": {
        "query": 'avg_over_time(nerc:dcgm_fb_used:avg5m{exported_pod="{pod_name}"}[{duration}])',
        "unit": "MiB",
    },
    "avg_gpu_power": {
        "query": 'avg_over_time(nerc:dcgm_power_usage:avg5m{exported_pod="{pod_name}"}[{duration}])',
        "unit": "watts",
    },
    "avg_cpu_usage": {
        "query": 'avg_over_time(rate(container_cpu_usage_seconds_total{pod="{pod_name}", container!="POD", container!=""}[5m])[{duration}:1m])',
        "unit": "cores",
    },
    "avg_memory_usage": {
        "query": 'avg_over_time(container_memory_working_set_bytes{pod="{pod_name}", container!="POD", container!=""}[{duration}])',
        "unit": "bytes",
    },
    "avg_network_receive": {
        "query": 'avg_over_time(rate(container_network_receive_bytes_total{pod="{pod_name}"}[5m])[{duration}:1m])',
        "unit": "bytes_per_sec",
    },
    "avg_network_transmit": {
        "query": 'avg_over_time(rate(container_network_transmit_bytes_total{pod="{pod_name}"}[5m])[{duration}:1m])',
        "unit": "bytes_per_sec",
    },
}

# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------


def load_json_file(path, description):
    p = Path(path)
    if not p.exists():
        print(f"  {description}: not found ({p})", file=sys.stderr)
        return None
    try:
        with open(p) as f:
            data = json.load(f)
        print(f"  {description}: loaded")
        return data
    except Exception as e:
        print(f"  {description}: failed to load ({e})", file=sys.stderr)
        return None


def load_discovery():
    data = load_json_file(f"{INPUT_DIR}/discovery.json", "Discovery data")
    if data is None:
        return []
    if isinstance(data, list):
        return data
    return [data]


def load_datasets():
    data = load_json_file(f"{INPUT_DIR}/dataset.json", "Dataset data")
    if data is None:
        return [], {}
    datasets = data.get("datasets", [])
    runtime_info = data.get("runtime_info", {})
    return datasets, runtime_info


def load_annotations():
    data = load_json_file(f"{INPUT_DIR}/annotations.json", "Annotations")
    if data is None:
        return {}
    return data


def load_storage():
    data = load_json_file(f"{INPUT_DIR}/storage.json", "InferenceService storage")
    if data is None:
        return {}
    return data


def load_containers():
    data = load_json_file(f"{INPUT_DIR}/containers.json", "Container specs")
    if data is None:
        return []
    if isinstance(data, list):
        return data
    return [data]


# ---------------------------------------------------------------------------
# Model detection (ported from coldpress/model_detector.py)
# ---------------------------------------------------------------------------

_QUANT_PATTERNS = [
    (r"GPTQ[_-]Int8", "gptq", 8),
    (r"GPTQ[_-]Int4", "gptq", 4),
    (r"gptq[_-]4bit", "gptq", 4),
    (r"GPTQ", "gptq", 4),
    (r"[_-]AWQ\b", "awq", 4),
    (r"[_-]awq\b", "awq", 4),
    (r"AQLM[_-](\d+)Bit", "aqlm", None),
    (r"AQLM", "aqlm", 2),
    (r"EXL2", "exl2", None),
    (r"SqueezeLLM[_-](\d+)bit", "squeezellm", None),
    (r"SqueezeLLM", "squeezellm", 4),
    (r"HQQ[_-](\d+)bit", "hqq", None),
    (r"HQQ", "hqq", 4),
    (r"QuIP", "quip", 2),
    (r"EETQ", "eetq", 8),
    (r"AutoRound", "autoround", 4),
    (r"[_-]NVFP4\b", "fp4", 4),
    (r"[_-]MXFP4\b", "fp4", 4),
    (r"[_-]FP4\b", "fp4", 4),
    (r"[_-]FP8\b", "fp8", 8),
    (r"[_-]fp8\b", "fp8", 8),
    (r"bnb[_-]4bit", "bitsandbytes", 4),
    (r"bnb[_-]8bit", "bitsandbytes", 8),
    (r"[_-]NF4\b", "bitsandbytes", 4),
    (r"[_-]nf4\b", "bitsandbytes", 4),
    (r"[_-]INT4\b", "int4", 4),
    (r"[_-]int4\b", "int4", 4),
    (r"[_-]INT8\b", "int8", 8),
    (r"[_-]int8\b", "int8", 8),
    (r"[_-]Marlin\b", "marlin", 4),
    (r"[_-]marlin\b", "marlin", 4),
    (r"GGUF", "gguf", None),
    (r"GGML", "ggml", None),
]

_COMPILED_QUANT_PATTERNS = [(re.compile(p), method, bits) for p, method, bits in _QUANT_PATTERNS]


def detect_quantization_from_name(model_name):
    if not model_name:
        return None
    for pattern, method, bits in _COMPILED_QUANT_PATTERNS:
        m = pattern.search(model_name)
        if m:
            result = {"quantization_method": method}
            if bits is not None:
                result["quantization_bits"] = bits
            elif m.lastindex and m.group(1).isdigit():
                result["quantization_bits"] = int(m.group(1))
            return result
    return None


def _coerce_scalar(s):
    """Best-effort scalar type coercion for values inside a key=value list."""
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    for conv in (int, float):
        try:
            return conv(s)
        except ValueError:
            pass
    return s


def _parse_json_or_kv(val):
    """Parse a flag value that's either a JSON blob or a comma-separated
    key=value list, e.g. vLLM's --speculative-config and
    --override-generation-config accept both forms. Returns None if the
    value can't be understood as either."""
    val = val.strip()
    if val.startswith("{"):
        try:
            return json.loads(val)
        except (ValueError, TypeError):
            return None
    result = {}
    for pair in val.split(","):
        if "=" not in pair:
            continue
        k, _, v = pair.partition("=")
        result[k.strip()] = _coerce_scalar(v.strip())
    return result or None


_VLLM_ARG_MAP = {
    "--model": ("model_name", str),
    "--served-model-name": ("served_model_name", str),
    "--quantization": ("quantization", str),
    "-q": ("quantization", str),
    "--dtype": ("dtype", str),
    "--max-model-len": ("max_model_len", int),
    "--tensor-parallel-size": ("tensor_parallel_size", int),
    "-tp": ("tensor_parallel_size", int),
    "--pipeline-parallel-size": ("pipeline_parallel_size", int),
    "-pp": ("pipeline_parallel_size", int),
    "--enable-expert-parallel": ("enable_expert_parallel", bool),
    "--data-parallel-size": ("data_parallel_size", int),
    "-dp": ("data_parallel_size", int),
    "--gpu-memory-utilization": ("gpu_memory_utilization", float),
    "--max-num-seqs": ("max_num_seqs", int),
    "--seed": ("seed", int),
    "--trust-remote-code": ("trust_remote_code", bool),
    "--enforce-eager": ("enforce_eager", bool),
    "--enable-prefix-caching": ("enable_prefix_caching", bool),
    "--port": ("port", int),
    "--speculative-model": ("speculative_model", str),
    "--num-speculative-tokens": ("num_speculative_tokens", int),
    "--speculative-config": ("speculative_config", _parse_json_or_kv),
    "--override-generation-config": ("generation_config_overrides", _parse_json_or_kv),
}

_BOOL_FLAGS = {k for k, (_, t) in _VLLM_ARG_MAP.items() if t is bool}


def detect_vllm_from_command(command):
    if not command:
        return None
    joined = " ".join(command)
    if "vllm" not in joined and "vllm.entrypoints" not in joined:
        return None

    result = {"serving_engine": "vllm"}

    for i, arg in enumerate(command):
        if "=" in arg:
            key, _, val = arg.partition("=")
        else:
            key = arg
            val = None

        if key in _BOOL_FLAGS:
            result[_VLLM_ARG_MAP[key][0]] = True
            continue

        if key not in _VLLM_ARG_MAP:
            continue

        name, conv = _VLLM_ARG_MAP[key]

        if val is None and i + 1 < len(command):
            val = command[i + 1]

        if val is None:
            continue

        try:
            converted = conv(val)
        except (ValueError, TypeError):
            converted = val

        if converted is not None:
            result[name] = converted

    if "quantization" not in result and "model_name" in result:
        quant = detect_quantization_from_name(result["model_name"])
        if quant:
            result.update(quant)

    # Normalize legacy --speculative-model/--num-speculative-tokens into the
    # same shape as the modern --speculative-config flag.
    if "speculative_config" not in result and (
        "speculative_model" in result or "num_speculative_tokens" in result
    ):
        spec_config = {}
        if "speculative_model" in result:
            spec_config["model"] = result.pop("speculative_model")
        if "num_speculative_tokens" in result:
            spec_config["num_speculative_tokens"] = result.pop("num_speculative_tokens")
        result["speculative_config"] = spec_config

    return result if len(result) > 1 else None


def _to_bool(val):
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("1", "true", "yes")


_TRL_ARG_MAP = {
    "--model_name_or_path": ("model_name", str),
    "--model-name-or-path": ("model_name", str),
    "--use_peft": ("use_peft", _to_bool),
    "--use-peft": ("use_peft", _to_bool),
    "--lora_r": ("lora_rank", int),
    "--lora-r": ("lora_rank", int),
    "--lora_alpha": ("lora_alpha", int),
    "--lora-alpha": ("lora_alpha", int),
    "--use_dora": ("use_dora", _to_bool),
    "--use-dora": ("use_dora", _to_bool),
    "--use_rslora": ("use_rslora", _to_bool),
    "--use-rslora": ("use_rslora", _to_bool),
    "--load_in_4bit": ("load_in_4bit", _to_bool),
    "--load-in-4bit": ("load_in_4bit", _to_bool),
    "--load_in_8bit": ("load_in_8bit", _to_bool),
    "--load-in-8bit": ("load_in_8bit", _to_bool),
    "--learning_rate": ("learning_rate", float),
    "--learning-rate": ("learning_rate", float),
    "--per_device_train_batch_size": ("batch_size", int),
    "--per-device-train-batch-size": ("batch_size", int),
    "--num_train_epochs": ("epochs", int),
    "--num-train-epochs": ("epochs", int),
    "--seed": ("random_seed", int),
}


def detect_trl_from_command(command):
    """Detect model/LoRA config from a `trl sft`/`trl dpo`-style CLI invocation."""
    if not command or not re.search(r"\btrl\b", " ".join(command)):
        return None

    result = {"training_framework": "trl"}

    for i, arg in enumerate(command):
        if arg.startswith("--") and "=" in arg:
            key, _, val = arg.partition("=")
        else:
            key = arg
            val = None

        if key not in _TRL_ARG_MAP:
            continue

        name, conv = _TRL_ARG_MAP[key]

        if val is None and i + 1 < len(command) and not command[i + 1].startswith("--"):
            val = command[i + 1]

        if val is None:
            continue

        try:
            converted = conv(val)
        except (ValueError, TypeError):
            converted = val

        result[name] = converted

    use_dora = result.pop("use_dora", False)
    use_rslora = result.pop("use_rslora", False)
    quantized = result.pop("load_in_4bit", False) or result.pop("load_in_8bit", False)

    if result.pop("use_peft", False):
        if "lora_rank" not in result:
            result["adaptation_method"] = "peft"
        elif use_dora:
            result["adaptation_method"] = "dora"
        elif use_rslora:
            result["adaptation_method"] = "rslora"
        elif quantized:
            result["adaptation_method"] = "qlora"
        else:
            result["adaptation_method"] = "lora"

    return result if len(result) > 1 else None


def _flatten_container_command(container):
    """Expand `sh -c "..."`/`bash -c "..."` wrappers into a flat token list.

    Jobs that need to `pip install` before running a training CLI (e.g. trl)
    wrap everything in a single shell string, which would otherwise hide the
    CLI flags from the per-token detectors below.
    """
    command = (container.get("command") or []) + (container.get("args") or [])
    if (
        len(command) >= 3
        and os.path.basename(command[0]) in ("sh", "bash")
        and command[1] in ("-c", "-lc", "-ec", "-cx")
    ):
        # Join shell line-continuations (`\` immediately followed by a
        # newline) before tokenizing -- shlex doesn't do this on its own,
        # and without it a backslash-newline survives as a spurious literal
        # token that can land right after a bare boolean flag (e.g.
        # `--use_peft \<newline>--lora_r`) and get misread as its value.
        script = re.sub(r"\\\n", " ", " ".join(command[2:]))
        try:
            return shlex.split(script)
        except ValueError:
            return command
    return command


_LAUNCHERS = {"accelerate", "deepspeed", "torchrun", "mpirun"}


def _find_flag_value(tokens, flag_names):
    for i, tok in enumerate(tokens):
        if tok.startswith("--") and "=" in tok:
            key, _, val = tok.partition("=")
            if key in flag_names:
                return val
        elif tok in flag_names and i + 1 < len(tokens):
            return tokens[i + 1]
    return None


# trl's own CLI (and similar tools built on HF Accelerate) accept these
# accelerate-launch arguments directly and spawn `accelerate launch`
# internally -- so "accelerate" never appears as its own token in the
# container's command, only these passthrough flags do.
_ACCELERATE_CONFIG_STRATEGIES = {
    "fsdp1": "fsdp",
    "fsdp2": "fsdp",
    "zero1": "deepspeed",
    "zero2": "deepspeed",
    "zero3": "deepspeed",
    "multi_gpu": "data_parallel",
    "single_gpu": None,
}


def detect_parallelization_from_command(tokens):
    """Best-effort detection of a distributed-training parallelization
    strategy, independent of which training tool (trl, a custom script, ...)
    is being launched. Covers three shapes:
      - an explicit launcher binary (accelerate/deepspeed/torchrun/mpirun)
      - a bare --fsdp/--deepspeed flag on the training command itself
      - accelerate-launch args (--num_processes, --accelerate_config)
        passed straight through to a CLI like `trl` that spawns
        `accelerate launch` internally, with no launcher token visible
    """
    if not tokens:
        return None

    launcher = next(
        (os.path.basename(tok) for tok in tokens if os.path.basename(tok) in _LAUNCHERS),
        None,
    )
    has_fsdp = any(t == "--fsdp" or t.startswith("--fsdp=") for t in tokens)
    has_deepspeed_flag = any(t == "--deepspeed" or t.startswith("--deepspeed=") for t in tokens)
    has_multi_gpu = "--multi_gpu" in tokens or "--multi-gpu" in tokens
    num_processes = _try_int(_find_flag_value(tokens, ("--num_processes", "--num-processes")))
    accelerate_config = _find_flag_value(tokens, ("--accelerate_config", "--accelerate-config"))
    accelerate_config_name = (
        os.path.splitext(os.path.basename(accelerate_config))[0] if accelerate_config else None
    )

    if has_fsdp:
        strategy = "fsdp"
    elif has_deepspeed_flag or launcher == "deepspeed":
        strategy = "deepspeed"
    elif accelerate_config_name in _ACCELERATE_CONFIG_STRATEGIES:
        strategy = _ACCELERATE_CONFIG_STRATEGIES[accelerate_config_name]
    elif has_multi_gpu or launcher in ("torchrun", "mpirun"):
        strategy = "data_parallel"
    elif num_processes and num_processes > 1:
        strategy = "data_parallel"
    else:
        strategy = None

    return {"parallelization_strategy": strategy} if strategy else None


def _parallelization_strategy_from_device_map(device_map):
    """Fallback for in-script sharding with no CLI/launcher/accelerate-config
    signal at all -- e.g. a raw transformers.Trainer or inference script that
    passes `device_map="auto"` (or another multi-device map) directly to
    `from_pretrained`, sharding the model across GPUs within a single
    process. Lowest-priority signal: any explicit launcher/CLI/accelerate
    detection is more authoritative than this heuristic."""
    if not device_map:
        return None
    if device_map in ("cpu", "cuda", "cuda:0"):
        return None
    return "model_parallel"


def detect_model_from_storage(storage):
    """Detect model identity from a KServe InferenceService's declared
    storage.path/storageUri (see watcher.go's resolveInferenceServiceStorage
    and storage.json). Predictor pods backed by an S3/MinIO data-connection
    bucket run a built-in serving-runtime container with a fixed
    --model=/mnt/models mount, so detect_vllm_from_command can't recover the
    real model identity from the CLI — only the InferenceService object
    declares it, as a bucket path string, e.g. "models/tinyllama-1.1b-chat".

    This is a best-effort identification, not a verification: the returned
    model_name is only the final path segment of the declared location. It is
    not derived from the actual file contents, and it does not confirm the
    bucket data matches the name (a renamed or generically-named prefix would
    be misreported), since resolving the data-connection Secret and reading
    the bucket is out of scope here.
    """
    if not storage:
        return None

    location = storage.get("storage_path") or storage.get("storage_uri")
    if not location:
        return None

    model_name = location.rstrip("/").split("/")[-1]
    if not model_name:
        return None

    result = {"model_name": model_name}
    quant = detect_quantization_from_name(model_name)
    if quant:
        result.update(quant)
    return result


def detect_model_from_containers(containers):
    model_result = None
    parallel_result = None
    for container in containers:
        tokens = _flatten_container_command(container)
        if model_result is None:
            model_result = detect_vllm_from_command(tokens) or detect_trl_from_command(tokens)
        if parallel_result is None:
            parallel_result = detect_parallelization_from_command(tokens)
        if model_result and parallel_result:
            break

    result = dict(model_result or {})
    if parallel_result:
        result.update(parallel_result)
    return result


_DATASET_ARG_MAP = {
    "--dataset_name": ("dataset_name", str),
    "--dataset-name": ("dataset_name", str),
    "--dataset_config_name": ("dataset_config", str),
    "--dataset-config-name": ("dataset_config", str),
    "--dataset_train_split": ("dataset_split", str),
    "--dataset-train-split": ("dataset_split", str),
}


def detect_dataset_from_command(tokens):
    """Detect the dataset requested on a training CLI invocation (e.g. `trl
    sft --dataset_name ...`), independent of which training tool is used."""
    if not tokens:
        return None

    result = {}
    for i, tok in enumerate(tokens):
        if tok.startswith("--") and "=" in tok:
            key, _, val = tok.partition("=")
        else:
            key = tok
            val = None

        if key not in _DATASET_ARG_MAP:
            continue

        name, conv = _DATASET_ARG_MAP[key]

        if val is None and i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            val = tokens[i + 1]

        if val is None:
            continue

        try:
            result[name] = conv(val)
        except (ValueError, TypeError):
            result[name] = val

    return result if result.get("dataset_name") else None


def detect_dataset_from_containers(containers):
    for container in containers:
        tokens = _flatten_container_command(container)
        result = detect_dataset_from_command(tokens)
        if result:
            return result
    return None


# ---------------------------------------------------------------------------
# Phase 1: Telemetry collection
# ---------------------------------------------------------------------------


def _prometheus_ssl_context():
    # Falls back to the default context (system trust store) if the service-ca
    # bundle isn't mounted, e.g. a plain-HTTP dev Prometheus (mock-openshift-cluster)
    # that doesn't need TLS at all — matches gpu-quota-operator's buildTransport().
    if os.path.exists(SERVICE_CA_CERT_FILE):
        return ssl.create_default_context(cafile=SERVICE_CA_CERT_FILE)
    return None


def _prometheus_auth_headers():
    # Read fresh on every call rather than once at startup: Kubernetes rotates a
    # projected ServiceAccount token in place roughly hourly, so caching it would
    # risk auth silently failing partway through a long-running postprocess Job.
    try:
        with open(SERVICE_ACCOUNT_TOKEN_FILE) as f:
            return {"Authorization": f"Bearer {f.read().strip()}"}
    except FileNotFoundError:
        return {}


def _query_prometheus(path, params, timeout):
    url = f"{PROMETHEUS_URL}{path}?{urllib.parse.urlencode(params)}"
    try:
        req = urllib.request.Request(url, headers=_prometheus_auth_headers())
        with urllib.request.urlopen(req, timeout=timeout, context=_prometheus_ssl_context()) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code}: {e.read().decode()}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Query failed: {e}", file=sys.stderr)
        return None


def _range_step_seconds(start_ms, end_ms, max_points=1000):
    span_s = max((end_ms - start_ms) / 1000, 1)
    return max(int(span_s / max_points), 15)


def query_prometheus_range(promql, start_ms, end_ms):
    return _query_prometheus(
        "/api/v1/query_range",
        {
            "query": promql,
            "start": start_ms / 1000,
            "end": end_ms / 1000,
            "step": _range_step_seconds(start_ms, end_ms),
        },
        timeout=30,
    )


def query_prometheus_instant(promql, time_ms):
    return _query_prometheus(
        "/api/v1/query",
        {"query": promql, "time": time_ms / 1000},
        timeout=30,
    )


def build_grafana_explore_url(grafana_url, datasource_uid, named_queries, start_ms, end_ms):
    # Purely presentational: builds a link into whatever Grafana instance you point it
    # at (via GRAFANA_URL/GRAFANA_DATASOURCE_UID), independent of the actual telemetry
    # queries above, which always go straight to PROMETHEUS_URL.
    end_ms_padded = end_ms + SCRAPE_INTERVAL_MS  # pad to capture the final scrape interval
    # Metrics span wildly different scales (%, MiB, watts, cores, bytes), so
    # plotting all of them by default produces an unreadable graph. Only the
    # first metric starts visible; the rest are hidden but still present as
    # toggleable query rows (Grafana persists a query's "hide" state in the
    # URL itself, so this stays shareable/bookmarkable).
    queries = [
        {
            "refId": chr(65 + i),
            "expr": promql,
            "datasource": {"uid": datasource_uid},
            "hide": i != 0,
        }
        for i, (_, promql) in enumerate(named_queries)
    ]
    explore_state = {
        "datasource": datasource_uid,
        "queries": queries,
        "range": {"from": str(start_ms), "to": str(end_ms_padded)},
    }
    return f"{grafana_url}/explore?left={urllib.parse.quote(json.dumps(explore_state))}"


def parse_range_response(response):
    if not response or response.get("status") != "success":
        return []
    results = []
    for series in response.get("data", {}).get("result", []):
        for ts, val in series.get("values", []):
            results.append(
                {
                    "timestamp": datetime.fromtimestamp(ts).isoformat(),
                    "value": float(val),
                }
            )
    return results


def parse_instant_value(response):
    if not response or response.get("status") != "success":
        return None
    # Only the first series, matching the pre-Prometheus-native behavior: a
    # multi-GPU pod can return one series per GPU-index label, and this has never
    # summed/averaged across them (the underlying queries aren't pre-aggregated).
    result = response.get("data", {}).get("result", [])
    if not result or "value" not in result[0]:
        return None
    return float(result[0]["value"][1])


def ms_to_promql_duration(ms):
    seconds = max(int(ms / 1000), 60)
    if seconds >= 3600:
        return f"{seconds // 3600}h"
    return f"{seconds // 60}m"


def collect_telemetry(discoveries):
    print(f"  Processing {len(discoveries)} pod(s)")

    telemetry_summary = {
        "collected_at": datetime.utcnow().isoformat() + "Z",
        "prometheus_url": PROMETHEUS_URL,
        "pods": [],
    }

    for discovery in discoveries:
        pod_metadata = discovery.get("pod_metadata", {})
        pod_uid = pod_metadata.get("uid")
        pod_name = pod_metadata.get("name")
        start_time = pod_metadata.get("start_time")

        if not pod_uid or pod_uid == "unknown":
            print(f"  WARNING: No pod UID, skipping", file=sys.stderr)
            continue

        gpu_count = discovery.get("gpu", {}).get("gpu_count")
        if (not gpu_count or str(gpu_count) == "0") and not DEBUG_TELEMETRY_ALL_PODS:
            print(f"  Skipping {pod_name} (no GPUs)")
            continue

        print(f"  Pod: {pod_name} ({pod_uid})")

        try:
            start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            print(f"  WARNING: Invalid start_time '{start_time}', skipping", file=sys.stderr)
            continue

        end_dt = datetime.utcnow()
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)

        metrics = {
            name: tmpl.replace("{pod_name}", pod_name)
            for name, tmpl in TELEMETRY_QUERIES.items()
        }

        pod_telemetry = {
            "pod_uid": pod_uid,
            "pod_name": pod_name,
            "start_time": start_time,
            "metrics": {},
        }
        if GRAFANA_URL and GRAFANA_DATASOURCE_UID:
            pod_telemetry["grafana_explore_url"] = build_grafana_explore_url(
                GRAFANA_URL, GRAFANA_DATASOURCE_UID, list(metrics.items()), start_ms, end_ms
            )

        for metric_name, promql in metrics.items():
            print(f"    Querying {metric_name}...")
            response = query_prometheus_range(promql, start_ms, end_ms)
            if response:
                data_points = parse_range_response(response)
                pod_telemetry["metrics"][metric_name] = {
                    "data_point_count": len(data_points),
                }
                print(f"      {len(data_points)} data points")
            else:
                print(f"      WARNING: No data returned")

        if SUMMARY_QUERIES:
            total_ms = end_ms - start_ms
            # Exclude the cold-start window (up to one scrape interval with no
            # updated observation from the hardware) from the averages. Capped
            # at half the run length rather than requiring a fixed minimum
            # runtime, so short runs still get a partial correction instead of
            # none at all.
            exclude_ms = min(SCRAPE_INTERVAL_MS, total_ms // 2)
            summary_start_ms = start_ms + exclude_ms
            includes_cold_start = exclude_ms < SCRAPE_INTERVAL_MS

            duration = ms_to_promql_duration(end_ms - summary_start_ms)
            print(
                f"    Running summary queries (duration={duration}, "
                f"excludes_cold_start={not includes_cold_start})..."
            )
            aggregated = {}
            for attempt in range(1, TELEMETRY_RETRY_ATTEMPTS + 1):
                pending = {
                    name: info for name, info in SUMMARY_QUERIES.items() if name not in aggregated
                }
                if not pending:
                    break
                if attempt > 1:
                    print(
                        f"      Retrying {len(pending)} summary quer"
                        f"{'y' if len(pending) == 1 else 'ies'} after possible ingestion "
                        f"delay (attempt {attempt}/{TELEMETRY_RETRY_ATTEMPTS}, "
                        f"waited {TELEMETRY_RETRY_DELAY_S}s)..."
                    )
                for sq_name, sq_info in pending.items():
                    promql = (
                        sq_info["query"]
                        .replace("{pod_name}", pod_name)
                        .replace("{duration}", duration)
                    )
                    response = query_prometheus_instant(promql, end_ms)
                    value = parse_instant_value(response)
                    if value is not None:
                        aggregated[sq_name] = {
                            "value": round(value, 2),
                            "unit": sq_info.get("unit", ""),
                        }
                        print(f"      {sq_name}: {round(value, 2)} {sq_info.get('unit', '')}")
                    else:
                        print(f"      {sq_name}: no data (attempt {attempt}/{TELEMETRY_RETRY_ATTEMPTS})")
                if len(aggregated) < len(SUMMARY_QUERIES) and attempt < TELEMETRY_RETRY_ATTEMPTS:
                    time.sleep(TELEMETRY_RETRY_DELAY_S)
            pod_telemetry["aggregated"] = aggregated
            pod_telemetry["summary_includes_cold_start"] = includes_cold_start

        telemetry_summary["pods"].append(pod_telemetry)

    print(f"  Pods processed: {len(telemetry_summary['pods'])}")
    return telemetry_summary


# ---------------------------------------------------------------------------
# Phase 2: AIBOM compilation
# ---------------------------------------------------------------------------


def safe_get(data, *keys, default=None):
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key, {})
        else:
            return default
    return data if data != {} else default


def compile_aibom(discoveries, detected_datasets, runtime_info, annotations, telemetry, detected_model=None, cli_dataset=None):
    print(f"  Discovery files: {len(discoveries)}")
    print(f"  Auto-detected datasets: {len(detected_datasets)}")
    if detected_model:
        print(f"  Detected model: {detected_model.get('model_name', 'unknown')}")
    if runtime_info:
        print(
            "  Runtime info: "
            + ", ".join(f"{k}={v}" for k, v in runtime_info.items())
        )
    print(f"  Telemetry: {'available' if telemetry else 'not available'}")

    aibom = {}

    # Experiment metadata from annotations
    aibom["experiment_intent"] = annotations.get("experiment-intent", "unknown")
    aibom["experiment_name"] = annotations.get("experiment-name") or JOB_NAME or None
    aibom["experiment_description"] = annotations.get("experiment-description")

    aibom["source_code"] = {
        "git_repository": annotations.get("git-repository"),
        "git_commit": annotations.get("git-commit"),
        "git_branch": annotations.get("git-branch"),
    }

    # Execution metadata from discovery
    pods = []
    for discovery in discoveries:
        pod_meta = discovery.get("pod_metadata", {})
        pods.append(
            {
                "pod_name": pod_meta.get("name"),
                "pod_uid": pod_meta.get("uid"),
                "pod_namespace": pod_meta.get("namespace"),
                "pod_ip": pod_meta.get("ip"),
                "node_name": pod_meta.get("node"),
                "start_time": pod_meta.get("start_time"),
            }
        )

    aibom["execution_metadata"] = {
        "job_id": JOB_NAME,
        "namespace": JOB_NAMESPACE,
        "pods": pods,
    }

    # Model info: auto-detected (container commands, then runtime hooks for
    # scripts that build TrainingArguments/from_pretrained directly in
    # Python with no corresponding CLI flags), then annotations override
    dm = detected_model or {}
    model_name = annotations.get("model-name") or dm.get("model_name") or runtime_info.get("model_name")
    quantization = (
        annotations.get("quantization")
        or dm.get("quantization")
        or dm.get("quantization_method")
        or runtime_info.get("quantization_method")
    )
    quantization_bits = (
        _try_int(annotations.get("quantization-bits"))
        or dm.get("quantization_bits")
        or runtime_info.get("quantization_bits")
    )
    aibom["model"] = {
        "name": model_name,
        "version": annotations.get("model-version"),
        "architecture": annotations.get("model-architecture") or runtime_info.get("model_architecture"),
        "framework": (
            annotations.get("model-framework")
            or dm.get("serving_engine")
            or dm.get("training_framework")
            or runtime_info.get("training_framework")
        ),
        "quantization": quantization,
        "quantization_bits": quantization_bits,
        "dtype": annotations.get("dtype") or dm.get("dtype") or runtime_info.get("dtype"),
    }
    if dm.get("speculative_config"):
        aibom["model"]["speculative_decoding"] = dm["speculative_config"]

    # Dataset section
    cli_ds = cli_dataset or {}
    declared_dataset = {
        "name": annotations.get("dataset-name") or cli_ds.get("dataset_name"),
        "version": annotations.get("dataset-version"),
        "source": annotations.get("dataset-source"),
        "license": annotations.get("dataset-license"),
    }
    if annotations.get("dataset-name"):
        declared_dataset["declared_via"] = "annotation"
    elif cli_ds.get("dataset_name"):
        declared_dataset["declared_via"] = "cli_arg"
    has_declared = bool(declared_dataset.get("name"))
    intent = aibom["experiment_intent"]

    if has_declared or detected_datasets or intent in ("training", "sft"):
        aibom["dataset"] = {"declared": declared_dataset}
        if detected_datasets:
            aibom["dataset"]["auto_detected"] = detected_datasets
            print(f"  Merged {len(detected_datasets)} auto-detected dataset(s)")
            if not declared_dataset.get("name"):
                first = detected_datasets[0]
                aibom["dataset"]["declared"]["name"] = first.get("dataset_name")
                aibom["dataset"]["declared"]["source"] = first.get("source")
                aibom["dataset"]["declared"]["declared_via"] = "inferred_from_runtime"
                if first.get("version"):
                    aibom["dataset"]["declared"]["version"] = first["version"]
                if first.get("license"):
                    aibom["dataset"]["declared"]["license"] = first["license"]

            declared_name = aibom["dataset"]["declared"].get("name")
            for entry in detected_datasets:
                entry["matches_declared"] = entry.get("dataset_name") == declared_name

    # Training config
    if intent in ("training", "sft"):
        aibom["training"] = {
            "optimizer": annotations.get("optimizer") or runtime_info.get("optimizer"),
            "learning_rate": _first_not_none(
                runtime_info.get("learning_rate"),
                dm.get("learning_rate"),
                _try_float(annotations.get("learning-rate")),
            ),
            "batch_size": _first_not_none(
                runtime_info.get("batch_size"),
                dm.get("batch_size"),
                _try_int(annotations.get("batch-size")),
            ),
            "epochs": _first_not_none(
                runtime_info.get("epochs"),
                dm.get("epochs"),
                _try_int(annotations.get("epochs")),
            ),
            "random_seed": _first_not_none(
                runtime_info.get("random_seed"),
                dm.get("random_seed"),
                _try_int(annotations.get("random-seed")),
            ),
            "parallelization_strategy": (
                annotations.get("parallelization-strategy")
                or runtime_info.get("parallelization_strategy")
                or dm.get("parallelization_strategy")
                or _parallelization_strategy_from_device_map(runtime_info.get("model_device_map"))
            ),
        }

    # Fine-tuning config
    if intent == "sft":
        aibom["fine_tuning"] = {
            "adaptation_method": (
                annotations.get("adaptation-method")
                or dm.get("adaptation_method")
                or runtime_info.get("adaptation_method")
            ),
            "lora_rank": (
                _try_int(annotations.get("lora-rank")) or dm.get("lora_rank") or runtime_info.get("lora_rank")
            ),
            "lora_alpha": (
                _try_int(annotations.get("lora-alpha")) or dm.get("lora_alpha") or runtime_info.get("lora_alpha")
            ),
        }

    # Inference config: auto-detected from container commands, then annotations override
    if intent == "inference":
        gen_overrides = dm.get("generation_config_overrides") or {}
        aibom["inference"] = {
            "serving_engine": annotations.get("serving-engine") or dm.get("serving_engine"),
            "max_model_len": _try_int(annotations.get("max-model-len")) or dm.get("max_model_len"),
            "tensor_parallel_size": _try_int(annotations.get("tensor-parallel-size")) or dm.get("tensor_parallel_size"),
            "pipeline_parallel_size": _try_int(annotations.get("pipeline-parallel-size")) or dm.get("pipeline_parallel_size"),
            "enable_expert_parallel": dm.get("enable_expert_parallel"),
            "data_parallel_size": _try_int(annotations.get("data-parallel-size")) or dm.get("data_parallel_size"),
            "gpu_memory_utilization": _try_float(annotations.get("gpu-memory-utilization")) or dm.get("gpu_memory_utilization"),
            "temperature": _try_float(annotations.get("temperature")) or gen_overrides.get("temperature"),
            "top_p": _try_float(annotations.get("top-p")) or gen_overrides.get("top_p"),
            "top_k": _try_int(annotations.get("top-k")) or gen_overrides.get("top_k"),
            "max_tokens": _try_int(annotations.get("max-tokens")),
        }

    # Environment from first discovery
    if discoveries:
        first = discoveries[0]
        gpu_info = first.get("gpu", {})
        system_info = first.get("system", {})

        fw_name = runtime_info.get("framework", annotations.get("model-framework"))
        fw_version = runtime_info.get("framework_version")
        fw_label = (
            f"{fw_name} {fw_version}"
            if fw_name and fw_version
            else (fw_version or fw_name)
        )

        gpu_models = gpu_info.get("gpu_models", "")
        if gpu_models and gpu_models.strip().lower() not in ("", "not available"):
            gpu_type = gpu_models.strip().split("\n")[0].strip()
        else:
            gpu_type = None
        mem_gb_str = system_info.get("memory_total_gb")
        mem_gb = round(float(mem_gb_str), 2) if mem_gb_str else None

        aibom["environment"] = {
            "gpu_type": gpu_type,
            "gpu_count": gpu_info.get("gpu_count"),
            "cpu_model": system_info.get("cpu_model"),
            "cpu_cores": system_info.get("cpu_count"),
            "memory_gb": mem_gb,
            "numa_nodes": system_info.get("numa_node_count"),
            "cuda_version": gpu_info.get("cuda_version"),
            "driver_version": gpu_info.get("gpu_driver_version"),
            "framework_version": fw_label,
            "kernel_version": safe_get(first, "system", "kernel_version"),
        }

    # Resource utilization from telemetry
    if telemetry and telemetry.get("pods"):
        all_aggregated = [p.get("aggregated", {}) for p in telemetry["pods"]]
        merged = {}
        for agg in all_aggregated:
            for key, info in agg.items():
                if isinstance(info, dict) and info.get("value") is not None:
                    merged.setdefault(key, []).append(info["value"])

        utilization = {"collected_at": telemetry.get("collected_at")}
        unit_map = {
            "avg_gpu_utilization": ("avg_gpu_utilization_pct", None),
            "avg_gpu_memory_used": ("avg_gpu_memory_used_mib", None),
            "avg_gpu_power": ("avg_gpu_power_watts", None),
            "avg_cpu_usage": ("avg_cpu_usage_cores", None),
            "avg_memory_usage": ("avg_memory_usage_gb", 1 / (1024**3)),
            "avg_network_receive": ("avg_network_receive_mbps", 8 / (1024 * 1024)),
            "avg_network_transmit": ("avg_network_transmit_mbps", 8 / (1024 * 1024)),
        }
        for metric_key, values in merged.items():
            if metric_key in unit_map:
                field_name, scale = unit_map[metric_key]
                avg = sum(values) / len(values)
                if scale:
                    avg *= scale
                utilization[field_name] = round(avg, 2)

        grafana_links = [
            {"pod_name": p["pod_name"], "explore_url": p["grafana_explore_url"]}
            for p in telemetry["pods"]
            if p.get("grafana_explore_url")
        ]
        if grafana_links:
            utilization["grafana_links"] = grafana_links

        # True if any pod's run was too short to exclude the cold-start
        # window, meaning the averages above may include a period of
        # stale/zero readings before the first scrape landed.
        utilization["summary_includes_cold_start"] = any(
            p.get("summary_includes_cold_start") for p in telemetry["pods"]
        )

        aibom["resource_utilization"] = utilization
    else:
        aibom["resource_utilization"] = {
            "note": "No telemetry data available.",
        }

    # Metadata
    aibom["_metadata"] = {
        "aibom_version": "0.1.0",
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": "aibom-webhook postprocess",
        "schema_compliance": "partial - focuses on reproducibility and telemetry fields",
        "dataset_detection": (
            "enabled" if detected_datasets else "no datasets detected"
        ),
    }

    return aibom


def _try_int(value):
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _try_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _first_not_none(*values):
    for v in values:
        if v is not None:
            return v
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    if not JOB_NAME:
        print("ERROR: AIBOM_JOB_NAME not set", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("AIBOM Post-Processing")
    print("=" * 60)
    print(f"Job: {JOB_NAMESPACE}/{JOB_NAME}")
    print(f"Input: {INPUT_DIR}")
    print()

    # Load input data
    print("--- Loading Input Data ---")
    discoveries = load_discovery()
    detected_datasets, runtime_info = load_datasets()
    annotations = load_annotations()
    containers = load_containers()
    storage = load_storage()
    print()

    # Model detection from container commands, then from a KServe
    # InferenceService's declared storage path/URI — the latter overrides
    # model_name since S3/MinIO-backed predictors have no CLI arg to parse it
    # from (see detect_model_from_storage).
    detected_model = detect_model_from_containers(containers)
    storage_model = detect_model_from_storage(storage)
    if storage_model:
        detected_model = {**(detected_model or {}), **storage_model}
    cli_dataset = detect_dataset_from_containers(containers)
    if detected_model:
        print(f"--- Model Detection ---")
        print(f"  Engine: {detected_model.get('serving_engine', 'unknown')}")
        if detected_model.get("model_name"):
            print(f"  Model: {detected_model['model_name']}")
        if detected_model.get("quantization_method"):
            print(f"  Quantization: {detected_model['quantization_method']} ({detected_model.get('quantization_bits', '?')}-bit)")
        if detected_model.get("speculative_config"):
            print(f"  Speculative decoding: {detected_model['speculative_config']}")
        if detected_model.get("generation_config_overrides"):
            print(f"  Generation config overrides: {detected_model['generation_config_overrides']}")
        print()

    # Telemetry
    telemetry = None
    if PROMETHEUS_URL:
        print("--- Phase 1: Telemetry Collection ---")
        try:
            telemetry = collect_telemetry(discoveries)
        except Exception as e:
            print(f"WARNING: Telemetry collection failed: {e}", file=sys.stderr)
        print()
    else:
        print("--- Phase 1: Skipped (no PROMETHEUS_URL) ---")
        print()

    # AIBOM compilation
    print("--- Phase 2: AIBOM Compilation ---")
    try:
        aibom = compile_aibom(
            discoveries, detected_datasets, runtime_info, annotations, telemetry,
            detected_model=detected_model, cli_dataset=cli_dataset,
        )
    except Exception as e:
        print(f"ERROR: AIBOM compilation failed: {e}", file=sys.stderr)
        sys.exit(1)
    print()

    # Output: create the AIBOM directly as a namespaced custom resource, rather
    # than printing to stdout for the watcher to scrape from pod logs.
    print("--- Phase 3: AIBOM Custom Resource Creation ---")
    aibom_cr = {
        "apiVersion": "aibom.io/v1alpha1",
        "kind": "AIBOM",
        "metadata": {
            "generateName": f"{JOB_NAME}-",
            "namespace": JOB_NAMESPACE,
            "labels": {"aibom.io/job-name": JOB_NAME},
        },
        "spec": {
            "jobName": JOB_NAME,
            "modelName": safe_get(aibom, "model", "name", default=""),
            "experimentIntent": aibom.get("experiment_intent") or "",
            "collectedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "data": aibom,
        },
    }
    try:
        created = k8s_api.create_custom_object(JOB_NAMESPACE, "aibom.io", "v1alpha1", "aiboms", aibom_cr)
    except Exception as e:
        print(f"ERROR: could not create AIBOM custom resource: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"  Created AIBOM/{JOB_NAMESPACE}/{created.get('metadata', {}).get('name', '?')}")
    print()

    # Summary
    print("--- Summary ---")
    print(f"  Experiment: {aibom.get('experiment_intent', 'unknown')}")
    print(f"  Job: {aibom.get('execution_metadata', {}).get('job_id', 'unknown')}")
    print(f"  Pods: {len(aibom.get('execution_metadata', {}).get('pods', []))}")
    env = aibom.get("environment", {})
    if env.get("gpu_type"):
        print(f"  GPU: {env['gpu_type']} x{env.get('gpu_count', '?')}")
    ds = aibom.get("dataset", {})
    if ds.get("auto_detected"):
        print(f"  Datasets detected: {len(ds['auto_detected'])}")
    util = aibom.get("resource_utilization", {})
    if util.get("avg_gpu_utilization_pct") is not None:
        print(f"  Avg GPU utilization: {util['avg_gpu_utilization_pct']}%")
    print()

    print("=" * 60)
    print("Post-processing complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
