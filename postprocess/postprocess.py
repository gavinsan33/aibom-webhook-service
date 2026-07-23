#!/usr/bin/env python3
"""AIBOM postprocess -- compile an AI Bill of Materials.

Runs as a Kubernetes Job after an instrumented workload completes.
Reads discovery and dataset data from a ConfigMap mount, optionally
queries Grafana for telemetry, and produces an AIBOM JSON document.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
import urllib.request
import urllib.parse
import urllib.error

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_DIR = os.environ.get("AIBOM_INPUT_DIR", "/data/input")
JOB_NAME = os.environ.get("AIBOM_JOB_NAME", "")
JOB_NAMESPACE = os.environ.get("AIBOM_JOB_NAMESPACE", "")
GRAFANA_URL = os.environ.get("GRAFANA_URL", "")
GRAFANA_API_TOKEN = os.environ.get("GRAFANA_API_TOKEN", "")
GRAFANA_DATASOURCE_UID = os.environ.get("GRAFANA_DATASOURCE_UID", "")

TELEMETRY_QUERIES = {
    "gpu_utilization": 'nerc:dcgm_gpu_util:avg5m{exported_pod="{pod_name}"}',
    "gpu_memory_used": 'nerc:dcgm_fb_used:avg5m{exported_pod="{pod_name}"}',
    "gpu_power": 'nerc:dcgm_power_usage:avg5m{exported_pod="{pod_name}"}',
    "cpu_usage": 'rate(container_cpu_usage_seconds_total{pod="{pod_name}", container!="POD", container!=""}[10m])',
    "memory_usage": 'container_memory_working_set_bytes{pod="{pod_name}", container!="POD", container!=""}',
    "network_receive": 'rate(container_network_receive_bytes_total{pod="{pod_name}"}[10m])',
    "network_transmit": 'rate(container_network_transmit_bytes_total{pod="{pod_name}"}[10m])',
}

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


# ---------------------------------------------------------------------------
# Phase 1: Telemetry collection
# ---------------------------------------------------------------------------


def discover_datasource_uid(grafana_url, api_token):
    try:
        req = urllib.request.Request(
            f"{grafana_url}/api/datasources",
            headers={"Authorization": f"Bearer {api_token}"},
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            datasources = json.loads(response.read())
            for ds in datasources:
                if ds.get("type") == "prometheus":
                    return ds.get("uid")
        print("WARNING: No Prometheus datasource found", file=sys.stderr)
        return None
    except Exception as e:
        print(f"WARNING: Could not auto-discover datasource: {e}", file=sys.stderr)
        return None


def query_grafana(grafana_url, api_token, datasource_uid, promql, start_ms, end_ms, instant=False):
    payload = {
        "queries": [
            {
                "datasource": {"uid": datasource_uid},
                "expr": promql,
                "refId": "A",
                "instant": instant,
                "range": not instant,
                "maxDataPoints": 1000,
            }
        ],
        "from": str(start_ms),
        "to": str(end_ms),
    }
    try:
        req = urllib.request.Request(
            f"{grafana_url}/api/ds/query",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {api_token}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code}: {e.read().decode()}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Query failed: {e}", file=sys.stderr)
        return None


def parse_grafana_response(response):
    if not response or "results" not in response:
        return []
    results = []
    for result_data in response["results"].values():
        for frame in result_data.get("frames", []):
            data_points = frame.get("data", {}).get("values", [])
            if len(data_points) >= 2:
                for ts, val in zip(data_points[0], data_points[1]):
                    results.append(
                        {
                            "timestamp": datetime.fromtimestamp(ts / 1000).isoformat(),
                            "value": val,
                        }
                    )
    return results


def parse_instant_value(response):
    if not response or "results" not in response:
        return None
    for result_data in response["results"].values():
        for frame in result_data.get("frames", []):
            data_points = frame.get("data", {}).get("values", [])
            if len(data_points) >= 2 and data_points[1]:
                return data_points[1][-1]
    return None


def ms_to_promql_duration(ms):
    seconds = max(int(ms / 1000), 60)
    if seconds >= 3600:
        return f"{seconds // 3600}h"
    return f"{seconds // 60}m"


def collect_telemetry(discoveries):
    grafana_url = GRAFANA_URL
    api_token = GRAFANA_API_TOKEN
    datasource_uid = GRAFANA_DATASOURCE_UID

    if not datasource_uid:
        print("  Auto-discovering Prometheus datasource...")
        datasource_uid = discover_datasource_uid(grafana_url, api_token)
        if not datasource_uid:
            print("ERROR: Could not find Prometheus datasource", file=sys.stderr)
            return None

    print(f"  Processing {len(discoveries)} pod(s)")

    telemetry_summary = {
        "collected_at": datetime.utcnow().isoformat() + "Z",
        "grafana_url": grafana_url,
        "datasource_uid": datasource_uid,
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
        if not gpu_count or str(gpu_count) == "0":
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

        for metric_name, promql in metrics.items():
            print(f"    Querying {metric_name}...")
            response = query_grafana(
                grafana_url, api_token, datasource_uid, promql, start_ms, end_ms
            )
            if response:
                data_points = parse_grafana_response(response)
                pod_telemetry["metrics"][metric_name] = {
                    "data_point_count": len(data_points),
                }
                print(f"      {len(data_points)} data points")
            else:
                print(f"      WARNING: No data returned")

        if SUMMARY_QUERIES:
            duration = ms_to_promql_duration(end_ms - start_ms)
            print(f"    Running summary queries (duration={duration})...")
            aggregated = {}
            for sq_name, sq_info in SUMMARY_QUERIES.items():
                promql = (
                    sq_info["query"]
                    .replace("{pod_name}", pod_name)
                    .replace("{duration}", duration)
                )
                response = query_grafana(
                    grafana_url, api_token, datasource_uid, promql,
                    start_ms, end_ms, instant=True,
                )
                value = parse_instant_value(response)
                if value is not None:
                    aggregated[sq_name] = {
                        "value": round(value, 2),
                        "unit": sq_info.get("unit", ""),
                    }
                    print(f"      {sq_name}: {round(value, 2)} {sq_info.get('unit', '')}")
                else:
                    print(f"      {sq_name}: no data")
            pod_telemetry["aggregated"] = aggregated

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


def compile_aibom(discoveries, detected_datasets, runtime_info, annotations, telemetry):
    print(f"  Discovery files: {len(discoveries)}")
    print(f"  Auto-detected datasets: {len(detected_datasets)}")
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

    # Model info from annotations
    model_name = annotations.get("model-name")
    aibom["model"] = {
        "name": model_name,
        "version": annotations.get("model-version"),
        "architecture": annotations.get("model-architecture"),
        "framework": annotations.get("model-framework"),
        "quantization": annotations.get("quantization"),
        "quantization_bits": _try_int(annotations.get("quantization-bits")),
        "dtype": annotations.get("dtype"),
    }

    # Dataset section
    declared_dataset = {
        "name": annotations.get("dataset-name"),
        "version": annotations.get("dataset-version"),
        "source": annotations.get("dataset-source"),
        "license": annotations.get("dataset-license"),
    }
    has_declared = any(declared_dataset.values())
    intent = aibom["experiment_intent"]

    if has_declared or detected_datasets or intent in ("training", "sft"):
        aibom["dataset"] = {"declared": declared_dataset}
        if detected_datasets:
            aibom["dataset"]["auto_detected"] = detected_datasets
            print(f"  Merged {len(detected_datasets)} auto-detected dataset(s)")
            if not declared_dataset.get("name") and detected_datasets:
                first = detected_datasets[0]
                aibom["dataset"]["declared"]["name"] = first.get("dataset_name")
                aibom["dataset"]["declared"]["source"] = first.get("source")
                if first.get("version"):
                    aibom["dataset"]["declared"]["version"] = first["version"]
                if first.get("license"):
                    aibom["dataset"]["declared"]["license"] = first["license"]

    # Training config
    if intent in ("training", "sft"):
        aibom["training"] = {
            "optimizer": annotations.get("optimizer"),
            "learning_rate": runtime_info.get(
                "learning_rate", _try_float(annotations.get("learning-rate"))
            ),
            "batch_size": runtime_info.get(
                "batch_size", _try_int(annotations.get("batch-size"))
            ),
            "epochs": runtime_info.get(
                "epochs", _try_int(annotations.get("epochs"))
            ),
            "random_seed": _try_int(annotations.get("random-seed")),
            "parallelization_strategy": annotations.get("parallelization-strategy"),
        }

    # Fine-tuning config
    if intent == "sft":
        aibom["fine_tuning"] = {
            "adaptation_method": annotations.get("adaptation-method"),
            "lora_rank": _try_int(annotations.get("lora-rank")),
            "lora_alpha": _try_int(annotations.get("lora-alpha")),
        }

    # Inference config
    if intent == "inference":
        aibom["inference"] = {
            "serving_engine": annotations.get("serving-engine"),
            "max_model_len": _try_int(annotations.get("max-model-len")),
            "tensor_parallel_size": _try_int(annotations.get("tensor-parallel-size")),
            "gpu_memory_utilization": _try_float(
                annotations.get("gpu-memory-utilization")
            ),
            "temperature": _try_float(annotations.get("temperature")),
            "top_p": _try_float(annotations.get("top-p")),
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
    print()

    # Telemetry
    telemetry = None
    if GRAFANA_API_TOKEN:
        print("--- Phase 1: Telemetry Collection ---")
        try:
            telemetry = collect_telemetry(discoveries)
        except Exception as e:
            print(f"WARNING: Telemetry collection failed: {e}", file=sys.stderr)
        print()
    else:
        print("--- Phase 1: Skipped (no GRAFANA_API_TOKEN) ---")
        print()

    # AIBOM compilation
    print("--- Phase 2: AIBOM Compilation ---")
    try:
        aibom = compile_aibom(
            discoveries, detected_datasets, runtime_info, annotations, telemetry
        )
    except Exception as e:
        print(f"ERROR: AIBOM compilation failed: {e}", file=sys.stderr)
        sys.exit(1)
    print()

    # Output
    aibom_json = json.dumps(aibom, indent=2)

    print("===AIBOM_RESULT_START===")
    print(aibom_json)
    print("===AIBOM_RESULT_END===")
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
