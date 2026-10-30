#!/usr/bin/env python3
"""AIBOM postprocess -- compile an AI Bill of Materials.

Runs as a Kubernetes Job after an instrumented workload completes.
Reads discovery and dataset data from a ConfigMap mount, optionally
queries Prometheus for telemetry, and produces an AIBOM JSON document.
"""

import base64
import hashlib
import json
import math
import os
import re
import secrets
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

# Path to the per-namespace Ed25519 private key (see charts/aibom-workload-namespace's
# templates/signing.yaml and CLAUDE.md's Compiled AIBOM Signing section) this Job signs
# the compiled AIBOM with before creating the custom resource. Empty/missing means the
# namespace's aibom-workload-namespace chart install predates this Secret -- the AIBOM
# is still created, just unsigned, mirroring how a missing discovery-signing key degrades
# to "unverifiable" rather than failing the workload (see generate_snapshot.py's
# sign_payload).
SIGNING_KEY_PATH = os.environ.get("AIBOM_SIGNING_KEY_PATH", "")

# The observability backend can lag behind real time before freshly-scraped
# samples become queryable, so a range query fired immediately after the
# workload's pod completes can race that ingestion delay and come back empty
# even though the same query succeeds moments later. Retry metrics with no
# data points with a delay rather than accepting the first empty result.
TELEMETRY_RETRY_ATTEMPTS = int(os.environ.get("AIBOM_TELEMETRY_RETRY_ATTEMPTS", "3"))
TELEMETRY_RETRY_DELAY_S = int(os.environ.get("AIBOM_TELEMETRY_RETRY_DELAY_S", "45"))

# Normally a pod with no detected GPU (gpu_count 0/missing, e.g. nvidia-smi found
# nothing) is skipped for telemetry entirely -- there's no GPU utilization to query.
# On a mock cluster (e.g. kind) with no real GPU hardware at all, that means every
# pod gets skipped and telemetry never gets exercised. Debug-only escape hatch to
# query telemetry for every pod regardless of detected GPU count.
DEBUG_TELEMETRY_ALL_PODS = os.environ.get("AIBOM_DEBUG_TELEMETRY_ALL_PODS", "").lower() == "true"

# Read image labels from the image's own registry when the cluster has no
# Image object for it (any non-OpenShift cluster, or an image OpenShift never
# imported). Anonymous pulls only; off with AIBOM_REGISTRY_LABEL_LOOKUP=false.
REGISTRY_LABEL_LOOKUP = os.environ.get("AIBOM_REGISTRY_LABEL_LOOKUP", "true").lower() != "false"

# Prometheus' scrape interval (OpenShift's default is 30s). The summary stats
# trim this much off the start of each pod's window as cold start, and the
# time-series step is floored at it (see SERIES_SCRAPE_INTERVAL_S below).
SCRAPE_INTERVAL_S = int(os.environ.get("AIBOM_SERIES_SCRAPE_INTERVAL_S", "30"))
SCRAPE_INTERVAL_MS = SCRAPE_INTERVAL_S * 1000

# The [5m] window every rate()/avg_over_time() query below uses. A point at
# time t summarizes (t-5m, t], so a pod's samples keep showing up in query
# results for up to this long after it stops -- which is why each pod's stats
# window ends at its own finish time, not at collection time.
RATE_WINDOW_MS = 5 * 60 * 1000

# Per-container cAdvisor series to count as the workload: not the pause
# container ("POD"), not the pod-level cgroup (""), and not this project's
# own injected aibom-dataset-sidecar. Shared by the stats and series queries.
_WORKLOAD_CONTAINERS = 'container!="POD", container!="", container!="aibom-dataset-sidecar"'

# Each query's raw range data points are kept (not just reduced to a single
# average) so stats -- min/max/p95 and a first/middle/last-third breakdown --
# can be derived from the same series a run's shape actually traced out,
# instead of needing a second `avg_over_time` query per metric. See
# compute_metric_stats() and CLAUDE.md's Telemetry Retries section.
#
# Every query aggregates to exactly one series per pod (sum, or avg for
# utilization), so compute_metric_stats sees one value per timestamp. Without
# that, a pod's per-container/per-GPU/per-interface series would be pooled and
# averaged together -- an 8 GiB app container next to a 50 MiB sidecar would
# report ~4 GiB of memory. The sums match SERIES_QUERIES' per-run `aggregate`
# lines. Placeholders: {namespace}, {pod_name}.
#
# Gauges (GPU util/mem/power, vLLM queue depth, KV-cache) are read raw, with
# NO avg_over_time wrapper: compute_metric_stats' max/min reduce these points,
# so a 5m trailing average flattened real bursts and made "Max GPU util" report
# e.g. 4% while the SERIES_QUERIES chart showed ~22%. Reading the gauge at each
# range step lets max/min reflect the true peak/trough the run actually hit.
# Rate queries (cpu/network/storage/vLLM throughput & latency) keep their [5m]
# window -- a rate needs a range vector.
# ponytail: step-sampled, so a spike strictly between two range steps is
# missed; switch max/min to max_over_time/min_over_time([step]) (as
# SERIES_QUERIES' {fn} does) if long-run peaks need to be exact.
TELEMETRY_QUERIES = {
    # Computed directly from dcgm-exporter's own raw metrics (present on any
    # standard DCGM install) rather than a "nerc:"-prefixed recording rule --
    # that rule is a PrometheusRule dependency specific to certain clusters
    # (e.g. NERC) and isn't present by default elsewhere, which silently
    # dropped GPU telemetry on any cluster that hadn't separately installed it.
    # DCGM's own `namespace`/`pod` labels are dcgm-exporter's; the workload's
    # are `exported_namespace`/`exported_pod`.
    "gpu_utilization": {
        "query": 'avg(DCGM_FI_DEV_GPU_UTIL{exported_namespace="{namespace}", exported_pod="{pod_name}"})',
        "unit": "percent",
    },
    "gpu_memory_used": {
        "query": 'sum(DCGM_FI_DEV_FB_USED{exported_namespace="{namespace}", exported_pod="{pod_name}"})',
        "unit": "MiB",
    },
    "gpu_power": {
        "query": 'sum(DCGM_FI_DEV_POWER_USAGE{exported_namespace="{namespace}", exported_pod="{pod_name}"})',
        "unit": "watts",
    },
    # rate()'s [5m] window matches what the pre-segmented-stats avg_* fields
    # used (a separate avg_over_time(rate(...[5m])[...]) summary query) --
    # keep it tight rather than widening it, since a wider window smooths out
    # exactly the mid-run detail compute_metric_stats' segments exist to show.
    "cpu_usage": {
        "query": (
            'sum(rate(container_cpu_usage_seconds_total{namespace="{namespace}", pod="{pod_name}", '
            + _WORKLOAD_CONTAINERS + '}[5m]))'
        ),
        "unit": "cores",
    },
    "memory_usage": {
        "query": (
            'sum(container_memory_working_set_bytes{namespace="{namespace}", pod="{pod_name}", '
            + _WORKLOAD_CONTAINERS + '})'
        ),
        "unit": "bytes",
    },
    # Network is only exposed at the pod level (one series per interface), so
    # there's no container filter here; sum across interfaces.
    "network_receive": {
        "query": 'sum(rate(container_network_receive_bytes_total{namespace="{namespace}", pod="{pod_name}"}[5m]))',
        "unit": "bytes_per_sec",
    },
    "network_transmit": {
        "query": 'sum(rate(container_network_transmit_bytes_total{namespace="{namespace}", pod="{pod_name}"}[5m]))',
        "unit": "bytes_per_sec",
    },
    # container_fs_* is labeled per-device, unlike network -- sum across
    # devices so this collapses to one series per pod like every other metric.
    # Depending on runtime/cgroup version, cAdvisor sometimes only exposes
    # these at the pod-level cgroup (container="") rather than per-container,
    # so the primary per-container sum falls back to the pod-level series via
    # `or` when the per-container one comes back empty for a pod -- the two
    # are never emitted simultaneously for the same pod, so this can't
    # double-count.
    "storage_read_throughput": {
        "query": (
            'sum by (pod) (rate(container_fs_reads_bytes_total{namespace="{namespace}", pod="{pod_name}", '
            + _WORKLOAD_CONTAINERS + '}[5m]))'
            ' or sum by (pod) (rate(container_fs_reads_bytes_total{namespace="{namespace}", pod="{pod_name}", container=""}[5m]))'
        ),
        "unit": "bytes_per_sec",
    },
    "storage_write_throughput": {
        "query": (
            'sum by (pod) (rate(container_fs_writes_bytes_total{namespace="{namespace}", pod="{pod_name}", '
            + _WORKLOAD_CONTAINERS + '}[5m]))'
            ' or sum by (pod) (rate(container_fs_writes_bytes_total{namespace="{namespace}", pod="{pod_name}", container=""}[5m]))'
        ),
        "unit": "bytes_per_sec",
    },
}

# A serving engine's own serving-level metrics (TTFT, ITL, queue depth,
# KV-cache usage, throughput) -- see CLAUDE.md's Inference Performance Telemetry
# section. Collected separately from TELEMETRY_QUERIES above and only for pods
# where detected_model.serving_engine is one we have queries for (main()); these
# are series the engine exposes on its own /metrics endpoint (not DCGM/cAdvisor)
# and land in aibom["inference"]["performance"] rather than resource_utilization
# -- a serving-level SLO ("were requests queueing, was the KV cache thrashing")
# is a different question than a hardware-utilization one.
#
# The engine's own metrics carry no pod/namespace label (vLLM emits just
# `engine`/`model_name`) -- the `namespace`/`pod` filters only work because the
# aibom-serving-metrics PodMonitor (charts/aibom-workload-namespace) scrapes the
# engine's port and Prometheus's target-discovery relabeling attaches those
# labels, the same mechanism the cAdvisor container_* queries above rely on.
#
# All four engines expose the same metric *shapes* -- TTFT/ITL histograms,
# running/waiting gauges, a KV-cache gauge, prompt/generation token counters --
# differing only in metric names, so the queries are generated from a per-engine
# name table (SERVING_ENGINE_METRICS) by _build_serving_queries rather than
# hand-written four times. Names confirmed against each engine's own docs
# (vLLM 0.17 live server; HF TGI metrics reference; SGLang production_metrics;
# LMDeploy metrics source), not guessed. TGI is the odd one: different names and
# no TTFT or KV-cache metric at all, so it reports a subset.
#
# TTFT/ITL are histograms; sum-rate-over-count-rate gives the average latency
# per time window, reduced by compute_metric_stats' min/max/avg/p95/segments
# like every gauge metric. That "p95" is honestly a p95 of the *windowed
# average* latency, not a true per-request percentile (which would need
# histogram_quantile() over the bucket series) -- a known simplification. A
# window with zero completed requests divides 0/0 (NaN); parse_range_response
# drops NaN samples rather than propagating them into min/max/avg.
#
# Where a metric was renamed across releases, both names are tried via `or` so
# older servers still report (vLLM: inter_token_latency_seconds was
# time_per_output_token_seconds, kv_cache_usage_perc was gpu_cache_usage_perc).
# Both sides aggregate to an empty label set, so `or` keeps only the new one.
#
# A KV-cache gauge is a 0-1 fraction despite its name; `scale: 100` multiplies
# it so the stored value matches its "percent" unit.
#
# kind: "hist" (sum/count rate ratio), "gauge" (instant value / _over_time),
# "counter" (rate of a _total). agg (gauge only): sum across series, or avg.
SERVING_ENGINE_METRICS = {
    "vllm": {
        "time_to_first_token_seconds": {"kind": "hist", "names": ["vllm:time_to_first_token_seconds"], "unit": "seconds"},
        "inter_token_latency_seconds": {"kind": "hist", "names": ["vllm:inter_token_latency_seconds", "vllm:time_per_output_token_seconds"], "unit": "seconds"},
        "num_requests_running": {"kind": "gauge", "names": ["vllm:num_requests_running"], "agg": "sum", "unit": "requests"},
        "num_requests_waiting": {"kind": "gauge", "names": ["vllm:num_requests_waiting"], "agg": "sum", "unit": "requests"},
        "kv_cache_usage": {"kind": "gauge", "names": ["vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"], "agg": "avg", "scale": 100, "unit": "percent"},
        "prompt_throughput": {"kind": "counter", "names": ["vllm:prompt_tokens_total"], "unit": "tokens_per_sec"},
        "generation_throughput": {"kind": "counter", "names": ["vllm:generation_tokens_total"], "unit": "tokens_per_sec"},
    },
    # LMDeploy mirrors vLLM's metric names under an lmdeploy: prefix.
    "lmdeploy": {
        "time_to_first_token_seconds": {"kind": "hist", "names": ["lmdeploy:time_to_first_token_seconds"], "unit": "seconds"},
        "inter_token_latency_seconds": {"kind": "hist", "names": ["lmdeploy:time_per_output_token_seconds"], "unit": "seconds"},
        "num_requests_running": {"kind": "gauge", "names": ["lmdeploy:num_requests_running"], "agg": "sum", "unit": "requests"},
        "num_requests_waiting": {"kind": "gauge", "names": ["lmdeploy:num_requests_waiting"], "agg": "sum", "unit": "requests"},
        "kv_cache_usage": {"kind": "gauge", "names": ["lmdeploy:gpu_cache_usage_perc"], "agg": "avg", "scale": 100, "unit": "percent"},
        "prompt_throughput": {"kind": "counter", "names": ["lmdeploy:prompt_tokens_total"], "unit": "tokens_per_sec"},
        "generation_throughput": {"kind": "counter", "names": ["lmdeploy:generation_tokens_total"], "unit": "tokens_per_sec"},
    },
    "sglang": {
        "time_to_first_token_seconds": {"kind": "hist", "names": ["sglang:time_to_first_token_seconds"], "unit": "seconds"},
        "inter_token_latency_seconds": {"kind": "hist", "names": ["sglang:request_time_per_output_token_seconds"], "unit": "seconds"},
        "num_requests_running": {"kind": "gauge", "names": ["sglang:num_running_reqs"], "agg": "sum", "unit": "requests"},
        "num_requests_waiting": {"kind": "gauge", "names": ["sglang:num_queue_reqs"], "agg": "sum", "unit": "requests"},
        # token_usage is a 0-1 fraction of the KV-cache pool in use.
        "kv_cache_usage": {"kind": "gauge", "names": ["sglang:token_usage"], "agg": "avg", "scale": 100, "unit": "percent"},
        "prompt_throughput": {"kind": "counter", "names": ["sglang:prompt_tokens_total"], "unit": "tokens_per_sec"},
        "generation_throughput": {"kind": "counter", "names": ["sglang:generation_tokens_total"], "unit": "tokens_per_sec"},
    },
    # TGI exposes no TTFT or KV-cache metric; it reports only what it has. It
    # times queueing and prefill separately (tgi_request_queue_duration per
    # request, tgi_batch_forward_duration per batch), but their sum isn't a
    # per-request TTFT, so none is synthesized. Its token counts are histograms
    # of tokens per request, so the _sum series is the cumulative token count
    # that vLLM's _total counters are. Names don't share vLLM's shape (no colon
    # prefix, different suffixes), which is exactly why the table carries names
    # per engine.
    "tgi": {
        "inter_token_latency_seconds": {"kind": "hist", "names": ["tgi_request_mean_time_per_token_duration"], "unit": "seconds"},
        "num_requests_running": {"kind": "gauge", "names": ["tgi_batch_current_size"], "agg": "sum", "unit": "requests"},
        "num_requests_waiting": {"kind": "gauge", "names": ["tgi_queue_size"], "agg": "sum", "unit": "requests"},
        "prompt_throughput": {"kind": "counter", "names": ["tgi_request_input_length_sum"], "unit": "tokens_per_sec"},
        "generation_throughput": {"kind": "counter", "names": ["tgi_request_generated_tokens_sum"], "unit": "tokens_per_sec"},
    },
}


def _build_serving_queries(metrics, series):
    """Generate the PromQL query dict for one engine's metric table, in either
    the per-pod stats shape (series=False) or the by-pod time-series shape
    (series=True). Reproduces the hand-verified vLLM queries exactly (see
    test_build_serving_queries_reproduces_vllm) so the generator inherits their
    confirmed correctness for every engine. {namespace}/{pod_name}/{pod_regex}/
    {fn}/{win} stay as literal placeholders for later substitution."""
    if series:
        fltr, by, sp = '{namespace="{namespace}", pod=~"{pod_regex}"}', " by (pod)", " "
    else:
        fltr, by, sp = '{namespace="{namespace}", pod="{pod_name}"}', "", ""
    sumfn = "sum" + by
    out = {}
    for key, d in metrics.items():
        kind, names, unit = d["kind"], d["names"], d["unit"]
        if kind == "hist":
            terms = [
                sumfn + sp + "(rate(" + n + "_sum" + fltr + "[5m]))"
                " / " + sumfn + sp + "(rate(" + n + "_count" + fltr + "[5m]))"
                for n in names
            ]
            query = terms[0] if len(terms) == 1 else " or ".join("(" + t + ")" for t in terms)
            info = {"query": query, "unit": unit}
            if series:
                info["aggregation"] = "avg"
        elif kind == "gauge":
            agg = d["agg"]
            if series:
                terms = ["{fn}" + by + sp + "({fn}_over_time(" + n + fltr + "[{win}s]))" for n in names]
            else:
                terms = [agg + "(" + n + fltr + ")" for n in names]
            query = " or ".join(terms)
            if len(terms) > 1:
                query = "(" + query + ")"
            if d.get("scale"):
                query = str(d["scale"]) + " * " + query
            info = {"query": query, "unit": unit}
            if series:
                info["aggregation"] = agg
                info["gauge"] = True
        else:  # counter
            query = sumfn + sp + "(rate(" + names[0] + fltr + "[5m]))"
            info = {"query": query, "unit": unit}
            if series:
                info["aggregation"] = "sum"
        out[key] = info
    return out


SERVING_TELEMETRY_QUERIES = {e: _build_serving_queries(m, series=False) for e, m in SERVING_ENGINE_METRICS.items()}
SERVING_SERIES_QUERIES = {e: _build_serving_queries(m, series=True) for e, m in SERVING_ENGINE_METRICS.items()}
# Back-compat aliases: vLLM was the only engine before multi-engine support.
VLLM_TELEMETRY_QUERIES = SERVING_TELEMETRY_QUERIES["vllm"]

# ---------------------------------------------------------------------------
# Persisted telemetry time series (see CLAUDE.md's Telemetry Time Series)
# ---------------------------------------------------------------------------
#
# Separate from TELEMETRY_QUERIES/VLLM_TELEMETRY_QUERIES on purpose: those feed
# the summary stats and must keep their exact semantics (cold-start-trimmed
# window, one query per pod, labels discarded). These are queried once per
# metric for the whole run (`pod=~"a|b|c"`, scoped to the workload namespace),
# over the full window, with series labels preserved.
#
# Placeholders: {namespace}, {pod_regex}, and for gauges {win} (the step, so
# buckets tile) and {fn} (avg, or max for the peak line). Raw base units are
# stored (bytes, bytes/s, cores...) -- the `unit` field says which -- rather
# than the display-scaled units resource_utilization uses.
#
# aibom-dataset-sidecar is excluded from per-container queries so the series
# describe the workload, not this project's own injected container (the same
# _WORKLOAD_CONTAINERS filter the stats queries use).
SERIES_QUERIES = {
    "gpu_utilization": {
        "query": (
            '{fn} by (exported_pod, gpu) ({fn}_over_time('
            'DCGM_FI_DEV_GPU_UTIL{exported_namespace="{namespace}", exported_pod=~"{pod_regex}"}[{win}s]))'
        ),
        "unit": "percent", "aggregation": "avg", "gauge": True,
    },
    "gpu_memory_used": {
        "query": (
            '{fn} by (exported_pod, gpu) ({fn}_over_time('
            'DCGM_FI_DEV_FB_USED{exported_namespace="{namespace}", exported_pod=~"{pod_regex}"}[{win}s]))'
        ),
        "unit": "MiB", "aggregation": "sum", "gauge": True,
    },
    "gpu_power": {
        "query": (
            '{fn} by (exported_pod, gpu) ({fn}_over_time('
            'DCGM_FI_DEV_POWER_USAGE{exported_namespace="{namespace}", exported_pod=~"{pod_regex}"}[{win}s]))'
        ),
        "unit": "watts", "aggregation": "sum", "gauge": True,
    },
    "cpu_usage": {
        "query": (
            'sum by (pod, container) (rate(container_cpu_usage_seconds_total{namespace="{namespace}", '
            'pod=~"{pod_regex}", ' + _WORKLOAD_CONTAINERS + '}[5m]))'
        ),
        "unit": "cores", "aggregation": "sum",
    },
    "memory_usage": {
        "query": (
            '{fn} by (pod, container) ({fn}_over_time(container_memory_working_set_bytes{namespace="{namespace}", '
            'pod=~"{pod_regex}", ' + _WORKLOAD_CONTAINERS + '}[{win}s]))'
        ),
        "unit": "bytes", "aggregation": "sum", "gauge": True,
    },
    "network_receive": {
        "query": (
            'sum by (pod, interface) (rate(container_network_receive_bytes_total{namespace="{namespace}", '
            'pod=~"{pod_regex}"}[5m]))'
        ),
        "unit": "bytes_per_sec", "aggregation": "sum",
    },
    "network_transmit": {
        "query": (
            'sum by (pod, interface) (rate(container_network_transmit_bytes_total{namespace="{namespace}", '
            'pod=~"{pod_regex}"}[5m]))'
        ),
        "unit": "bytes_per_sec", "aggregation": "sum",
    },
    # Same per-container-or-pod-level fallback as TELEMETRY_QUERIES.
    "storage_read_throughput": {
        "query": (
            'sum by (pod) (rate(container_fs_reads_bytes_total{namespace="{namespace}", pod=~"{pod_regex}", '
            + _WORKLOAD_CONTAINERS + '}[5m]))'
            ' or sum by (pod) (rate(container_fs_reads_bytes_total{namespace="{namespace}", '
            'pod=~"{pod_regex}", container=""}[5m]))'
        ),
        "unit": "bytes_per_sec", "aggregation": "sum",
    },
    "storage_write_throughput": {
        "query": (
            'sum by (pod) (rate(container_fs_writes_bytes_total{namespace="{namespace}", pod=~"{pod_regex}", '
            + _WORKLOAD_CONTAINERS + '}[5m]))'
            ' or sum by (pod) (rate(container_fs_writes_bytes_total{namespace="{namespace}", '
            'pod=~"{pod_regex}", container=""}[5m]))'
        ),
        "unit": "bytes_per_sec", "aggregation": "sum",
    },
}

VLLM_SERIES_QUERIES = SERVING_SERIES_QUERIES["vllm"]

SERIES_SCHEMA_VERSION = 1
SERIES_OBJECT_KIND = "AIBOMTelemetry"
SERIES_API_GROUP = "aibom.io"
SERIES_API_VERSION = "v1alpha1"
SERIES_PLURAL = "aibomtelemetries"
# Roughly the number of points kept per metric (the query step is derived from it).
SERIES_TARGET_POINTS = int(os.environ.get("AIBOM_SERIES_TARGET_POINTS", "200"))
# The step (and so the _over_time window of gauge queries, which equals it) is
# floored at the Prometheus scrape interval: a bucket narrower than that can
# contain no sample at all and come back empty.
SERIES_SCRAPE_INTERVAL_S = SCRAPE_INTERVAL_S
# Hard ceiling on the stored document. A custom resource shares etcd's ~1.5 MB
# object limit (and the CRD's spec.seriesJson maxLength); stay well under.
SERIES_MAX_BYTES = int(os.environ.get("AIBOM_SERIES_MAX_BYTES", "900000"))
# A metric with more per-pod/per-GPU series than this keeps only its aggregate line.
SERIES_MAX_SERIES_PER_METRIC = int(os.environ.get("AIBOM_SERIES_MAX_SERIES_PER_METRIC", "64"))
# When the document is over SERIES_MAX_BYTES, resolution is reduced first (see
# _fit_series_doc) but never below this many points per line; only then is
# per-pod/per-GPU detail dropped.
SERIES_MIN_POINTS = int(os.environ.get("AIBOM_SERIES_MIN_POINTS", "100"))

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
    # Data written by a pod predating the redaction in runtime_detector.py
    # may still carry credentials; compile_aibom logs runtime_info as-is (#102).
    if runtime_info.get("git_repository"):
        runtime_info["git_repository"] = redact_git_url(runtime_info["git_repository"])
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

# A name segment ends where the next character isn't alphanumeric. Not `\b`:
# underscore is a word character, so `\b` would reject `fp8_e4m3`.
_END = r"(?![A-Za-z0-9])"

# Matched case-insensitively, first match wins -- so the more specific
# patterns (GPTQ with its bit width, GGUF block types) precede the generic ones.
_QUANT_PATTERNS = [
    (r"GPTQ[_-]Int8", "gptq", 8),
    (r"GPTQ[_-]Int4", "gptq", 4),
    (r"gptq[_-]4bit", "gptq", 4),
    (r"GPTQ", "gptq", 4),
    (r"[_-]AWQ" + _END, "awq", 4),
    (r"AQLM[_-](\d+)Bit", "aqlm", None),
    (r"AQLM", "aqlm", 2),
    (r"EXL2", "exl2", None),
    (r"SqueezeLLM[_-](\d+)bit", "squeezellm", None),
    (r"SqueezeLLM", "squeezellm", 4),
    (r"HQQ[_-](\d+)bit", "hqq", None),
    (r"HQQ", "hqq", 4),
    (r"(?<![A-Za-z])QuIP", "quip", 2),
    (r"EETQ", "eetq", 8),
    (r"AutoRound", "autoround", 4),
    (r"[_-]NVFP4" + _END, "fp4", 4),
    (r"[_-]MXFP4" + _END, "fp4", 4),
    (r"[_-]FP4" + _END, "fp4", 4),
    (r"[_-]FP8" + _END, "fp8", 8),
    (r"bnb[_-]4bit", "bitsandbytes", 4),
    (r"bnb[_-]8bit", "bitsandbytes", 8),
    (r"[_-]NF4" + _END, "bitsandbytes", 4),
    (r"[_-]INT4" + _END, "int4", 4),
    (r"[_-]INT8" + _END, "int8", 8),
    (r"[_-]Marlin" + _END, "marlin", 4),
    # llm-compressor / compressed-tensors naming (e.g. RedHatAI's
    # `...-quantized.w4a16`): the bits are the weight bits.
    (r"[._-]W(\d+)A\d+" + _END, "compressed-tensors", None),
    # llama.cpp block types: Q4_K_M, Q8_0, IQ3_XS, ...
    (r"(?<![A-Za-z0-9])I?Q(\d)_(?:K(?:_[SML])?|[01]|XXS|XS|NL)" + _END, "gguf", None),
    (r"GGUF", "gguf", None),
    (r"GGML", "ggml", None),
]

_COMPILED_QUANT_PATTERNS = [
    (re.compile(p, re.IGNORECASE), method, bits) for p, method, bits in _QUANT_PATTERNS
]


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


# ---------------------------------------------------------------------------
# Command tokenizing and scoping
#
# Every detector below works on a flat token list, but a shell-wrapped
# command line is usually several commands (`pip install trl && trl sft ...`),
# so each detector first narrows to the tokens of the one invocation it cares
# about. Scanning the whole script instead matched "vllm" in `trl[vllm]`,
# read pip's `-q` as vLLM's `--quantization`, and took `pip install deepspeed`
# for a DeepSpeed launch.
# ---------------------------------------------------------------------------

_SHELLS = {"sh", "bash", "zsh", "dash", "ash"}
_SEPARATOR_CHARS = frozenset(";&|\n")
_UNEXPANDED_VAR_RE = re.compile(r"\$[{(A-Za-z_]")
_NUMBER_RE = re.compile(r"^-\d+(\.\d+)?$")
_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_COMMAND_PREFIXES = {"exec", "nohup", "env", "time", "sudo", "command"}
_PACKAGE_MANAGERS = {
    "pip", "pip3", "conda", "mamba", "micromamba", "apt", "apt-get", "apk",
    "yum", "dnf", "poetry",
}
_BOOL_WORDS = {"true", "false", "1", "0", "yes", "no"}


def _has_unexpanded_var(value):
    """True for a `$VAR`/`${VAR}`/`$(VAR)` that was never expanded -- the
    literal text isn't a model name or a number."""
    return bool(_UNEXPANDED_VAR_RE.search(value))


def _is_flag_token(tok):
    return tok.startswith("-") and tok != "-" and not _NUMBER_RE.match(tok)


def _split_flag(tok, sep):
    """Split `--key=value` into (key, value-or-None), writing the key's
    internal `-`/`_` as `sep` so `--max-model-len` and `--max_model_len` look
    up the same entry. Short flags (`-tp`) are left as they are."""
    key, eq, val = tok.partition("=")
    if key.startswith("--"):
        key = "--" + key[2:].replace("-", "_").replace("_", sep)
    return key, (val if eq else None)


def _finite_float(val):
    """float() without `nan`/`inf`, which would make the AIBOM invalid JSON
    and unsignable under JCS."""
    f = float(val)
    if not math.isfinite(f):
        raise ValueError(val)
    return f


def _int_or_float(val):
    """`3` -> 3, `0.5` -> 0.5 (e.g. --num_train_epochs)."""
    f = _finite_float(val)
    return int(f) if f.is_integer() else f


_HUMAN_INT_RE = re.compile(r"^(\d+(?:\.\d+)?)([kmgtKMGT]?)$")


def _human_readable_int(val):
    """vLLM's `--max-model-len` syntax: `4096`, `32k` (x1000), `32K` (x1024).
    Like vLLM, a fraction is only allowed with a decimal suffix (`25.6k`)."""
    m = _HUMAN_INT_RE.match(val.strip())
    if not m:
        raise ValueError(val)
    num, suffix = m.groups()
    if not suffix:
        return int(num)
    if suffix.isupper():
        return int(num) * 1024 ** ("KMGT".index(suffix) + 1)
    return int(float(num) * 1000 ** ("kmgt".index(suffix) + 1))


def _to_bool(val):
    return val.strip().lower() in ("1", "true", "yes")


def _parse_flags(args, arg_map, sep, bool_values=False):
    """Read the flags in `arg_map` out of `args` into ({name: value}, first
    positional). `--no<sep>flag` negates a bool flag; with `bool_values`, a
    bool flag also takes a following `true`/`false`. The positional is only
    recognized while every flag so far was one we know the arity of."""
    bool_flags = {k for k, (_, t) in arg_map.items() if t is bool}
    result = {}
    positional = None
    expect_positional = True
    i = 0
    while i < len(args):
        tok = args[i]
        i += 1
        if not _is_flag_token(tok):
            if expect_positional:
                positional = tok
            expect_positional = False
            continue

        key, val = _split_flag(tok, sep)
        negated = key.startswith("--no" + sep) and "--" + key[5:] in bool_flags
        if negated:
            key = "--" + key[5:]
        spec = arg_map.get(key)
        if spec is None:
            expect_positional = False
            continue

        name, conv = spec
        if conv is bool:
            if bool_values and val is None and i < len(args) and args[i].lower() in _BOOL_WORDS:
                val = args[i]
                i += 1
            result[name] = not negated and (val is None or _to_bool(val))
            continue
        if val is None and i < len(args) and not _is_flag_token(args[i]):
            val = args[i]
            i += 1
        if not val or _has_unexpanded_var(val):
            continue
        try:
            converted = conv(val)
        except (ValueError, TypeError, OverflowError):
            continue  # `--max-model-len abc` is a typo, not a value to record
        if converted is not None:
            result[name] = converted
    return result, positional


def _shell_script(command):
    """The script text of `sh -c "..."`, or None if `command` isn't a shell
    wrapper. Accepts any flag cluster containing `c` (`-c`, `-lc`, `-euc`,
    `-xc`) after other flags (`bash -l -c`, `bash -o pipefail -c`)."""
    if len(command) < 3 or os.path.basename(command[0]) not in _SHELLS:
        return None
    i = 1
    while i < len(command) - 1:
        tok = command[i]
        if tok in ("-o", "+o"):
            i += 2
            continue
        if not tok.startswith(("-", "+")):
            return None  # a script file, not an inline script
        if not tok.startswith("--") and "c" in tok[1:]:
            return " ".join(command[i + 1:])
        i += 1
    return None


def _flatten_container_command(container):
    """Expand `sh -c "..."`/`bash -c "..."` wrappers into a flat token list.

    Jobs that need to `pip install` before running a training CLI (e.g. trl)
    wrap everything in a single shell string, which would otherwise hide the
    CLI flags from the per-token detectors below. Command separators
    (`&&`, `||`, `;`, `|`, `&`, and unquoted newlines) come out as their own
    tokens; `_split_segments` cuts on them.
    """
    command = (container.get("command") or []) + (container.get("args") or [])
    script = _shell_script(command)
    if script is None:
        return command
    script = _drop_comment_lines_and_heredocs(script)
    # Join shell line-continuations (`\` immediately followed by a
    # newline) before tokenizing -- shlex doesn't do this on its own,
    # and without it a backslash-newline survives as a spurious literal
    # token that can land right after a bare boolean flag (e.g.
    # `--use_peft \<newline>--lora_r`) and get misread as its value.
    script = re.sub(r"\\\n", " ", script)
    try:
        # An unquoted newline separates commands, as in a shell; a quoted
        # one stays inside its token.
        lex = shlex.shlex(script, posix=True, punctuation_chars=";&|\n")
        lex.whitespace = " \t\r"
        lex.whitespace_split = True
        lex.commenters = ""
        return _strip_comments(list(lex))
    except ValueError:
        return command


_HEREDOC_RE = re.compile(r"(?<!<)<<(?!<)-?\s*(['\"]?)(\w+)\1")


def _drop_comment_lines_and_heredocs(script):
    """Remove whole-line comments and heredoc bodies before tokenizing. Both
    are literal text to the shell, but an apostrophe in them (`# it's`)
    leaves shlex with an unbalanced quote and nothing gets tokenized."""
    out = []
    terminator = None
    for line in script.split("\n"):
        if terminator is not None:
            if line.strip() == terminator:
                terminator = None
            continue
        if line.lstrip().startswith("#"):
            continue
        out.append(line)
        m = _HEREDOC_RE.search(line)
        if m:
            terminator = m.group(2)
    return "\n".join(out)


def _strip_comments(tokens):
    """Drop `# ...` up to the next newline. shlex's own `commenters` can't be
    used: it swallows the newline (merging the lines on either side into one
    command) and cuts `a/b#c` mid-word. A *quoted* argument starting with `#`
    can't be told apart here and is dropped too."""
    out = []
    in_comment = False
    for tok in tokens:
        if "\n" in tok and set(tok) <= _SEPARATOR_CHARS:
            in_comment = False
        elif in_comment or tok.startswith("#"):
            in_comment = True
            continue
        out.append(tok)
    return out


def _split_segments(tokens):
    """Cut a flat token list into one list per command."""
    segments = []
    current = []
    for tok in tokens:
        if tok and set(tok) <= _SEPARATOR_CHARS:
            if current:
                segments.append(current)
                current = []
        else:
            current.append(tok)
    if current:
        segments.append(current)
    return segments


def _command_start(segment):
    """Index of the command word, past `FOO=bar` assignments and wrappers
    like `exec`/`nohup`/`env`."""
    i = 0
    while i < len(segment) and (
        _ASSIGNMENT_RE.match(segment[i]) or segment[i] in _COMMAND_PREFIXES
    ):
        i += 1
    return i


def _is_package_install(segment):
    start = _command_start(segment)
    if start >= len(segment):
        return False
    word = os.path.basename(segment[start])
    rest = segment[start + 1:start + 3]
    if word in _PACKAGE_MANAGERS:
        return True
    if word == "uv":
        return bool(rest) and rest[0] in ("pip", "add", "sync")
    return word.startswith("python") and rest[:2] == ["-m", "pip"]


def _invocation_segments(tokens):
    """Segments that might run a workload: not package installs."""
    for seg in _split_segments(tokens or []):
        if not _is_package_install(seg):
            yield seg


# ---------------------------------------------------------------------------
# vLLM
# ---------------------------------------------------------------------------

_VLLM_ARG_MAP = {
    "--model": ("model_name", str),
    "--served-model-name": ("served_model_name", str),
    "--quantization": ("quantization", str),
    "-q": ("quantization", str),
    "--dtype": ("dtype", str),
    "--max-model-len": ("max_model_len", _human_readable_int),
    "--tensor-parallel-size": ("tensor_parallel_size", int),
    "-tp": ("tensor_parallel_size", int),
    "--pipeline-parallel-size": ("pipeline_parallel_size", int),
    "-pp": ("pipeline_parallel_size", int),
    "--enable-expert-parallel": ("enable_expert_parallel", bool),
    "--data-parallel-size": ("data_parallel_size", int),
    "-dp": ("data_parallel_size", int),
    "--gpu-memory-utilization": ("gpu_memory_utilization", _finite_float),
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

def _vllm_server_args(tokens):
    """The arguments of the first vLLM *server* invocation (`vllm serve ...`
    or `python -m vllm.entrypoints...api_server`), or None. `vllm bench ...`
    and friends are clients, and a bare `vllm` inside `pip install vllm` or
    `trl[vllm]` isn't an invocation at all."""
    for seg in _invocation_segments(tokens):
        for i, tok in enumerate(seg):
            if os.path.basename(tok) == "vllm" and seg[i + 1:i + 2] == ["serve"]:
                return seg[i + 2:]
            if tok.startswith("vllm.entrypoints.") and "api_server" in tok:
                return seg[i + 1:]
    return None


def _parse_vllm_args(args):
    # `vllm serve <model>`: the model is the first positional.
    parsed, positional = _parse_flags(args, _VLLM_ARG_MAP, "-")
    result = {"serving_engine": "vllm", **parsed}

    if "model_name" not in result and positional and not _has_unexpanded_var(positional):
        result["model_name"] = positional

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

    return result


def detect_vllm_from_command(command):
    args = _vllm_server_args(command)
    return None if args is None else _parse_vllm_args(args)


# llm-d's xpu/cpu/rocm model-server images set vLLM's api_server module as
# their ENTRYPOINT, so an args-only pod on one of them is vLLM too. Its CUDA
# and AWS images don't (their ENTRYPOINT is NVIDIA's generic wrapper), so pods
# on those spell out `vllm serve` and are matched by command instead. Only these
# names match: the same org also publishes the routing sidecar
# (llm-d-router-disagg-sidecar), the scheduler/endpoint picker, the inference
# simulator and an SGLang image (llm-d-xpu-sglang), and none of them is vLLM
# (#165).
_VLLM_IMAGE_RE = re.compile(
    r"(?:^|/)vllm-openai$|(?:^|/)vllm/[^/]+$"
    r"|(?:^|/)llm-d-(?:xpu|cpu|rocm)(?:-dev)?$"
)


def _strip_tag_digest(ref):
    """Image reference without its :tag and @digest (a registry :port stays)."""
    head, sep, last = ref.split("@", 1)[0].rpartition("/")
    return head + sep + last.split(":", 1)[0]


def _is_vllm_image(image):
    if not image:
        return False
    return bool(_VLLM_IMAGE_RE.search(_strip_tag_digest(image)))


def _detect_vllm_from_entrypoint_args(container):
    """The vllm-openai image sets `vllm serve` / the api_server module as its
    ENTRYPOINT, so a spec that only sets `args` (`["--model", "x", ...]`) has
    no "vllm" token anywhere. Only applies when `command` is unset, i.e. the
    image's own ENTRYPOINT runs."""
    if container.get("command") or not _is_vllm_image(container.get("image")):
        return None
    args = list(container.get("args") or [])
    if args[:1] == ["serve"]:
        args = args[1:]
    return _parse_vllm_args(args) if args else None


# ---------------------------------------------------------------------------
# Other serving engines (TGI, SGLang, LMDeploy)
#
# Same shape as the vLLM detector: find the server invocation in the (possibly
# shell-wrapped) command, parse the flags that map onto AIBOM fields, and emit
# `serving_engine` + `model_name` (+ a few shared flags). The output keys match
# vLLM's (model_name/dtype/quantization/tensor_parallel_size/...) so the model
# and inference sections and experiment_intent inference pick them up unchanged.
# Each flag map is intentionally small -- the model name is what matters most
# (it drives model.name, quantization-from-name, and intent); engine-specific
# tuning flags beyond the shared ones aren't exhaustively mapped.
# ---------------------------------------------------------------------------

def _parse_server_args(args, engine, arg_map, positional_model=False):
    parsed, positional = _parse_flags(args, arg_map, "-")
    result = {"serving_engine": engine, **parsed}
    if positional_model and "model_name" not in result and positional and not _has_unexpanded_var(positional):
        result["model_name"] = positional
    if "quantization" not in result and "model_name" in result:
        quant = detect_quantization_from_name(result["model_name"])
        if quant:
            result.update(quant)
    return result


# TGI: `text-generation-launcher --model-id <m> ...` (the official image's own
# ENTRYPOINT, so KServe-style args-only specs are common -- handled below).
_TGI_ARG_MAP = {
    "--model-id": ("model_name", str),
    "--quantize": ("quantization", str),
    "--dtype": ("dtype", str),
    "--num-shard": ("tensor_parallel_size", int),
    "--max-total-tokens": ("max_model_len", int),
}


def _tgi_server_args(tokens):
    for seg in _invocation_segments(tokens):
        for i, tok in enumerate(seg):
            if os.path.basename(tok) == "text-generation-launcher":
                return seg[i + 1:]
    return None


def detect_tgi_from_command(command):
    args = _tgi_server_args(command)
    return None if args is None else _parse_server_args(args, "tgi", _TGI_ARG_MAP)


# SGLang: `python -m sglang.launch_server --model-path <m> ...`.
_SGLANG_ARG_MAP = {
    "--model-path": ("model_name", str),
    "--served-model-name": ("served_model_name", str),
    "--dtype": ("dtype", str),
    "--quantization": ("quantization", str),
    "--tp-size": ("tensor_parallel_size", int),
    "--tp": ("tensor_parallel_size", int),
    "--dp-size": ("data_parallel_size", int),
    "--dp": ("data_parallel_size", int),
    "--context-length": ("max_model_len", int),
    # SGLang's closest analogue to vLLM's --gpu-memory-utilization: the fraction
    # of GPU memory reserved for the static (KV-cache) pool.
    "--mem-fraction-static": ("gpu_memory_utilization", _finite_float),
}


def _sglang_server_args(tokens):
    for seg in _invocation_segments(tokens):
        for i, tok in enumerate(seg):
            if i > 0 and seg[i - 1] == "-m" and (tok == "sglang.launch_server" or tok.startswith("sglang.launch_server")):
                return seg[i + 1:]
    return None


def detect_sglang_from_command(command):
    args = _sglang_server_args(command)
    return None if args is None else _parse_server_args(args, "sglang", _SGLANG_ARG_MAP)


# LMDeploy: `lmdeploy serve api_server <model> ...` (model is positional).
_LMDEPLOY_ARG_MAP = {
    "--model-name": ("served_model_name", str),
    "--model-format": ("quantization", str),  # awq/gptq -- its weight-quant flag
    "--dtype": ("dtype", str),
    "--tp": ("tensor_parallel_size", int),
    "--session-len": ("max_model_len", int),
    # Closest analogue to vLLM's --gpu-memory-utilization: fraction of free GPU
    # memory given to the KV cache.
    "--cache-max-entry-count": ("gpu_memory_utilization", _finite_float),
}


def _lmdeploy_server_args(tokens):
    for seg in _invocation_segments(tokens):
        for i, tok in enumerate(seg):
            is_bin = os.path.basename(tok) == "lmdeploy"
            is_mod = i > 0 and seg[i - 1] == "-m" and tok == "lmdeploy"
            if (is_bin or is_mod) and seg[i + 1:i + 3] == ["serve", "api_server"]:
                return seg[i + 3:]
    return None


def detect_lmdeploy_from_command(command):
    args = _lmdeploy_server_args(command)
    return None if args is None else _parse_server_args(args, "lmdeploy", _LMDEPLOY_ARG_MAP, positional_model=True)


# TGI's image sets text-generation-launcher as ENTRYPOINT, so a KServe-style
# predictor often sets only `args`, with no launcher token to match on -- same
# args-only case _detect_vllm_from_entrypoint_args handles for vLLM. (SGLang and
# LMDeploy images expect an explicit command, so they have no entrypoint-only
# fallback.)
_TGI_IMAGE_RE = re.compile(r"(?:^|/)text-generation-inference$|(?:^|/)tgi$")


def _detect_tgi_from_entrypoint_args(container):
    if container.get("command") or not container.get("image"):
        return None
    if not _TGI_IMAGE_RE.search(_strip_tag_digest(container["image"])):
        return None
    args = list(container.get("args") or [])
    return _parse_server_args(args, "tgi", _TGI_ARG_MAP) if args else None


# ---------------------------------------------------------------------------
# trl
# ---------------------------------------------------------------------------

# Hyphenated spellings are normalized to these underscore forms first.
_TRL_ARG_MAP = {
    "--model_name_or_path": ("model_name", str),
    "--use_peft": ("use_peft", bool),
    "--lora_r": ("lora_rank", int),
    "--lora_alpha": ("lora_alpha", int),
    "--use_dora": ("use_dora", bool),
    "--use_rslora": ("use_rslora", bool),
    "--load_in_4bit": ("load_in_4bit", bool),
    "--load_in_8bit": ("load_in_8bit", bool),
    "--learning_rate": ("learning_rate", _finite_float),
    "--per_device_train_batch_size": ("batch_size", int),
    "--num_train_epochs": ("epochs", _int_or_float),
    "--seed": ("random_seed", int),
}

_TRL_SUBCOMMAND_RE = re.compile(r"^[a-z][a-z_]*$")


def _trl_args(tokens):
    """The arguments of the first `trl <subcommand>` / `python -m trl...`
    invocation, or None."""
    for seg in _invocation_segments(tokens):
        for i, tok in enumerate(seg):
            if (
                os.path.basename(tok) == "trl"
                and i + 1 < len(seg)
                and _TRL_SUBCOMMAND_RE.match(seg[i + 1])
            ):
                return seg[i + 2:]
            if i > 0 and seg[i - 1] == "-m" and (tok == "trl" or tok.startswith("trl.")):
                return seg[i + 1:]
    return None


def detect_trl_from_command(command):
    """Detect model/LoRA config from a `trl sft`/`trl dpo`-style CLI invocation."""
    args = _trl_args(command)
    if args is None:
        return None

    # `--use_peft`, `--use_peft true` and `--use_peft=false` all occur.
    parsed, _ = _parse_flags(args, _TRL_ARG_MAP, "_", bool_values=True)
    result = {"training_framework": "trl", **parsed}

    use_dora = result.pop("use_dora", False)
    use_rslora = result.pop("use_rslora", False)
    load_in_4bit = result.pop("load_in_4bit", False)
    load_in_8bit = result.pop("load_in_8bit", False)
    quantized = load_in_4bit or load_in_8bit

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

    return result


# ---------------------------------------------------------------------------
# Axolotl (fine-tuning)
#
# `axolotl train <config.yml> [overrides]` (modern Click CLI, kebab-case
# overrides), or the legacy `python -m axolotl.cli.train <config>` /
# `accelerate launch -m axolotl.cli.train` (snake_case). The real config lives
# in the YAML, which runs in the training container postprocess can't read, so
# -- exactly like `trl --config` -- only values passed as CLI overrides are
# seen; the detector still reports `training_framework` (and so an intent) when
# none are. `_split_flag` normalizes kebab and snake to one spelling, so both
# CLI styles map through the same (snake-keyed) table. Only `train` is a
# training run (not preprocess/inference/merge-lora/...).
# ---------------------------------------------------------------------------

_AXOLOTL_ARG_MAP = {
    "--base_model": ("model_name", str),
    "--learning_rate": ("learning_rate", _finite_float),
    "--micro_batch_size": ("batch_size", int),
    "--num_epochs": ("epochs", _int_or_float),
    "--lora_r": ("lora_rank", int),
    "--lora_alpha": ("lora_alpha", int),
    "--adapter": ("adapter", str),
    "--load_in_4bit": ("load_in_4bit", bool),
    "--load_in_8bit": ("load_in_8bit", bool),
    "--seed": ("random_seed", int),
}


def _axolotl_args(tokens):
    for seg in _invocation_segments(tokens):
        for i, tok in enumerate(seg):
            if os.path.basename(tok) == "axolotl" and seg[i + 1:i + 2] == ["train"]:
                return seg[i + 2:]
            if i > 0 and seg[i - 1] == "-m" and tok == "axolotl.cli.train":
                return seg[i + 1:]
    return None


def detect_axolotl_from_command(command):
    """Detect model/LoRA config from an `axolotl train`-style CLI invocation."""
    args = _axolotl_args(command)
    if args is None:
        return None
    parsed, _ = _parse_flags(args, _AXOLOTL_ARG_MAP, "_", bool_values=True)
    result = {"training_framework": "axolotl", **parsed}

    adapter = (result.pop("adapter", "") or "").lower()
    quantized = result.pop("load_in_4bit", False) or result.pop("load_in_8bit", False)
    if adapter in ("lora", "qlora", "relora"):
        # axolotl's own QLoRA is `adapter: qlora`; a bare `lora` + 4/8-bit load
        # is QLoRA in effect too.
        result["adaptation_method"] = "qlora" if (adapter == "lora" and quantized) else adapter

    return result


# ---------------------------------------------------------------------------
# LLaMA-Factory (fine-tuning)
#
# `llamafactory-cli train <config.yaml>` or `llamafactory-cli train --flag ...`.
# It's built on transformers, so its flag names mirror trl's HF-style set. A
# config-file run exposes nothing on the CLI (not read, same as trl/axolotl); a
# flag-style run is parsed here. `--finetuning_type` selects lora/full/freeze;
# only `lora` carries an adaptation method (QLoRA when `--quantization_bit` is
# set, DoRA/rsLoRA with their flags).
# ---------------------------------------------------------------------------

_LLAMAFACTORY_ARG_MAP = {
    "--model_name_or_path": ("model_name", str),
    "--finetuning_type": ("finetuning_type", str),
    "--lora_rank": ("lora_rank", int),
    "--lora_alpha": ("lora_alpha", int),
    "--use_dora": ("use_dora", bool),
    "--use_rslora": ("use_rslora", bool),
    "--quantization_bit": ("quantization_bit", int),
    "--learning_rate": ("learning_rate", _finite_float),
    "--per_device_train_batch_size": ("batch_size", int),
    "--num_train_epochs": ("epochs", _int_or_float),
    "--seed": ("random_seed", int),
}


def _llamafactory_args(tokens):
    for seg in _invocation_segments(tokens):
        for i, tok in enumerate(seg):
            if os.path.basename(tok) == "llamafactory-cli" and seg[i + 1:i + 2] == ["train"]:
                return seg[i + 2:]
    return None


def detect_llamafactory_from_command(command):
    """Detect model/LoRA config from a `llamafactory-cli train`-style CLI."""
    args = _llamafactory_args(command)
    if args is None:
        return None
    parsed, _ = _parse_flags(args, _LLAMAFACTORY_ARG_MAP, "_", bool_values=True)
    result = {"training_framework": "llama-factory", **parsed}

    ft = (result.pop("finetuning_type", "") or "").lower()
    use_dora = result.pop("use_dora", False)
    use_rslora = result.pop("use_rslora", False)
    quantized = bool(result.pop("quantization_bit", None))
    if ft == "lora":
        if use_dora:
            result["adaptation_method"] = "dora"
        elif use_rslora:
            result["adaptation_method"] = "rslora"
        elif quantized:
            result["adaptation_method"] = "qlora"
        else:
            result["adaptation_method"] = "lora"

    return result


# ---------------------------------------------------------------------------
# Parallelization
# ---------------------------------------------------------------------------

_LAUNCHERS = {
    "accelerate": "accelerate",
    "deepspeed": "deepspeed",
    "torchrun": "torchrun",
    "mpirun": "mpirun",
    "mpiexec": "mpirun",
}
_LAUNCHER_MODULES = {
    "torch.distributed.run": "torchrun",
    "torch.distributed.launch": "torchrun",
    "accelerate.commands.launch": "accelerate",
}


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


def _launcher_of(segment):
    """The distributed launcher this command runs (`torchrun ...`,
    `python -m torch.distributed.run ...`), or None. Only the command word
    counts: `pip install deepspeed` doesn't launch DeepSpeed."""
    start = _command_start(segment)
    if start >= len(segment):
        return None
    word = os.path.basename(segment[start])
    if word.startswith("python"):
        for i, tok in enumerate(segment[start + 1:], start + 1):
            if tok == "-m" and i + 1 < len(segment):
                return _LAUNCHER_MODULES.get(segment[i + 1])
    return _LAUNCHERS.get(word)


def _is_single_process_launch(launcher, tokens):
    """`torchrun --nproc_per_node=1` / `mpirun -np 1`: a launcher wrapping one
    process isn't data parallelism."""
    if launcher == "torchrun":
        nproc = _find_flag_value(tokens, ("--nproc_per_node",))
        nnodes = _find_flag_value(tokens, ("--nnodes",)) or "1"
        return nproc == "1" and nnodes in ("1", "1:1")
    if launcher == "mpirun":
        return _find_flag_value(tokens, ("-np", "-n", "--np")) == "1"
    return False


def _underscore_flag(tok):
    """`--num-processes=4` -> `--num_processes=4`; the value is untouched."""
    key, val = _split_flag(tok, "_")
    return key if val is None else key + "=" + val


def _segment_parallelization(segment):
    tokens = [_underscore_flag(t) for t in segment]
    launcher = _launcher_of(segment)
    has_fsdp = any(t in ("--fsdp", "--use_fsdp") or t.startswith("--fsdp=") for t in tokens)
    has_deepspeed_flag = any(
        t in ("--deepspeed", "--use_deepspeed")
        or t.startswith(("--deepspeed=", "--deepspeed_config_file"))
        for t in tokens
    )
    has_multi_gpu = "--multi_gpu" in tokens
    num_processes = _try_int(_find_flag_value(tokens, ("--num_processes",)))
    config_flags = ("--accelerate_config", "--config_file") if launcher == "accelerate" else (
        "--accelerate_config",
    )
    accelerate_config = _find_flag_value(tokens, config_flags)
    accelerate_config_name = (
        os.path.splitext(os.path.basename(accelerate_config))[0] if accelerate_config else None
    )

    if has_fsdp:
        return "fsdp"
    if has_deepspeed_flag or launcher == "deepspeed":
        return "deepspeed"
    if accelerate_config_name in _ACCELERATE_CONFIG_STRATEGIES:
        return _ACCELERATE_CONFIG_STRATEGIES[accelerate_config_name]
    if has_multi_gpu:
        return "data_parallel"
    if launcher in ("torchrun", "mpirun") and not _is_single_process_launch(launcher, tokens):
        return "data_parallel"
    if num_processes and num_processes > 1:
        return "data_parallel"
    return None


def detect_parallelization_from_command(tokens):
    """Best-effort detection of a distributed-training parallelization
    strategy, independent of which training tool (trl, a custom script, ...)
    is being launched. Covers three shapes:
      - an explicit launcher binary (accelerate/deepspeed/torchrun/mpirun) as
        the command word of its own command
      - a bare --fsdp/--deepspeed flag on the training command itself
      - accelerate-launch args (--num_processes, --accelerate_config)
        passed straight through to a CLI like `trl` that spawns
        `accelerate launch` internally, with no launcher token visible
    Each command in a shell script is judged on its own, so a flag or a
    package name in an unrelated command doesn't count.
    """
    for seg in _invocation_segments(tokens):
        strategy = _segment_parallelization(seg)
        if strategy:
            return {"parallelization_strategy": strategy}
    return None


_HF_URI_RE = re.compile(r"^([^/@:]+(?:/[^/@:]+)?)(?:[@:]([^/]+))?(?:/.*)?$")


def _parse_storage_uri(uri):
    """Split a scheme-aware storageUri into (name, revision, declared_via),
    or None for schemes that only get the generic last-segment treatment.

    hf://org/model[:revision] -> repo id + optional pinned revision.
    oci://registry/repo[:tag|@digest] -> image reference without tag/digest;
    the tag is not a revision (it moves), the digest is resolved separately
    from the pod's imageID in detect_model_from_storage.
    """
    if not uri:
        return None
    if uri.startswith("hf://"):
        # hf://org/model[:rev | @rev][/subpath] -- only org/model is the repo id.
        m = _HF_URI_RE.match(uri[len("hf://"):].strip("/"))
        if m:
            return m.group(1), m.group(2), "hf_uri"
    elif uri.startswith("oci://"):
        ref = _strip_tag_digest(uri[len("oci://"):])
        if ref:
            return ref, None, "oci_uri"
    return None


def detect_model_from_storage(storage, containers=None):
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

    # model_files comes from the discovery init container reading the
    # pre-pulled model directory itself (pvc:// storageUri) -- see
    # generate_snapshot.py's read_model_source_files. Its repo_id, when
    # present, is a better name than a folder someone chose.
    files = storage.get("model_files") or {}
    location = storage.get("storage_path") or storage.get("storage_uri")

    model_name = None
    name_source = None
    uri_revision = None
    parsed = None if storage.get("storage_path") else _parse_storage_uri(storage.get("storage_uri"))
    if files.get("repo_id"):
        model_name, name_source = files["repo_id"], "model_files_readme"
    elif parsed:
        model_name, uri_revision, name_source = parsed
        if name_source == "oci_uri":
            # KServe's ModelCar container runs the model image itself, so its
            # spec image equals the URI sans scheme; the resolved digest is
            # what was actually pulled (the tag can move).
            image = storage["storage_uri"][len("oci://"):]
            for c in containers or []:
                if c.get("image") == image and _image_digest(c.get("image_id")):
                    uri_revision = _image_digest(c["image_id"])
                    break
    elif location:
        model_name = location.rstrip("/").split("/")[-1]
        name_source = "storage_path"
    if not model_name:
        return None

    result = {"model_name": model_name, "model_name_declared_via": name_source}
    quant = detect_quantization_from_name(model_name)
    if quant:
        result.update(quant)

    if files.get("dtype"):
        result["dtype"] = files["dtype"]
    if files.get("architectures"):
        result["architecture"] = files["architectures"][0]
    if files.get("revision") or uri_revision:
        result["model_revision"] = files.get("revision") or uri_revision
    if files.get("base_model"):
        result["base_model"] = files["base_model"]
    if files.get("total_size_bytes"):
        result["model_size_bytes"] = files["total_size_bytes"]
    return result


def detect_model_from_containers(containers):
    model_result = None
    parallel_result = None
    for container in containers:
        tokens = _flatten_container_command(container)
        if model_result is None:
            model_result = (
                detect_vllm_from_command(tokens)
                or detect_tgi_from_command(tokens)
                or detect_sglang_from_command(tokens)
                or detect_lmdeploy_from_command(tokens)
                or detect_trl_from_command(tokens)
                or detect_axolotl_from_command(tokens)
                or detect_llamafactory_from_command(tokens)
                or _detect_vllm_from_entrypoint_args(container)
                or _detect_tgi_from_entrypoint_args(container)
            )
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


def _dataset_names_match(detected, declared):
    """Case-insensitive, and a bare name matches the same name under an org
    or directory (`imdb` ~ `stanfordnlp/imdb`, `train.json` ~
    `/data/train.json`), but two different orgs don't."""
    if not detected or not declared:
        return False
    a, b = (str(x).strip().rstrip("/").lower() for x in (detected, declared))
    if a == b:
        return True
    if "/" in a and "/" in b:
        return False
    return a.rpartition("/")[2] == b.rpartition("/")[2]


def detect_dataset_from_containers(containers):
    for container in containers:
        tokens = _flatten_container_command(container)
        result = detect_dataset_from_command(tokens)
        if result:
            return result
    return None


# ---------------------------------------------------------------------------
# Git provenance detection: a `git clone`/`checkout` invocation in a
# container's own command/args -- covers workloads that pull their training
# code at runtime (e.g. `sh -c "git clone <url> && python train.py"`) rather
# than baking it into the image. Independent of, and a fallback below, the
# .git-directory runtime hook in runtime_detector.py, which reflects the
# actual final checked-out state rather than just the command's stated intent.
# ---------------------------------------------------------------------------

# SECURITY: keep this logic identical to runtime_detector.py's _redact_git_url -- the two can't share a
# module (see tests/test_redact_git_url_sync.py, which fails if they drift).
def redact_git_url(url):
    """Strip credentials from a git remote URL before it's recorded (#102).

    `git clone https://user:TOKEN@host/repo` stores that URL verbatim in
    .git/config and in the container command, and it would otherwise land in
    the signed, immutable AIBOM. For http(s)/ftp the whole userinfo goes (a
    bare username is often the token itself, e.g. a GitHub PAT); for other
    schemes (ssh://git@host/...) only a password is dropped, since the
    username isn't secret. scp-style `git@host:org/repo` has no scheme and is
    returned as-is for the same reason. Query string and fragment are dropped
    entirely (`?token=`, `?private_token=`), since they never identify the
    repository.
    """
    if not isinstance(url, str) or "://" not in url:
        return url
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        # Unparseable (e.g. malformed IPv6 host): drop anything that looks
        # like userinfo, query and fragment rather than risk keeping a secret.
        return re.sub(r"[?#].*$", "", re.sub(r"^([^:/]+://)[^/]*@", r"\1", url))
    netloc = parts.netloc
    if "@" in netloc:
        userinfo, _, hostport = netloc.rpartition("@")
        if parts.scheme.lower() in ("http", "https", "ftp", "ftps"):
            netloc = hostport
        else:
            user = userinfo.partition(":")[0]
            netloc = f"{user}@{hostport}" if user else hostport
    return f"{parts.scheme}://{netloc}{parts.path}"


# A SHA-1 (abbreviated or full) or a full SHA-256 commit. All-digit refs
# (`20240101`) are taken as tags; a hex-looking branch name (`cafe123`) is
# still misread as a commit -- nothing in the command says which it is.
_COMMIT_SHA_RE = re.compile(r"[0-9a-fA-F]{7,40}|[0-9a-fA-F]{64}")
# git's own global options that take a value (`git -C repo checkout ...`).
_GIT_GLOBAL_VALUE_OPTS = {"-C", "-c", "--git-dir", "--work-tree", "--namespace"}
# `git clone` options that take a separate value, so the value isn't read as
# the repository (`--reference /cache/mirror.git URL`).
_GIT_CLONE_VALUE_OPTS = {
    "-b", "--branch", "-o", "--origin", "--depth", "--reference", "--reference-if-able",
    "-u", "--upload-pack", "--template", "-c", "--config", "--separate-git-dir",
    "--shallow-since", "--shallow-exclude", "-j", "--jobs", "--filter",
    "--server-option", "--bundle-uri", "--revision",
}
# `git checkout`/`switch` options naming a new branch (`-b exp origin/exp`).
_GIT_NEW_BRANCH_OPTS = ("-b", "-B", "-c", "-C", "--orphan")
# Refs that say nothing about which commit ended up checked out.
_GIT_UNPINNED_REFS = {"HEAD", "FETCH_HEAD", "ORIG_HEAD", "-", "@"}


def _git_invocation(segment):
    """(subcommand, its args, the `-C` directory) for a `git ...` command."""
    start = _command_start(segment)
    if start >= len(segment) or os.path.basename(segment[start]) != "git":
        return None
    i, cdir = start + 1, None
    while i < len(segment) and segment[i].startswith("-"):
        if segment[i] in _GIT_GLOBAL_VALUE_OPTS and i + 1 < len(segment):
            if segment[i] == "-C":
                cdir = os.path.join(cdir or "", segment[i + 1])
            i += 2
        else:
            i += 1
    if i >= len(segment):
        return None
    return segment[i], segment[i + 1:], cdir


def _positionals(args, value_opts, stop_at_dashdash=False):
    """Arguments that aren't options or option values. After `--`, the rest
    are positional for clone and pathspecs (ignored) for checkout."""
    out, i = [], 0
    while i < len(args):
        a = args[i]
        if a == "--":
            if not stop_at_dashdash:
                out.extend(args[i + 1:])
            break
        if a.startswith("-") and a != "-":
            i += 2 if a in value_opts else 1
            continue
        out.append(a)
        i += 1
    return out


def _clone_dir_name(repo):
    """The directory `git clone <repo>` creates when none is given."""
    name = re.split(r"[/:]", repo.rstrip("/"))[-1]
    return name[:-len(".git")] if name.endswith(".git") else name


def _usable_ref(ref):
    return bool(ref) and ref not in _GIT_UNPINNED_REFS and not _has_unexpanded_var(ref)


def _is_commit_ref(ref):
    return _usable_ref(ref) and bool(_COMMIT_SHA_RE.fullmatch(ref)) and not ref.isdigit()


def detect_git_clone_from_command(tokens):
    """`git clone` and later `git checkout`/`switch` in a command (#113).

    Each checkout is applied to the clone it runs in, by following `cd` and
    `git -C`; with only one clone, a checkout whose location can't be told is
    applied to it. With several clones, the one the script last `cd`s (or
    `-C`s) into is reported, else the first: a later clone is usually a
    dataset or tool, not the code that runs.
    ponytail: directories are matched by basename, so two clones with the
    same name in different parents are confused.
    """
    if not tokens:
        return None
    clones = []  # dicts: repo, dir, commit, branch
    cwd = None
    primary = None

    def _clone_at(path):
        name = os.path.basename(os.path.normpath(path)) if path else None
        return next((c for c in clones if c["dir"] == name), None)

    for segment in _split_segments(tokens):
        start = _command_start(segment)
        if start < len(segment) and segment[start] == "cd":
            target = segment[start + 1] if start + 1 < len(segment) else None
            cwd = None if target in (None, "-", "~") else os.path.join(cwd or "", target)
            primary = _clone_at(cwd) or primary
            continue
        inv = _git_invocation(segment)
        if not inv:
            continue
        sub, args, cdir = inv
        if sub == "clone":
            pos = _positionals(args, _GIT_CLONE_VALUE_OPTS)
            if not pos or _has_unexpanded_var(pos[0]):
                continue
            repo = pos[0]
            branch = _find_flag_value(args, ("-b", "--branch"))
            clones.append({
                "repo": repo,
                "dir": os.path.basename(os.path.normpath(pos[1])) if len(pos) > 1 else _clone_dir_name(repo),
                "commit": None,
                "branch": branch if _usable_ref(branch) else None,
            })
        elif sub in ("checkout", "switch"):
            where = os.path.join(cwd or "", cdir) if cdir else cwd
            clone = _clone_at(where) or (clones[0] if len(clones) == 1 else None)
            if not clone:
                continue
            if cdir:
                primary = clone
            new_branch = _find_flag_value(args, _GIT_NEW_BRANCH_OPTS)
            pos = _positionals(args, set(_GIT_NEW_BRANCH_OPTS), stop_at_dashdash=True)
            target = pos[0] if pos else None
            # A checkout of something unknown (FETCH_HEAD, $REF) clears
            # what an earlier clone -b/checkout said, rather than keeping it.
            if new_branch:
                # `checkout -b exp [start]`: on branch exp, at start if a SHA.
                clone["branch"] = new_branch if _usable_ref(new_branch) else None
                clone["commit"] = target if target and _is_commit_ref(target) else None
            elif target:
                is_commit = _is_commit_ref(target)
                clone["commit"] = target if is_commit else None
                clone["branch"] = target if _usable_ref(target) and not is_commit else None

    chosen = primary or (clones[0] if clones else None)
    if not chosen:
        return None
    result = {"git_repository": redact_git_url(chosen["repo"])}
    if chosen["commit"]:
        result["git_commit"] = chosen["commit"]
    if chosen["branch"]:
        result["git_branch"] = chosen["branch"]
    return result


def detect_git_clone_from_containers(containers):
    for container in containers:
        tokens = _flatten_container_command(container)
        result = detect_git_clone_from_command(tokens)
        if result:
            result["detected_via"] = "cli_arg"
            return result
    return None


def detect_git_provenance_from_runtime_info(runtime_info):
    """Git provenance captured by runtime_detector.py's .git-directory read
    inside the training container (see its _capture_git_provenance). Ranked
    above the CLI-parsed `git clone` tier: this reflects the actual final
    checked-out state, not just the command's stated intent."""
    if not runtime_info.get("git_commit") and not runtime_info.get("git_repository"):
        return None
    result = {
        "git_commit": runtime_info.get("git_commit"),
        "git_repository": redact_git_url(runtime_info.get("git_repository")),
        "git_branch": runtime_info.get("git_branch"),
        "detected_via": "git_directory",
    }
    if runtime_info.get("git_dirty") is not None:
        result["git_dirty"] = runtime_info["git_dirty"]
    return result


def _normalize_repo(url):
    """Comparable form of a remote: no scheme, userinfo, `.git` or case, and
    scp-style `git@host:org/repo` written as `host/org/repo`."""
    u = (redact_git_url(url) or "").strip().lower()
    rest = re.sub(r"^[a-z][a-z0-9+.-]*://", "", u)
    rest = re.sub(r"^[^@/]*@", "", rest)
    if "://" not in u:  # scp-style host:path
        rest = rest.replace(":", "/", 1)
    return re.sub(r"(\.git)?/*$", "", rest)


def _same_provenance_value(key, a, b):
    if key == "git_commit":
        a, b = a.lower(), b.lower()
        return a.startswith(b) or b.startswith(a)  # an abbreviated SHA matches its full one
    return _normalize_repo(a) == _normalize_repo(b)


def merge_provenance(primary, fallback):
    """`primary`, with fields it lacks filled from `fallback` -- unless the two
    name different commits or repositories, since then they describe
    different checkouts and mixing them would pair one checkout's commit with
    another's repo, branch or dirty flag (#113)."""
    if not primary:
        return fallback
    if not fallback:
        return primary
    for key in ("git_commit", "git_repository"):
        if primary.get(key) and fallback.get(key) and not _same_provenance_value(key, primary[key], fallback[key]):
            return primary
    merged = dict(primary)
    for key in ("git_commit", "git_repository", "git_branch", "git_dirty"):
        if merged.get(key) is None and fallback.get(key) is not None:
            merged[key] = fallback[key]
    return merged


# ---------------------------------------------------------------------------
# Git provenance detection (from commit labels baked onto a container's
# image at build time -- OpenShift BuildConfig's own labels, or the
# vendor-neutral OCI equivalent set by other CI systems)
# ---------------------------------------------------------------------------

_BUILD_LABEL_COMMIT = "io.openshift.build.commit.id"
_BUILD_LABEL_REF = "io.openshift.build.commit.ref"
_BUILD_LABEL_SOURCE = "io.openshift.build.source-location"

# Vendor-neutral fallback: populated by tooling other than an OpenShift
# BuildConfig (GitHub Actions' docker/metadata-action, `docker buildx build
# --label`, Cloud Native Buildpacks, ko, Jib, ...). No standard OCI label for
# the branch, so this tier only ever yields commit + repository.
_OCI_LABEL_REVISION = "org.opencontainers.image.revision"
_OCI_LABEL_SOURCE = "org.opencontainers.image.source"


def _image_digest(image_id):
    """Extract the sha256 digest from a container's imageID
    ("registry/repo@sha256:..."), or None if it isn't digest-pinned."""
    if not image_id or "@sha256:" not in image_id:
        return None
    return image_id.rsplit("@", 1)[-1]


_NON_WORKLOAD_CONTAINERS = {
    "istio-proxy", "linkerd-proxy", "queue-proxy", "oauth-proxy", "kube-rbac-proxy",
    "storage-initializer", "aibom-discovery", "aibom-dataset-sidecar",
}


def _image_labels(digest, follow_manifest_list=True):
    """Config labels of the Image object for `digest`. A manifest list's
    Image has none of its own; its per-platform children (same build, so the
    same labels) are tried in turn (#113)."""
    try:
        image = k8s_api.get_cluster_object("image.openshift.io", "v1", "images", digest)
    except Exception as e:
        print(f"  WARNING: could not resolve image {digest}: {e}", file=sys.stderr)
        return {}
    if not image:
        return {}
    # dockerImageMetadata mirrors Docker's own image config JSON schema
    # verbatim (capitalized field names), not Kubernetes camelCase.
    labels = safe_get(image, "dockerImageMetadata", "Config", "Labels") or {}
    if labels or not follow_manifest_list:
        return labels
    for manifest in image.get("dockerImageManifests") or []:
        if manifest.get("digest"):
            labels = _image_labels(manifest["digest"], follow_manifest_list=False)
            if labels:
                return labels
    return {}


_REGISTRY_TIMEOUT_S = 10
_REGISTRY_MAX_BYTES = 4 * 1024 * 1024
_REGISTRY_MANIFEST_ACCEPT = ", ".join((
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.docker.distribution.manifest.v2+json",
))


def _parse_image_ref(ref):
    """(registry host, repository) of an image reference, filling in Docker
    Hub's implicit host and `library/` prefix. None for a bare digest."""
    name = _strip_tag_digest(re.sub(r"^[a-z-]+://", "", ref or ""))  # docker-pullable://
    if not name or name.startswith("sha256"):
        return None
    first, _, rest = name.partition("/")
    if rest and ("." in first or ":" in first or first == "localhost"):
        host, repo = first, rest
    else:
        host, repo = "docker.io", name
    if host in ("docker.io", "index.docker.io"):
        host = "registry-1.docker.io"
        if "/" not in repo:
            repo = "library/" + repo
    return host, repo


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


# Redirects are followed by hand so the bearer token isn't forwarded to the
# blob storage a registry redirects to.
_registry_opener = urllib.request.build_opener(_NoRedirect)


def _registry_get(url, accept=None, token=None, hops=0):
    if urllib.parse.urlsplit(url).scheme != "https":
        raise ValueError(f"refusing non-https registry URL {url}")
    headers = {"Accept": accept} if accept else {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        with _registry_opener.open(urllib.request.Request(url, headers=headers), timeout=_REGISTRY_TIMEOUT_S) as r:
            body = r.read(_REGISTRY_MAX_BYTES + 1)
    except urllib.error.HTTPError as e:
        location = e.headers.get("Location") if e.headers else None
        if e.code in (301, 302, 303, 307, 308) and location and hops < 3:
            target = urllib.parse.urljoin(url, location)
            same_host = urllib.parse.urlsplit(target).netloc == urllib.parse.urlsplit(url).netloc
            return _registry_get(target, accept, token if same_host else None, hops + 1)
        raise
    if len(body) > _REGISTRY_MAX_BYTES:
        raise ValueError(f"registry response from {url} exceeds {_REGISTRY_MAX_BYTES} bytes")
    return body


def _registry_token(challenge, repo):
    """Anonymous bearer token for a `WWW-Authenticate: Bearer realm=...`
    challenge, or None if the registry wants some other scheme."""
    if not challenge.lower().startswith("bearer "):
        return None
    params = dict(re.findall(r'(\w+)="([^"]*)"', challenge))
    realm = params.pop("realm", None)
    if not realm:
        return None
    params.setdefault("scope", f"repository:{repo}:pull")
    query = urllib.parse.urlencode({k: v for k, v in params.items() if k in ("service", "scope")})
    data = json.loads(_registry_get(f"{realm}{'&' if '?' in realm else '?'}{query}"))
    return data.get("token") or data.get("access_token")


def _registry_blob(host, repo, kind, digest, auth):
    """A manifest or blob by digest, checked against that digest. `auth`
    caches the token across calls for one image."""
    url = f"https://{host}/v2/{repo}/{kind}/{digest}"
    accept = _REGISTRY_MANIFEST_ACCEPT if kind == "manifests" else None
    try:
        body = _registry_get(url, accept, auth.get("token"))
    except urllib.error.HTTPError as e:
        if e.code != 401 or "token" in auth:
            raise
        auth["token"] = _registry_token(e.headers.get("WWW-Authenticate", ""), repo)
        if not auth["token"]:
            raise
        body = _registry_get(url, accept, auth["token"])
    actual = "sha256:" + hashlib.sha256(body).hexdigest()
    if actual != digest:
        raise ValueError(f"{url} returned content with digest {actual}")
    return json.loads(body)


def _registry_image_labels(image_ref):
    """Config labels of a digest-pinned image, read from its registry. An
    index's per-platform images come from the same build, so the first one
    that isn't an attestation stands in for all of them."""
    if not REGISTRY_LABEL_LOOKUP:
        return {}
    parsed = _parse_image_ref(image_ref)
    digest = _image_digest(image_ref)
    if not parsed or not digest:
        return {}
    host, repo = parsed
    auth = {}
    try:
        manifest = _registry_blob(host, repo, "manifests", digest, auth)
        if "manifests" in manifest:
            children = [m for m in manifest["manifests"] if (m.get("platform") or {}).get("os") != "unknown"]
            if not children:
                return {}
            manifest = _registry_blob(host, repo, "manifests", children[0]["digest"], auth)
        config_digest = safe_get(manifest, "config", "digest")
        if not config_digest:
            return {}
        config = _registry_blob(host, repo, "blobs", config_digest, auth)
    except Exception as e:
        print(f"  WARNING: could not read labels for {host}/{repo}@{digest} from its registry: {e}", file=sys.stderr)
        return {}
    return safe_get(config, "config", "Labels") or {}


def detect_git_provenance_from_containers(containers):
    """Best-effort git provenance from commit labels baked onto a
    container's image at build time. Only resolves anything if the image was
    actually built in-cluster (so an OpenShift-specific or OCI-standard
    commit label exists) -- looked up by the image's digest (containers.json's
    image_id, from ContainerStatuses, not the mutable image tag) against the
    cluster-scoped Image object, so a re-tagged image can't spoof the labels.
    See CLAUDE.md's git provenance section for why this is identification,
    not an authorization/trust guarantee: it reports what the build
    controller recorded, not whether the source was vetted. Without an Image
    object, the labels are read from the image's registry instead.
    """
    for container in containers:
        # Sidecars (service mesh, KServe, ours) aren't the workload's code.
        if container.get("name") in _NON_WORKLOAD_CONTAINERS:
            continue
        # A bare `sha256:` image_id (some runtimes) is the config digest, not
        # the manifest digest Image objects are named by; a digest-pinned
        # spec image still identifies it (#113).
        ref = next((r for r in (container.get("image_id"), container.get("image")) if _image_digest(r)), None)
        if not ref:
            continue
        digest = _image_digest(ref)
        labels = _image_labels(digest) or _registry_image_labels(ref)

        commit = labels.get(_BUILD_LABEL_COMMIT)
        oci_commit = labels.get(_OCI_LABEL_REVISION)
        # Labels inherit through FROM, so one of two disagreeing commits
        # belongs to a base image, and nothing says which (#113).
        if commit and oci_commit and commit != oci_commit:
            print(
                f"  WARNING: image {digest} has conflicting commit labels "
                f"({_BUILD_LABEL_COMMIT}={commit}, {_OCI_LABEL_REVISION}={oci_commit}); ignoring both",
                file=sys.stderr,
            )
            continue
        if commit:
            return {
                "git_commit": commit,
                "git_repository": redact_git_url(labels.get(_BUILD_LABEL_SOURCE)),
                "git_branch": labels.get(_BUILD_LABEL_REF),
                "detected_via": "openshift_build_label",
            }

        if oci_commit:
            return {
                "git_commit": oci_commit,
                "git_repository": redact_git_url(labels.get(_OCI_LABEL_SOURCE)),
                "git_branch": None,
                "detected_via": "oci_image_label",
            }
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


def query_prometheus_range(promql, start_ms, end_ms, step_seconds=None):
    return _query_prometheus(
        "/api/v1/query_range",
        {
            "query": promql,
            "start": start_ms / 1000,
            "end": end_ms / 1000,
            "step": step_seconds or _range_step_seconds(start_ms, end_ms),
        },
        timeout=30,
    )


def query_prometheus_instant(promql, time_ms):
    return _query_prometheus(
        "/api/v1/query",
        {"query": promql, "time": time_ms / 1000},
        timeout=30,
    )


def parse_range_response(response):
    if not response or response.get("status") != "success":
        return []
    results = []
    for series in response.get("data", {}).get("result", []):
        for ts, val in series.get("values", []):
            fval = float(val)
            # A ratio-of-rates query (e.g. VLLM_TELEMETRY_QUERIES' TTFT/ITL
            # sum-rate/count-rate) divides 0/0 into NaN whenever a window had
            # zero completed requests. Drop it here rather than in
            # compute_metric_stats, since a NaN mixed into min()/max()/sum()
            # silently corrupts the whole reduction depending on comparison
            # order -- a dropped sample is just a data point this window
            # didn't have, same as if the window had no scrape at all.
            if math.isnan(fval):
                continue
            results.append(
                {
                    "timestamp": datetime.fromtimestamp(ts).isoformat(),
                    "value": fval,
                }
            )
    return results


def _chunk_avg(values):
    return sum(values) / len(values) if values else None


def _round_stat(value):
    """Rounding for a reported stat, applied once, after display scaling:
    2 decimals at magnitude >= 1, 3 significant figures below that. Rounding
    raw values to 2 decimals (as this used to, before scaling) turned a
    14 ms inter-token latency into 0.01 s and 0.0006 cores into 0.0."""
    if value is None:
        return None
    if value == 0 or abs(value) >= 1:
        return round(value, 2)
    return float(f"{value:.3g}")


def compute_metric_stats(data_points):
    """Reduce a metric's raw range data points to min/max/avg/p95 plus a
    first/middle/last-third breakdown, instead of a single run-wide average.
    A flat average can't distinguish a run that held steady from one that
    started high and degraded (thermal throttling, a stalled data loader,
    checkpoint pauses); the three segments make that shape visible without
    storing the full series. See CLAUDE.md's Segmented Performance Stats
    section. Values are left unrounded here; aggregate_pod_metrics rounds
    once, after scaling to the display unit."""
    if not data_points:
        return None
    values = [p["value"] for p in sorted(data_points, key=lambda p: p["timestamp"])]
    n = len(values)
    sorted_values = sorted(values)
    p95_index = min(n - 1, math.ceil(0.95 * n) - 1)
    third = n // 3
    return {
        "min": min(values),
        "max": max(values),
        "avg": sum(values) / n,
        "p95": sorted_values[p95_index],
        "segments": {
            "first_third": _chunk_avg(values[:third]),
            "middle_third": _chunk_avg(values[third : 2 * third]),
            "last_third": _chunk_avg(values[2 * third :]),
        },
    }


def _collect_metrics_with_retry(query_defs, metrics, stats_start_ms, end_ms):
    """Runs one pod's worth of range queries with the shared retry-on-empty
    logic (see CLAUDE.md's Telemetry Retries) -- only metrics still
    missing on a given attempt are re-queried, not the whole batch. query_defs
    is the {name: {"query": ..., "unit": ...}} dict (e.g. TELEMETRY_QUERIES)
    that `metrics` (name -> pod-substituted PromQL string) was built from."""
    collected = {}
    for attempt in range(1, TELEMETRY_RETRY_ATTEMPTS + 1):
        pending = {name: q for name, q in metrics.items() if name not in collected}
        if not pending:
            break
        if attempt > 1:
            print(
                f"      Retrying {len(pending)} metric"
                f"{'s' if len(pending) != 1 else ''} after possible ingestion "
                f"delay (attempt {attempt}/{TELEMETRY_RETRY_ATTEMPTS}, "
                f"waited {TELEMETRY_RETRY_DELAY_S}s)..."
            )
        for metric_name, promql in pending.items():
            print(f"    Querying {metric_name}...")
            response = query_prometheus_range(promql, stats_start_ms, end_ms)
            data_points = parse_range_response(response) if response else []
            stats = compute_metric_stats(data_points)
            if stats:
                collected[metric_name] = {
                    "data_point_count": len(data_points),
                    "unit": query_defs[metric_name]["unit"],
                    **stats,
                }
                print(f"      {len(data_points)} data points, avg={stats['avg']:.4g}")
            else:
                print(f"      no data (attempt {attempt}/{TELEMETRY_RETRY_ATTEMPTS})")
        if len(collected) < len(metrics) and attempt < TELEMETRY_RETRY_ATTEMPTS:
            time.sleep(TELEMETRY_RETRY_DELAY_S)
    return collected


def _utc_now():
    return datetime.now(timezone.utc)


def _utc_iso_z(dt):
    """ISO-8601 UTC timestamp with a "Z" suffix, e.g. 2026-01-01T00:00:00.123456Z."""
    return dt.astimezone(timezone.utc).replace(tzinfo=None).isoformat() + "Z"


def _pod_finished_ms(pod_name, containers):
    """The latest container finished_at for pod_name (from containers.json,
    the watcher's read of ContainerStatuses[].State.Terminated), in epoch ms,
    or None if any of the pod's containers has no finish time -- still
    running (e.g. the bare-pod finalizer path, where the pod may not have
    stopped yet) or its status wasn't captured."""
    pod_containers = [c for c in containers or [] if c.get("pod_name") == pod_name]
    if not pod_containers:
        return None
    finished = []
    for c in pod_containers:
        try:
            finished.append(_parse_start_utc(c["finished_at"]))
        except (KeyError, ValueError, AttributeError, TypeError):
            return None
    return int(max(finished).timestamp() * 1000)


def _pod_stats_window(pod_name, start_time, containers):
    """Returns (start_ms, end_ms, stats_start_ms, includes_cold_start) for one
    pod's summary stats, or None if start_time can't be parsed.

    start_time is parsed as UTC (a suffix-less value too -- see
    _parse_start_utc), the same as the time-series path. The window ends when
    the pod's containers finished, not at collection time: the [5m] rate and
    avg_over_time windows keep returning a stopped pod's last samples for up
    to RATE_WINDOW_MS afterwards, which would otherwise skew the last third
    of short runs, and more so for pods processed later in a multi-pod loop.
    Falls back to now when the pod hasn't finished.

    The first scrape interval is excluded as cold start (no observation from
    the hardware yet), capped at half the run so a very short run still gets
    a partial correction; includes_cold_start flags a run too short for the
    full exclusion."""
    try:
        start_ms = int(_parse_start_utc(start_time).timestamp() * 1000)
    except (ValueError, AttributeError, TypeError):
        return None
    now_ms = int(_utc_now().timestamp() * 1000)
    finished_ms = _pod_finished_ms(pod_name, containers)
    end_ms = now_ms if finished_ms is None else min(finished_ms, now_ms)
    end_ms = max(end_ms, start_ms)
    total_ms = end_ms - start_ms
    exclude_ms = min(SCRAPE_INTERVAL_MS, total_ms // 2)
    return start_ms, end_ms, start_ms + exclude_ms, exclude_ms < SCRAPE_INTERVAL_MS


def _substitute_pod_query(promql, pod_name):
    return promql.replace("{namespace}", JOB_NAMESPACE).replace("{pod_name}", pod_name)


def _dcgm_gpu_hardware(pod_name):
    """Best-effort GPU model + per-device memory for one pod from DCGM, sourced
    by the trusted postprocess Job from cluster metrics rather than the in-pod
    discovery container's nvidia-smi. dcgm-exporter tags DCGM_FI_DEV_FB_TOTAL
    (the device's total framebuffer, MiB) with a `modelName` label and the
    workload's `exported_pod`, so one instant query yields both, needing no RBAC
    beyond the cluster-monitoring-view the Job already holds. Returns
    {"gpu_models": "A100\\nA100", "gpu_memory_mb": [40960, 40960]} (one entry per
    GPU, ordered by GPU index), or {} when DCGM isn't present or the pod has no
    GPU series -- callers fall back to the discovery values. Never raises."""
    if not JOB_NAMESPACE or not pod_name:
        return {}
    promql = _substitute_pod_query(
        'DCGM_FI_DEV_FB_TOTAL{exported_namespace="{namespace}", exported_pod="{pod_name}"}',
        pod_name,
    )
    try:
        resp = query_prometheus_instant(promql, int(time.time() * 1000))
    except Exception as e:  # network/auth/parse -- hardware stays discovery-sourced
        print(f"  WARNING: DCGM GPU hardware query failed for {pod_name}: {e}", file=sys.stderr)
        return {}
    if not resp or resp.get("status") != "success":
        return {}
    devices = {}  # gpu index (or UUID) -> {"model": str|None, "mem_mb": int|None}
    for series in resp.get("data", {}).get("result", []):
        metric = series.get("metric", {})
        key = metric.get("gpu") or metric.get("UUID") or str(len(devices))
        try:
            mem_mb = int(round(float(series.get("value", [None, None])[1])))
        except (TypeError, ValueError, IndexError):
            mem_mb = None
        devices[key] = {"model": metric.get("modelName"), "mem_mb": mem_mb}
    if not devices:
        return {}
    # gpu indices are strings; sort them numerically so "10" comes after "9",
    # with non-numeric keys (UUIDs) after them.
    def _device_order(k):
        try:
            return (0, int(k), "")
        except ValueError:
            return (1, 0, k)

    ordered = [devices[k] for k in sorted(devices, key=_device_order)]
    models = [d["model"] for d in ordered if d["model"]]
    mems = [d["mem_mb"] for d in ordered if d["mem_mb"] is not None]
    out = {}
    if models:
        out["gpu_models"] = "\n".join(models)
    if mems:
        out["gpu_memory_mb"] = mems
    return out


def collect_telemetry(discoveries, containers=None):
    print(f"  Processing {len(discoveries)} pod(s)")

    telemetry_summary = {
        "collected_at": _utc_iso_z(_utc_now()),
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

        spec_gpu_count = pod_gpu_count(pod_name, containers)
        gpu_count = spec_gpu_count if spec_gpu_count is not None else discovery.get("gpu", {}).get("gpu_count")
        if (not gpu_count or str(gpu_count) == "0") and not DEBUG_TELEMETRY_ALL_PODS:
            print(f"  Skipping {pod_name} (no GPUs)")
            continue

        print(f"  Pod: {pod_name} ({pod_uid})")

        window = _pod_stats_window(pod_name, start_time, containers)
        if window is None:
            print(f"  WARNING: Invalid start_time '{start_time}', skipping", file=sys.stderr)
            continue
        start_ms, end_ms, stats_start_ms, includes_cold_start = window

        metrics = {
            name: _substitute_pod_query(info["query"], pod_name)
            for name, info in TELEMETRY_QUERIES.items()
        }

        pod_telemetry = {
            "pod_uid": pod_uid,
            "pod_name": pod_name,
            "start_time": start_time,
            "end_ms": end_ms,
            "metrics": {},
            "includes_cold_start": includes_cold_start,
        }

        pod_telemetry["metrics"] = _collect_metrics_with_retry(
            TELEMETRY_QUERIES, metrics, stats_start_ms, end_ms
        )

        telemetry_summary["pods"].append(pod_telemetry)

    print(f"  Pods processed: {len(telemetry_summary['pods'])}")
    return telemetry_summary


def collect_vllm_telemetry(discoveries, containers=None, engine="vllm"):
    """Collects a serving engine's own serving-level metrics (see
    SERVING_TELEMETRY_QUERIES) for each pod, mirroring collect_telemetry's
    per-pod loop and retry logic but querying a disjoint set of series.
    `engine` selects the per-engine query set. Callers should only invoke this
    when detected_model.serving_engine is one we have queries for -- there's no
    cheap way to tell from discovery.json alone whether a given pod is actually
    running that engine, and querying its metrics for a pod that isn't just
    costs a few empty Prometheus queries, so the guard belongs at the caller
    (main()), not here."""
    queries = SERVING_TELEMETRY_QUERIES[engine]
    print(f"  Processing {len(discoveries)} pod(s)")

    telemetry_summary = {
        "collected_at": _utc_iso_z(_utc_now()),
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

        print(f"  Pod: {pod_name} ({pod_uid})")

        window = _pod_stats_window(pod_name, start_time, containers)
        if window is None:
            print(f"  WARNING: Invalid start_time '{start_time}', skipping", file=sys.stderr)
            continue
        _start_ms, end_ms, stats_start_ms, includes_cold_start = window

        metrics = {
            name: _substitute_pod_query(info["query"], pod_name)
            for name, info in queries.items()
        }

        pod_telemetry = {
            "pod_uid": pod_uid,
            "pod_name": pod_name,
            "start_time": start_time,
            "end_ms": end_ms,
            "metrics": {},
            "includes_cold_start": includes_cold_start,
        }

        pod_telemetry["metrics"] = _collect_metrics_with_retry(
            queries, metrics, stats_start_ms, end_ms
        )

        telemetry_summary["pods"].append(pod_telemetry)

    print(f"  Pods processed: {len(telemetry_summary['pods'])}")
    return telemetry_summary


_POD_NAME_RE = re.compile(r"^[a-z0-9]([-a-z0-9.]*[a-z0-9])?$")
_SERIES_LABEL_SOURCES = (
    ("exported_pod", "pod"),  # DCGM: `pod` there is dcgm-exporter's own pod
    ("pod", "pod"),
    ("container", "container"),
    ("interface", "interface"),
    ("gpu", "gpu"),
)


def _round_sig(value):
    rounded = float(f"{value:.6g}")
    return int(rounded) if rounded == int(rounded) else rounded


def _series_labels(metric):
    labels = {}
    for source, dest in _SERIES_LABEL_SOURCES:
        if metric.get(source) and dest not in labels:
            labels[dest] = metric[source]
    return labels


def parse_range_series(response):
    """Keeps each returned series separate (unlike parse_range_response, which
    flattens them for the stats) as {"labels": {...}, "points": [[ts, v]]},
    with integer unix-second timestamps -- unambiguous UTC, no timezone
    suffix to misread. NaN/Inf samples are dropped."""
    if not response or response.get("status") != "success":
        return []
    out = []
    for result in response.get("data", {}).get("result", []):
        points = []
        for ts, val in result.get("values", []):
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            if math.isnan(fval) or math.isinf(fval):
                continue
            points.append([int(ts), _round_sig(fval)])
        if points:
            out.append({"labels": _series_labels(result.get("metric", {})), "points": points})
    out.sort(key=lambda s: sorted(s["labels"].items()))
    return out


def _aggregate_points(series_list, how):
    by_ts = {}
    for s in series_list:
        for ts, value in s["points"]:
            by_ts.setdefault(ts, []).append(value)
    reducers = {"sum": sum, "max": max, "avg": lambda vs: sum(vs) / len(vs)}
    return [[ts, _round_sig(reducers[how](vs))] for ts, vs in sorted(by_ts.items())]


def _parse_start_utc(start_time):
    """Parses a pod start_time as UTC. A suffix-less value (what the discovery
    init container writes) is taken as UTC too, not this process's local zone."""
    dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _pod_regex(pod_names):
    # Pod names are DNS-1123, so validating them means only "." needs escaping
    # (and a backslash would need double-escaping inside a PromQL string).
    valid = [n for n in pod_names if n and _POD_NAME_RE.match(n)]
    return "|".join(n.replace(".", "[.]") for n in valid)


def _collect_series_metrics(query_defs, namespace, pod_names, start_ms, end_ms, step):
    pod_regex = _pod_regex(pod_names)
    if not pod_regex:
        return {}

    def run(defn, fn):
        promql = (
            defn["query"]
            .replace("{namespace}", namespace)
            .replace("{pod_regex}", pod_regex)
            .replace("{win}", str(step))
            .replace("{fn}", fn)
        )
        return parse_range_series(query_prometheus_range(promql, start_ms, end_ms, step_seconds=step))

    metrics = {}
    for name, defn in query_defs.items():
        print(f"    Querying series {name}...")
        series = run(defn, "avg")
        if not series:
            print("      no data")
            continue
        entry = {
            "unit": defn["unit"],
            "aggregation": defn["aggregation"],
            "aggregate": _aggregate_points(series, defn["aggregation"]),
        }
        if defn.get("gauge"):
            # Per-bucket peak, so a spike shorter than the step (an OOM-adjacent
            # memory climb) survives the downsampling. For sum-aggregated
            # metrics this is the sum of each series' own bucket peak -- a
            # slight upper bound on the true peak of the total.
            peak = run(defn, "max")
            if peak:
                entry["aggregate_max"] = _aggregate_points(
                    peak, "sum" if defn["aggregation"] == "sum" else "max"
                )
        if len(series) > SERIES_MAX_SERIES_PER_METRIC:
            entry["series_omitted"] = True
        else:
            entry["series"] = series
        metrics[name] = entry
        print(f"      {len(series)} series, {len(entry['aggregate'])} points")
    return metrics


def _encode_series_doc(doc):
    return json.dumps(doc, separators=(",", ":"), sort_keys=True)


def _rebucket_points(points, anchor, bucket_s, how):
    """Merges points onto a coarser grid of bucket_s-second buckets starting at
    anchor, reducing each bucket by "avg" or "max". A bucket is labeled with
    its start time, which stays on the original grid (the source points sit at
    anchor + i * step)."""
    buckets = {}
    for ts, value in points:
        buckets.setdefault((ts - anchor) // bucket_s, []).append(value)
    reducer = max if how == "max" else (lambda vs: sum(vs) / len(vs))
    return [[anchor + k * bucket_s, _round_sig(reducer(vs))] for k, vs in sorted(buckets.items())]


def _rebucket_series_doc(doc, factor):
    """Returns a copy of doc at `factor` times coarser resolution. Lines and
    per-series points are averaged per bucket; aggregate_max (already a
    per-bucket peak) takes the max, so a spike survives as the peak of its
    merged bucket instead of being averaged away."""
    window = doc["window"]
    anchor, bucket_s = window["start"], window["step_seconds"] * factor
    out = {**doc, "window": {**window, "step_seconds": bucket_s}, "metrics": {}}
    for name, metric in doc["metrics"].items():
        merged = {k: v for k, v in metric.items() if k not in ("aggregate", "aggregate_max", "series")}
        merged["aggregate"] = _rebucket_points(metric["aggregate"], anchor, bucket_s, "avg")
        if "aggregate_max" in metric:
            merged["aggregate_max"] = _rebucket_points(metric["aggregate_max"], anchor, bucket_s, "max")
        if "series" in metric:
            merged["series"] = [
                {"labels": s["labels"], "points": _rebucket_points(s["points"], anchor, bucket_s, "avg")}
                for s in metric["series"]
            ]
        out["metrics"][name] = merged
    return out


def _fit_series_doc(doc, max_bytes):
    """Serializes doc, shrinking it until it fits max_bytes, in this order:

    1. Reduce resolution (2x, 3x, ... coarser), as long as every line keeps at
       least SERIES_MIN_POINTS points. Per-GPU/per-pod detail is the one thing
       only this feature can show, so it's worth more than full resolution.
    2. Drop per-pod/per-GPU detail, largest metric first.

    The aggregate lines are never dropped. Returns the encoded string, or None
    if even the aggregates alone don't fit."""
    encoded = _encode_series_doc(doc)
    if len(encoded.encode("utf-8")) <= max_bytes:
        return encoded

    original = doc
    points = max((len(m["aggregate"]) for m in original["metrics"].values()), default=0)
    factor = 2
    while -(-points // factor) >= SERIES_MIN_POINTS:  # ceil(points / factor)
        # Always re-bucketed from the original, never from the previous
        # candidate, so factors don't compound.
        doc = _rebucket_series_doc(original, factor)
        encoded = _encode_series_doc(doc)
        if len(encoded.encode("utf-8")) <= max_bytes:
            print(f"  Reduced telemetry series resolution {factor}x (to {doc['window']['step_seconds']}s steps) to fit the size cap")
            return encoded
        factor += 1
    # `doc` is now the coarsest allowed resolution (or the original, if none was possible).

    while len(encoded.encode("utf-8")) > max_bytes:
        sizes = [
            (len(json.dumps(m["series"])), name) for name, m in doc["metrics"].items() if "series" in m
        ]
        if not sizes:
            return None
        _, largest = max(sizes)
        del doc["metrics"][largest]["series"]
        doc["metrics"][largest]["series_omitted"] = True
        encoded = _encode_series_doc(doc)
    return encoded


def collect_telemetry_series(telemetry, vllm_telemetry, engine="vllm"):
    """Builds the downsampled time-series document persisted alongside the
    AIBOM (see CLAUDE.md's Telemetry Time Series). Returns the encoded JSON
    string, or None if nothing came back. Never raises for missing data --
    an AIBOM without series is still a complete AIBOM."""
    resource_pods = (telemetry or {}).get("pods") or []
    vllm_pods = (vllm_telemetry or {}).get("pods") or []
    all_pods = resource_pods + vllm_pods
    starts = []
    for p in all_pods:
        try:
            starts.append(_parse_start_utc(p["start_time"]))
        except (ValueError, AttributeError, KeyError, TypeError):
            print(f"  WARNING: Invalid start_time for {p.get('pod_name')}, ignoring for series window", file=sys.stderr)
    if not starts:
        return None

    start_ms = int(min(starts).timestamp() * 1000)
    # Ends when the last pod finished (each pod's end_ms, see
    # _pod_stats_window), not at collection time.
    pod_ends = [p["end_ms"] for p in all_pods if p.get("end_ms")]
    end_ms = max(pod_ends) if pod_ends else int(_utc_now().timestamp() * 1000)
    if end_ms <= start_ms:
        return None
    span_s = (end_ms - start_ms) / 1000
    step = max(math.ceil(span_s / SERIES_TARGET_POINTS), SERIES_SCRAPE_INTERVAL_S)

    metrics = {}
    metrics.update(_collect_series_metrics(
        SERIES_QUERIES, JOB_NAMESPACE, [p.get("pod_name") for p in resource_pods], start_ms, end_ms, step
    ))
    if vllm_pods and engine in SERVING_SERIES_QUERIES:
        metrics.update(_collect_series_metrics(
            SERVING_SERIES_QUERIES[engine], JOB_NAMESPACE, [p.get("pod_name") for p in vllm_pods], start_ms, end_ms, step
        ))
    if not metrics:
        return None

    doc = {
        "schema_version": SERIES_SCHEMA_VERSION,
        "window": {"start": start_ms // 1000, "end": end_ms // 1000, "step_seconds": step},
        "pods": sorted({p["pod_name"] for p in all_pods if p.get("pod_name")}),
        "metrics": metrics,
    }
    encoded = _fit_series_doc(doc, SERIES_MAX_BYTES)
    if encoded is None:
        print("  WARNING: telemetry series exceed the size cap even without detail, skipping", file=sys.stderr)
    return encoded


def publish_series_object(encoded):
    """Stores the encoded series document in an AIBOMTelemetry custom resource
    and returns the reference to embed in the (signed) AIBOM data. The
    payload is kept as the exact string (spec.seriesJson), not structured
    JSON, so a reader can hash precisely what was stored; the reference
    carries that sha256, making the series tamper-evident for any verifier
    that checks it. Returns None on failure."""
    name = f"{JOB_NAME}-telemetry-{secrets.token_hex(4)}"
    raw = encoded.encode("utf-8")
    window = json.loads(encoded)["window"]
    body = {
        "apiVersion": f"{SERIES_API_GROUP}/{SERIES_API_VERSION}",
        "kind": SERIES_OBJECT_KIND,
        "metadata": {"name": name, "namespace": JOB_NAMESPACE, "labels": {"aibom.io/job-name": JOB_NAME}},
        "spec": {
            "schemaVersion": SERIES_SCHEMA_VERSION,
            "sizeBytes": len(raw),
            "window": {"start": window["start"], "end": window["end"], "stepSeconds": window["step_seconds"]},
            "seriesJson": encoded,
        },
    }
    try:
        k8s_api.create_custom_object(JOB_NAMESPACE, SERIES_API_GROUP, SERIES_API_VERSION, SERIES_PLURAL, body)
    except Exception as e:
        print(f"WARNING: could not store telemetry series object: {e}", file=sys.stderr)
        return None
    return {
        "schema_version": SERIES_SCHEMA_VERSION,
        "kind": SERIES_OBJECT_KIND,
        "name": name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "window": window,
    }


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


def pod_status_from_containers(pod_name, containers):
    """Reduce a pod's container statuses (terminated_reason/exit_code, captured by
    the watcher from the live Pod object at postprocess time -- see
    buildPostprocessInputs in watcher.go) to a single pod-level status/exit_code,
    the same way `kubectl get pods` picks one representative status per pod.
    OOMKilled takes priority over any other non-zero exit even if only one of
    several containers in the pod hit it, since it's the more actionable signal.
    Returns (status, exit_code), both None if no container in this pod reported
    a terminated state yet (e.g. the watcher couldn't read Pod status, or --
    for the bare-pod finalizer path -- the pod hadn't actually stopped)."""
    pod_containers = [c for c in containers if c.get("pod_name") == pod_name and c.get("terminated_reason")]
    if not pod_containers:
        return None, None
    for c in pod_containers:
        if c["terminated_reason"] == "OOMKilled":
            return "OOMKilled", c.get("exit_code")
    for c in pod_containers:
        if c.get("exit_code") not in (0, None):
            return c["terminated_reason"], c.get("exit_code")
    return pod_containers[0]["terminated_reason"], pod_containers[0].get("exit_code")


def pod_job_result(pod_name, containers):
    """The terminal condition ("Complete"/"Failed") of the Job that owns this
    pod, as the watcher recorded it in containers.json, or None for a bare pod
    or a Job that hadn't finished."""
    for c in containers:
        if c.get("pod_name") == pod_name and c.get("job_result"):
            return c["job_result"]
    return None


def pod_serving_role(pod_name, containers):
    """The pod's llm-d prefill/decode/encode role (its llm-d.ai/role label, as
    the watcher recorded it in containers.json), or None (#167)."""
    for c in containers:
        if c.get("pod_name") == pod_name and c.get("serving_role"):
            return c["serving_role"]
    return None


def pod_restart_info(pod_name, containers):
    """What a pod's containers went through before their current state, from
    the container statuses' lastState (#112). With restartPolicy OnFailure a
    container that was OOM-killed and then restarted successfully reports a
    clean current state, so the kill is only visible here. Returns a dict with
    restart_count/last_termination_reason/last_exit_code, or {} if nothing in
    the pod restarted. As with pod_status_from_containers, OOMKilled wins over
    other non-zero exits."""
    restarted = [
        c for c in containers
        if c.get("pod_name") == pod_name and c.get("restart_count") and c.get("last_terminated_reason")
    ]
    if not restarted:
        return {}
    chosen = next(
        (c for c in restarted if c["last_terminated_reason"] == "OOMKilled"),
        next((c for c in restarted if c.get("last_exit_code") not in (0, None)), restarted[0]),
    )
    return {
        "restart_count": sum(c["restart_count"] for c in restarted),
        "last_termination_reason": chosen["last_terminated_reason"],
        "last_exit_code": chosen.get("last_exit_code"),
    }


# Maps a resource_utilization metric name to the containers.json field
# holding its configured limit -- only memory/cpu have a Kubernetes resource
# limit concept; GPU/network/storage don't, so those metrics never get a
# "limit" key.
_METRIC_LIMIT_KEYS = {
    "memory_usage": "memory_limit_bytes",
    "cpu_usage": "cpu_limit_millis",
}


def pod_resource_limit(pod_name, containers, resource_key):
    """Sum one pod's own container limits for a single resource -- the same
    cgroup-level rollup the kubelet enforces when a pod has multiple
    containers sharing one memory/cpu ceiling. None if no container in this
    pod reports a limit for that resource (not the same as a limit of 0,
    which would mean "no memory available at all")."""
    limits = [c[resource_key] for c in containers if c.get("pod_name") == pod_name and c.get(resource_key) is not None]
    return sum(limits) if limits else None


def pod_gpu_count(pod_name, containers):
    """A pod's GPU count from its containers' nvidia.com/gpu claims in
    containers.json (gpu_limit, summed across the pod's containers) -- the
    number the scheduler actually placed the pod against. Preferred over the
    in-pod discovery container's nvidia-smi count, which can miscount (a
    driver-error line read as a device, #115), sees only the share one
    container was handed, and is set from inside the workload. None if no
    container in the pod claims a GPU, so the discovery value is used as a
    fallback."""
    counts = [c["gpu_limit"] for c in containers or [] if c.get("pod_name") == pod_name and c.get("gpu_limit") is not None]
    return sum(counts) if counts else None


def compute_metric_limit(metric_name, pod_names, containers, scale):
    """Resolves resource_utilization.metrics.<metric_name>.limit: the
    configured ceiling usage was measured against, scaled the same way the
    metric's own min/max/avg already are so the two are directly comparable
    (e.g. "used 7.8 of an 8.0 GiB limit"). When a JobSet's sibling pods carry
    different limits, the tightest one wins -- the ceiling closest to
    actually constraining usage -- mirroring the same "pods can disagree"
    tension CLAUDE.md's Segmented Performance Stats section already notes
    for avg/p95 across pods. None if the metric has no limit concept
    (_METRIC_LIMIT_KEYS) or no pod in this workload reported one."""
    resource_key = _METRIC_LIMIT_KEYS.get(metric_name)
    if not resource_key or not containers:
        return None
    pod_limits = [pod_resource_limit(pod_name, containers, resource_key) for pod_name in pod_names]
    pod_limits = [limit for limit in pod_limits if limit is not None]
    if not pod_limits:
        return None
    limit = min(pod_limits)
    if resource_key == "cpu_limit_millis":
        limit = limit / 1000  # millicores -> cores, matching cpu_usage's own display unit
    return _round_stat(limit * scale)


def aggregate_pod_metrics(pods, unit_map):
    """Reduces a collect_telemetry()/collect_vllm_telemetry()-shaped
    telemetry["pods"] list to one {metric_name: {...}} dict, cross-pod (e.g.
    a JobSet's sibling pods). avg/p95/segments are averaged across pods (an
    approximation -- a true cross-pod percentile would need every pod's raw
    series merged first); min/max take the true extreme across all of them,
    since a single pod's outlier is still real. See CLAUDE.md's Segmented
    Performance Stats section. Shared by resource_utilization (which adds a
    k8s resource `limit` on top, a concept these metrics alone don't have)
    and inference.performance."""
    metric_details = {}
    for metric_name, (scale, display_unit) in unit_map.items():
        scale = scale or 1
        per_pod_stats = [p["metrics"][metric_name] for p in pods if p.get("metrics", {}).get(metric_name)]
        if not per_pod_stats:
            continue

        segments = {}
        for seg in ("first_third", "middle_third", "last_third"):
            seg_values = [s["segments"][seg] for s in per_pod_stats if s["segments"][seg] is not None]
            segments[seg] = _round_stat(_chunk_avg(seg_values) * scale) if seg_values else None
        metric_details[metric_name] = {
            "unit": display_unit,
            "min": _round_stat(min(s["min"] for s in per_pod_stats) * scale),
            "max": _round_stat(max(s["max"] for s in per_pod_stats) * scale),
            "avg": _round_stat(_chunk_avg([s["avg"] for s in per_pod_stats]) * scale),
            "p95": _round_stat(_chunk_avg([s["p95"] for s in per_pod_stats]) * scale),
            "segments": segments,
        }
    return metric_details


_SEED_GENERATORS = ("python", "numpy", "torch")
_DETERMINISM_FLAG_TYPES = {
    "cudnn_deterministic": bool,
    "cudnn_benchmark": bool,
    "deterministic_algorithms": bool,
    "deterministic_algorithms_warn_only": bool,
    "cublas_workspace_config": str,
    "python_hash_seed": str,
}


def _seed_or_none(value):
    # runtime_info comes from the workload's own container; keep only the
    # shapes runtime_detector.py writes (an int, or a string for a str seed
    # or an int too large for JCS, which would leave the AIBOM unsigned).
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return None
    if isinstance(value, int) and abs(value) > 2 ** 53 - 1:
        return str(value)
    return value[:64] if isinstance(value, str) else value


def training_determinism(runtime_info):
    """training.determinism from the runtime hooks' seed and flag
    observations, or None if they saw nothing."""
    block = {}
    seeds = runtime_info.get("seeds")
    if isinstance(seeds, dict):
        values = seeds.get("values") if isinstance(seeds.get("values"), dict) else {}
        values = {g: _seed_or_none(values.get(g)) for g in _SEED_GENERATORS}
        values = {g: v for g, v in values.items() if v is not None}
        if values:
            block["seeds"] = values
        history = seeds.get("history") if isinstance(seeds.get("history"), dict) else {}
        history = {
            g: [v for v in history[g] if _seed_or_none(v) is not None]
            for g in _SEED_GENERATORS if isinstance(history.get(g), list)
        }
        history = {g: h for g, h in history.items() if len(h) > 1}
        if history:
            block["seed_history"] = history
    data_seed = _seed_or_none(runtime_info.get("data_seed"))
    if data_seed is not None:
        block["data_seed"] = data_seed
    if runtime_info.get("full_determinism") is True:
        block["full_determinism"] = True
    flags = runtime_info.get("determinism_flags")
    if isinstance(flags, dict):
        for key, kind in _DETERMINISM_FLAG_TYPES.items():
            if isinstance(flags.get(key), kind):
                block[key] = flags[key]
    return block or None


def _agreed_seed(determinism):
    """The one value every seeded generator was last seeded with, for a
    script that seeds directly instead of through TrainingArguments."""
    values = set(((determinism or {}).get("seeds") or {}).values())
    if len(values) != 1:
        return None
    value = values.pop()
    return value if isinstance(value, int) else None


def compile_aibom(
    discoveries, detected_datasets, runtime_info, annotations, telemetry,
    detected_model=None, cli_dataset=None, detected_provenance=None, containers=None,
    vllm_telemetry=None,
):
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

    # Computed once and reused for both _metadata.generated_at and the
    # duration_seconds calculation below, so the two can't drift apart.
    generated_at_dt = _utc_now()
    generated_at = generated_at_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    aibom = {}

    # Experiment metadata from annotations
    #
    # An explicit annotation always wins; otherwise fall back to what
    # detect_model_from_containers already inferred from the container's own
    # CLI (a vLLM invocation implies inference; a trl invocation implies
    # training, or sft specifically if --use_peft resolved an
    # adaptation_method) -- mirrors dataset.declared/source_code's
    # annotation-first-with-auto-detected-fallback pattern instead of only
    # ever defaulting straight to "unknown" when nobody set the annotation.
    dm = detected_model or {}
    declared_intent = annotations.get("experiment-intent")
    if declared_intent:
        aibom["experiment_intent"] = declared_intent
        aibom["experiment_intent_declared_via"] = "annotation"
    elif dm.get("serving_engine"):
        aibom["experiment_intent"] = "inference"
        aibom["experiment_intent_declared_via"] = "inferred_from_model_detection"
    elif dm.get("training_framework"):
        aibom["experiment_intent"] = "sft" if dm.get("adaptation_method") else "training"
        aibom["experiment_intent_declared_via"] = "inferred_from_model_detection"
    else:
        aibom["experiment_intent"] = "unknown"
        aibom["experiment_intent_declared_via"] = None
    aibom["experiment_name"] = annotations.get("experiment-name") or JOB_NAME or None
    aibom["experiment_description"] = annotations.get("experiment-description")

    # Git provenance: an explicit annotation always wins; otherwise fall back
    # to whatever was auto-detected (a CLI-parsed `git clone`, a runtime
    # .git read, or an image's build-time commit label -- see
    # detect_git_clone_from_containers/detect_git_provenance_from_containers).
    # declared_via records which source won, mirroring dataset.declared.declared_via.
    # Detected fields fill in what the annotations leave out only when they
    # describe the same checkout (merge_provenance), and any git annotation
    # makes declared_via "annotation" (#113).
    dp = detected_provenance or {}
    annotated = {
        key: annotations.get(name)
        for key, name in (
            ("git_commit", "git-commit"), ("git_repository", "git-repository"), ("git_branch", "git-branch"),
        )
        if annotations.get(name)
    }
    prov = merge_provenance(annotated, dp) or {}
    if annotated:
        declared_via = "annotation"
    elif dp.get("git_commit") or dp.get("git_repository"):
        declared_via = dp.get("detected_via")
    else:
        declared_via = None

    aibom["source_code"] = {
        # Redacted again here, not just in each detector: an annotation is
        # free text, and any future detection tier would have the same
        # problem (#102).
        "git_repository": redact_git_url(prov.get("git_repository")),
        "git_commit": prov.get("git_commit"),
        "git_branch": prov.get("git_branch"),
        "declared_via": declared_via,
    }
    # Only the runtime .git-directory tier can know this; kept only while
    # that tier's checkout is the one reported.
    if prov.get("git_dirty") is not None:
        aibom["source_code"]["dirty"] = prov["git_dirty"]

    # Execution metadata from discovery
    pods = []
    for discovery in discoveries:
        pod_meta = discovery.get("pod_metadata", {})
        pod_name = pod_meta.get("name")
        status, exit_code = pod_status_from_containers(pod_name, containers or [])
        pod_entry = {
            "pod_name": pod_name,
            "pod_uid": pod_meta.get("uid"),
            "pod_namespace": pod_meta.get("namespace"),
            "pod_ip": pod_meta.get("ip"),
            "node_name": pod_meta.get("node"),
            "start_time": pod_meta.get("start_time"),
            "status": status,
            "exit_code": exit_code,
        }
        pod_entry.update(pod_restart_info(pod_name, containers or []))
        serving_role = pod_serving_role(pod_name, containers or [])
        if serving_role:
            pod_entry["serving_role"] = serving_role
        # A failed pod of a Job that went on to complete is an earlier attempt
        # (restartPolicy Never + backoffLimit). Its own status stays above as
        # attempt detail, but it must not make the whole workload look failed.
        if pod_job_result(pod_name, containers or []) == "Complete" and status not in (None, "Completed"):
            pod_entry["retried_attempt"] = True
        pods.append(pod_entry)

    # duration_seconds spans from the earliest pod's start (a JobSet can have
    # sibling pods that started at slightly different times) to now --
    # postprocess runs immediately after the workload's Job completes/is
    # deleted, so "now" is the closest available proxy for when it finished.
    # Only the duration itself is stored here: the start/end timestamps it's
    # derived from already exist as pods[].start_time above and _metadata's
    # generated_at below, and duplicating them would give this AIBOM two
    # sources of truth for the same fact -- permanently, since spec is
    # immutable once created.
    # Earliest pod start to the last pod's finish (or to now, if any pod
    # hasn't finished), both as UTC.
    pod_start_times = [p["start_time"] for p in pods if p.get("start_time")]
    duration_seconds = None
    if pod_start_times:
        try:
            earliest_dt = min(_parse_start_utc(t) for t in pod_start_times)
            pod_finishes = [_pod_finished_ms(p.get("pod_name"), containers) for p in pods]
            if pod_finishes and all(f is not None for f in pod_finishes):
                end_s = min(max(pod_finishes) / 1000, generated_at_dt.timestamp())
            else:
                end_s = generated_at_dt.timestamp()
            duration_seconds = round(end_s - earliest_dt.timestamp())
        except (ValueError, AttributeError):
            print(f"  WARNING: Invalid pod start_time in {pod_start_times}, omitting duration_seconds", file=sys.stderr)

    # status rolls up pods[].status to a single value for the whole workload --
    # OOMKilled if any pod hit it (the most actionable failure mode, even if
    # only one pod of a JobSet OOMed while its siblings completed normally),
    # else any other non-Completed status, else "Completed" once every pod that
    # reported a status did so cleanly, else None if no pod reported one at all.
    # Failed attempts of a Job that then completed (retried_attempt) don't
    # count: the Job's own outcome is what the workload did (#112).
    pod_statuses = [p["status"] for p in pods if p.get("status") and not p.get("retried_attempt")]
    if "OOMKilled" in pod_statuses:
        status = "OOMKilled"
    elif any(s != "Completed" for s in pod_statuses):
        status = next(s for s in pod_statuses if s != "Completed")
    elif pod_statuses:
        status = "Completed"
    else:
        status = None
    # A Job that completed is Completed even if every pod we hold is a failed
    # earlier attempt, or none reported a terminated state.
    if status is None and any(
        pod_job_result(p.get("pod_name"), containers or []) == "Complete" for p in pods
    ):
        status = "Completed"
    # A Job can fail without any container reporting a failure (a deadline hit
    # while pods were being killed, say); its own condition still says so.
    if status in (None, "Completed") and any(
        pod_job_result(p.get("pod_name"), containers or []) == "Failed" for p in pods
    ):
        status = "Failed"
    # Pods killed for running past the Job's activeDeadlineSeconds report the
    # kill itself ("Error", exit 143); the deadline is what actually ended it.
    if status != "OOMKilled" and any(
        c.get("job_failure_reason") == "DeadlineExceeded" for c in containers or []
    ):
        status = "DeadlineExceeded"

    aibom["execution_metadata"] = {
        "job_id": JOB_NAME,
        "namespace": JOB_NAMESPACE,
        "pods": pods,
        "duration_seconds": duration_seconds,
        "status": status,
    }

    # Model info: auto-detected (container commands, then runtime hooks for
    # scripts that build TrainingArguments/from_pretrained directly in
    # Python with no corresponding CLI flags), then annotations override
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
        "architecture": (
            annotations.get("model-architecture")
            or runtime_info.get("model_architecture")
            or dm.get("architecture")
        ),
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
    # Identity details read from the model's own files on a PVC (see
    # detect_model_from_storage). Only emitted when present, and the name's
    # source is recorded unless an annotation overrode it.
    if dm.get("model_name_declared_via"):
        aibom["model"]["name_declared_via"] = (
            "annotation" if annotations.get("model-name") else dm["model_name_declared_via"]
        )
    # Those details describe the detected model; an annotation naming a
    # different one makes them someone else's (#109).
    if annotations.get("model-name") in (None, "", dm.get("model_name")):
        for src, dst in (
            ("model_revision", "revision"),
            ("base_model", "base_model"),
            ("model_size_bytes", "size_bytes"),
        ):
            if dm.get(src):
                aibom["model"][dst] = dm[src]

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
        for src, dst in (("dataset_config", "config_name"), ("dataset_split", "split")):
            if cli_ds.get(src):
                declared_dataset[dst] = cli_ds[src]
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

            # An inferred name was copied from the first entry, so there's
            # nothing independent to compare against (#114).
            declared = aibom["dataset"]["declared"]
            inferred = declared.get("declared_via") == "inferred_from_runtime"
            for entry in detected_datasets:
                entry["matches_declared"] = None if inferred else _dataset_names_match(
                    entry.get("dataset_name"), declared.get("name"))

    # Training config
    if intent in ("training", "sft"):
        determinism = training_determinism(runtime_info)
        aibom["training"] = {
            "optimizer": annotations.get("optimizer") or runtime_info.get("optimizer"),
            "learning_rate": _first_not_none(
                _try_float(annotations.get("learning-rate")),
                runtime_info.get("learning_rate"),
                dm.get("learning_rate"),
            ),
            "batch_size": _first_not_none(
                _try_int(annotations.get("batch-size")),
                runtime_info.get("batch_size"),
                dm.get("batch_size"),
            ),
            "epochs": _first_not_none(
                _try_int(annotations.get("epochs")),
                runtime_info.get("epochs"),
                dm.get("epochs"),
            ),
            "random_seed": _first_not_none(
                _try_int(annotations.get("random-seed")),
                runtime_info.get("random_seed"),
                dm.get("random_seed"),
                _agreed_seed(determinism),
            ),
            "determinism": determinism,
            "mixed_precision": runtime_info.get("mixed_precision"),
            "parallelization_strategy": (
                annotations.get("parallelization-strategy")
                or runtime_info.get("parallelization_strategy")
                or dm.get("parallelization_strategy")
                # Lowest priority: in-script sharding (device_map) seen by the runtime hook.
                or ("model_parallel" if (runtime_info.get("model_device_count") or 0) > 1 else None)
            ),
        }

    # Fine-tuning config
    if intent == "sft":
        runtime_adaptation = runtime_info.get("adaptation_method")
        # The peft hook can't tell lora from qlora itself: the LoraConfig is
        # often built before the quantized model loads (#109).
        if runtime_adaptation == "lora" and (
            runtime_info.get("quantization_method") or runtime_info.get("quantization_bits")
        ):
            runtime_adaptation = "qlora"
        aibom["fine_tuning"] = {
            "adaptation_method": (
                annotations.get("adaptation-method")
                or dm.get("adaptation_method")
                or runtime_adaptation
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
            **{k: dm.get(k) for k in _VLLM_EXTRA_KEYS},
        }
        # An llm-d prefill or decode pod only does part of each request, so its
        # performance numbers below need the role to be read correctly. Set
        # only when every pod that has a role agrees on it.
        roles = {p["serving_role"] for p in pods if p.get("serving_role")}
        if len(roles) == 1:
            aibom["inference"]["serving_role"] = roles.pop()

        # Serving-level SLOs (TTFT, ITL, queue depth, KV-cache usage,
        # throughput) -- a distinct top-level section from
        # resource_utilization since these describe the application's own
        # behavior, not the hardware underneath it (see CLAUDE.md's
        # Inference Performance Telemetry section). Populated for any engine in
        # SERVING_TELEMETRY_QUERIES (vLLM, SGLang, LMDeploy, TGI); the
        # units below are the same canonical set across engines, so a metric a
        # given engine doesn't export is simply absent from the pod telemetry
        # and skipped here (e.g. TGI has no TTFT or KV-cache metric).
        if vllm_telemetry and vllm_telemetry.get("pods"):
            vllm_unit_map = {
                "time_to_first_token_seconds": (None, "seconds"),
                "inter_token_latency_seconds": (None, "seconds"),
                "num_requests_running": (None, "requests"),
                "num_requests_waiting": (None, "requests"),
                "kv_cache_usage": (None, "percent"),
                "prompt_throughput": (None, "tokens_per_sec"),
                "generation_throughput": (None, "tokens_per_sec"),
            }
            performance = {
                "collected_at": vllm_telemetry.get("collected_at"),
                "metrics": aggregate_pod_metrics(vllm_telemetry["pods"], vllm_unit_map),
            }
            performance["summary_includes_cold_start"] = any(
                p.get("includes_cold_start") for p in vllm_telemetry["pods"]
            )
            aibom["inference"]["performance"] = performance

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

        # GPU model/count/memory are preferred from trusted out-of-pod sources
        # (the pod-spec claim and DCGM) over the in-pod discovery container's
        # nvidia-smi, falling back to discovery when those aren't available
        # (#110). Nothing else in environment has such a source without extra
        # RBAC, so cpu_model/numa/cuda/kernel stay discovery-sourced.
        first_pod = safe_get(first, "pod_metadata", "name")
        dcgm_hw = _dcgm_gpu_hardware(first_pod)
        spec_gpu_count = pod_gpu_count(first_pod, containers)

        gpu_models = dcgm_hw.get("gpu_models") or gpu_info.get("gpu_models", "")
        if gpu_models and gpu_models.strip().lower() not in ("", "not available"):
            gpu_type = gpu_models.strip().split("\n")[0].strip()
        else:
            gpu_type = None
        mem_gb_str = system_info.get("memory_total_gb")
        mem_gb = round(float(mem_gb_str), 2) if mem_gb_str else None

        aibom["environment"] = {
            "gpu_type": gpu_type,
            "gpu_count": spec_gpu_count if spec_gpu_count is not None else gpu_info.get("gpu_count"),
            "cpu_model": system_info.get("cpu_model"),
            "cpu_cores": system_info.get("cpu_count"),
            "memory_gb": mem_gb,
            "numa_nodes": system_info.get("numa_node_count"),
            "cuda_version": gpu_info.get("cuda_version"),
            "driver_version": gpu_info.get("gpu_driver_version"),
            "framework_version": fw_label,
            "kernel_version": safe_get(first, "system", "kernel_version"),
        }
        aibom["environment"].update(_discovery_details(first))
        if dcgm_hw.get("gpu_memory_mb"):
            aibom["environment"]["gpu_memory_mb"] = dcgm_hw["gpu_memory_mb"]
    # From the runtime hooks, so present even when discovery data isn't.
    software = software_environment(runtime_info)
    if software:
        aibom.setdefault("environment", {}).update(software)

    # Resource utilization from telemetry
    if telemetry and telemetry.get("pods"):
        # display_unit reflects the *scaled* value stored below, not the raw
        # per_pod_stats unit collect_telemetry recorded (e.g. "bytes") -- the
        # two diverge for memory/network/storage. Memory is binary (GiB, the
        # same unit as a Kubernetes "8Gi" limit); network and storage rates
        # are decimal, so "Mbps"/"MBps" mean what they say.
        unit_map = {
            "gpu_utilization": (None, "percent"),
            "gpu_memory_used": (None, "MiB"),
            "gpu_power": (None, "watts"),
            "cpu_usage": (None, "cores"),
            "memory_usage": (1 / (1024**3), "GiB"),
            "network_receive": (8 / 1e6, "Mbps"),
            "network_transmit": (8 / 1e6, "Mbps"),
            "storage_read_throughput": (1 / 1e6, "MBps"),
            "storage_write_throughput": (1 / 1e6, "MBps"),
        }

        utilization = {"collected_at": telemetry.get("collected_at")}
        metric_details = aggregate_pod_metrics(telemetry["pods"], unit_map)
        for metric_name, (scale, _display_unit) in unit_map.items():
            if metric_name not in metric_details:
                continue
            pod_names = [
                p["pod_name"] for p in telemetry["pods"] if p.get("metrics", {}).get(metric_name)
            ]
            limit = compute_metric_limit(metric_name, pod_names, containers or [], scale or 1)
            if limit is not None:
                metric_details[metric_name]["limit"] = limit

        utilization["metrics"] = metric_details

        # True if any pod's run was too short to exclude the cold-start
        # window, meaning the stats above may include a period of
        # stale/zero readings before the first scrape landed.
        utilization["summary_includes_cold_start"] = any(
            p.get("includes_cold_start") for p in telemetry["pods"]
        )

        aibom["resource_utilization"] = utilization
    else:
        aibom["resource_utilization"] = {
            "note": "No telemetry data available.",
        }

    # Metadata
    aibom["_metadata"] = {
        "aibom_version": "0.1.0",
        "generated_at": generated_at,
        "generator": "aibom-webhook postprocess",
        "schema_compliance": "partial - focuses on reproducibility and telemetry fields",
        "dataset_detection": (
            "enabled" if detected_datasets else "no datasets detected"
        ),
    }

    return aibom


_VLLM_EXTRA_KEYS = (
    "served_model_name", "max_num_seqs", "seed", "trust_remote_code",
    "enforce_eager", "enable_prefix_caching",
)

# Discovery fields surfaced into environment.*. The rest of discovery.json is
# captured but deliberately left out: node-level tuning, point-in-time readings
# and values that are empty or wrong from inside a container (cgroup v1 paths,
# hardcoded sda, /dev/nvme*, the init container's own ulimits).
_DISCOVERY_DETAILS = {
    "cpu": ("system", ("cpu_architecture", "cpu_cores_per_socket", "cpu_threads_per_core", "cache_l3")),
    "network": ("network", ("rdma_devices", "rdma_device_count", "primary_mtu")),
    "storage": ("storage", ("block_devices",)),
    "kernel_config": ("performance_config", ("cpu_governor", "numa_balancing",
                                             "transparent_hugepages", "max_map_count")),
}


_SOFTWARE_BUILD_FIELDS = (
    ("torch_cuda", "torch_cuda_version"),
    ("torch_hip", "torch_hip_version"),
    ("cudnn", "cudnn_version"),
    ("nccl", "nccl_version"),
)
_MAX_LIBRARIES = 64


def software_environment(runtime_info):
    """environment fields from the runtime hooks' software_versions: the
    Python version, the versions of the ML libraries the workload imported,
    and the CUDA/HIP toolkit, cuDNN and NCCL torch was built with. Strings
    only, length-capped: runtime_info is written inside the workload's own
    container."""
    sv = runtime_info.get("software_versions")
    if not isinstance(sv, dict):
        return {}
    out = {}
    if isinstance(sv.get("python"), str) and sv["python"]:
        out["python_version"] = sv["python"][:32]
    packages = sv.get("packages")
    if isinstance(packages, dict):
        libraries = {
            name[:64]: version[:64]
            for name, version in sorted(packages.items())
            if isinstance(name, str) and name and isinstance(version, str) and version
        }
        if libraries:
            out["libraries"] = dict(list(libraries.items())[:_MAX_LIBRARIES])
    for key, field in _SOFTWARE_BUILD_FIELDS:
        if isinstance(sv.get(key), str) and sv[key]:
            out[field] = sv[key][:32]
    return out


def _discovery_details(disc):
    """Extra environment sections from one discovery snapshot; empty/'N/A' values dropped."""
    out = {}
    for name, (section, keys) in _DISCOVERY_DETAILS.items():
        src = disc.get(section) or {}
        vals = {k: src[k] for k in keys if src.get(k) not in (None, "", "N/A", "None")}
        if vals:
            out[name] = vals
    mem = [
        int(float(m)) for m in str(safe_get(disc, "gpu", "gpu_memory_per_device_mb", default="")).split()
        if _try_float(m) is not None
    ]
    if mem:
        out["gpu_memory_mb"] = mem
    return out


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
# Compiled AIBOM signing
# ---------------------------------------------------------------------------


def sign_aibom(aibom):
    """Ed25519-sign aibom's canonical JSON serialization with the
    per-namespace key at SIGNING_KEY_PATH, so a copy of this AIBOM read
    outside the cluster entirely (an archived export, or the CR itself
    fetched by a tool with no reason to trust the cluster it came from) can
    still be checked against tampering.

    Asymmetric rather than the HMAC used for discovery/dataset data (see
    generate_snapshot.py's sign_payload): the intended verifiers here --
    oc-aibom, or anything reading an independently archived copy -- are
    genuinely untrusted relative to the signer, unlike the watcher verifying
    discovery/dataset data, which is an equally trusted, project-controlled
    process. Handing every verifier the same HMAC key here would let any of
    them forge a signature too.

    Returns (signature_b64, public_key_b64), or (None, None) if
    SIGNING_KEY_PATH isn't set or the key file isn't mounted -- e.g. a
    namespace whose aibom-workload-namespace chart install predates
    signing.yaml's aibom-compiled-signing-key Secret. The AIBOM is still
    created unsigned in that case rather than failing the Job.
    """
    if not SIGNING_KEY_PATH:
        return None, None
    try:
        with open(SIGNING_KEY_PATH, "rb") as f:
            key_pem = f.read()
    except OSError:
        return None, None

    import rfc8785
    from cryptography.hazmat.primitives import serialization

    private_key = serialization.load_pem_private_key(key_pem, password=None)
    public_key = private_key.public_key()
    public_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )

    # RFC 8785 (JSON Canonicalization Scheme), not a hand-rolled sort_keys/
    # separators convention -- the verifier here (oc-aibom, or anything
    # reading an archived copy) is written in Go, a different language's
    # JSON encoder, which doesn't format floats or escape non-ASCII the same
    # way Python's json module does by default. JCS exists specifically to
    # make two independent implementations agree on canonical bytes for the
    # same logical JSON value; both `rfc8785` (this) and the reference-
    # lineage `gowebpki/jcs` Go package were verified to produce identical
    # output for representative AIBOM-shaped fixtures (floats, unicode,
    # nesting, empty collections) before this was wired up.
    canonical = rfc8785.dumps(aibom)
    signature = private_key.sign(canonical)
    return base64.b64encode(signature).decode("ascii"), base64.b64encode(public_bytes).decode("ascii")


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
    storage_model = detect_model_from_storage(storage, containers)
    if storage_model:
        # Storage wins for model_name (the CLI only sees /mnt/models), but
        # dtype/architecture come from the model's own config.json -- an
        # explicit --dtype on the command line must not be overridden by it.
        merged = {**(detected_model or {}), **storage_model}
        for key in ("dtype", "architecture"):
            if (detected_model or {}).get(key):
                merged[key] = detected_model[key]
        detected_model = merged
    cli_dataset = detect_dataset_from_containers(containers)
    # Precedence among auto-detected sources (annotations always override,
    # handled separately in compile_aibom): the runtime .git-directory read
    # reflects the actual final checked-out state, ahead of a CLI-parsed
    # `git clone` (which only reflects the command's stated intent, not
    # what ended up on disk), ahead of a label baked onto the image at
    # build time (which may just describe an unrelated base image's own
    # source rather than the code actually cloned and run at pod startup).
    # A higher tier missing a field (a .git with no remote) is filled from
    # the next one, if it describes the same checkout (#113). The image
    # lookup only runs while commit or repository is still unknown.
    detected_provenance = merge_provenance(
        detect_git_provenance_from_runtime_info(runtime_info), detect_git_clone_from_containers(containers)
    )
    if not (detected_provenance and detected_provenance.get("git_commit") and detected_provenance.get("git_repository")):
        detected_provenance = merge_provenance(detected_provenance, detect_git_provenance_from_containers(containers))
    if detected_provenance:
        print("--- Git Provenance Detection ---")
        print(f"  Detected via: {detected_provenance.get('detected_via', 'unknown')}")
        if detected_provenance.get("git_commit"):
            print(f"  Commit: {detected_provenance['git_commit']}")
        if detected_provenance.get("git_branch"):
            print(f"  Branch: {detected_provenance['git_branch']}")
        if detected_provenance.get("git_repository"):
            print(f"  Repository: {detected_provenance['git_repository']}")
        if detected_provenance.get("git_dirty") is not None:
            print(f"  Working tree dirty: {detected_provenance['git_dirty']}")
        print()
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
    vllm_telemetry = None
    series_encoded = None
    serving_engine = (detected_model or {}).get("serving_engine")
    if PROMETHEUS_URL:
        print("--- Phase 1: Telemetry Collection ---")
        try:
            telemetry = collect_telemetry(discoveries, containers)
        except Exception as e:
            print(f"WARNING: Telemetry collection failed: {e}", file=sys.stderr)
        if serving_engine in SERVING_TELEMETRY_QUERIES:
            print(f"--- Phase 1b: {serving_engine} Telemetry Collection ---")
            try:
                vllm_telemetry = collect_vllm_telemetry(discoveries, containers, serving_engine)
            except Exception as e:
                print(f"WARNING: {serving_engine} telemetry collection failed: {e}", file=sys.stderr)
        # Runs after the stats collection above (and its retries), so the
        # backend has had the longest chance to ingest the run's final scrapes.
        if telemetry or vllm_telemetry:
            print("--- Phase 1c: Telemetry Time Series ---")
            try:
                series_encoded = collect_telemetry_series(telemetry, vllm_telemetry, serving_engine or "vllm")
            except Exception as e:
                print(f"WARNING: telemetry series collection failed: {e}", file=sys.stderr)
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
            detected_provenance=detected_provenance, containers=containers,
            vllm_telemetry=vllm_telemetry,
        )
    except Exception as e:
        print(f"ERROR: AIBOM compilation failed: {e}", file=sys.stderr)
        sys.exit(1)
    print()

    # Output: create the AIBOM directly as a namespaced custom resource, rather
    # than printing to stdout for the watcher to scrape from pod logs.
    print("--- Phase 3: AIBOM Custom Resource Creation ---")
    # The reference goes into aibom["telemetry_series_ref"] before signing,
    # so the series' sha256 is covered by the signature (spec is immutable, so
    # it can't be added afterwards). The AIBOMTelemetry object is created first
    # for that reason; it gets its ownerReference once the AIBOM's uid exists.
    series_object_name = None
    if series_encoded:
        series_ref = publish_series_object(series_encoded)
        if series_ref:
            aibom["telemetry_series_ref"] = series_ref
            series_object_name = series_ref["name"]
            print(f"  Stored telemetry series in {SERIES_OBJECT_KIND}/{series_object_name} ({series_ref['size_bytes']} bytes)")
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
            # Same value as spec.data._metadata.generated_at -- computed once
            # in compile_aibom rather than a second independent datetime.now()
            # call, so there's a single canonical "when did this finish"
            # timestamp instead of two that could drift apart.
            "collectedAt": aibom["_metadata"]["generated_at"],
            "data": aibom,
        },
    }
    try:
        signature, public_key = sign_aibom(aibom)
    except Exception as e:
        print(f"WARNING: could not sign AIBOM: {e}", file=sys.stderr)
        signature, public_key = None, None
    if signature:
        aibom_cr["spec"]["signature"] = signature
        aibom_cr["spec"]["signaturePublicKey"] = public_key
        # Idempotent create-or-merge (see k8s_api.patch_configmap) -- the public
        # key never changes once the Secret above is generated, so this is a
        # no-op after the first successful run in this namespace. Published
        # so a verifier working from a live cluster (rather than an archived,
        # self-contained copy carrying its own signaturePublicKey) has a
        # cluster-side anchor to cross-check against.
        try:
            k8s_api.patch_configmap(
                JOB_NAMESPACE, "aibom-compiled-signing-public-key", {"ed25519-public-key": public_key}
            )
        except Exception as e:
            print(f"WARNING: could not publish signing public key: {e}", file=sys.stderr)
        print("  Signed with Ed25519 (see aibom-compiled-signing-public-key ConfigMap)")
    else:
        print("  Skipped signing (no AIBOM_SIGNING_KEY_PATH configured for this namespace)")
    try:
        created = k8s_api.create_custom_object(JOB_NAMESPACE, "aibom.io", "v1alpha1", "aiboms", aibom_cr)
    except Exception as e:
        print(f"ERROR: could not create AIBOM custom resource: {e}", file=sys.stderr)
        if series_object_name:
            # This identity has no delete on aibomtelemetries (rbac.yaml), so the
            # object can't be cleaned up here; the Job's retry stores a fresh one.
            print(
                f"WARNING: {SERIES_OBJECT_KIND}/{series_object_name} is now orphaned (no AIBOM owns it); "
                f"remove it with: oc delete aibomtel {series_object_name} -n {JOB_NAMESPACE}",
                file=sys.stderr,
            )
        sys.exit(1)
    created_meta = created.get("metadata", {})
    print(f"  Created AIBOM/{JOB_NAMESPACE}/{created_meta.get('name', '?')}")
    if series_object_name:
        # blockOwnerDeletion needs update on aiboms/finalizers under OpenShift's
        # owner-reference permission enforcement -- granted to this Job's Role
        # in rbac.yaml for exactly this.
        try:
            k8s_api.set_custom_object_owner(
                JOB_NAMESPACE,
                SERIES_API_GROUP,
                SERIES_API_VERSION,
                SERIES_PLURAL,
                series_object_name,
                {
                    "apiVersion": "aibom.io/v1alpha1",
                    "kind": "AIBOM",
                    "name": created_meta["name"],
                    "uid": created_meta["uid"],
                    "blockOwnerDeletion": True,
                },
            )
        except Exception as e:
            print(
                f"WARNING: could not set ownerReference on {SERIES_OBJECT_KIND}/{series_object_name} "
                f"(it will not be garbage-collected with the AIBOM): {e}",
                file=sys.stderr,
            )
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
    gpu_util = aibom.get("resource_utilization", {}).get("metrics", {}).get("gpu_utilization")
    if gpu_util is not None:
        print(f"  Avg GPU utilization: {gpu_util['avg']}%")
    print()

    print("=" * 60)
    print("Post-processing complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
