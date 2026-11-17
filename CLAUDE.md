# CLAUDE.md

Read `README.md` for the high-level picture. The rationale and edge-case behavior behind each mechanism lives in `IMPLEMENTATION_NOTES.md`; it is long, so read the section for the area you're changing (index below) instead of the whole file, and update it when behavior changes.

## Layout

- `cmd/webhook`, `internal/webhook` (mutator, admission guards), `internal/watcher` (Job/pod watcher, postprocess Job creation), `internal/aibomdata` (names and GPU-resource rules shared by both).
- `scripts/aibom-scripts/`: `generate_snapshot.py` (discovery init container), `runtime_detector.py` (installed as `sitecustomize` in the user's container), `dataset_sidecar.py`, `k8s_api.py`.
- `postprocess/postprocess.py`: compiles, signs and creates the `AIBOM` CR.
- `charts/aibom-webhook`, `charts/aibom-workload-namespace`; `docs/` is the published docs site (`mkdocs.yml`).

## Testing

- Python: `.venv-test/bin/python -m pytest -q tests` (system Python has no pytest).
- Go: `go test ./...`.

## Rules that span files or repos

- **Stdlib only** in `generate_snapshot.py` (the discovery image is admin-configurable) and `runtime_detector.py` (runs in the user's interpreter). `_redact_git_url` is duplicated in the detector for that reason; keep it in sync with `redact_git_url` in `postprocess.py`.
- **Hooks never import.** `runtime_detector.py` hooks only read `sys.modules`; forcing an import from a hook breaks half-initialized packages (see Pod Matching and Injection, "Hooks never import").
- **Names computed on both sides:** `k8s_api.py`'s `_postprocess_job_name` and Go's `truncatedTriggerBase` must produce identical strings (pinned by `TestTruncatedNamesCarryHashOfFullName` and `test_resolve_data_configmap_name_truncates_long_pod_names`).
- **RBAC:** the per-job Role must stay within the workload-namespace chart's `aibom-webhook` Role (`TestWorkloadNamespaceRoleCoversPerJobRole`). The webhook's ClusterRole is read-only; writes come from per-namespace Roles.
- **Signing:** `spec` is CEL-immutable, so the signature is computed before create. Anything in `spec.data` must be JCS-canonicalizable: finite numbers, ints within 2^53 (use strings past that), or the AIBOM silently loses its signature.
- **Never trust workload-supplied metadata** for decisions: `aibom.io/instrumented` is stripped from non-qualifying pods, `aibom.io/postprocess-for` is checked against `request.userInfo`.
- **Changing `spec.data`'s shape** means updating `docs/reference/schema.md`, `docs/CAPABILITIES.md`, `docs/reference/cyclonedx-mapping.md`, and the readers in `oc-aibom` (`internal/aibom/types.go`, `cyclonedx.go`, `describe`) and `aibom-console-plugin` (`src/types/aibom.ts`). Readers must keep accepting older records' shapes (string-typed environment values, missing fields).
- **Data from inside the pod is self-attested.** Discovery/dataset HMAC signing stops the app container from forging, not the pod author (#149). Prefer sources outside the pod when cheap.
- **Fail open.** The webhook (`failurePolicy: Ignore`), the discovery init container, and postprocess's optional steps (telemetry, series, signing) degrade instead of failing the workload or the AIBOM.
- **Finalizers:** anything that adds `aibom.io/log-extraction(-pod)` needs a release path, including opted-out and deleting namespaces (Stale Finalizers, Job Pods Held Until Postprocessed).
- The legacy (pre-#130) data ConfigMap name fallback in `buildPostprocessInputs` can be removed once no pre-#130 pod can be running.

## `IMPLEMENTATION_NOTES.md` sections

- **Pod Matching and Injection**: which pods are instrumented, GPU claim mirroring and which resources count as GPUs (#110), AMD, NetworkPolicy for admission, webhook RBAC (#148), long-running pods, label trust, how the Python hooks are installed (#103, #106).
- **`aibom.io/postprocess-for` Trust Boundary**: why the label is sanitized on Job admission and the trusted-identity flags.
- **Runtime Detection Output Durability (#107)**: locking, debounced flush, SIGTERM, fork, git provenance staleness.
- **Stale Finalizers (#104)** and **Job Pods Held Until Postprocessed (#164)**.
- **Discovery Init Container Fails Open (#104)**.
- **Discovery Data Signing**: HMAC keys, what out-of-pod sources replace, multi-pod `environment` (#115), typed values (#209).
- **Dataset Data Signing**: the native sidecar, shutdown race, self-attestation limit.
- **Compiled AIBOM Signing**: Ed25519 + JCS, key placement, limits.
- **Postprocess Flow**: data/inputs ConfigMaps, cleanup, same-named reruns, long names (#130).
- **Workload Selection and Grouping**: what qualifies, triggers, JobSets, bare pods, PyTorchJob.
- **Pod Termination Status**: status rollup, retries (#112), deadlines (#164), llm-d role (#167), container images (#208).
- **Model Auto-Detection**: CLI parsing per tool, runtime hooks (#109), KServe storage, `hf://`/`oci://`, shell parsing (#108), quantization from names.
- **Determinism (#175)**, **DataLoader Sampling (#177)**, **Optimizer and Scheduler (#176)**, **Numerical Precision (#186)**, **Distributed Topology (#178)**, **Software Versions (#187)**: what each runtime hook records and why.
- **Experiment Intent Detection**, **Dataset Declaration and Reconciliation**, **Git Provenance Detection** (tiers, registry label lookup, credential redaction #102).
- **Prometheus Telemetry**, **Telemetry Retries**, **Segmented Performance Stats**, **Telemetry Time Series** (`AIBOMTelemetry` CR, size cap), **Inference Performance Telemetry** (per-engine metric names, PodMonitor ports).
- **Non-OpenShift Clusters** and **Deploy Versioning (`--version`)**.
