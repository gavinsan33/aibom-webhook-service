# Development and Publishing

Building from source, alternative deploy recipes, image/chart publishing, and repository layout.

## Deploying with `just` recipes

There are three ways to get images into the cluster — pick whichever fits:

| Situation | Recipe |
|---|---|
| Default — Quay's GitHub build triggers already build both images on every push to `main` | `just deploy` |
| No egress to quay.io, or no external registry account | `just deploy-buildconfig` (in-cluster OpenShift BuildConfig) |
| Iterating locally, don't want to wait on Quay or a git push | `just deploy-local --repo=<repo>` |

If your account doesn't have cluster-scoped permission to create/patch CRDs, pass `--skip-crds` to any of the three; the `aiboms.aibom.io` CRD and `aibom-system` namespace must then already exist (created once via `oc apply -f charts/aibom-webhook/crds/aibom-crd.yaml` and `oc create namespace aibom-system`).

```bash
# Default: pull whatever Quay's build triggers most recently built and pushed
just deploy

# Pin to an immutable SHA tag instead of the mutable "latest" Quay keeps overwriting
just deploy --version=<sha>

# Point at a different quay.io org/user than the values.yaml default
just deploy --repo=quay.io/<your-org>

# Or, build both images in-cluster from source instead (no quay.io dependency)
just deploy-buildconfig

# Roll back: rebuilds and redeploys that exact historical commit
just deploy-buildconfig --version=<older-sha>

# Or, build locally and push to quay.io yourself — for iterating without
# waiting on Quay's build trigger or pushing a commit
just deploy-local --repo=quay.io/<your-org>

# Set up a workload namespace (label, image pull access, scripts ConfigMap)
just setup-namespace --namespace=my-ai-workloads

# Verify: submit a Job, check the pod for the init container
oc get pod <pod-name> -n my-ai-workloads -o jsonpath='{.spec.initContainers[*].name}'
# Should output: aibom-discovery

# Check dataset detector env vars
oc get pod <pod-name> -n my-ai-workloads -o jsonpath='{.spec.containers[0].env[*].name}'
# Should include: AIBOM_DATASET_DETECT AIBOM_DEBUG AIBOM_DATASET_OUTPUT PYTHONPATH
```

To remove a workload namespace's setup: `just uninstall-namespace --namespace=<ns>` (runs `helm uninstall aibom-ns-<ns>` and removes the `aibom.io/enabled` label, after a confirmation prompt).

To remove the deployment: `just undeploy` (runs `helm uninstall aibom-webhook`, after a confirmation prompt). Helm installs CRDs once but never upgrades or removes them automatically, so `just deploy`/`deploy-buildconfig`/`deploy-local` each explicitly `oc apply -f charts/aibom-webhook/crds/aibom-crd.yaml` before the `helm upgrade --install` step (skipped along with everything else cluster-scoped when `--skip-crds` is passed — see that flag's own note below). `helm uninstall` still leaves the CRD (and any AIBOM custom resources) in place regardless.

## Setting Up Quay Auto-Build

One-time setup, done in the Quay web UI (not scriptable — it requires a GitHub OAuth authorization). Repeat for both `aibom-webhook-service` and `aibom-postprocess` repos on quay.io:

1. Repo → **Builds** tab → **Add Build Trigger** → **GitHub Repository Push**, authorizing Quay against GitHub if prompted
2. Source repo: this repo; branch filter restricted to `main` only
3. Dockerfile location: `/Dockerfile` for `aibom-webhook-service`, `/postprocess/Dockerfile` for `aibom-postprocess`; context `/` for both (the postprocess Dockerfile `COPY`s files from outside its own directory)
4. Tagging options: add a template so each build produces both `latest` and a short-commit-SHA tag, keeping rollback ("redeploy an older SHA") consistent with `just deploy-buildconfig`'s path

Once set up, every push to `main` produces new `latest` and `<sha>` tags automatically — `just deploy` (no arguments) always deploys whatever was built most recently.

## Setting Up Chart Publishing

One-time setup so `helm upgrade --install ... oci://quay.io/<org>/aibom-webhook` (and `.../aibom-workload-namespace`) work for anyone, without a checked-out copy of this repo:

1. `helm registry login quay.io` with an account (or robot account) that has push access under your Quay org
2. `just chart-push` — packages both charts (embedding `scripts/aibom-scripts/*.py` into the workload-namespace chart so it doesn't need `--set-file`) and pushes each one to `oci://quay.io/<org>/aibom-webhook` / `.../aibom-workload-namespace` under two tags, mirroring `just deploy`'s mutable `latest`/immutable `<sha>` split for images:
   - `<Chart.yaml version>` (e.g. `0.1.0`) — mutable, overwritten on every `chart-push` (nothing bumps `Chart.yaml`'s `version:` automatically). `helm upgrade --install` with no `--version` resolves here.
   - `<Chart.yaml version>-<git sha>` (e.g. `0.1.0-abc1234`, `-dirty`-suffixed the same way `deploy-local`'s tag is for an uncommitted working tree) — immutable, one per `chart-push`, for pinning/rollback. This uses SemVer *prerelease* syntax (hyphen), which has strictly lower precedence than the plain release per the SemVer spec — that's what makes "no `--version`" reliably resolve to the mutable tag above instead of an unpredictable tie (an earlier version of this used build-metadata syntax, `+<sha>`, which SemVer precedence ignores entirely — the two tags were then equal-precedence and which one `helm` picked with no `--version` was undefined)

   Defaults to pushing under `quay.io/gsanders`; pass a different org as the first argument.
3. In the Quay web UI, make both chart repos public (or otherwise arrange pull credentials) so `helm upgrade --install` can pull them anonymously

Re-run `just chart-push` any time `charts/aibom-webhook`, `charts/aibom-workload-namespace`, or `scripts/aibom-scripts/*.py` changes.

**Automating it**: unlike the images, Quay's own GitHub build trigger can't drive this (it only knows how to run a `docker build`), so `.github/workflows/chart-publish.yml` runs `just chart-push` in GitHub Actions instead — triggered only on a push to `main` (i.e. a merge, not every feature-branch commit) that touches `charts/aibom-webhook/**`, `charts/aibom-workload-namespace/**`, or `scripts/aibom-scripts/**`. One-time setup: create a Quay **robot account** scoped to push access on the `aibom-webhook`/`aibom-workload-namespace` repos, then add its username/token as the `QUAY_ROBOT_USERNAME`/`QUAY_ROBOT_TOKEN` repo secrets (Settings → Secrets and variables → Actions).

## Local testing (without a cluster)

```bash
# Start the server
just run

# In another terminal, send a test admission review
curl -sk -X POST https://localhost:8443/mutate \
  -H "Content-Type: application/json" \
  -d '{
    "apiVersion": "admission.k8s.io/v1",
    "kind": "AdmissionReview",
    "request": {
      "uid": "test",
      "resource": {"group": "", "version": "v1", "resource": "pods"},
      "object": {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
          "name": "test-pod",
          "namespace": "default",
          "ownerReferences": [{"kind": "Job", "name": "my-job", "apiVersion": "batch/v1", "uid": "abc"}]
        },
        "spec": {
          "containers": [{"name": "train", "image": "pytorch:latest"}]
        }
      }
    }
  }'

# Health check
curl -sk https://localhost:8443/healthz
```

## Project Structure

```
cmd/webhook/main.go                # Entrypoint: TLS, HTTP server, watcher, graceful shutdown
internal/
  webhook/
    handler.go                      # AdmissionReview HTTP handler
    mutator.go                      # Pod matching + JSON patch construction
    handler_test.go                 # Unit tests
  watcher/
    watcher.go                      # Job completion watcher + postprocess Job creation
    watcher_test.go                 # Unit tests
  config/config.go                  # Configuration struct
  aibomdata/aibomdata.go            # Shared postprocess Job/ConfigMap naming convention
postprocess/
  postprocess.py                    # AIBOM compiler; creates the AIBOM CR directly (runs in postprocess Job)
  Dockerfile                        # Postprocess container image; also COPYs in scripts/aibom-scripts/k8s_api.py
charts/
  aibom-webhook/                    # Cluster-level install: namespace, CRD, RBAC, certs, Deployment/Service,
    crds/aibom-crd.yaml               # webhook config, OpenShift BuildConfig/ImageStream (just deploy/undeploy)
    templates/
      serviceaccount.yaml
      clusterrole.yaml
      clusterrolebinding.yaml
      certificates.yaml             # cert-manager Issuer + Certificate
      deployment.yaml                # Deployment + Service
      webhook-configuration.yaml
      build.yaml                    # OpenShift BuildConfig + ImageStream
  aibom-workload-namespace/         # Per-namespace install: RBAC + scripts ConfigMap (just setup-namespace)
    templates/
      serviceaccount.yaml
      rbac.yaml
      scripts-configmap.yaml
      monitoring.yaml                # service-ca ConfigMap + cluster-monitoring-view ClusterRoleBinding
scripts/
  generate-certs.sh                 # Self-signed TLS cert generation for local dev only (cluster deploy uses cert-manager)
  remote-build-sha.sh                # Resolves the short SHA `just deploy`'s --version defaults to (justfile helper)
  aibom-scripts/
    generate_snapshot.py             # Hardware discovery script (from coldpress)
    runtime_detector.py               # Dataset detection + training runtime hooks (from coldpress)
    k8s_api.py                       # Stdlib-only in-cluster REST client shared by these scripts
    dataset_sidecar.py               # Signs + publishes dataset detection data from outside the app container
examples/
  vllm-inference.yaml               # Example JobSet: vLLM server + guidellm benchmark
  vllm-inference-rhoai.yaml         # Same model via a RHOAI/KServe InferenceService
  granite-lora-finetune.yaml        # Example Job: single-GPU LoRA fine-tuning via trl sft
  granite-lora-finetune-multigpu.yaml  # Same, but 2 GPUs via trl's --num_processes passthrough
  granite-lora-finetune-raw-trainer.yaml  # Same, but via raw transformers.Trainer + peft.LoraConfig (no CLI at all)
tests/
  postprocess/test_postprocess.py   # Unit tests for postprocess.py's CLI-arg detectors and compile_aibom
  aibom_scripts/                    # Unit tests for runtime_detector.py's hooks (fake torch/datasets/transformers/peft modules)
pyproject.toml                      # pytest config (pythonpath into postprocess/ and scripts/aibom-scripts/)
requirements-dev.txt                # Test-only deps (pytest, pyyaml) — production scripts stay dependency-free
Dockerfile                          # Multi-stage build (distroless)
justfile                            # Build, test, deploy recipes (just test, just test go, just test python)
Makefile                            # Bootstrap only: `make install-just` installs the just task runner
```
