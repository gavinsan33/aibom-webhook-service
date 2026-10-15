# Development and Publishing

Building from source, alternative deploy recipes, image/chart publishing, and repository layout.

## Deploying with `just` recipes

There are three ways to get images into the cluster — pick whichever fits:

| Situation | Recipe |
|---|---|
| Default — `.github/workflows/images.yml` already builds both images on every push to `main` | `just deploy` |
| No egress to quay.io, or no external registry account | `just deploy-buildconfig` (in-cluster OpenShift BuildConfig) |
| Iterating locally, don't want to wait on Quay or a git push | `just deploy-local --repo=<repo>` |

The recipes target OpenShift: they call `oc` and impersonate `system:admin`. Set `AIBOM_PLATFORM=kubernetes` to use `kubectl` with your own credentials instead. `just deploy-buildconfig` works only on OpenShift.

If your account doesn't have cluster-scoped permission to create/patch CRDs, pass `--skip-crds` to any of the three; the `aiboms.aibom.io` CRD and `project-aibom` namespace must then already exist (created once via `oc apply -f charts/aibom-webhook/crds/aibom-crd.yaml` and `oc create namespace project-aibom`).

```bash
# Default: pull whatever images.yml most recently built and pushed
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
# waiting on a CI build or pushing a commit
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

## Image builds

`.github/workflows/images.yml` rebuilds both images on every push to `main` and pushes them to `quay.io/project-aibom` as `latest` and `<short-sha>`. `just deploy` with no arguments deploys `latest`, and `--version=<sha>` deploys or rolls back to a specific commit's build. It uses the same Quay robot secrets as the release workflow (see below). Quay's own GitHub build triggers aren't used: GitHub rejects the deploy key they generate as too short.

## Releases and chart publishing

**Releases.** Push a `vX.Y.Z` tag. `.github/workflows/release.yml` builds both images as `quay.io/project-aibom/aibom-webhook-service:X.Y.Z` and `.../aibom-postprocess:X.Y.Z`, then runs `just chart-push --version=X.Y.Z`, which packages both charts with `version` and `appVersion` set to `X.Y.Z` (embedding `scripts/aibom-scripts/*.py` into the workload-namespace chart so it doesn't need `--set-file`) and pushes them to `oci://quay.io/project-aibom`. The workflow fails before pushing anything if an image tag already exists, and `chart-push --version` refuses a chart version that's already published, so a release is only ever pushed once. After tagging, bump `version` in both `Chart.yaml` files to the next planned release, so dev builds sort after it.

**Dev builds.** `.github/workflows/chart-publish.yml` runs `just chart-push` on every push to `main` that touches `charts/aibom-webhook/**`, `charts/aibom-workload-namespace/**` or `scripts/aibom-scripts/**`. It publishes `<Chart.yaml version>-<git sha>` (e.g. `0.1.0-abc1234`, `-dirty`-suffixed for an uncommitted tree), whose images are `latest`, built by `images.yml`. The hyphen makes it a SemVer prerelease, so `helm install` with no `--version` skips it. Run `just chart-push` by hand the same way, after `helm registry login quay.io`.

**One-time setup.** Create a Quay robot account with push access to the two image repos and the two chart repos (used by all three workflows), and add its username and token as the `QUAY_ROBOT_USERNAME`/`QUAY_ROBOT_TOKEN` repo secrets (Settings → Secrets and variables → Actions). Make all four repos public, or arrange pull credentials, so installs can pull anonymously.

**Helm repository.** Each release is also added to a classic Helm repository, an `index.yaml` on the `gh-pages` branch, served at `https://gavinsan33.github.io/aibom-webhook-service`. That's what an OpenShift `HelmChartRepository` points at (see [Getting Started](../user-guide/getting-started.md#install-from-the-openshift-console)); it doesn't take OCI registries. One-time setup: after the first release creates the branch, enable GitHub Pages for this repo from the `gh-pages` branch (Settings → Pages).

**Checking a release for the catalog.** `just chart-verify --version=X.Y.Z` packages both charts and runs Red Hat's chart-verifier with the community profile, writing a report per chart to `chart-verifier/`. Only `helm-lint` is mandatory for a community chart. The webhook chart's `not-contains-crds` and both charts' `images-are-certified` fail and are expected to; they're mandatory only for certified partner charts.

## Publishing to the OpenShift Helm catalog

Both charts are submitted to [openshift-helm-charts/charts](https://github.com/openshift-helm-charts/charts) as community charts, which puts them in every OpenShift cluster's Developer Catalog. Pull requests there are checked by its CI and merged automatically when they pass.

1. **Once, register the charts.** In a fork of that repository, add the two `OWNERS` files from `catalog/` at `charts/community/project-aibom/aibom-webhook/OWNERS` and `charts/community/project-aibom/aibom-workload-namespace/OWNERS`, and open a pull request to `main`. Wait for it to merge. Only the GitHub users listed in `users` can submit versions.
2. **For each release,** add the release's chart tarballs as `charts/community/project-aibom/<chart>/X.Y.Z/<chart>-X.Y.Z.tgz`, one directory per chart, and open a pull request. The tarballs are on the `gh-pages` branch, or rebuild them with `just _chart-package X.Y.Z <dir>`. Their CI runs chart-verifier itself, so no report is needed. A published version can't be changed; release a new one instead.

Catalog users upgrade from the console's Upgrade action on the Helm release. Helm still doesn't update CRDs on upgrade, so each release's notes should say when the CRDs changed.

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
      certificates.yaml             # cert-manager Issuer + Certificate (off OpenShift; service-ca on it)
      deployment.yaml                # Deployment + Service
      webhook-configuration.yaml
      build.yaml                    # OpenShift BuildConfig + ImageStream
  aibom-workload-namespace/         # Per-namespace install: RBAC + scripts ConfigMap (just setup-namespace)
    templates/
      rbac.yaml
      signing.yaml                   # HMAC keys + the watcher's read access to them
      webhook-access.yaml            # the webhook's writes in this namespace
      scripts-configmap.yaml
      monitoring.yaml                # vLLM PodMonitor
scripts/
  generate-certs.sh                 # Self-signed TLS cert generation for local dev only (cluster deploy uses service-ca or cert-manager)
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
