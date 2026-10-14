# Upgrading and Compatibility

## Upgrade order

Run these in order, as a cluster-admin, for each upgrade:

1. **Update the CRDs.** Helm installs the contents of a chart's `crds/` directory once and never updates them, so `helm upgrade` leaves the `AIBOM` and `AIBOMTelemetry` CRDs as they were. From a checkout, `just deploy` re-applies the CRD file on every run. With the published chart, apply it yourself:

    ```bash
    helm show crds oci://quay.io/gsanders/aibom-webhook | oc apply -f -
    ```

    This matters because the apiserver silently drops fields a CRD schema doesn't know. An AIBOM created against an old CRD loses any newer field, such as the signature fields, with no error. `spec` is immutable, so it can't be repaired afterward.

2. **Upgrade the webhook chart** in `project-aibom` with the same `helm upgrade --install` command used to install it. This also updates the `aibom-view` role that lets users read AIBOMs and their telemetry.
3. **Upgrade each enabled namespace** by re-running the `aibom-workload-namespace` install. This refreshes the `aibom-scripts` ConfigMap and the namespace's RBAC. Until a namespace is upgraded, new features that rely on its Roles, the CRD or the scripts can be missing from its AIBOMs, such as stored telemetry. Workloads still run and produce AIBOMs.

Existing AIBOMs are not changed by any of this.

## Postprocess moved to the webhook namespace

Postprocess Jobs, the `aibom-postprocess` ServiceAccount and the compiled-AIBOM signing key now live in the webhook's namespace, so a pod in a workload namespace can't mount the key or run as that identity.

- **Copy each namespace's signing key before upgrading its namespace chart.** The chart no longer owns `aibom-compiled-signing-key`, so the upgrade deletes it. Without the copy, the watcher generates a new key, the published public key changes, and `oc aibom describe` reports earlier AIBOMs as `KEY MISMATCH`.

    ```bash
    oc get secret aibom-compiled-signing-key -n <namespace> -o jsonpath='{.data.ed25519-key}' | base64 -d \
      | oc create secret generic aibom-compiled-signing-key-<namespace> -n project-aibom --from-file=ed25519-key=/dev/stdin
    ```

- **Upgrade both charts before the next workload finishes.** New postprocess Jobs run as the webhook namespace's ServiceAccount, which gets its access in each workload namespace from the namespace chart. Until a namespace is upgraded, its AIBOMs fail with a Forbidden error.
- **`monitoringAccess.*` moved** from the namespace chart to the webhook chart, and `just setup-namespace --skip-monitoring-access` is gone.
- Postprocess Jobs created before the upgrade are left in the workload namespace and are no longer cleaned up. Delete them with `oc delete job -l aibom.io/postprocess-for -n <namespace>`.
- A failed postprocess Job's logs are in the webhook namespace now. The workload's namespace gets an `AIBOMPostprocessFailed` Event naming it.

## Versions

Both charts are at version `0.1.0`. The plain `0.1.0` tag is republished on every chart push, and each push also gets an immutable `0.1.0-<git-sha>` tag. With no `--version`, Helm takes the mutable tag. To make an upgrade repeatable, pin the SHA tag. See [Getting Started](../user-guide/getting-started.md).

The CRDs are `aibom.io/v1alpha1`. There is no published matrix of which webhook, CLI and console plugin versions work together. The components are built from `main` in separate repositories, and the CLI and plugin read fields the webhook writes. Upgrade the webhook first, then the CLI and plugin.

## Cluster versions

- **Console plugin:** its docs state OpenShift 4.12 or later.
- **Webhook:** the dataset sidecar is a Kubernetes native sidecar, which is generally available from Kubernetes 1.29 (OpenShift 4.16). OpenShift isn't required; see [Kubernetes without OpenShift](../user-guide/getting-started.md#kubernetes-without-openshift). The webhook chart's other requirements are in [Getting Started](../user-guide/getting-started.md#cluster-requirements).
