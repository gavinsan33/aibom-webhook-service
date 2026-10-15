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

## Versions

Releases are published from `vX.Y.Z` git tags. Each release publishes both charts at version `X.Y.Z` and both images tagged `X.Y.Z`, and the charts' image tags default to their own version, so installing a chart version always runs the same images. A release is never republished. With no `--version`, Helm installs the newest release; pass `--version=X.Y.Z` to pin one.

Merges to `main` also publish dev builds of the charts as `<version>-<git-sha>`, using the `latest` images. Helm skips these unless you pass `--devel` or the exact version.

The CRDs are `aibom.io/v1alpha1`. There is no published matrix of which webhook, CLI and console plugin versions work together. The components are built from `main` in separate repositories, and the CLI and plugin read fields the webhook writes. Upgrade the webhook first, then the CLI and plugin.

## Cluster versions

- **Console plugin:** its docs state OpenShift 4.12 or later.
- **Webhook:** the dataset sidecar is a Kubernetes native sidecar, which is generally available from Kubernetes 1.29 (OpenShift 4.16). OpenShift isn't required; see [Kubernetes without OpenShift](../user-guide/getting-started.md#kubernetes-without-openshift). The webhook chart's other requirements are in [Getting Started](../user-guide/getting-started.md#cluster-requirements).
