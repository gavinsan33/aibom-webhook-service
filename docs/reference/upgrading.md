# Upgrading and Compatibility

## Upgrade order

Run these in order, as a cluster-admin, for each upgrade:

1. **Update the CRDs.** Helm installs the contents of a chart's `crds/` directory once and never updates them, so `helm upgrade` leaves the `AIBOM` and `AIBOMTelemetry` CRDs as they were. From a checkout, `just deploy` re-applies the CRD file on every run. With the published chart, apply it yourself:

    ```bash
    helm show crds oci://quay.io/project-aibom/aibom-webhook | oc apply -f -
    ```

    This matters because the apiserver silently drops fields a CRD schema doesn't know. An AIBOM created against an old CRD loses any newer field, such as the signature fields, with no error. `spec` is immutable, so it can't be repaired afterward.

2. **Upgrade the webhook chart** in `project-aibom` with the same `helm upgrade --install` command used to install it. This also updates the `aibom-view` role that lets users read AIBOMs and their telemetry.
3. **Upgrade each enabled namespace** by re-running the `aibom-workload-namespace` install. This refreshes the `aibom-scripts` ConfigMap and the namespace's RBAC. Until a namespace is upgraded, new features that rely on its Roles, the CRD or the scripts can be missing from its AIBOMs, such as stored telemetry. Workloads still run and produce AIBOMs.

Existing AIBOMs are not changed by any of this.

## Short name `aibomrun`

The `AIBOM` CRD's short name changed from `aibom` to `aibomrun`, so the project no longer claims [k8s-aibom](https://github.com/GoogleCloudPlatform/k8s-aibom)'s short name. It takes effect once the CRDs are re-applied (step 1 above). After that, `oc get aibom` fails unless k8s-aibom is installed, so update scripts to `oc get aibomrun` or `oc get aiboms.aibom.io`. The CLI was renamed at the same time and is now `oc aibomrun`; reinstall it as described in its docs.

## Record format version

`_metadata.aibom_version` is now the record format version, `1.0.0`, and is bumped whenever `spec.data`'s shape changes. Earlier records all say `0.1.0` across several shapes, so a script that needs to know a pre-1.0.0 record's shape should check its fields rather than this value.

## Typed `environment` values

`environment.cpu_cores`, `numa_nodes`, `cpu.cpu_cores_per_socket`, `cpu.cpu_threads_per_core`, `network.rdma_device_count` and `network.primary_mtu` are now numbers, `network.rdma_devices` is a list of device names, and `storage.block_devices` is a list of `{name, size}` objects. AIBOMs created before the upgrade keep the strings (`"512"`, `"mlx5_0,mlx5_1"`, raw `lsblk` output), so a script reading both should accept either form. Update `oc aibomrun` and the console plugin to versions that display both.

## Long workload names

A workload name longer than 39 characters now gets a short hash in its data ConfigMap and per-job identity names, so two long names that share a prefix (KServe replicas of one predictor, `<experiment>-run-1` and `-run-2`) no longer share them. Names of 39 characters or fewer are unchanged.

Pods admitted before the upgrade wrote to the old, unhashed name. The watcher still reads a pod's data from there when the new ConfigMap has none for it, which also covers bare pods (KServe predictors) in namespaces whose `aibom-scripts` haven't been refreshed by step 3 yet. It never deletes by the old name, because the old name of a long workload can be the current name of another workload called exactly its first 39 characters. So each long-named workload that was running at the upgrade leaves its old data ConfigMap behind, and, for Jobs, its old identity (a ServiceAccount, Role, RoleBinding and Secret sharing one name). Once every workload that was running at upgrade time has finished and been postprocessed, any `*-aibom-postprocess-data` ConfigMap or `*-aibom-workload-identity` object created before the upgrade is safe to delete:

```bash
oc get cm,sa -n <namespace> --sort-by=.metadata.creationTimestamp | grep -E 'aibom-(postprocess-data|workload-identity)'
oc delete cm <name> -n <namespace>
oc delete sa,role,rolebinding,secret <identity-name> -n <namespace>
```

## Image caching (OpenShift)

On OpenShift, `cacheImages.enabled` (default `true`) creates `ImageStream`s that mirror the injected images into the cluster's internal registry, and points the injection flags at that registry. On a `helm upgrade` into this from an earlier version:

- The ImageStreams are created and OpenShift imports the images from Quay. The import is asynchronous, so a workload admitted in the first few seconds, before the import finishes, can briefly see `ImagePullBackOff` until the kubelet retries against the now-populated registry.
- Only workloads admitted after the webhook pod rolls with its new flags get the cached references. Pods already running keep their original image references.
- No change is needed in `aibom-workload-namespace`; it already creates the `system:image-puller` RoleBinding that lets workload namespaces pull from the internal registry.

Set `cacheImages.enabled=false` to keep pulling the injected images from Quay.

## Versions

Releases are published from `vX.Y.Z` git tags. Each release publishes both charts at version `X.Y.Z` and both images tagged `X.Y.Z`, and the charts' image tags default to their own version, so installing a chart version always runs the same images. A release is never republished. With no `--version`, Helm installs the newest release; pass `--version=X.Y.Z` to pin one.

Merges to `main` also publish dev builds of the charts, versioned after the last release (e.g. `0.1.1-dev.5.gabc1234`), using the `latest` images. Helm skips these unless you pass `--devel` or the exact version. To install the newest dev build, add `--devel` to both install commands:

```bash
helm upgrade --install aibom-webhook oci://quay.io/project-aibom/aibom-webhook \
  -n project-aibom --create-namespace --kube-as-user=system:admin --devel
helm upgrade --install aibom-ns-<namespace> oci://quay.io/project-aibom/aibom-workload-namespace \
  -n <namespace> --kube-as-user=system:admin --devel
```

Both charts are published together on each merge, so `--devel` picks the same build for each.

The CRDs are `aibom.io/v1`. Releases up to 0.2.0 used `v1alpha1`, and there is no migration: a cluster with existing `v1alpha1` records can't take the new CRDs until those records are deleted or migrated, and the CLI and console plugin read only `v1`. There is no published matrix of which webhook, CLI and console plugin versions work together. The components are built from `main` in separate repositories, and the CLI and plugin read fields the webhook writes. Upgrade the webhook first, then the CLI and plugin.

## Cluster versions

- **Console plugin:** its docs state OpenShift 4.12 or later.
- **Webhook:** the dataset sidecar is a Kubernetes native sidecar, which is generally available from Kubernetes 1.29 (OpenShift 4.16). OpenShift isn't required; see [Kubernetes without OpenShift](../user-guide/getting-started.md#kubernetes-without-openshift). The webhook chart's other requirements are in [Getting Started](../user-guide/getting-started.md#cluster-requirements).
