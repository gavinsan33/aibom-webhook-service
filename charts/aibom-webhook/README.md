# AIBOM Webhook

Records an AI Bill of Materials (AIBOM) for AI workloads. A mutating admission webhook instruments pods in opted-in namespaces with a hardware-discovery init container and dataset-detection hooks; when the workload finishes, a postprocess Job compiles an `AIBOM` custom resource covering hardware, model and training configuration, datasets, git provenance and Prometheus telemetry. Workload manifests don't change.

Full documentation: https://gavinsan33.github.io/project-aibom/

## Installing from the OpenShift Helm catalog? Install the CRDs first

**The chart in the OpenShift Helm catalog does not include its CRDs, and nothing works without them.** The catalog doesn't accept charts that contain CRDs. Before you install this chart from the Developer Catalog, a cluster-admin must apply the `AIBOM` and `AIBOMTelemetry` CRDs for the same version:

```bash
oc apply -f https://github.com/gavinsan33/aibom-webhook-service/releases/download/v<version>/aibom-crds.yaml
```

If they're missing, the install fails and prints this command. **Apply them again before every upgrade**, with the new version in the URL: Helm never installs or updates CRDs for this chart, and an AIBOM created against an older CRD silently loses any field the CRD doesn't know.

The chart from `oci://quay.io/project-aibom/aibom-webhook` includes the CRDs and installs them itself.

## Requirements

- OpenShift 4.16 or later, or Kubernetes 1.29 or later.
- On Kubernetes, [cert-manager](https://cert-manager.io/), which issues the webhook's TLS certificate. On OpenShift the built-in service CA issues it, so nothing extra is needed.
- A cluster-admin to install. The chart creates ClusterRoles, ClusterRoleBindings, a MutatingWebhookConfiguration and, from quay.io, the `AIBOM`/`AIBOMTelemetry` CRDs.

## Install

```bash
helm upgrade --install aibom-webhook oci://quay.io/project-aibom/aibom-webhook \
  -n project-aibom --create-namespace --version <version>
```

Then enable each namespace that runs workloads: label it `aibom.io/enabled=true` and install the `aibom-workload-namespace` chart in it. Helm can't label a namespace it didn't create, so the label is a separate step:

```bash
oc label namespace <namespace> aibom.io/enabled=true
helm upgrade --install aibom-ns-<namespace> oci://quay.io/project-aibom/aibom-workload-namespace \
  -n <namespace> --version <version>
```

If you installed this chart somewhere other than `project-aibom`, also set `aibomSystemNamespace` on the namespace chart.

Run `helm test aibom-webhook -n project-aibom` to check that the webhook answers and the CRDs are installed.

## Things to know

- **Upgrades don't update the CRDs.** Helm installs a chart's CRDs once and never updates them. Before upgrading, apply the new ones: `helm show crds oci://quay.io/project-aibom/aibom-webhook --version <version> | oc apply -f -`, or, without `helm`, the `oc apply -f` command above with the new version.
- **A `PYTHONPATH` set in the image is overridden.** The webhook sets `PYTHONPATH` on instrumented containers to load its detection hooks, and can't see an `ENV PYTHONPATH` baked into the image. Declare it with the `aibom.io/python-path` pod annotation, for example `/app:/app/lib`, and it's appended.
- **The webhook fails open.** If it's unreachable, pods are created uninstrumented and no AIBOM is recorded.
- **Instrumented pods pull a discovery image.** It's `quay.io/project-aibom/aibom-discovery`, about 230 MB, at the chart's version. If the cluster can't reach Quay, set `image.discovery.repository` and `.tag` to a copy it can pull; any image with `python3`, `bash`, `lscpu` and `lsblk` works.
- **Argo CD and other offline renders** must set `platform: openshift` (or `kubernetes`) on both charts, since `auto` can't detect the cluster without access to it, and `signing.createKeys: false` on the namespace chart (see the [Argo CD install guide](https://gavinsan33.github.io/project-aibom/aibom-webhook-service/user-guide/argocd/)). Argo CD also doesn't run this chart's `post-delete` hook, which removes the webhook's finalizers on uninstall. Before deleting the app, delete the workloads that are still running, or afterwards run the webhook image with `--strip-finalizers` as a cluster-admin.
- **Image labels are read from registries.** With `gitProvenance.registryLookup` on (the default), the postprocess Job reads image labels anonymously from the image's registry when the cluster has no `Image` object for it, to find the git commit the image was built from. Set it to `false` to stop these outbound requests.

Every value is described in `values.schema.json` and in the [configuration reference](https://gavinsan33.github.io/project-aibom/aibom-webhook-service/reference/configuration/).
