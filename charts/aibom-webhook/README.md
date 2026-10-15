# AIBOM Webhook

Records an AI Bill of Materials (AIBOM) for AI workloads. A mutating admission webhook instruments pods in opted-in namespaces with a hardware-discovery init container and dataset-detection hooks; when the workload finishes, a postprocess Job compiles an `AIBOM` custom resource covering hardware, model and training configuration, datasets, git provenance and Prometheus telemetry. Workload manifests don't change.

Full documentation: https://gavinsan33.github.io/project-aibom/

## Requirements

- OpenShift 4.16 or later, or Kubernetes 1.29 or later.
- On Kubernetes, [cert-manager](https://cert-manager.io/), which issues the webhook's TLS certificate. On OpenShift the built-in service CA issues it, so nothing extra is needed.
- A cluster-admin to install. The chart creates ClusterRoles, ClusterRoleBindings, a MutatingWebhookConfiguration and the `AIBOM`/`AIBOMTelemetry` CRDs.

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

- **Upgrades don't update the CRDs.** Helm installs a chart's CRDs once and never updates them. Before upgrading, apply the new ones: `helm show crds oci://quay.io/project-aibom/aibom-webhook --version <version> | oc apply -f -`.
- **A `PYTHONPATH` set in the image is overridden.** The webhook sets `PYTHONPATH` on instrumented containers to load its detection hooks, and can't see an `ENV PYTHONPATH` baked into the image. Declare it with the `aibom.io/python-path` pod annotation, for example `/app:/app/lib`, and it's appended.
- **The webhook fails open.** If it's unreachable, pods are created uninstrumented and no AIBOM is recorded.
- **Instrumented pods pull a discovery image.** It's `quay.io/project-aibom/aibom-discovery`, about 230 MB, at the chart's version. If the cluster can't reach Quay, set `image.discovery.repository` and `.tag` to a copy it can pull; any image with `python3`, `bash`, `lscpu` and `lsblk` works.
- **Registry label lookup is off.** With `gitProvenance.registryLookup=true`, the postprocess Job reads image labels anonymously from the image's registry when the cluster has no `Image` object for it, to find the git commit the image was built from. It's off because those are outbound requests to whatever registry an image came from.

Every value is described in `values.schema.json` and in the [configuration reference](https://gavinsan33.github.io/project-aibom/aibom-webhook-service/reference/configuration/).
