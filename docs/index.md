# AIBOM Webhook Service

A Kubernetes mutating admission webhook that automatically instruments AI workloads with AIBOM (AI Bill of Materials) metadata collection. When a pod is created in an opted-in namespace, the webhook injects hardware discovery, dataset detection, and tracking labels. No changes to the user's manifests are required.

To filter, inspect, and compare the resulting `AIBOM` custom resources, see the [`oc-aibom`](https://github.com/gavinsan33/oc-aibom) `kubectl`/`oc` plugin.

## How it works

1. An admin labels a namespace: `oc label namespace my-ns aibom.io/enabled=true`.
2. An admin installs the `aibom-workload-namespace` chart in that namespace, once per namespace. It adds the `aibom-scripts` ConfigMap, the RBAC workloads and the webhook need there, and the HMAC signing keys (see [Getting Started](user-guide/getting-started.md#enable-a-workload-namespace)).
3. A user submits a Job, JobSet, PyTorchJob, or RayJob in that namespace.
4. The Kubernetes API server calls the webhook before creating the pod.
5. The webhook injects an `aibom-discovery` init container (hardware snapshot), dataset detection hooks, and an `aibom.io/instrumented: "true"` label.
6. The pod is created with the injections. The user's original YAML is untouched.
7. When the Job completes (or is deleted, for long-running pods such as KServe predictors), the **watcher** creates a postprocess Job that compiles the AIBOM.

Pods are matched if they are owned directly by a Job or PyTorchJob, **or** if any container requests `nvidia.com/gpu` resources. JobSet pods match through their child Jobs. Ray pods are matched through a GPU request.

The webhook always fails open (`failurePolicy: Ignore`): if the service is down, pods are created normally.

## What gets injected

**Init container (`aibom-discovery`)**

- Captures CPU model, cores and cache; GPU model, count, VRAM and CUDA version; memory; network (RDMA); storage; kernel config; and cgroup limits.
- Writes the result into the workload's data ConfigMap, signed with a per-namespace HMAC key that is mounted only into this init container, never into the application container.

**Runtime detector (in each application container)**

- Loaded as `sitecustomize.py` on `PYTHONPATH`, so it runs at Python startup with no code changes.
- Hooks PyTorch `DataLoader`, HuggingFace `datasets.load_dataset`, torchvision datasets, webdataset, `transformers.TrainingArguments`, `transformers.PreTrainedModel.from_pretrained`, `peft.LoraConfig`, and the global seeding functions (`random.seed`, `numpy.random.seed`, `torch.manual_seed`), and records the versions of the ML libraries the process imported.
- Captures dataset name, version, split, fingerprint, license, and training arguments.
- Writes its result to a local file; it never talks to the Kubernetes API itself.

**Sidecar (`aibom-dataset-sidecar`)**

- A Kubernetes native sidecar that starts without blocking the application container and terminates only after every application container has exited.
- Signs the runtime detector's file with its own per-namespace HMAC key and publishes it to the data ConfigMap. It is the only component that writes dataset data there.

## Related projects

[k8s-aibom](https://github.com/GoogleCloudPlatform/k8s-aibom) is an unprivileged controller that inventories the AI workloads running in a cluster and writes one CycloneDX ML-BOM per serving workload, with evidence and a confidence tier on every attribute. It is the serving-side inventory for people who didn't run the workload; this project is the run record for the people who did.

Both projects define a kind `AIBOM` with plural `aiboms`, in different API groups (`aibom.io` here, `aibom.k8saibom.dev` there), so they can be installed on the same cluster. Use the short name `aibomrun` (or `aiboms.aibom.io`) for this project's resources; `aibom` is k8s-aibom's short name, and a bare `kubectl get aiboms` resolves to whichever CRD the API server lists first. The `oc-aibom` CLI installs as `oc aibomrun`, leaving `kubectl aibom` to k8s-aibom's krew plugin.

## Where to go next

- [Getting Started](user-guide/getting-started.md): install the webhook and enable a namespace.
- [Annotations](user-guide/annotations.md): add experiment metadata to your Jobs.
- [Troubleshooting](user-guide/troubleshooting.md): workloads running uninstrumented.
- [Detected Fields](CAPABILITIES.md): every field the pipeline can populate and how.
- [AIBOM Schema](reference/schema.md): how completed AIBOMs are stored.
- [Signing Keys](reference/signing.md): the per-namespace Secrets, rotation effects, and the GitOps caveat.
