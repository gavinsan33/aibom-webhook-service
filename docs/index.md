# AIBOM Webhook Service

A Kubernetes mutating admission webhook that automatically instruments AI workloads with AIBOM (AI Bill of Materials) metadata collection. When a pod is created in an opted-in namespace, the webhook injects hardware discovery, dataset detection, and tracking labels. No changes to the user's manifests are required.

To filter, inspect, and compare the resulting `AIBOM` custom resources, see the [`oc-aibom`](https://github.com/gavinsan33/oc-aibom) `kubectl`/`oc` plugin.

## How it works

1. An admin labels a namespace: `oc label namespace my-ns aibom.io/enabled=true`.
2. An admin creates the `aibom-scripts` ConfigMap in that namespace (see [Getting Started](user-guide/getting-started.md)).
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
- Runs benchmarks: CPU compute, memory bandwidth, disk I/O throughput, and context switch latency.
- Writes the result into the workload's data ConfigMap, signed with a per-namespace HMAC key that is mounted only into this init container, never into the application container.

**Runtime detector (in each application container)**

- Loaded as `sitecustomize.py` on `PYTHONPATH`, so it runs at Python startup with no code changes.
- Hooks PyTorch `DataLoader`, HuggingFace `datasets.load_dataset`, torchvision datasets, webdataset, `transformers.TrainingArguments`, `transformers.PreTrainedModel.from_pretrained`, and `peft.LoraConfig`.
- Captures dataset name, version, split, fingerprint, license, and training arguments.
- Writes its result to a local file; it never talks to the Kubernetes API itself.

**Sidecar (`aibom-dataset-sidecar`)**

- A Kubernetes native sidecar that starts without blocking the application container and terminates only after every application container has exited.
- Signs the runtime detector's file with its own per-namespace HMAC key and publishes it to the data ConfigMap. It is the only component that writes dataset data there.

## Where to go next

- [Getting Started](user-guide/getting-started.md): install the webhook and enable a namespace.
- [Annotations](user-guide/annotations.md): add experiment metadata to your Jobs.
- [Troubleshooting](user-guide/troubleshooting.md): workloads running uninstrumented.
- [Detected Fields](CAPABILITIES.md): every field the pipeline can populate and how.
- [AIBOM Schema](reference/schema.md): how completed AIBOMs are stored.
