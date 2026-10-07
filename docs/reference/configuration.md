# Configuration

<!-- TODO: generate the chart values tables from charts/*/values.yaml -->

The webhook server accepts these flags. When deploying through the chart, set the images through the chart values rather than passing the flags directly.

| Flag | Default | Description |
|------|---------|-------------|
| `--tls-cert` | `/certs/tls.crt` | Path to the TLS certificate |
| `--tls-key` | `/certs/tls.key` | Path to the TLS private key |
| `--port` | `8443` | Server port |
| `--discovery-image` | `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime` | Image for the discovery init container. Set via `image.discovery.repository` and `.tag` in `charts/aibom-webhook/values.yaml`. It only needs `python3` and `bash`; `nvidia-smi` is injected at runtime by the NVIDIA Container Toolkit on GPU pods. You can swap in any image already available in-cluster to avoid an external pull per pod. |
| `--dataset-detection` | `true` | Inject dataset detection hooks into application containers |
| `--enable-watcher` | `true` | Start the Job completion watcher |
| `--postprocess-image` | `busybox:latest` | Image for AIBOM postprocess Jobs. This default is a placeholder only; the chart always overrides it via `image.postprocess.repository` and `.tag`. |
| `--dataset-sidecar-image` | `python:3.12-slim` | Image for the dataset-signing sidecar container |
