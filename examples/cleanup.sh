#!/bin/bash
set -euo pipefail

NAMESPACE="${1:-gavin-test}"
JOBSET="${2:-aibom-vllm-benchmark}"

echo "Cleaning up '$JOBSET' in namespace '$NAMESPACE'..."

echo "  Deleting JobSet..."
oc delete jobset "$JOBSET" -n "$NAMESPACE" --ignore-not-found

echo "  Deleting postprocess jobs..."
oc delete jobs -n "$NAMESPACE" -l "aibom.io/postprocess-for" --ignore-not-found

echo "  Deleting postprocess ConfigMaps..."
oc delete configmaps -n "$NAMESPACE" -l "aibom.io/postprocess-for" --ignore-not-found

echo "Done. Re-apply with: oc apply -f examples/vllm-inference.yaml"
