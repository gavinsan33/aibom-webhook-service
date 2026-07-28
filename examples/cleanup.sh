#!/bin/bash
# Cleans up every example manifest in this directory, plus any AIBOM
# postprocess jobs/configmaps left behind. Each *.yaml file's own
# metadata.namespace governs what gets deleted, so a mismatched
# NAMESPACE argument here only affects the postprocess-label cleanup.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="${1:-gavin-test}"

echo "Cleaning up example workloads..."
for f in "$SCRIPT_DIR"/*.yaml; do
  echo "  Deleting resources from $(basename "$f")..."
  oc delete -f "$f" --ignore-not-found
done

echo "Cleaning up AIBOM postprocess leftovers in namespace '$NAMESPACE'..."

echo "  Deleting postprocess jobs..."
oc delete jobs -n "$NAMESPACE" -l "aibom.io/postprocess-for" --ignore-not-found

echo "  Deleting postprocess ConfigMaps..."
oc delete configmaps -n "$NAMESPACE" -l "aibom.io/postprocess-for" --ignore-not-found

echo "Done. Re-apply an example with: oc apply -f examples/<file>.yaml"
