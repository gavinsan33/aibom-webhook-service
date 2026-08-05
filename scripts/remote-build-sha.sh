#!/usr/bin/env bash
set -euo pipefail

# Prints the short SHA of the commit the in-cluster BuildConfig will actually
# build — the tip of charts/aibom-webhook/values.yaml's build.gitRepo/build.gitRef.
# Used by the justfile to default `deploy`'s --version argument, since local HEAD
# can differ from that remote (unpushed or uncommitted changes). Reads gitRepo/
# gitRef straight out of values.yaml instead of hardcoding a second copy here, so
# there's exactly one place that ever needs updating if they change.
VALUES_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/charts/aibom-webhook/values.yaml"
REPO="$(grep -m1 '^\s*gitRepo:' "${VALUES_FILE}" | sed 's/^\s*gitRepo:\s*//')"
REF="$(grep -m1 '^\s*gitRef:' "${VALUES_FILE}" | sed 's/^\s*gitRef:\s*//')"

git ls-remote "${REPO}" "${REF}" | cut -c1-7
