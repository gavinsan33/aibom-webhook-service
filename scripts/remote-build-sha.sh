#!/usr/bin/env bash
set -euo pipefail

# Prints the short SHA of the commit the in-cluster BuildConfig will actually
# build — the tip of charts/aibom-webhook/values.yaml's build.gitRepo/build.gitRef.
# Used by the justfile to default `deploy`/`redeploy`'s tag argument, since local
# HEAD can differ from that remote (unpushed or uncommitted changes). Keep the
# repo/ref below in sync with values.yaml.
REPO="https://github.com/gavinsan33/aibom-webhook-service.git"
REF="master"

git ls-remote "${REPO}" "${REF}" | cut -c1-7
