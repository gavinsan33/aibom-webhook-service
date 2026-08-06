#!/usr/bin/env bash
set -euo pipefail

# Prints the short SHA of the tip of a git ref on charts/aibom-webhook/values.yaml's
# build.gitRepo — the ref itself is either the optional $1 argument, or (if omitted)
# build.gitRef from that same values.yaml, i.e. what the in-cluster BuildConfig
# builds by default. Used by the justfile to default `deploy`'s --version argument
# (no argument) and to resolve `deploy --branch=<name>` to an immutable SHA (with
# an argument), since local HEAD can differ from the remote (unpushed or
# uncommitted changes) and a branch name alone isn't a stable, rollback-able
# version. Reads gitRepo/gitRef straight out of values.yaml instead of hardcoding
# a second copy here, so there's exactly one place that ever needs updating if
# they change.
VALUES_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/charts/aibom-webhook/values.yaml"
REPO="$(grep -m1 '^\s*gitRepo:' "${VALUES_FILE}" | sed 's/^\s*gitRepo:\s*//')"
REF="${1:-$(grep -m1 '^\s*gitRef:' "${VALUES_FILE}" | sed 's/^\s*gitRef:\s*//')}"

sha="$(git ls-remote "${REPO}" "${REF}" | cut -c1-7)"
if [[ -z "${sha}" ]]; then
    echo "error: ref '${REF}' not found on ${REPO} — has it been pushed?" >&2
    exit 1
fi
echo "${sha}"
