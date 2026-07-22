#!/usr/bin/env bash
set -euo pipefail

# Generate self-signed TLS certs for the AIBOM webhook service.
# Usage: ./scripts/generate-certs.sh [--local]
#   --local: also add localhost to SAN (for local dev testing)

CERT_DIR="${CERT_DIR:-certs}"
NAMESPACE="${NAMESPACE:-aibom-system}"
SERVICE="${SERVICE:-aibom-webhook}"
SECRET_NAME="${SECRET_NAME:-aibom-webhook-certs}"

SAN="DNS:${SERVICE}.${NAMESPACE}.svc,DNS:${SERVICE}.${NAMESPACE}.svc.cluster.local"
if [[ "${1:-}" == "--local" ]]; then
    SAN="${SAN},DNS:localhost,IP:127.0.0.1"
fi

mkdir -p "${CERT_DIR}"

# Generate CA
openssl genrsa -out "${CERT_DIR}/ca.key" 2048 2>/dev/null
openssl req -x509 -new -nodes \
    -key "${CERT_DIR}/ca.key" \
    -sha256 -days 365 \
    -out "${CERT_DIR}/ca.crt" \
    -subj "/CN=aibom-webhook-ca"

# Generate server cert
openssl genrsa -out "${CERT_DIR}/tls.key" 2048 2>/dev/null
openssl req -new \
    -key "${CERT_DIR}/tls.key" \
    -out "${CERT_DIR}/server.csr" \
    -subj "/CN=${SERVICE}.${NAMESPACE}.svc" \
    -addext "subjectAltName=${SAN}"

openssl x509 -req \
    -in "${CERT_DIR}/server.csr" \
    -CA "${CERT_DIR}/ca.crt" \
    -CAkey "${CERT_DIR}/ca.key" \
    -CAcreateserial \
    -out "${CERT_DIR}/tls.crt" \
    -days 365 -sha256 \
    -copy_extensions copy

# Output CA bundle for webhook config
CA_BUNDLE=$(base64 -w0 < "${CERT_DIR}/ca.crt")
echo ""
echo "CA bundle (paste into deploy/webhook-config.yaml caBundle field):"
echo "${CA_BUNDLE}"
echo ""

# Detect CLI (prefer oc, fall back to kubectl)
CLI="$(command -v oc 2>/dev/null || command -v kubectl 2>/dev/null || true)"

if [[ -n "${CLI}" ]] && "${CLI}" cluster-info &>/dev/null 2>&1; then
    "${CLI}" create namespace "${NAMESPACE}" --dry-run=client -o yaml | "${CLI}" apply -f -
    "${CLI}" -n "${NAMESPACE}" create secret tls "${SECRET_NAME}" \
        --cert="${CERT_DIR}/tls.crt" \
        --key="${CERT_DIR}/tls.key" \
        --dry-run=client -o yaml | "${CLI}" apply -f -
    echo "Secret ${SECRET_NAME} created in namespace ${NAMESPACE}"

    # Patch the webhook config with the CA bundle
    if [[ -f deploy/webhook-config.yaml ]]; then
        sed -i "s|caBundle:.*|caBundle: ${CA_BUNDLE}|" deploy/webhook-config.yaml
        echo "Patched deploy/webhook-config.yaml with CA bundle"
    fi
else
    echo "oc/kubectl not available or cluster not reachable — skipping secret creation"
    echo "Manually create the secret and patch webhook-config.yaml with the CA bundle above"
fi
