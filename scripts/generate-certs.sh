#!/usr/bin/env bash
set -euo pipefail

# Generate self-signed TLS certs for running the webhook locally (just run).
# Cluster deployments get their certs from cert-manager instead (see
# charts/aibom-webhook/templates/certificates.yaml) — this script never touches
# a cluster.
# Usage: ./scripts/generate-certs.sh

CERT_DIR="${CERT_DIR:-certs}"
NAMESPACE="${NAMESPACE:-aibom-system}"
SERVICE="${SERVICE:-aibom-webhook}"

SAN="DNS:${SERVICE}.${NAMESPACE}.svc,DNS:${SERVICE}.${NAMESPACE}.svc.cluster.local,DNS:localhost,IP:127.0.0.1"

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

echo ""
echo "Certs written to ${CERT_DIR}/ — run 'just run' to start the webhook against them."
