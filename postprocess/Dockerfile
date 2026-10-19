# ubi9/python-312-minimal: the same Red Hat UBI base the discovery image uses
# (certification requires a UBI base; python:3.12-slim is Debian and can't be
# certified). postprocess.py is otherwise dependency-free stdlib (see k8s_api.py's
# own comment on why -- it also runs inside arbitrary workload images). This image
# is fully project-controlled, unlike those, so it can afford real dependencies:
# Ed25519 signing (sign_aibom) has no stdlib equivalent, and RFC 8785 (JSON
# Canonicalization Scheme) canonicalization needs to match byte-for-byte with
# oc-aibom's Go-side verifier, which a hand-rolled sort_keys/separators convention
# can't guarantee across languages.
FROM registry.access.redhat.com/ubi9/python-312-minimal@sha256:0ee75328b9fd716773396a1e9f86a9ca8b3cde9818078c43f920d4e556f78c2e
WORKDIR /app
USER 0
RUN pip install --no-cache-dir cryptography==43.* rfc8785==0.1.*
COPY postprocess/postprocess.py .
COPY scripts/aibom-scripts/k8s_api.py .
USER 1001
ENTRYPOINT ["python3", "/app/postprocess.py"]
