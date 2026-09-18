FROM python:3.12-slim
WORKDIR /app
# postprocess.py is otherwise dependency-free stdlib (see k8s_api.py's own
# comment on why -- it also runs inside arbitrary workload images). This
# image is fully project-controlled, unlike those, so it can afford real
# dependencies: Ed25519 signing (sign_aibom) has no stdlib equivalent, and
# RFC 8785 (JSON Canonicalization Scheme) canonicalization needs to match
# byte-for-byte with oc-aibom's Go-side verifier, which a hand-rolled
# sort_keys/separators convention can't guarantee across languages.
RUN pip install --no-cache-dir cryptography==43.* rfc8785==0.1.*
COPY postprocess/postprocess.py .
COPY scripts/aibom-scripts/k8s_api.py .
USER 65532:65532
ENTRYPOINT ["python3", "/app/postprocess.py"]
