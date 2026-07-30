FROM python:3.12-slim
WORKDIR /app
COPY postprocess/postprocess.py .
COPY scripts/aibom-scripts/k8s_api.py .
USER 65532:65532
ENTRYPOINT ["python3", "/app/postprocess.py"]
