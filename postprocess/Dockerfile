FROM python:3.12-slim
WORKDIR /app
COPY postprocess.py .
USER 65532:65532
ENTRYPOINT ["python3", "/app/postprocess.py"]
