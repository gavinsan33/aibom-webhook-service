FROM golang:1.26 AS builder
WORKDIR /workspace
COPY go.mod go.sum ./
RUN go mod download
COPY cmd/ cmd/
COPY internal/ internal/
RUN CGO_ENABLED=0 GOOS=linux go build -o webhook-server ./cmd/webhook/

# ubi-micro: no package manager or shell, just the base filesystem -- enough for
# a static (CGO_ENABLED=0) Go binary, and the smallest UBI base. Replaces
# distroless/static:nonroot so the image is Red Hat UBI-based (certification
# requires a UBI base; distroless can't be certified).
FROM registry.access.redhat.com/ubi9/ubi-micro@sha256:5be877a2a280ae36eb9ab80ca4efb28bbf77c57abbba38137cdc37a03b8465e6
WORKDIR /
COPY --from=builder /workspace/webhook-server /webhook-server
# Non-root; OpenShift assigns an arbitrary UID at runtime regardless.
USER 1001
ENTRYPOINT ["/webhook-server"]
