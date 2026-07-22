BINARY_NAME := webhook-server
IMG ?= aibom-webhook-service:latest
CLI ?= $(shell command -v oc 2>/dev/null || command -v $(CLI) 2>/dev/null)

NAMESPACE ?= default

.PHONY: build test run docker-build docker-push deploy undeploy generate-certs create-scripts-configmap clean fmt vet

build:
	go build -v -o bin/$(BINARY_NAME) ./cmd/webhook/

test:
	go test ./internal/... -v -count=1

run: build
	./bin/$(BINARY_NAME) --tls-cert=certs/tls.crt --tls-key=certs/tls.key --port=8443

fmt:
	go fmt ./...

vet:
	go vet ./...

docker-build:
	docker build -t $(IMG) .

docker-push:
	docker push $(IMG)

generate-certs:
	./scripts/generate-certs.sh

deploy: generate-certs
	$(CLI) apply -f deploy/namespace.yaml
	$(CLI) apply -f deploy/rbac.yaml
	$(CLI) apply -f deploy/deployment.yaml
	$(CLI) apply -f deploy/webhook-config.yaml

undeploy:
	$(CLI) delete -f deploy/webhook-config.yaml --ignore-not-found
	$(CLI) delete -f deploy/deployment.yaml --ignore-not-found
	$(CLI) delete -f deploy/rbac.yaml --ignore-not-found
	$(CLI) delete -f deploy/namespace.yaml --ignore-not-found

create-scripts-configmap:
	$(CLI) create configmap aibom-scripts \
		--from-file=generate_snapshot.py=scripts/aibom-scripts/generate_snapshot.py \
		--from-file=dataset_detector.py=scripts/aibom-scripts/dataset_detector.py \
		-n $(NAMESPACE) --dry-run=client -o yaml | $(CLI) apply -f -

clean:
	rm -rf bin/
	rm -rf certs/
