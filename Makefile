BINARY_NAME := webhook-server
IMG ?= aibom-webhook-service:latest

.PHONY: build test run docker-build docker-push deploy undeploy generate-certs clean fmt vet

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
	kubectl apply -f deploy/namespace.yaml
	kubectl apply -f deploy/rbac.yaml
	kubectl apply -f deploy/deployment.yaml
	kubectl apply -f deploy/webhook-config.yaml

undeploy:
	kubectl delete -f deploy/webhook-config.yaml --ignore-not-found
	kubectl delete -f deploy/deployment.yaml --ignore-not-found
	kubectl delete -f deploy/rbac.yaml --ignore-not-found
	kubectl delete -f deploy/namespace.yaml --ignore-not-found

clean:
	rm -rf bin/
	rm -rf certs/
