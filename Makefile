BINARY_NAME := webhook-server
IMG ?= aibom-webhook-service:latest
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
	oc apply -f deploy/namespace.yaml
	oc apply -f deploy/rbac.yaml
	oc apply -f deploy/deployment.yaml
	oc apply -f deploy/webhook-config.yaml

redeploy:
	oc apply -f deploy/rbac.yaml
	oc apply -f deploy/deployment.yaml
	oc apply -f deploy/webhook-config.yaml
	oc -n aibom-system rollout restart deployment/aibom-webhook

undeploy:
	oc delete -f deploy/webhook-config.yaml --ignore-not-found
	oc delete -f deploy/deployment.yaml --ignore-not-found
	oc delete -f deploy/rbac.yaml --ignore-not-found
	oc delete -f deploy/namespace.yaml --ignore-not-found

create-scripts-configmap:
	oc create configmap aibom-scripts \
		--from-file=generate_snapshot.py=scripts/aibom-scripts/generate_snapshot.py \
		--from-file=dataset_detector.py=scripts/aibom-scripts/dataset_detector.py \
		-n $(NAMESPACE) --dry-run=client -o yaml | oc apply -f -

clean:
	rm -rf bin/
	rm -rf certs/
