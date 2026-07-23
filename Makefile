BINARY_NAME := webhook-server
IMG ?= aibom-webhook-service:latest
POSTPROCESS_IMG ?= aibom-postprocess:latest
NAMESPACE ?= default

.PHONY: build test run docker-build docker-push docker-build-postprocess docker-push-postprocess deploy undeploy generate-certs create-scripts-configmap clean fmt vet

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

docker-build-postprocess:
	docker build -t $(POSTPROCESS_IMG) postprocess/

docker-push-postprocess:
	docker push $(POSTPROCESS_IMG)

generate-certs:
	./scripts/generate-certs.sh

deploy: generate-certs
	oc apply -f deploy/namespace.yaml
	oc apply -f deploy/rbac.yaml
	oc apply -f deploy/deployment.yaml
	oc apply -f deploy/webhook-config.yaml

redeploy:
	oc apply -f deploy/rbac.yaml
	oc apply -f deploy/build.yaml
	oc apply -f deploy/deployment.yaml
	oc apply -f deploy/webhook-config.yaml
	oc -n aibom-system start-build aibom-webhook-service --wait
	oc -n aibom-system start-build aibom-postprocess --wait
	oc -n aibom-system delete pods -l openshift.io/build.name --field-selector=status.phase==Succeeded
	oc -n aibom-system rollout restart deployment/aibom-webhook
	oc -n aibom-system rollout status deployment/aibom-webhook --timeout=120s
	@echo "NOTE: Run 'make setup-namespace NAMESPACE=<ns>' for each workload namespace"

undeploy:
	oc delete -f deploy/webhook-config.yaml --ignore-not-found
	oc delete -f deploy/deployment.yaml --ignore-not-found
	oc delete -f deploy/rbac.yaml --ignore-not-found
	oc delete -f deploy/namespace.yaml --ignore-not-found

setup-namespace:
	oc label namespace $(NAMESPACE) aibom.io/enabled=true --overwrite
	oc policy add-role-to-group system:image-puller system:serviceaccounts:$(NAMESPACE) -n aibom-system
	oc create configmap aibom-scripts \
		--from-file=generate_snapshot.py=scripts/aibom-scripts/generate_snapshot.py \
		--from-file=dataset_detector.py=scripts/aibom-scripts/dataset_detector.py \
		-n $(NAMESPACE) --dry-run=client -o yaml | oc apply -f -

clean:
	rm -rf bin/
	rm -rf certs/
