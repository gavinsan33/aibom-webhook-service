BINARY_NAME := webhook-server
IMG ?= aibom-webhook-service:latest
POSTPROCESS_IMG ?= aibom-postprocess:latest
NAMESPACE ?= default
# The only place aibom-webhook's namespace is configured — the chart itself has no
# separate namespace value, it always installs into the release namespace (-n below).
WEBHOOK_NAMESPACE ?= aibom-system
# Tag used for both the BuildConfig output ImageStreamTag and the Deployment's image
# reference (see values.yaml image.*.tag). Defaults to "latest" to keep first-time
# deploys simple; pass a commit SHA (e.g. `make redeploy IMAGE_TAG=$(git rev-parse
# --short HEAD)`) to get an immutable, rollback-able tag per build instead of always
# overwriting "latest" in place.
IMAGE_TAG ?= latest

.PHONY: build test test-python run docker-build docker-push docker-build-postprocess docker-push-postprocess deploy deploy-no-crds redeploy redeploy-no-crds _rollout undeploy setup-namespace clean fmt vet

build:
	go build -v -o bin/$(BINARY_NAME) ./cmd/webhook/

test:
	go test ./internal/... -v -count=1

test-python:
	python3 -m venv .venv-test
	.venv-test/bin/pip install -q -r requirements-dev.txt
	.venv-test/bin/python -m pytest

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
	docker build -t $(POSTPROCESS_IMG) -f postprocess/Dockerfile .

docker-push-postprocess:
	docker push $(POSTPROCESS_IMG)

# -n sets both where Helm records this release AND where every resource in the chart
# gets created (templates use .Release.Namespace, not a values.yaml override) — without
# it, both would default to whatever namespace your oc context happens to be on.
# --create-namespace creates it if missing; a no-op if it already exists.
#
# Requires cert-manager installed in the cluster (see charts/aibom-webhook/templates/certificates.yaml).
deploy:
	helm upgrade --install aibom-webhook charts/aibom-webhook -n $(WEBHOOK_NAMESPACE) --create-namespace \
		--set image.webhook.tag=$(IMAGE_TAG) --set image.postprocess.tag=$(IMAGE_TAG)

# For accounts without cluster-scoped CRD/Namespace create-or-patch permission: skips the
# aiboms.aibom.io CRD and never touches the Namespace object (--create-namespace does a
# server-side apply even when the namespace already exists, which still needs patch
# permission on Namespace). Both must already exist, created once by a cluster-admin via
# oc apply -f charts/aibom-webhook/crds/aibom-crd.yaml and oc create namespace <ns>.
deploy-no-crds:
	helm upgrade --install aibom-webhook charts/aibom-webhook -n $(WEBHOOK_NAMESPACE) --skip-crds \
		--set image.webhook.tag=$(IMAGE_TAG) --set image.postprocess.tag=$(IMAGE_TAG)

redeploy: deploy _rollout

redeploy-no-crds: deploy-no-crds _rollout

_rollout:
	oc -n $(WEBHOOK_NAMESPACE) start-build aibom-webhook-service --wait
	oc -n $(WEBHOOK_NAMESPACE) start-build aibom-postprocess --wait
	oc -n $(WEBHOOK_NAMESPACE) delete pods -l openshift.io/build.name --field-selector=status.phase==Succeeded
	oc -n $(WEBHOOK_NAMESPACE) rollout restart deployment/aibom-webhook
	oc -n $(WEBHOOK_NAMESPACE) rollout status deployment/aibom-webhook --timeout=120s
	@echo "NOTE: Run 'make setup-namespace NAMESPACE=<ns>' for each workload namespace"

undeploy:
	helm uninstall aibom-webhook -n $(WEBHOOK_NAMESPACE)

# Namespace must already exist. Labels it first so it's never left instrumented by
# RBAC/ConfigMap alone without actually being opted into the webhook.
setup-namespace:
	oc label namespace $(NAMESPACE) aibom.io/enabled=true --overwrite
	helm upgrade --install aibom-ns-$(NAMESPACE) charts/aibom-workload-namespace -n $(NAMESPACE) \
		--set-file scripts.generateSnapshot=scripts/aibom-scripts/generate_snapshot.py \
		--set-file scripts.runtimeDetector=scripts/aibom-scripts/runtime_detector.py \
		--set-file scripts.k8sApi=scripts/aibom-scripts/k8s_api.py

clean:
	rm -rf bin/
	rm -rf certs/
	rm -rf .venv-test/
