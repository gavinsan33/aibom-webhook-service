// Package aibomdata holds the naming convention for the per-workload data
// ConfigMap, shared between the webhook (which injects the ConfigMap name
// into workload pods so they can write discovery/dataset data directly into
// it) and the watcher (which reads/aggregates that same ConfigMap once the
// workload completes).
package aibomdata

import "strings"

const (
	MaxJobNameLength  = 63
	PostprocessSuffix = "-aibom-postprocess"
	ConfigMapSuffix   = "-data"

	// LabelPostprocessFor is set on postprocess Jobs (and now their pods) to name
	// the workload they were generated for. The webhook checks this on pods to
	// avoid re-instrumenting a postprocess Job's own pod, which would otherwise
	// derive a second-generation data ConfigMap name from the postprocess Job's
	// own name (see mutator.go's shouldMutate).
	LabelPostprocessFor = "aibom.io/postprocess-for"

	// LabelKServeInferenceService is the label KServe applies to every predictor
	// pod, naming the owning InferenceService. For a predictor pod already
	// instrumented via the requestsGPU fallback, it lets the watcher look up
	// that InferenceService to resolve model identity for storage.key/path-based
	// (S3/MinIO data-connection) deployments, which carry no CLI args to parse.
	LabelKServeInferenceService = "serving.kserve.io/inferenceservice"

	// DiscoverySigningKeySecretName is created per workload namespace by the
	// aibom-workload-namespace chart (templates/signing.yaml). The webhook
	// mounts it only into the discovery init container (never an app
	// container) so generate_snapshot.py can HMAC-sign discovery-<pod>.json;
	// the watcher reads the same Secret (via RBAC scoped to this exact name,
	// see clusterrole.yaml) to verify that signature before trusting a pod's
	// hardware data enough to merge it into the aggregate discovery.json.
	DiscoverySigningKeySecretName = "aibom-discovery-hmac-key"

	// DiscoverySigningKeyDataKey is the key within that Secret's data map.
	DiscoverySigningKeyDataKey = "hmac-key"
)

// PostprocessJobName returns the deterministic postprocess Job name for a
// given trigger name (an owning Job's name, or a bare pod's own name),
// truncated to fit Kubernetes' 63-character name limit.
func PostprocessJobName(triggerName string) string {
	maxBase := MaxJobNameLength - len(PostprocessSuffix)
	if len(triggerName) > maxBase {
		triggerName = triggerName[:maxBase]
	}
	triggerName = strings.TrimRight(triggerName, "-")
	return triggerName + PostprocessSuffix
}

// ConfigMapName returns the deterministic data ConfigMap name for a given
// trigger name, truncated to fit Kubernetes' 253-character name limit.
func ConfigMapName(triggerName string) string {
	name := PostprocessJobName(triggerName) + ConfigMapSuffix
	if len(name) > 253 {
		name = strings.TrimRight(name[:253], "-")
	}
	return name
}
