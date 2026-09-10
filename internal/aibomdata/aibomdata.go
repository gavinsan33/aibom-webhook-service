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

	// WorkloadIdentitySuffix names the per-job ServiceAccount/Role/
	// RoleBinding/Secret the webhook provisions at admission time (see
	// internal/webhook/identity.go's ensureWorkloadIdentity) so the
	// discovery init container -- and, where the app container's own
	// standard-path token mount can be safely replaced, the app container
	// too -- authenticate as an identity scoped via resourceNames to
	// exactly this job's own data ConfigMap, instead of sharing whatever
	// ServiceAccount the pod runs as (which also carries any image-pull or
	// cloud IAM federation that identity needs, so it can't just be
	// overridden). Only ever created for Job-owned pods, where triggerName
	// is known at admission; bare GPU pods (e.g. KServe predictors) fall
	// back to the broader namespace-wide aibom-workload-data Role, since
	// their final pod name -- and so the ConfigMap this Role would need to
	// name -- doesn't exist yet at admission time.
	WorkloadIdentitySuffix = "-aibom-workload-identity"
)

// truncatedTriggerBase truncates triggerName to the narrowest budget any of
// PostprocessJobName/ConfigMapName/WorkloadIdentityName need (i.e. the
// longest suffix among them, currently WorkloadIdentitySuffix), and applies
// it uniformly. All three names are derived from this single shared base
// rather than each re-truncating triggerName to their own suffix's budget:
// otherwise two distinct trigger names sharing a prefix up to the
// shorter-suffix cutoff but differing beyond it would produce the same
// WorkloadIdentityName while getting different ConfigMapNames -- letting
// one job's identity cleanup (see watcher.go's collectAIBOM) delete the
// ServiceAccount/Role/RoleBinding/Secret out from under an unrelated job
// that happens to collide on the truncated name.
func truncatedTriggerBase(triggerName string) string {
	maxBase := MaxJobNameLength - len(WorkloadIdentitySuffix)
	if len(triggerName) > maxBase {
		triggerName = triggerName[:maxBase]
	}
	return strings.TrimRight(triggerName, "-")
}

// PostprocessJobName returns the deterministic postprocess Job name for a
// given trigger name (an owning Job's name, or a bare pod's own name),
// truncated to fit Kubernetes' 63-character name limit.
func PostprocessJobName(triggerName string) string {
	return truncatedTriggerBase(triggerName) + PostprocessSuffix
}

// ConfigMapName returns the deterministic data ConfigMap name for a given
// trigger name, truncated to fit Kubernetes' 253-character name limit.
func ConfigMapName(triggerName string) string {
	name := truncatedTriggerBase(triggerName) + PostprocessSuffix + ConfigMapSuffix
	if len(name) > 253 {
		name = strings.TrimRight(name[:253], "-")
	}
	return name
}

// WorkloadIdentityName returns the deterministic name for the per-job
// ServiceAccount/Role/RoleBinding/Secret quartet, truncated to fit
// Kubernetes' 63-character name limit (ServiceAccount names, like Job
// names, are also used as label values elsewhere).
func WorkloadIdentityName(triggerName string) string {
	return truncatedTriggerBase(triggerName) + WorkloadIdentitySuffix
}
