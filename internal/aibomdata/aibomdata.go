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
