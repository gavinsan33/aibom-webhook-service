package aibomdata

import (
	"strings"
	"testing"
)

func TestWorkloadIdentityName(t *testing.T) {
	got := WorkloadIdentityName("my-training-job")
	want := "my-training-job" + WorkloadIdentitySuffix
	if got != want {
		t.Errorf("WorkloadIdentityName() = %q, want %q", got, want)
	}
}

func TestWorkloadIdentityNameTruncatesLongTriggerNames(t *testing.T) {
	longName := strings.Repeat("a", 100)
	got := WorkloadIdentityName(longName)
	if len(got) > MaxJobNameLength {
		t.Errorf("WorkloadIdentityName() length = %d, want <= %d", len(got), MaxJobNameLength)
	}
	if !strings.HasSuffix(got, WorkloadIdentitySuffix) {
		t.Errorf("WorkloadIdentityName() = %q, want suffix %q", got, WorkloadIdentitySuffix)
	}
}

func TestConfigMapNameAndWorkloadIdentityNameShareTruncatedBase(t *testing.T) {
	// The two must truncate a long trigger name to the exact same base (see
	// truncatedTriggerBase's comment) -- otherwise a job's identity cleanup
	// could delete the ServiceAccount/Role/RoleBinding/Secret of a different
	// job that happens to collide on WorkloadIdentityName alone. This is
	// also the exact invariant scripts/aibom-scripts/k8s_api.py's
	// resolve_data_configmap_name() must independently reproduce for
	// bare/ReplicaSet-owned pods (see aibomdata.go's truncatedTriggerBase
	// comment) -- a past mismatch there silently broke discovery data for
	// every such pod.
	longName := strings.Repeat("a", 100)
	cmBase := strings.TrimSuffix(strings.TrimSuffix(ConfigMapName(longName), ConfigMapSuffix), PostprocessSuffix)
	identityBase := strings.TrimSuffix(WorkloadIdentityName(longName), WorkloadIdentitySuffix)
	if cmBase != identityBase {
		t.Errorf("ConfigMapName base = %q, WorkloadIdentityName base = %q, want equal", cmBase, identityBase)
	}
}

func TestWorkloadIdentityNameTrimsTrailingHyphenAfterTruncation(t *testing.T) {
	maxBase := MaxJobNameLength - len(WorkloadIdentitySuffix)
	triggerName := strings.Repeat("a", maxBase-1) + "--extra"
	got := WorkloadIdentityName(triggerName)
	base := strings.TrimSuffix(got, WorkloadIdentitySuffix)
	if strings.HasSuffix(base, "-") {
		t.Errorf("WorkloadIdentityName() = %q, base %q should not end in a hyphen", got, base)
	}
}
