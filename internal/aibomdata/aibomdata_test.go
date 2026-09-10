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

func TestWorkloadIdentityNameTrimsTrailingHyphenAfterTruncation(t *testing.T) {
	maxBase := MaxJobNameLength - len(WorkloadIdentitySuffix)
	triggerName := strings.Repeat("a", maxBase-1) + "--extra"
	got := WorkloadIdentityName(triggerName)
	base := strings.TrimSuffix(got, WorkloadIdentitySuffix)
	if strings.HasSuffix(base, "-") {
		t.Errorf("WorkloadIdentityName() = %q, base %q should not end in a hyphen", got, base)
	}
}
