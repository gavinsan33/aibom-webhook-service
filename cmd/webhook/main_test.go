package main

import "testing"

func TestOwnNamespacePrefersPodNamespaceEnv(t *testing.T) {
	t.Setenv("POD_NAMESPACE", "  aibom-system  ")
	if got := ownNamespace(); got != "aibom-system" {
		t.Fatalf("POD_NAMESPACE override: got %q, want %q", got, "aibom-system")
	}
}

func TestOwnNamespaceEmptyWhenUndeterminable(t *testing.T) {
	// No env override, and the service-account namespace file doesn't exist in
	// the test environment, so this must report "" -- which main treats as a
	// fatal, fail-fast condition rather than continuing with an empty namespace.
	t.Setenv("POD_NAMESPACE", "")
	if got := ownNamespace(); got != "" {
		t.Fatalf("expected empty namespace when undeterminable, got %q", got)
	}
}
