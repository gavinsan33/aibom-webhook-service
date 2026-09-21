package webhook

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func signingPublicKeyConfigMap(namespace string) *corev1.ConfigMap {
	return &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "aibom-compiled-signing-public-key",
			Namespace: namespace,
		},
		Data: map[string]string{"ed25519-public-key": "attacker-controlled-value"},
	}
}

func TestValidateSigningPublicKeyConfigMap_AllowsTrustedIdentity(t *testing.T) {
	cm := signingPublicKeyConfigMap("ml-team")
	reason := ValidateSigningPublicKeyConfigMap(cm, "system:serviceaccount:ml-team:aibom-postprocess")
	if reason != "" {
		t.Fatalf("expected allow (empty reason) for trusted identity, got %q", reason)
	}
}

func TestValidateSigningPublicKeyConfigMap_DeniesUntrustedRequester(t *testing.T) {
	cm := signingPublicKeyConfigMap("ml-team")
	reason := ValidateSigningPublicKeyConfigMap(cm, "system:serviceaccount:ml-team:some-training-pod-sa")
	if reason == "" {
		t.Fatal("expected a denial reason for an untrusted requester, got allow")
	}
}

func TestValidateSigningPublicKeyConfigMap_DeniesCrossNamespaceIdentity(t *testing.T) {
	// aibom-postprocess in a *different* namespace must not be trusted to
	// write this namespace's anchor -- each namespace's postprocess Job
	// should only ever be able to publish its own.
	cm := signingPublicKeyConfigMap("ml-team")
	reason := ValidateSigningPublicKeyConfigMap(cm, "system:serviceaccount:other-team:aibom-postprocess")
	if reason == "" {
		t.Fatal("expected a denial reason for a cross-namespace aibom-postprocess identity, got allow")
	}
}

func TestValidateSigningPublicKeyConfigMap_IgnoresOtherConfigMaps(t *testing.T) {
	cm := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "train-job-aibom-postprocess-data",
			Namespace: "ml-team",
		},
	}
	reason := ValidateSigningPublicKeyConfigMap(cm, "system:serviceaccount:ml-team:some-training-pod-sa")
	if reason != "" {
		t.Fatalf("expected allow (empty reason) for an unrelated ConfigMap name, got %q", reason)
	}
}
