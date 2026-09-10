package webhook

import (
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
	"testing"

	"github.com/gavinsan33/aibom-webhook-service/internal/aibomdata"
)

func newTestMutatorWithClientset() (*Mutator, *fake.Clientset) {
	m := newTestMutator()
	clientset := newFakeClientsetWithTokens()
	m.Clientset = clientset
	return m, clientset
}

func findVolume(patches []PatchOperation, name string) *corev1.Volume {
	for _, p := range patches {
		vols, ok := p.Value.([]corev1.Volume)
		if ok {
			for i := range vols {
				if vols[i].Name == name {
					return &vols[i]
				}
			}
			continue
		}
		vol, ok := p.Value.(corev1.Volume)
		if ok && vol.Name == name {
			return &vol
		}
	}
	return nil
}

func TestMutate_UsesPerJobIdentityTokenWhenClientsetConfigured(t *testing.T) {
	m, _ := newTestMutatorWithClientset()
	patches, err := m.Mutate(podWithOwner("Job"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	vol := findVolume(patches, "aibom-token")
	if vol == nil {
		t.Fatal("aibom-token volume patch not found")
	}
	src := vol.VolumeSource.Projected.Sources[0]
	if src.Secret == nil {
		t.Fatal("expected aibom-token's first source to be a SecretProjection when a per-job identity was provisioned")
	}
	if src.Secret.Name != aibomdata.WorkloadIdentityName("test-job") {
		t.Errorf("secret source name = %q, want %q", src.Secret.Name, aibomdata.WorkloadIdentityName("test-job"))
	}
	if src.ServiceAccountToken != nil {
		t.Error("should not fall back to a ServiceAccountTokenProjection once identity provisioning succeeds")
	}
}

func TestMutate_ProvisionsIdentityResourcesInWorkloadNamespace(t *testing.T) {
	m, clientset := newTestMutatorWithClientset()
	if _, err := m.Mutate(podWithOwner("Job")); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	name := aibomdata.WorkloadIdentityName("test-job")
	if _, err := clientset.CoreV1().ServiceAccounts("default").Get(t.Context(), name, metav1.GetOptions{}); err != nil {
		t.Errorf("expected per-job ServiceAccount to be provisioned: %v", err)
	}
	if _, err := clientset.RbacV1().Roles("default").Get(t.Context(), name, metav1.GetOptions{}); err != nil {
		t.Errorf("expected per-job Role to be provisioned: %v", err)
	}
}

func TestMutate_FallsBackToSharedTokenWhenOwnerUnknown(t *testing.T) {
	m, _ := newTestMutatorWithClientset()
	// A bare GPU pod has no matching owner, so its own name (and so its data
	// ConfigMap name) isn't known yet at admission time -- see
	// dataConfigMapEnvVar's doc comment. ensurePodWorkloadIdentity must skip
	// identity provisioning here rather than erroring.
	patches, err := m.Mutate(podWithGPU())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	vol := findVolume(patches, "aibom-token")
	if vol == nil {
		t.Fatal("aibom-token volume patch not found")
	}
	src := vol.VolumeSource.Projected.Sources[0]
	if src.ServiceAccountToken == nil {
		t.Error("expected fallback to a ServiceAccountTokenProjection for a bare pod with no matching owner")
	}
}

// TestMutate_DoesNotReplaceAppContainerTokenMountEvenWhenIdentityProvisioned
// guards the #47 behavior change: previously, a per-job identity being
// provisioned meant the app container's own pre-existing default-SA token
// mount got retargeted to it. Since the app container no longer talks to
// the Kubernetes API for anything, that's no longer necessary or done --
// its own mount, whatever it is, is left completely alone.
func TestMutate_DoesNotReplaceAppContainerTokenMountEvenWhenIdentityProvisioned(t *testing.T) {
	m, _ := newTestMutatorWithClientset()
	pod := podWithOwner("Job")
	pod.Spec.Containers[0].VolumeMounts = []corev1.VolumeMount{
		{Name: "kube-api-access-abcde", MountPath: "/var/run/secrets/kubernetes.io/serviceaccount", ReadOnly: true},
	}

	patches, err := m.Mutate(pod)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	for _, p := range patches {
		if p.Op == "replace" && p.Path == "/spec/containers/0/volumeMounts/0/name" {
			t.Errorf("unexpected replace patch %+v -- app container mounts should never be retargeted anymore", p)
		}
	}
}

// TestMutate_DatasetSidecarSharesIdentityWithDiscoveryInitContainer confirms
// the aibom-dataset-sidecar container (see #47) gets the same job-scoped
// identity (aibom-token) and its own separate dataset-signing key, mirroring
// the discovery init container's mounts.
func TestMutate_DatasetSidecarSharesIdentityWithDiscoveryInitContainer(t *testing.T) {
	m, _ := newTestMutatorWithClientset()
	patches, err := m.Mutate(podWithOwner("Job"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	sidecar := findInitContainer(patches, "aibom-dataset-sidecar")
	if sidecar == nil {
		t.Fatal("aibom-dataset-sidecar init container not found")
	}
	if sidecar.RestartPolicy == nil || *sidecar.RestartPolicy != corev1.ContainerRestartPolicyAlways {
		t.Error("expected aibom-dataset-sidecar to be a native sidecar (RestartPolicy: Always)")
	}

	var hasToken, hasDatasetKey bool
	for _, mount := range sidecar.VolumeMounts {
		if mount.Name == "aibom-token" {
			hasToken = true
		}
		if mount.Name == "aibom-dataset-signing-key" {
			hasDatasetKey = true
		}
	}
	if !hasToken {
		t.Error("expected aibom-dataset-sidecar to mount aibom-token")
	}
	if !hasDatasetKey {
		t.Error("expected aibom-dataset-sidecar to mount aibom-dataset-signing-key")
	}
}

func findInitContainer(patches []PatchOperation, name string) *corev1.Container {
	for _, p := range patches {
		if containers, ok := p.Value.([]corev1.Container); ok {
			for i := range containers {
				if containers[i].Name == name {
					return &containers[i]
				}
			}
			continue
		}
		if c, ok := p.Value.(corev1.Container); ok && c.Name == name {
			return &c
		}
	}
	return nil
}
