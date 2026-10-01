package webhook

import (
	"testing"

	"github.com/gavinsan33/aibom-webhook-service/internal/aibomdata"
	corev1 "k8s.io/api/core/v1"
)

func kservePod(sourceURI string) *corev1.Pod {
	pod := gpuPod(gpuContainer("kserve-container", 1, true))
	pod.Labels = map[string]string{aibomdata.LabelKServeInferenceService: "qwen-32b"}
	if sourceURI != "" {
		pod.Annotations = map[string]string{aibomdata.AnnotationKServeStorageSourceURI: sourceURI}
	}
	return pod
}

func TestModelSourcePVC(t *testing.T) {
	tests := []struct {
		name      string
		pod       *corev1.Pod
		wantClaim string
		wantSub   string
		wantOK    bool
	}{
		{"claim and subpath", kservePod("pvc://llm-serving-storage/qwen-32b"), "llm-serving-storage", "qwen-32b", true},
		{"nested subpath, trailing slash", kservePod("pvc://c/models/qwen/"), "c", "models/qwen", true},
		{"claim only", kservePod("pvc://c"), "c", "", true},
		{"s3 source", kservePod("s3://bucket/model"), "", "", false},
		{"no annotation", kservePod(""), "", "", false},
		{"path traversal", kservePod("pvc://c/../other"), "", "", false},
		{"empty claim", kservePod("pvc:///x"), "", "", false},
		{"not a kserve predictor", func() *corev1.Pod {
			p := kservePod("pvc://c/x")
			p.Labels = nil
			return p
		}(), "", "", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			claim, sub, ok := modelSourcePVC(tt.pod)
			if claim != tt.wantClaim || sub != tt.wantSub || ok != tt.wantOK {
				t.Errorf("got (%q, %q, %v), want (%q, %q, %v)", claim, sub, ok, tt.wantClaim, tt.wantSub, tt.wantOK)
			}
		})
	}
}

func TestMutate_MountsModelPVCReadOnlyIntoDiscoveryOnly(t *testing.T) {
	m := newTestMutator()
	patches, err := m.Mutate(kservePod("pvc://llm-serving-storage/qwen-32b"), "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	c := findInitContainer(patches, "aibom-discovery")
	if c == nil {
		t.Fatal("aibom-discovery init container not found")
	}
	var found bool
	for _, vm := range c.VolumeMounts {
		if vm.Name == modelSourceVolumeName {
			found = true
			if !vm.ReadOnly || vm.SubPath != "qwen-32b" || vm.MountPath != modelSourceMountPath {
				t.Errorf("unexpected model mount: %+v", vm)
			}
		}
	}
	if !found {
		t.Fatal("model source volume not mounted into the discovery init container")
	}
	var envOK bool
	for _, e := range c.Env {
		if e.Name == "AIBOM_MODEL_DIR" && e.Value == modelSourceMountPath {
			envOK = true
		}
	}
	if !envOK {
		t.Error("AIBOM_MODEL_DIR not set on discovery init container")
	}
}

func TestMutate_NoModelPVCForNonPVCPod(t *testing.T) {
	m := newTestMutator()
	patches, err := m.Mutate(kservePod("s3://bucket/model"), "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	c := findInitContainer(patches, "aibom-discovery")
	for _, vm := range c.VolumeMounts {
		if vm.Name == modelSourceVolumeName {
			t.Error("model source volume must not be mounted for a non-pvc source")
		}
	}
}
