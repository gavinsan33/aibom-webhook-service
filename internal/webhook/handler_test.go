package webhook

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	admissionv1 "k8s.io/api/admission/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

func newTestMutator() *Mutator {
	return NewMutator("busybox:latest")
}

func podWithOwner(kind string) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod",
			Namespace: "default",
			OwnerReferences: []metav1.OwnerReference{
				{Kind: kind, Name: "test-job"},
			},
		},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{Name: "train", Image: "pytorch:latest"},
			},
		},
	}
}

func podWithGPU() *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "gpu-pod",
			Namespace: "default",
		},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{
					Name:  "train",
					Image: "pytorch:latest",
					Resources: corev1.ResourceRequirements{
						Limits: corev1.ResourceList{
							"nvidia.com/gpu": resource.MustParse("1"),
						},
					},
				},
			},
		},
	}
}

func podAlreadyInstrumented() *corev1.Pod {
	pod := podWithOwner("Job")
	pod.Labels = map[string]string{"aibom.io/instrumented": "true"}
	return pod
}

func podNoMatch() *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "web-pod",
			Namespace: "default",
			OwnerReferences: []metav1.OwnerReference{
				{Kind: "Deployment", Name: "web-app"},
			},
		},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{Name: "web", Image: "nginx:latest"},
			},
		},
	}
}

// --- shouldMutate tests ---

func TestShouldMutate_JobOwner(t *testing.T) {
	m := newTestMutator()
	if !m.shouldMutate(podWithOwner("Job")) {
		t.Error("expected pod with Job owner to match")
	}
}

func TestShouldMutate_JobSetOwner(t *testing.T) {
	m := newTestMutator()
	if !m.shouldMutate(podWithOwner("JobSet")) {
		t.Error("expected pod with JobSet owner to match")
	}
}

func TestShouldMutate_PyTorchJobOwner(t *testing.T) {
	m := newTestMutator()
	if !m.shouldMutate(podWithOwner("PyTorchJob")) {
		t.Error("expected pod with PyTorchJob owner to match")
	}
}

func TestShouldMutate_RayJobOwner(t *testing.T) {
	m := newTestMutator()
	if !m.shouldMutate(podWithOwner("RayJob")) {
		t.Error("expected pod with RayJob owner to match")
	}
}

func TestShouldMutate_GPURequest(t *testing.T) {
	m := newTestMutator()
	if !m.shouldMutate(podWithGPU()) {
		t.Error("expected pod with GPU request to match")
	}
}

func TestShouldMutate_AlreadyInstrumented(t *testing.T) {
	m := newTestMutator()
	if m.shouldMutate(podAlreadyInstrumented()) {
		t.Error("expected already-instrumented pod to be skipped")
	}
}

func TestShouldMutate_NoMatch(t *testing.T) {
	m := newTestMutator()
	if m.shouldMutate(podNoMatch()) {
		t.Error("expected Deployment-owned pod without GPU to be skipped")
	}
}

// --- Mutate tests ---

func TestMutate_InjectsInitContainer(t *testing.T) {
	m := newTestMutator()
	patches, err := m.Mutate(podWithOwner("Job"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	found := false
	for _, p := range patches {
		if p.Path == "/spec/initContainers" {
			found = true
			containers, ok := p.Value.([]corev1.Container)
			if !ok {
				t.Fatal("initContainers patch value is not []Container")
			}
			if containers[0].Name != "aibom-discovery" {
				t.Errorf("expected init container name 'aibom-discovery', got %q", containers[0].Name)
			}
			if len(containers[0].Env) != 5 {
				t.Errorf("expected 5 env vars, got %d", len(containers[0].Env))
			}
		}
	}
	if !found {
		t.Error("expected initContainers patch")
	}
}

func TestMutate_InjectsVolume(t *testing.T) {
	m := newTestMutator()
	patches, err := m.Mutate(podWithOwner("Job"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	found := false
	for _, p := range patches {
		if p.Path == "/spec/volumes" {
			found = true
			volumes, ok := p.Value.([]corev1.Volume)
			if !ok {
				t.Fatal("volumes patch value is not []Volume")
			}
			if volumes[0].Name != "aibom-data" {
				t.Errorf("expected volume name 'aibom-data', got %q", volumes[0].Name)
			}
			if volumes[0].EmptyDir == nil {
				t.Error("expected emptyDir volume source")
			}
		}
	}
	if !found {
		t.Error("expected volumes patch")
	}
}

func TestMutate_AddsLabel(t *testing.T) {
	m := newTestMutator()
	patches, err := m.Mutate(podWithOwner("Job"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	found := false
	for _, p := range patches {
		if p.Path == "/metadata/labels" {
			found = true
			labels, ok := p.Value.(map[string]string)
			if !ok {
				t.Fatal("labels patch value is not map[string]string")
			}
			if labels["aibom.io/instrumented"] != "true" {
				t.Error("expected aibom.io/instrumented label")
			}
		}
	}
	if !found {
		t.Error("expected labels patch")
	}
}

func TestMutate_ExistingLabels(t *testing.T) {
	m := newTestMutator()
	pod := podWithOwner("Job")
	pod.Labels = map[string]string{"app": "training"}

	patches, err := m.Mutate(pod)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	found := false
	for _, p := range patches {
		if p.Path == "/metadata/labels/aibom.io~1instrumented" {
			found = true
			if p.Value != "true" {
				t.Errorf("expected label value 'true', got %v", p.Value)
			}
		}
	}
	if !found {
		t.Error("expected escaped label path patch when labels already exist")
	}
}

func TestMutate_ExistingInitContainers(t *testing.T) {
	m := newTestMutator()
	pod := podWithOwner("Job")
	pod.Spec.InitContainers = []corev1.Container{
		{Name: "existing-init", Image: "busybox"},
	}

	patches, err := m.Mutate(pod)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	found := false
	for _, p := range patches {
		if p.Path == "/spec/initContainers/-" {
			found = true
		}
		if p.Path == "/spec/initContainers" {
			t.Error("should append with /- when initContainers already exist, not replace")
		}
	}
	if !found {
		t.Error("expected append patch at /spec/initContainers/-")
	}
}

func TestMutate_NoMutationNeeded(t *testing.T) {
	m := newTestMutator()
	patches, err := m.Mutate(podNoMatch())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if patches != nil {
		t.Errorf("expected nil patches for non-matching pod, got %d", len(patches))
	}
}

// --- Handler round-trip tests ---

func buildAdmissionReview(pod *corev1.Pod) admissionv1.AdmissionReview {
	podBytes, _ := json.Marshal(pod)
	return admissionv1.AdmissionReview{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "admission.k8s.io/v1",
			Kind:       "AdmissionReview",
		},
		Request: &admissionv1.AdmissionRequest{
			UID: "test-uid",
			Resource: metav1.GroupVersionResource{
				Group: "", Version: "v1", Resource: "pods",
			},
			Object: runtime.RawExtension{Raw: podBytes},
		},
	}
}

func TestHandleAdmission_MutatesPod(t *testing.T) {
	h := NewHandler(newTestMutator())
	review := buildAdmissionReview(podWithOwner("Job"))

	body, _ := json.Marshal(review)
	req := httptest.NewRequest(http.MethodPost, "/mutate", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()

	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}

	var resp admissionv1.AdmissionReview
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to unmarshal response: %v", err)
	}

	if !resp.Response.Allowed {
		t.Error("expected Allowed=true")
	}
	if resp.Response.Patch == nil {
		t.Error("expected non-nil patch for matching pod")
	}
	if resp.Response.PatchType == nil || *resp.Response.PatchType != admissionv1.PatchTypeJSONPatch {
		t.Error("expected JSONPatch patch type")
	}
}

func TestHandleAdmission_NoMutationForDeployment(t *testing.T) {
	h := NewHandler(newTestMutator())
	review := buildAdmissionReview(podNoMatch())

	body, _ := json.Marshal(review)
	req := httptest.NewRequest(http.MethodPost, "/mutate", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()

	h.ServeHTTP(rr, req)

	var resp admissionv1.AdmissionReview
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to unmarshal response: %v", err)
	}

	if !resp.Response.Allowed {
		t.Error("expected Allowed=true")
	}
	if resp.Response.Patch != nil {
		t.Error("expected nil patch for non-matching pod")
	}
}

func TestHandleAdmission_WrongMethod(t *testing.T) {
	h := NewHandler(newTestMutator())
	req := httptest.NewRequest(http.MethodGet, "/mutate", nil)
	rr := httptest.NewRecorder()

	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusMethodNotAllowed {
		t.Errorf("expected 405, got %d", rr.Code)
	}
}

func TestHandleAdmission_WrongContentType(t *testing.T) {
	h := NewHandler(newTestMutator())
	req := httptest.NewRequest(http.MethodPost, "/mutate", bytes.NewReader([]byte("{}")))
	req.Header.Set("Content-Type", "text/plain")
	rr := httptest.NewRecorder()

	h.ServeHTTP(rr, req)

	if rr.Code != http.StatusUnsupportedMediaType {
		t.Errorf("expected 415, got %d", rr.Code)
	}
}
