package webhook

import (
	"bytes"
	"encoding/json"
	"fmt"
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
	return NewMutator("pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime", true)
}

func newTestMutatorNoDataset() *Mutator {
	return NewMutator("pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime", false)
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

func podWithExistingPythonPath() *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod",
			Namespace: "default",
			OwnerReferences: []metav1.OwnerReference{
				{Kind: "Job", Name: "test-job"},
			},
		},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{
					Name:  "train",
					Image: "pytorch:latest",
					Env: []corev1.EnvVar{
						{Name: "PYTHONPATH", Value: "/usr/local/lib/python3.10"},
						{Name: "OTHER_VAR", Value: "keep"},
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

func TestMutate_KServePredictor_AddsInferenceServiceNameEnv(t *testing.T) {
	m := newTestMutator()
	pod := podWithGPU()
	pod.Labels = map[string]string{"serving.kserve.io/inferenceservice": "granite-model"}

	patches, err := m.Mutate(pod)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	for _, p := range patches {
		if p.Path != "/spec/initContainers" {
			continue
		}
		c := p.Value.([]corev1.Container)[0]
		for _, env := range c.Env {
			if env.Name == "INFERENCESERVICE_NAME" {
				if env.ValueFrom == nil || env.ValueFrom.FieldRef == nil {
					t.Fatalf("INFERENCESERVICE_NAME should be a downward API field ref, got %+v", env)
				}
				want := "metadata.labels['serving.kserve.io/inferenceservice']"
				if env.ValueFrom.FieldRef.FieldPath != want {
					t.Errorf("field path = %q, want %q", env.ValueFrom.FieldRef.FieldPath, want)
				}
				return
			}
		}
		t.Fatal("expected INFERENCESERVICE_NAME env var on a KServe predictor pod")
	}
	t.Fatal("init container patch not found")
}

func TestMutate_NonKServePod_NoInferenceServiceNameEnv(t *testing.T) {
	m := newTestMutator()
	patches, err := m.Mutate(podWithGPU())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	for _, p := range patches {
		if p.Path != "/spec/initContainers" {
			continue
		}
		c := p.Value.([]corev1.Container)[0]
		for _, env := range c.Env {
			if env.Name == "INFERENCESERVICE_NAME" {
				t.Error("INFERENCESERVICE_NAME should not be set for a pod without the KServe predictor label — a downward API field ref to a missing label fails pod admission")
			}
		}
		return
	}
	t.Fatal("init container patch not found")
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

func TestShouldMutate_PostprocessPod(t *testing.T) {
	m := newTestMutator()
	pod := podWithOwner("Job")
	pod.Labels = map[string]string{"aibom.io/postprocess-for": "train-job"}
	if m.shouldMutate(pod) {
		t.Error("expected a postprocess Job's own pod to be skipped despite its Job owner")
	}
}

// --- Discovery init container tests ---

func TestMutate_DiscoveryScriptCommand(t *testing.T) {
	m := newTestMutator()
	patches, err := m.Mutate(podWithOwner("Job"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	for _, p := range patches {
		if p.Path == "/spec/initContainers" {
			containers := p.Value.([]corev1.Container)
			c := containers[0]
			if c.Command[0] != "/bin/bash" {
				t.Errorf("expected /bin/bash command, got %q", c.Command[0])
			}
			if c.Args[0] != "python3 /scripts/generate_snapshot.py" {
				t.Errorf("expected python3 script command, got %q", c.Args[0])
			}
			return
		}
	}
	t.Error("init container patch not found")
}

func TestMutate_InjectsInitContainer(t *testing.T) {
	m := newTestMutator()
	patches, err := m.Mutate(podWithOwner("Job"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	for _, p := range patches {
		if p.Path == "/spec/initContainers" {
			containers, ok := p.Value.([]corev1.Container)
			if !ok {
				t.Fatal("initContainers patch value is not []Container")
			}
			if containers[0].Name != "aibom-discovery" {
				t.Errorf("expected init container name 'aibom-discovery', got %q", containers[0].Name)
			}
			if len(containers[0].Env) != 6 {
				t.Errorf("expected 6 env vars, got %d", len(containers[0].Env))
			}
			if len(containers[0].VolumeMounts) != 3 {
				t.Errorf("expected 3 volume mounts (aibom-data + aibom-scripts + aibom-token), got %d", len(containers[0].VolumeMounts))
			}
			return
		}
	}
	t.Error("initContainers patch not found")
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

	for _, p := range patches {
		if p.Path == "/spec/initContainers/-" {
			return // correct: appending
		}
		if p.Path == "/spec/initContainers" {
			t.Error("should append with /- when initContainers already exist, not replace")
		}
	}
	t.Error("expected append patch at /spec/initContainers/-")
}

// --- Volume tests ---

func TestMutate_InjectsAIBOMDataVolume(t *testing.T) {
	m := newTestMutator()
	patches, err := m.Mutate(podWithOwner("Job"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	for _, p := range patches {
		if p.Path == "/spec/volumes" {
			volumes, ok := p.Value.([]corev1.Volume)
			if !ok {
				t.Fatal("volumes patch value is not []Volume")
			}
			if volumes[0].Name != "aibom-data" {
				t.Errorf("expected first volume 'aibom-data', got %q", volumes[0].Name)
			}
			if volumes[0].EmptyDir == nil {
				t.Error("expected emptyDir volume source")
			}
			return
		}
	}
	t.Error("volumes patch not found")
}

func TestMutate_InjectsScriptsVolume(t *testing.T) {
	m := newTestMutator()
	patches, err := m.Mutate(podWithOwner("Job"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	for _, p := range patches {
		if p.Path == "/spec/volumes/-" {
			vol, ok := p.Value.(corev1.Volume)
			if !ok {
				continue
			}
			if vol.Name == "aibom-scripts" {
				if vol.ConfigMap == nil {
					t.Error("expected ConfigMap volume source")
				} else if vol.ConfigMap.Name != "aibom-scripts" {
					t.Errorf("expected ConfigMap name 'aibom-scripts', got %q", vol.ConfigMap.Name)
				}
				return
			}
		}
	}
	t.Error("aibom-scripts volume patch not found")
}

// --- Label / annotation tests ---

func TestMutate_AddsLabel(t *testing.T) {
	m := newTestMutator()
	patches, err := m.Mutate(podWithOwner("Job"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	for _, p := range patches {
		if p.Path == "/metadata/labels" {
			labels, ok := p.Value.(map[string]string)
			if !ok {
				t.Fatal("labels patch value is not map[string]string")
			}
			if labels["aibom.io/instrumented"] != "true" {
				t.Error("expected aibom.io/instrumented label")
			}
			return
		}
	}
	t.Error("expected labels patch")
}

func TestMutate_ExistingLabels(t *testing.T) {
	m := newTestMutator()
	pod := podWithOwner("Job")
	pod.Labels = map[string]string{"app": "training"}

	patches, err := m.Mutate(pod)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	for _, p := range patches {
		if p.Path == "/metadata/labels/aibom.io~1instrumented" {
			if p.Value != "true" {
				t.Errorf("expected label value 'true', got %v", p.Value)
			}
			return
		}
	}
	t.Error("expected escaped label path patch when labels already exist")
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

// --- Dataset detector tests ---

func TestMutate_InjectsDatasetDetectorEnvVars(t *testing.T) {
	m := newTestMutator()
	patches, err := m.Mutate(podWithOwner("Job"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	envVarNames := map[string]bool{}
	for _, p := range patches {
		if p.Path == "/spec/containers/0/env" {
			envs, ok := p.Value.([]corev1.EnvVar)
			if !ok {
				t.Fatal("env patch value is not []EnvVar")
			}
			for _, e := range envs {
				envVarNames[e.Name] = true
			}
		}
	}

	for _, expected := range []string{"AIBOM_DATASET_DETECT", "AIBOM_DEBUG", "AIBOM_DATASET_OUTPUT", "PYTHONPATH"} {
		if !envVarNames[expected] {
			t.Errorf("expected env var %q in dataset detector patches", expected)
		}
	}
}

// collectContainerVolumeMountPatches gathers every VolumeMount added to a
// container's volumeMounts, regardless of whether the patch replaced the
// whole array ([]VolumeMount, when the container started with none) or
// appended individual entries (VolumeMount, one patch per entry, when it
// already had some) — see buildDatasetDetectorPatches.
func collectContainerVolumeMountPatches(patches []PatchOperation, containerIdx int) []corev1.VolumeMount {
	prefix := fmt.Sprintf("/spec/containers/%d/volumeMounts", containerIdx)
	var mounts []corev1.VolumeMount
	for _, p := range patches {
		if p.Path == prefix {
			mounts = append(mounts, p.Value.([]corev1.VolumeMount)...)
		} else if p.Path == prefix+"/-" {
			mounts = append(mounts, p.Value.(corev1.VolumeMount))
		}
	}
	return mounts
}

func TestMutate_AddsTokenMountToAppContainer(t *testing.T) {
	m := newTestMutator()
	patches, err := m.Mutate(podWithOwner("Job"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	mounts := collectContainerVolumeMountPatches(patches, 0)
	if len(mounts) == 0 {
		t.Fatal("container volumeMounts patch not found")
	}
	for _, mount := range mounts {
		if mount.Name == "aibom-token" && mount.MountPath == "/var/run/secrets/kubernetes.io/serviceaccount" {
			return
		}
	}
	t.Fatal("expected aibom-token mount when the container has no existing token mount")
}

// TestMutate_SkipsTokenMountWhenAlreadyPresent guards against the real
// failure mode this exists to avoid: if automountServiceAccountToken wasn't
// disabled, the built-in ServiceAccount admission controller already mounted
// a token at this exact path before our webhook ran — a second volumeMount
// at an identical path fails pod admission outright.
func TestMutate_SkipsTokenMountWhenAlreadyPresent(t *testing.T) {
	m := newTestMutator()
	pod := podWithOwner("Job")
	pod.Spec.Containers[0].VolumeMounts = []corev1.VolumeMount{
		{Name: "kube-api-access-abcde", MountPath: "/var/run/secrets/kubernetes.io/serviceaccount", ReadOnly: true},
	}

	patches, err := m.Mutate(pod)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	mounts := collectContainerVolumeMountPatches(patches, 0)
	if len(mounts) == 0 {
		t.Fatal("container volumeMounts patch not found")
	}
	for _, mount := range mounts {
		if mount.Name == "aibom-token" {
			t.Fatal("should not add a second volumeMount at a path the container already mounts")
		}
	}
}

func TestMutate_DatasetDetectorVolumeMount(t *testing.T) {
	m := newTestMutator()
	patches, err := m.Mutate(podWithOwner("Job"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	for _, p := range patches {
		if p.Path == "/spec/containers/0/volumeMounts" {
			mounts, ok := p.Value.([]corev1.VolumeMount)
			if !ok {
				t.Fatal("volumeMounts patch value is not []VolumeMount")
			}
			foundDetector := false
			for _, mount := range mounts {
				if mount.MountPath == "/aibom-hooks/usercustomize.py" && mount.SubPath == "runtime_detector.py" {
					foundDetector = true
				}
			}
			if !foundDetector {
				t.Error("expected usercustomize.py mount with subPath runtime_detector.py")
			}
			return
		}
	}
	t.Error("container volumeMounts patch not found")
}

func TestMutate_PythonPathAppend(t *testing.T) {
	m := newTestMutator()
	patches, err := m.Mutate(podWithExistingPythonPath())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	for _, p := range patches {
		if p.Path == "/spec/containers/0/env/0/value" {
			val, ok := p.Value.(string)
			if !ok {
				t.Fatal("PYTHONPATH replace value is not string")
			}
			if val != "/aibom-hooks:/usr/local/lib/python3.10" {
				t.Errorf("expected prepended PYTHONPATH, got %q", val)
			}
			return
		}
	}
	t.Error("expected PYTHONPATH replace patch")
}

func TestMutate_DatasetDetectionDisabled(t *testing.T) {
	m := newTestMutatorNoDataset()
	patches, err := m.Mutate(podWithOwner("Job"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	for _, p := range patches {
		if p.Path == "/spec/containers/0/env" || p.Path == "/spec/containers/0/env/-" {
			t.Error("dataset detector env vars should not be injected when disabled")
		}
		if p.Path == "/spec/containers/0/volumeMounts" || p.Path == "/spec/containers/0/volumeMounts/-" {
			t.Error("dataset detector volume mounts should not be injected when disabled")
		}
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
