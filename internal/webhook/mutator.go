package webhook

import (
	"fmt"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

var matchedOwnerKinds = map[string]bool{
	"Job":        true,
	"JobSet":     true,
	"PyTorchJob": true,
	"RayJob":     true,
}

type Mutator struct {
	DiscoveryImage   string
	DatasetDetection bool
}

type PatchOperation struct {
	Op    string      `json:"op"`
	Path  string      `json:"path"`
	Value interface{} `json:"value,omitempty"`
}

func NewMutator(discoveryImage string, datasetDetection bool) *Mutator {
	return &Mutator{
		DiscoveryImage:   discoveryImage,
		DatasetDetection: datasetDetection,
	}
}

func (m *Mutator) Mutate(pod *corev1.Pod) ([]PatchOperation, error) {
	if !m.shouldMutate(pod) {
		return nil, nil
	}

	var patches []PatchOperation

	// Add aibom-data emptyDir volume
	patches = appendVolume(patches, pod, buildAIBOMVolume())

	// Add aibom-scripts ConfigMap volume
	patches = appendVolume(patches, pod, buildScriptsVolume())

	// Add discovery init container
	initContainer := m.buildDiscoveryInitContainer()
	if len(pod.Spec.InitContainers) == 0 {
		patches = append(patches, PatchOperation{
			Op:    "add",
			Path:  "/spec/initContainers",
			Value: []corev1.Container{initContainer},
		})
	} else {
		patches = append(patches, PatchOperation{
			Op:    "add",
			Path:  "/spec/initContainers/-",
			Value: initContainer,
		})
	}

	// Inject dataset detector into application containers
	if m.DatasetDetection {
		for i := range pod.Spec.Containers {
			patches = append(patches, m.buildDatasetDetectorPatches(pod, i)...)
		}
	}

	// Add instrumented label
	if pod.Labels == nil {
		patches = append(patches, PatchOperation{
			Op:   "add",
			Path: "/metadata/labels",
			Value: map[string]string{
				"aibom.io/instrumented": "true",
			},
		})
	} else {
		patches = append(patches, PatchOperation{
			Op:    "add",
			Path:  "/metadata/labels/aibom.io~1instrumented",
			Value: "true",
		})
	}

	// Add instrumented-by annotation
	if pod.Annotations == nil {
		patches = append(patches, PatchOperation{
			Op:   "add",
			Path: "/metadata/annotations",
			Value: map[string]string{
				"aibom.io/instrumented-by": "webhook",
			},
		})
	} else {
		patches = append(patches, PatchOperation{
			Op:    "add",
			Path:  "/metadata/annotations/aibom.io~1instrumented-by",
			Value: "webhook",
		})
	}

	return patches, nil
}

func (m *Mutator) shouldMutate(pod *corev1.Pod) bool {
	if alreadyInstrumented(pod) {
		return false
	}
	return hasMatchingOwner(pod) || requestsGPU(pod)
}

func alreadyInstrumented(pod *corev1.Pod) bool {
	if pod.Labels == nil {
		return false
	}
	return pod.Labels["aibom.io/instrumented"] == "true"
}

func hasMatchingOwner(pod *corev1.Pod) bool {
	for _, ref := range pod.OwnerReferences {
		if matchedOwnerKinds[ref.Kind] {
			return true
		}
	}
	return false
}

func requestsGPU(pod *corev1.Pod) bool {
	gpuResource := corev1.ResourceName("nvidia.com/gpu")
	for i := range pod.Spec.Containers {
		c := &pod.Spec.Containers[i]
		if q, ok := c.Resources.Limits[gpuResource]; ok && q.Cmp(resource.MustParse("0")) > 0 {
			return true
		}
		if q, ok := c.Resources.Requests[gpuResource]; ok && q.Cmp(resource.MustParse("0")) > 0 {
			return true
		}
	}
	return false
}

func (m *Mutator) buildDiscoveryInitContainer() corev1.Container {
	return corev1.Container{
		Name:    "aibom-discovery",
		Image:   m.DiscoveryImage,
		Command: []string{"/bin/bash", "-c"},
		Args:    []string{"python3 /scripts/generate_snapshot.py"},
		Env: []corev1.EnvVar{
			downwardAPIEnv("POD_NAME", "metadata.name"),
			downwardAPIEnv("POD_UID", "metadata.uid"),
			downwardAPIEnv("POD_NAMESPACE", "metadata.namespace"),
			downwardAPIEnv("POD_IP", "status.podIP"),
			downwardAPIEnv("NODE_NAME", "spec.nodeName"),
		},
		VolumeMounts: []corev1.VolumeMount{
			{Name: "aibom-data", MountPath: "/tmp/result"},
			{Name: "aibom-scripts", MountPath: "/scripts", ReadOnly: true},
		},
	}
}

// buildDatasetDetectorPatches creates JSON patches to inject dataset detection
// into a specific application container. It adds env vars for activation and
// mounts the detector script as usercustomize.py so Python auto-imports it.
func (m *Mutator) buildDatasetDetectorPatches(pod *corev1.Pod, containerIdx int) []PatchOperation {
	var patches []PatchOperation
	container := &pod.Spec.Containers[containerIdx]

	// Build PYTHONPATH value, prepending to any existing value
	pythonPath := "/aibom-hooks"
	for _, env := range container.Env {
		if env.Name == "PYTHONPATH" && env.Value != "" {
			pythonPath = "/aibom-hooks:" + env.Value
			break
		}
	}

	envVars := []corev1.EnvVar{
		{Name: "AIBOM_DATASET_DETECT", Value: "1"},
		{Name: "AIBOM_DEBUG", Value: "1"},
		{Name: "AIBOM_DATASET_OUTPUT", Value: "/tmp/aibom/dataset_detected.json"},
		{Name: "PYTHONPATH", Value: pythonPath},
	}

	envPath := fmt.Sprintf("/spec/containers/%d/env", containerIdx)
	if len(container.Env) == 0 {
		patches = append(patches, PatchOperation{
			Op:    "add",
			Path:  envPath,
			Value: envVars,
		})
	} else {
		// If PYTHONPATH already exists, replace it; add the rest
		pythonPathExists := false
		for j, env := range container.Env {
			if env.Name == "PYTHONPATH" {
				patches = append(patches, PatchOperation{
					Op:    "replace",
					Path:  fmt.Sprintf("%s/%d/value", envPath, j),
					Value: pythonPath,
				})
				pythonPathExists = true
				break
			}
		}
		for _, env := range envVars {
			if env.Name == "PYTHONPATH" && pythonPathExists {
				continue
			}
			patches = append(patches, PatchOperation{
				Op:    "add",
				Path:  envPath + "/-",
				Value: env,
			})
		}
	}

	// Mount usercustomize.py (dataset detector) and aibom-data volume
	mounts := []corev1.VolumeMount{
		{
			Name:      "aibom-scripts",
			MountPath: "/aibom-hooks/usercustomize.py",
			SubPath:   "dataset_detector.py",
			ReadOnly:  true,
		},
		{
			Name:      "aibom-data",
			MountPath: "/tmp/aibom",
		},
	}

	mountPath := fmt.Sprintf("/spec/containers/%d/volumeMounts", containerIdx)
	if len(container.VolumeMounts) == 0 {
		patches = append(patches, PatchOperation{
			Op:    "add",
			Path:  mountPath,
			Value: mounts,
		})
	} else {
		for _, mount := range mounts {
			patches = append(patches, PatchOperation{
				Op:    "add",
				Path:  mountPath + "/-",
				Value: mount,
			})
		}
	}

	return patches
}

func downwardAPIEnv(name, fieldPath string) corev1.EnvVar {
	return corev1.EnvVar{
		Name: name,
		ValueFrom: &corev1.EnvVarSource{
			FieldRef: &corev1.ObjectFieldSelector{FieldPath: fieldPath},
		},
	}
}

func buildAIBOMVolume() corev1.Volume {
	return corev1.Volume{
		Name: "aibom-data",
		VolumeSource: corev1.VolumeSource{
			EmptyDir: &corev1.EmptyDirVolumeSource{},
		},
	}
}

func buildScriptsVolume() corev1.Volume {
	return corev1.Volume{
		Name: "aibom-scripts",
		VolumeSource: corev1.VolumeSource{
			ConfigMap: &corev1.ConfigMapVolumeSource{
				LocalObjectReference: corev1.LocalObjectReference{Name: "aibom-scripts"},
			},
		},
	}
}

// appendVolume adds a volume patch, handling nil vs existing volumes array.
// It tracks the running count so subsequent appends use the correct operation.
func appendVolume(patches []PatchOperation, pod *corev1.Pod, vol corev1.Volume) []PatchOperation {
	existingCount := len(pod.Spec.Volumes)
	// Count how many volume patches we've already added
	for _, p := range patches {
		if p.Path == "/spec/volumes" || p.Path == "/spec/volumes/-" {
			existingCount++
		}
	}

	if existingCount == 0 {
		return append(patches, PatchOperation{
			Op:    "add",
			Path:  "/spec/volumes",
			Value: []corev1.Volume{vol},
		})
	}
	return append(patches, PatchOperation{
		Op:    "add",
		Path:  "/spec/volumes/-",
		Value: vol,
	})
}
