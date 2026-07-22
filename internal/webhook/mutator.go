package webhook

import (
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
	DiscoveryImage string
}

type PatchOperation struct {
	Op    string      `json:"op"`
	Path  string      `json:"path"`
	Value interface{} `json:"value,omitempty"`
}

func NewMutator(discoveryImage string) *Mutator {
	return &Mutator{DiscoveryImage: discoveryImage}
}

func (m *Mutator) Mutate(pod *corev1.Pod) ([]PatchOperation, error) {
	if !m.shouldMutate(pod) {
		return nil, nil
	}

	var patches []PatchOperation

	volume := buildAIBOMVolume()
	if len(pod.Spec.Volumes) == 0 {
		patches = append(patches, PatchOperation{
			Op:    "add",
			Path:  "/spec/volumes",
			Value: []corev1.Volume{volume},
		})
	} else {
		patches = append(patches, PatchOperation{
			Op:    "add",
			Path:  "/spec/volumes/-",
			Value: volume,
		})
	}

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
		Command: []string{"/bin/sh", "-c"},
		Args:    []string{"echo 'AIBOM discovery placeholder' && sleep 1"},
		Env: []corev1.EnvVar{
			downwardAPIEnv("POD_NAME", "metadata.name"),
			downwardAPIEnv("POD_UID", "metadata.uid"),
			downwardAPIEnv("POD_NAMESPACE", "metadata.namespace"),
			downwardAPIEnv("POD_IP", "status.podIP"),
			downwardAPIEnv("NODE_NAME", "spec.nodeName"),
		},
		VolumeMounts: []corev1.VolumeMount{
			{Name: "aibom-data", MountPath: "/tmp/aibom"},
		},
	}
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
