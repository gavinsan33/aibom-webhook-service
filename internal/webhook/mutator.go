package webhook

import (
	"fmt"

	"github.com/gavinsan33/aibom-webhook-service/internal/aibomdata"
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

	// Add our own Kubernetes API token volume — see buildTokenVolume's doc
	// comment for why the pod's own (possibly absent) automounted token
	// can't be relied on for containers this webhook adds.
	patches = appendVolume(patches, pod, buildTokenVolume())

	// Add discovery init container
	initContainer := m.buildDiscoveryInitContainer(pod)
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
	if alreadyInstrumented(pod) || isPostprocessPod(pod) {
		return false
	}
	return hasMatchingOwner(pod) || requestsGPU(pod)
}

// isPostprocessPod reports whether this pod belongs to a postprocess Job
// itself (labeled by the watcher via its pod template, see watcher.go's
// createPostprocessJobCore). Without this check, the postprocess Job's own
// pod — owned by a plain batch/v1 Job like any other matched workload — would
// get instrumented too, deriving a second-generation, truncated data
// ConfigMap name from the postprocess Job's own name instead of the original
// workload's.
func isPostprocessPod(pod *corev1.Pod) bool {
	return pod.Labels[aibomdata.LabelPostprocessFor] != ""
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

// triggerName returns the identity the watcher will later use to name the
// postprocess Job/data ConfigMap for this pod: the owning Job's name for
// Job/JobSet/PyTorchJob/RayJob-owned pods, or the pod's own name for bare
// GPU pods (e.g. KServe predictors) — mirroring watcher.go's onJobEvent
// (Job path) and onPodEvent (bare pod path).
func triggerName(pod *corev1.Pod) string {
	for _, ref := range pod.OwnerReferences {
		if matchedOwnerKinds[ref.Kind] {
			return ref.Name
		}
	}
	return pod.Name
}

// dataConfigMapEnvVar returns the static AIBOM_DATA_CONFIGMAP env var, but
// only when triggerName(pod) is reliably known at admission time — i.e. the
// pod has a matching owner (its name comes from ownerReferences, already set
// before admission). For a bare/ReplicaSet-owned pod with no such owner
// (e.g. a KServe predictor), triggerName falls back to pod.Name, which is
// EMPTY at this point for any pod created via generateName — the API server
// hasn't assigned the real name yet when this webhook runs. Baking in
// aibomdata.ConfigMapName("") here would silently point every write at a
// malformed "-aibom-postprocess-data" ConfigMap. Instead, ok is false and the
// caller omits the env var entirely; k8s_api.resolve_data_configmap_name()
// derives the same name at runtime from POD_NAME (a downward API value,
// resolved by the kubelet after the real name exists).
func dataConfigMapEnvVar(pod *corev1.Pod) (corev1.EnvVar, bool) {
	if !hasMatchingOwner(pod) {
		return corev1.EnvVar{}, false
	}
	return corev1.EnvVar{Name: "AIBOM_DATA_CONFIGMAP", Value: aibomdata.ConfigMapName(triggerName(pod))}, true
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

func (m *Mutator) buildDiscoveryInitContainer(pod *corev1.Pod) corev1.Container {
	env := []corev1.EnvVar{
		downwardAPIEnv("POD_NAME", "metadata.name"),
		downwardAPIEnv("POD_UID", "metadata.uid"),
		downwardAPIEnv("POD_NAMESPACE", "metadata.namespace"),
		downwardAPIEnv("POD_IP", "status.podIP"),
		downwardAPIEnv("NODE_NAME", "spec.nodeName"),
	}
	if dataConfigMapEnv, ok := dataConfigMapEnvVar(pod); ok {
		env = append(env, dataConfigMapEnv)
	}
	// Only pods KServe itself already labeled as a predictor get this one —
	// a single-field label downward API reference fails pod admission
	// outright if the referenced label isn't present on the pod, so this
	// can't be added unconditionally for every workload kind (Job/JobSet/
	// PyTorchJob/RayJob pods have no such label).
	if pod.Labels[aibomdata.LabelKServeInferenceService] != "" {
		env = append(env, downwardAPIEnv(
			"INFERENCESERVICE_NAME",
			fmt.Sprintf("metadata.labels['%s']", aibomdata.LabelKServeInferenceService),
		))
	}

	c := corev1.Container{
		Name:    "aibom-discovery",
		Image:   m.DiscoveryImage,
		Command: []string{"/bin/bash", "-c"},
		Args:    []string{"python3 /scripts/generate_snapshot.py"},
		Env:     env,
		VolumeMounts: []corev1.VolumeMount{
			{Name: "aibom-data", MountPath: "/tmp/result"},
			{Name: "aibom-scripts", MountPath: "/scripts", ReadOnly: true},
			aibomTokenVolumeMount(),
		},
	}

	if gpuRes := podGPUResource(pod); gpuRes != nil {
		c.Resources = corev1.ResourceRequirements{
			Limits: corev1.ResourceList{corev1.ResourceName("nvidia.com/gpu"): *gpuRes},
		}
	}

	return c
}

func podGPUResource(pod *corev1.Pod) *resource.Quantity {
	gpuResource := corev1.ResourceName("nvidia.com/gpu")
	for i := range pod.Spec.Containers {
		if q, ok := pod.Spec.Containers[i].Resources.Limits[gpuResource]; ok && q.Cmp(resource.MustParse("0")) > 0 {
			return &q
		}
		if q, ok := pod.Spec.Containers[i].Resources.Requests[gpuResource]; ok && q.Cmp(resource.MustParse("0")) > 0 {
			return &q
		}
	}
	return nil
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
		downwardAPIEnv("POD_NAME", "metadata.name"),
		downwardAPIEnv("POD_NAMESPACE", "metadata.namespace"),
		{Name: "PYTHONPATH", Value: pythonPath},
	}
	if dataConfigMapEnv, ok := dataConfigMapEnvVar(pod); ok {
		envVars = append(envVars, dataConfigMapEnv)
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

	// Mount usercustomize.py (runtime detector), its k8s_api.py import
	// dependency, and the aibom-data volume
	mounts := []corev1.VolumeMount{
		{
			Name:      "aibom-scripts",
			MountPath: "/aibom-hooks/usercustomize.py",
			SubPath:   "runtime_detector.py",
			ReadOnly:  true,
		},
		{
			Name:      "aibom-scripts",
			MountPath: "/aibom-hooks/k8s_api.py",
			SubPath:   "k8s_api.py",
			ReadOnly:  true,
		},
		{
			Name:      "aibom-data",
			MountPath: "/tmp/aibom",
		},
	}
	// Unlike the discovery init container (which we add fresh and so never
	// has a pre-existing mount to collide with), this is the workload's own
	// container — if automountServiceAccountToken wasn't disabled, the
	// built-in ServiceAccount admission controller already mounted a token
	// at this same path before our webhook ran, and a second volumeMount at
	// an identical path fails pod admission outright.
	if !hasVolumeMountAtPath(container.VolumeMounts, aibomTokenVolumeMount().MountPath) {
		mounts = append(mounts, aibomTokenVolumeMount())
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

// buildTokenVolume provisions our own copy of the standard "kube-api-access"
// projected volume — the same three sources (SA token, cluster CA bundle,
// namespace) the built-in ServiceAccount admission controller normally
// projects automatically. That controller only mounts it into containers
// already present in the pod spec when it runs; since we add the discovery
// init container (and, for dataset detection, hooks into app containers)
// via a mutating webhook patch afterward, those newly-added containers never
// get the automatic one — this is true regardless of the pod's own
// automountServiceAccountToken setting, since a container only gets a token
// if it has an explicit volumeMount naming a token volume. Without this,
// k8s_api.py (used by both generate_snapshot.py and runtime_detector.py) has
// no token to authenticate with at all.
func buildTokenVolume() corev1.Volume {
	expirationSeconds := int64(3600)
	return corev1.Volume{
		Name: "aibom-token",
		VolumeSource: corev1.VolumeSource{
			Projected: &corev1.ProjectedVolumeSource{
				Sources: []corev1.VolumeProjection{
					{
						ServiceAccountToken: &corev1.ServiceAccountTokenProjection{
							Path:              "token",
							ExpirationSeconds: &expirationSeconds,
						},
					},
					{
						ConfigMap: &corev1.ConfigMapProjection{
							LocalObjectReference: corev1.LocalObjectReference{Name: "kube-root-ca.crt"},
							Items:                []corev1.KeyToPath{{Key: "ca.crt", Path: "ca.crt"}},
						},
					},
					{
						DownwardAPI: &corev1.DownwardAPIProjection{
							Items: []corev1.DownwardAPIVolumeFile{
								{Path: "namespace", FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.namespace"}},
							},
						},
					},
				},
			},
		},
	}
}

// aibomTokenVolumeMount mounts buildTokenVolume at the exact path k8s_api.py
// expects (_SA_DIR), so it's indistinguishable from the token the
// ServiceAccount admission controller would have auto-mounted.
func aibomTokenVolumeMount() corev1.VolumeMount {
	return corev1.VolumeMount{
		Name:      "aibom-token",
		MountPath: "/var/run/secrets/kubernetes.io/serviceaccount",
		ReadOnly:  true,
	}
}

func hasVolumeMountAtPath(mounts []corev1.VolumeMount, path string) bool {
	for _, m := range mounts {
		if m.MountPath == path {
			return true
		}
	}
	return false
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
