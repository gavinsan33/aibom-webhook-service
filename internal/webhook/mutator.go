package webhook

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/gavinsan33/aibom-webhook-service/internal/aibomdata"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/client-go/kubernetes"
)

// workloadIdentityProvisionTimeout bounds ensurePodWorkloadIdentity's
// Kubernetes API calls, which run synchronously in the admission path — a
// hanging apiserver call here must not indefinitely delay pod admission.
const workloadIdentityProvisionTimeout = 5 * time.Second

var matchedOwnerKinds = map[string]bool{
	"Job":        true,
	"JobSet":     true,
	"PyTorchJob": true,
	"RayJob":     true,
}

type Mutator struct {
	DiscoveryImage   string
	DatasetDetection bool

	// Clientset, if set, lets Mutate provision a per-job workload identity
	// (see identity.go's ensureWorkloadIdentity) scoped to exactly this
	// job's own data ConfigMap. Left nil in tests that don't exercise this
	// path and treated the same as any other identity-provisioning failure:
	// Mutate falls back to the pod's own (shared, namespace-wide) identity
	// rather than failing admission.
	Clientset kubernetes.Interface
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
		// A workload that doesn't qualify for instrumentation may still have
		// pre-set aibom.io/instrumented / aibom.io/instrumented-by itself
		// (see shouldMutate's doc comment on why the value can't be
		// trusted). Left alone, that false claim would still reach etcd on
		// this pod, and the watcher selects pods to postprocess by
		// aibom.io/instrumented=true (see watcher.go) -- so an uninstrumented
		// pod could masquerade as having been properly collected, and the
		// watcher would compile an AIBOM from data that was never actually
		// gathered. isPostprocessPod's own pods never carry this label at
		// all, so this is always safe to run on the "not qualifying" path.
		return stripSpoofedInstrumentationClaims(pod), nil
	}

	var patches []PatchOperation

	// Provision (or fall back from) a per-job identity scoped to exactly
	// this job's own data ConfigMap -- see ensurePodWorkloadIdentity and
	// identity.go's ensureWorkloadIdentity. identitySecretName is "" when
	// this pod has no owner with a name known at admission time (e.g. a
	// bare KServe predictor pod) or provisioning failed, in which case
	// buildTokenVolume below falls back to today's behavior: a projected
	// token for the pod's own (shared, namespace-wide) ServiceAccount.
	identitySecretName := m.ensurePodWorkloadIdentity(pod)

	// Add aibom-data emptyDir volume
	patches = appendVolume(patches, pod, buildAIBOMVolume())

	// Add aibom-scripts ConfigMap volume
	patches = appendVolume(patches, pod, buildScriptsVolume())

	// Add our own Kubernetes API token volume — see buildTokenVolume's doc
	// comment for why the pod's own (possibly absent) automounted token
	// can't be relied on for containers this webhook adds.
	patches = appendVolume(patches, pod, buildTokenVolume(identitySecretName))

	// Add the discovery signing key volume — mounted only into the discovery
	// init container below (buildDiscoveryInitContainer), never into an app
	// container, so the workload's own code never has access to it.
	patches = appendVolume(patches, pod, buildDiscoverySigningKeyVolume())

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
			patches = append(patches, m.buildDatasetDetectorPatches(pod, i, identitySecretName)...)
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

// shouldMutate reports whether pod should be instrumented. It deliberately
// does NOT consult any aibom.io/instrumented value already present on the
// incoming pod: this webhook's MutatingWebhookConfiguration only matches
// CREATE operations with reinvocationPolicy: Never, so there is no
// legitimate scenario where this webhook has already run once and set that
// label earlier in the same admission chain. Labels are part of the object
// the requester submits, so trusting a pre-existing "true" value here would
// let any workload dodge instrumentation for free simply by pre-setting the
// label the webhook itself would otherwise add.
func (m *Mutator) shouldMutate(pod *corev1.Pod) bool {
	if isPostprocessPod(pod) {
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

// stripSpoofedInstrumentationClaims returns JSON patches removing any
// aibom.io/instrumented label and aibom.io/instrumented-by annotation
// already present on a pod the webhook has decided not to instrument. See
// Mutate's call site for why a requester-supplied claim here can't be left
// in place. JSON Patch "remove" fails admission if the target path doesn't
// exist, so each removal is only emitted when the key is actually present.
func stripSpoofedInstrumentationClaims(pod *corev1.Pod) []PatchOperation {
	var patches []PatchOperation
	if _, ok := pod.Labels["aibom.io/instrumented"]; ok {
		patches = append(patches, PatchOperation{
			Op:   "remove",
			Path: "/metadata/labels/aibom.io~1instrumented",
		})
	}
	if _, ok := pod.Annotations["aibom.io/instrumented-by"]; ok {
		patches = append(patches, PatchOperation{
			Op:   "remove",
			Path: "/metadata/annotations/aibom.io~1instrumented-by",
		})
	}
	return patches
}

func hasMatchingOwner(pod *corev1.Pod) bool {
	for _, ref := range pod.OwnerReferences {
		if matchedOwnerKinds[ref.Kind] {
			return true
		}
	}
	return false
}

// ensurePodWorkloadIdentity provisions a per-job identity for pod (see
// identity.go's ensureWorkloadIdentity) and returns the Secret name to
// mount as this pod's token, or "" if that isn't applicable -- no
// Clientset configured, or the pod has no owner whose name is known at
// admission time (see dataConfigMapEnvVar's doc comment: a bare pod's own
// name doesn't exist yet when it's created via generateName) -- or
// provisioning failed. Any failure here is logged and swallowed, never
// returned as a Mutate error: this service fails open (failurePolicy:
// Ignore), and a Kubernetes API hiccup while provisioning RBAC must not
// block the pod it's trying to instrument.
func (m *Mutator) ensurePodWorkloadIdentity(pod *corev1.Pod) string {
	if m.Clientset == nil {
		return ""
	}
	if !hasMatchingOwner(pod) {
		return ""
	}
	ctx, cancel := context.WithTimeout(context.Background(), workloadIdentityProvisionTimeout)
	defer cancel()

	trigger := triggerName(pod)
	configMapName := aibomdata.ConfigMapName(trigger)
	secretName, err := ensureWorkloadIdentity(ctx, m.Clientset, pod.Namespace, trigger, configMapName)
	if err != nil {
		log.Printf("warning: could not provision per-job workload identity for %s/%s (job %s): %v; falling back to shared ServiceAccount token", pod.Namespace, pod.Name, trigger, err)
		return ""
	}
	return secretName
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
			discoverySigningKeyVolumeMount(),
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
func (m *Mutator) buildDatasetDetectorPatches(pod *corev1.Pod, containerIdx int, identitySecretName string) []PatchOperation {
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
	existingTokenMountIdx := volumeMountIndexAtPath(container.VolumeMounts, aibomTokenVolumeMount().MountPath)
	switch {
	case existingTokenMountIdx == -1:
		mounts = append(mounts, aibomTokenVolumeMount())
	case identitySecretName != "":
		// A per-job identity was provisioned (see ensurePodWorkloadIdentity),
		// so the built-in default-SA mount already occupying this path would
		// otherwise leave this container able to fall back to its own
		// (broader, namespace-wide) ServiceAccount for ConfigMap writes,
		// defeating the point of scoping this identity to one job's
		// ConfigMap in the first place. Retarget it to our volume instead of
		// leaving it in place.
		patches = append(patches, PatchOperation{
			Op:    "replace",
			Path:  fmt.Sprintf("/spec/containers/%d/volumeMounts/%d/name", containerIdx, existingTokenMountIdx),
			Value: aibomTokenVolumeMount().Name,
		})
	}
	// else: no per-job identity available (bare pod, no Clientset, or
	// provisioning failed) -- leave the pre-existing default-SA mount alone,
	// same as today.

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
//
// identitySecretName, when non-empty, swaps the first source from a
// ServiceAccountTokenProjection (necessarily for the pod's own
// spec.serviceAccountName — see ensurePodWorkloadIdentity's doc comment for
// why that identity can't just be overridden) to a SecretProjection reading
// the token ensureWorkloadIdentity minted for a per-job identity scoped to
// exactly this job's data ConfigMap. Either way the result lands at the
// same "token" path, so k8s_api.py doesn't need to know which one it got.
func buildTokenVolume(identitySecretName string) corev1.Volume {
	var tokenSource corev1.VolumeProjection
	if identitySecretName != "" {
		tokenSource = corev1.VolumeProjection{
			Secret: &corev1.SecretProjection{
				LocalObjectReference: corev1.LocalObjectReference{Name: identitySecretName},
				Items:                []corev1.KeyToPath{{Key: workloadIdentityTokenSecretKey, Path: "token"}},
			},
		}
	} else {
		expirationSeconds := int64(3600)
		tokenSource = corev1.VolumeProjection{
			ServiceAccountToken: &corev1.ServiceAccountTokenProjection{
				Path:              "token",
				ExpirationSeconds: &expirationSeconds,
			},
		}
	}

	return corev1.Volume{
		Name: "aibom-token",
		VolumeSource: corev1.VolumeSource{
			Projected: &corev1.ProjectedVolumeSource{
				Sources: []corev1.VolumeProjection{
					tokenSource,
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

// buildDiscoverySigningKeyVolume is mounted as optional: a namespace that
// hasn't been upgraded to a chart version carrying signing.yaml yet simply
// has no such Secret, and generate_snapshot.py falls back to writing
// unsigned discovery data (see its own missing-key handling) rather than
// the pod failing to start.
func buildDiscoverySigningKeyVolume() corev1.Volume {
	optional := true
	return corev1.Volume{
		Name: "aibom-discovery-signing-key",
		VolumeSource: corev1.VolumeSource{
			Secret: &corev1.SecretVolumeSource{
				SecretName: aibomdata.DiscoverySigningKeySecretName,
				Optional:   &optional,
			},
		},
	}
}

func discoverySigningKeyVolumeMount() corev1.VolumeMount {
	return corev1.VolumeMount{
		Name:      "aibom-discovery-signing-key",
		MountPath: "/var/run/secrets/aibom/discovery-signing",
		ReadOnly:  true,
	}
}

// volumeMountIndexAtPath returns the index of the volumeMount in mounts
// whose MountPath matches path, or -1 if none does.
func volumeMountIndexAtPath(mounts []corev1.VolumeMount, path string) int {
	for i, m := range mounts {
		if m.MountPath == path {
			return i
		}
	}
	return -1
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
