package watcher

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"strings"
	"time"

	"github.com/gavinsan33/aibom-webhook-service/internal/aibomdata"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
)

const (
	LabelEnabled             = "aibom.io/enabled"
	LabelInstrumented        = "aibom.io/instrumented"
	LabelPostprocessFor      = aibomdata.LabelPostprocessFor
	AnnotationPostprocess    = "aibom.io/postprocess-job"
	AnnotationAIBOMCollected = "aibom.io/aibom-collected"

	annotationPrefix = "aibom.io/"

	initContainerName = "aibom-discovery"

	finalizerName = "aibom.io/log-extraction"

	// podFinalizerName is distinct from finalizerName so that `kubectl get -o yaml`
	// self-documents which mechanism (Job-level vs Pod-level) placed a given finalizer.
	podFinalizerName = "aibom.io/log-extraction-pod"

	postprocessContainerName      = "aibom-postprocess"
	postprocessServiceAccountName = "aibom-postprocess"

	resyncPeriod      = 30 * time.Second
	maxJobNameLength  = aibomdata.MaxJobNameLength
	postprocessSuffix = aibomdata.PostprocessSuffix
	configMapSuffix   = aibomdata.ConfigMapSuffix
)

type Watcher struct {
	clientset        kubernetes.Interface
	postprocessImage string
	factory          informers.SharedInformerFactory
	// podFactory is a separate, server-side label-selector-scoped factory for the Pod
	// informer. Unlike Jobs (which have no label capturing "qualifies for
	// postprocessing", so watch-everything-then-filter is unavoidable), pods are far
	// higher cardinality cluster-wide and the webhook already applies LabelInstrumented
	// before this feature runs, so scoping the watch server-side avoids needless
	// watch-cache load from every unrelated pod in the cluster.
	podFactory informers.SharedInformerFactory
}

func New(clientset kubernetes.Interface, postprocessImage string) *Watcher {
	w := &Watcher{
		clientset:        clientset,
		postprocessImage: postprocessImage,
		factory:          informers.NewSharedInformerFactory(clientset, resyncPeriod),
		podFactory: informers.NewSharedInformerFactoryWithOptions(
			clientset, resyncPeriod,
			informers.WithTweakListOptions(func(opts *metav1.ListOptions) {
				opts.LabelSelector = fmt.Sprintf("%s=true,!batch.kubernetes.io/job-name", LabelInstrumented)
			}),
		),
	}

	w.factory.Batch().V1().Jobs().Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    w.onJobEvent,
		UpdateFunc: func(_, newObj interface{}) { w.onJobEvent(newObj) },
		DeleteFunc: w.onJobEvent,
	})

	w.podFactory.Core().V1().Pods().Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    w.onPodEvent,
		UpdateFunc: func(_, newObj interface{}) { w.onPodEvent(newObj) },
		DeleteFunc: w.onPodEvent,
	})

	// Ensure the namespace informer is created so it syncs with the factory.
	w.factory.Core().V1().Namespaces().Informer()

	return w
}

func (w *Watcher) Start(ctx context.Context) error {
	w.factory.Start(ctx.Done())
	w.podFactory.Start(ctx.Done())

	synced := w.factory.WaitForCacheSync(ctx.Done())
	for gvr, ok := range synced {
		if !ok {
			return fmt.Errorf("informer failed to sync: %v", gvr)
		}
	}
	podSynced := w.podFactory.WaitForCacheSync(ctx.Done())
	for gvr, ok := range podSynced {
		if !ok {
			return fmt.Errorf("pod informer failed to sync: %v", gvr)
		}
	}

	log.Println("watcher started, watching for completed instrumented Jobs and long-running instrumented Pods")
	<-ctx.Done()
	w.factory.Shutdown()
	w.podFactory.Shutdown()
	return nil
}

func (w *Watcher) onJobEvent(obj interface{}) {
	job, ok := obj.(*batchv1.Job)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			return
		}
		job, ok = tombstone.Obj.(*batchv1.Job)
		if !ok {
			return
		}
	}

	if !w.isNamespaceEnabled(job.Namespace) {
		return
	}

	if job.Labels[LabelPostprocessFor] != "" {
		alreadyCollected := job.Annotations != nil && job.Annotations[AnnotationAIBOMCollected] != ""
		if w.isJobComplete(job) && !alreadyCollected {
			w.collectAIBOM(context.TODO(), job)
		}
		return
	}

	readyForPostprocess := w.isJobComplete(job) || job.DeletionTimestamp != nil

	if !readyForPostprocess {
		// Path A: job is new/running — add finalizer if it qualifies
		if hasFinalizer(job) {
			return
		}
		if !w.shouldPostprocess(job) {
			return
		}
		if err := w.addFinalizer(context.TODO(), job); err != nil {
			log.Printf("warning: could not add finalizer to %s/%s: %v", job.Namespace, job.Name, err)
		}
		return
	}

	// Path B: job is complete or being deleted — run postprocessing
	if job.Annotations != nil && job.Annotations[AnnotationPostprocess] != "" {
		if hasFinalizer(job) {
			w.removeFinalizer(context.TODO(), job)
		}
		return
	}

	if !hasFinalizer(job) && !w.isJobComplete(job) {
		return
	}

	if !w.shouldPostprocess(job) {
		if hasFinalizer(job) {
			w.removeFinalizer(context.TODO(), job)
		}
		return
	}

	if err := w.createPostprocessJob(context.TODO(), job); err != nil {
		log.Printf("failed to create postprocess job for %s/%s: %v", job.Namespace, job.Name, err)
	}

	if hasFinalizer(job) {
		w.removeFinalizer(context.TODO(), job)
	}
}

// onPodEvent is the Pod-level equivalent of onJobEvent, for bare/ReplicaSet-owned pods
// (e.g. KServe InferenceService predictors) that have no owning Job to hang a finalizer
// on. These pods never "complete" — the only postprocessing trigger is deletion.
func (w *Watcher) onPodEvent(obj interface{}) {
	pod, ok := obj.(*corev1.Pod)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			return
		}
		pod, ok = tombstone.Obj.(*corev1.Pod)
		if !ok {
			return
		}
	}

	if !w.isNamespaceEnabled(pod.Namespace) {
		return
	}

	// Pods owned by a Job (or a JobSet's Jobs) are handled by onJobEvent's
	// Job-level finalizer path — the Job controller sets this label on every
	// pod it creates, so it reliably identifies pods already covered there.
	if pod.Labels["batch.kubernetes.io/job-name"] != "" {
		return
	}

	if pod.Labels[LabelInstrumented] != "true" {
		return
	}

	if pod.DeletionTimestamp == nil {
		// Path A: pod is running — add finalizer if it qualifies
		if hasPodFinalizer(pod) {
			return
		}
		if !shouldPostprocessPod(pod) {
			return
		}
		if err := w.addPodFinalizer(context.TODO(), pod); err != nil {
			log.Printf("warning: could not add finalizer to pod %s/%s: %v", pod.Namespace, pod.Name, err)
		}
		return
	}

	// Path B: pod is being deleted — run postprocessing
	if pod.Annotations != nil && pod.Annotations[AnnotationPostprocess] != "" {
		if hasPodFinalizer(pod) {
			w.removePodFinalizer(context.TODO(), pod)
		}
		return
	}

	if !hasPodFinalizer(pod) {
		// Never qualified while running (or missed the add event) — nothing to do.
		return
	}

	if !shouldPostprocessPod(pod) {
		w.removePodFinalizer(context.TODO(), pod)
		return
	}

	if err := w.createPostprocessJobForPod(context.TODO(), pod); err != nil {
		log.Printf("failed to create postprocess job for pod %s/%s: %v", pod.Namespace, pod.Name, err)
	}

	w.removePodFinalizer(context.TODO(), pod)
}

func (w *Watcher) shouldPostprocess(job *batchv1.Job) bool {
	pods, err := w.getInstrumentedPods(job)
	if err != nil || len(pods) == 0 {
		return false
	}
	return podsRequestGPU(pods) || len(collectAIBOMAnnotations(job.Annotations)) > 0
}

func shouldPostprocessPod(pod *corev1.Pod) bool {
	return podsRequestGPU([]corev1.Pod{*pod}) || len(collectAIBOMAnnotations(pod.Annotations)) > 0
}

func hasFinalizer(job *batchv1.Job) bool {
	for _, f := range job.Finalizers {
		if f == finalizerName {
			return true
		}
	}
	return false
}

func (w *Watcher) addFinalizer(ctx context.Context, job *batchv1.Job) error {
	finalizers := append(job.Finalizers, finalizerName)
	finalizersJSON, _ := json.Marshal(finalizers)
	patch := fmt.Sprintf(`{"metadata":{"finalizers":%s}}`, finalizersJSON)
	_, err := w.clientset.BatchV1().Jobs(job.Namespace).Patch(ctx, job.Name, types.MergePatchType, []byte(patch), metav1.PatchOptions{})
	if err != nil {
		return fmt.Errorf("add finalizer to %s/%s: %w", job.Namespace, job.Name, err)
	}
	log.Printf("added finalizer to %s/%s", job.Namespace, job.Name)
	return nil
}

func (w *Watcher) removeFinalizer(ctx context.Context, job *batchv1.Job) {
	var remaining []string
	for _, f := range job.Finalizers {
		if f != finalizerName {
			remaining = append(remaining, f)
		}
	}
	finalizersJSON, _ := json.Marshal(remaining)
	if remaining == nil {
		finalizersJSON = []byte("[]")
	}
	patch := fmt.Sprintf(`{"metadata":{"finalizers":%s}}`, finalizersJSON)
	_, err := w.clientset.BatchV1().Jobs(job.Namespace).Patch(ctx, job.Name, types.MergePatchType, []byte(patch), metav1.PatchOptions{})
	if err != nil {
		log.Printf("warning: could not remove finalizer from %s/%s: %v", job.Namespace, job.Name, err)
	} else {
		log.Printf("removed finalizer from %s/%s", job.Namespace, job.Name)
	}
}

func hasPodFinalizer(pod *corev1.Pod) bool {
	for _, f := range pod.Finalizers {
		if f == podFinalizerName {
			return true
		}
	}
	return false
}

func (w *Watcher) addPodFinalizer(ctx context.Context, pod *corev1.Pod) error {
	finalizers := append(pod.Finalizers, podFinalizerName)
	finalizersJSON, _ := json.Marshal(finalizers)
	patch := fmt.Sprintf(`{"metadata":{"finalizers":%s}}`, finalizersJSON)
	_, err := w.clientset.CoreV1().Pods(pod.Namespace).Patch(ctx, pod.Name, types.MergePatchType, []byte(patch), metav1.PatchOptions{})
	if err != nil {
		return fmt.Errorf("add finalizer to pod %s/%s: %w", pod.Namespace, pod.Name, err)
	}
	log.Printf("added finalizer to pod %s/%s", pod.Namespace, pod.Name)
	return nil
}

func (w *Watcher) removePodFinalizer(ctx context.Context, pod *corev1.Pod) {
	var remaining []string
	for _, f := range pod.Finalizers {
		if f != podFinalizerName {
			remaining = append(remaining, f)
		}
	}
	finalizersJSON, _ := json.Marshal(remaining)
	if remaining == nil {
		finalizersJSON = []byte("[]")
	}
	patch := fmt.Sprintf(`{"metadata":{"finalizers":%s}}`, finalizersJSON)
	_, err := w.clientset.CoreV1().Pods(pod.Namespace).Patch(ctx, pod.Name, types.MergePatchType, []byte(patch), metav1.PatchOptions{})
	if err != nil {
		log.Printf("warning: could not remove finalizer from pod %s/%s: %v", pod.Namespace, pod.Name, err)
	} else {
		log.Printf("removed finalizer from pod %s/%s", pod.Namespace, pod.Name)
	}
}

func (w *Watcher) isNamespaceEnabled(namespace string) bool {
	ns, err := w.factory.Core().V1().Namespaces().Lister().Get(namespace)
	if err != nil {
		return false
	}
	return ns.Labels[LabelEnabled] == "true"
}

func (w *Watcher) isJobComplete(job *batchv1.Job) bool {
	for _, c := range job.Status.Conditions {
		if c.Type == batchv1.JobComplete && c.Status == corev1.ConditionTrue {
			return true
		}
	}
	return false
}

func podsRequestGPU(pods []corev1.Pod) bool {
	gpuResource := corev1.ResourceName("nvidia.com/gpu")
	zero := resource.MustParse("0")
	for i := range pods {
		for j := range pods[i].Spec.Containers {
			c := &pods[i].Spec.Containers[j]
			if q, ok := c.Resources.Limits[gpuResource]; ok && q.Cmp(zero) > 0 {
				return true
			}
			if q, ok := c.Resources.Requests[gpuResource]; ok && q.Cmp(zero) > 0 {
				return true
			}
		}
	}
	return false
}

func (w *Watcher) getInstrumentedPods(job *batchv1.Job) ([]corev1.Pod, error) {
	pods, err := w.clientset.CoreV1().Pods(job.Namespace).List(context.TODO(), metav1.ListOptions{
		LabelSelector: fmt.Sprintf("batch.kubernetes.io/job-name=%s,%s=true", job.Name, LabelInstrumented),
	})
	if err != nil {
		return nil, fmt.Errorf("list pods for job %s/%s: %w", job.Namespace, job.Name, err)
	}
	return pods.Items, nil
}

// extractDataFromPod reads a pod's contribution to the AIBOM data ConfigMap.
// Both discovery and dataset data are written directly into dataCM by the
// aibom-discovery init container and the app container's runtime-detector
// hook respectively (keyed "discovery-<pod-name>.json"/"dataset-<pod-name>.json")
// rather than scraped from logs — dataCM is nil if the ConfigMap doesn't exist
// yet (e.g. neither has run/flushed yet).
func extractDataFromPod(pod *corev1.Pod, dataCM *corev1.ConfigMap) (discoveryJSON, datasetJSON string) {
	if dataCM != nil {
		discoveryJSON = dataCM.Data[fmt.Sprintf("discovery-%s.json", pod.Name)]
		datasetJSON = dataCM.Data[fmt.Sprintf("dataset-%s.json", pod.Name)]
	}

	return discoveryJSON, datasetJSON
}

// collectAIBOMAnnotations returns annotations with the aibom.io/ prefix stripped,
// excluding internal bookkeeping keys.
func collectAIBOMAnnotations(annotations map[string]string) map[string]string {
	result := make(map[string]string)
	for key, value := range annotations {
		if strings.HasPrefix(key, annotationPrefix) {
			stripped := strings.TrimPrefix(key, annotationPrefix)
			if stripped != "" && stripped != "instrumented" && stripped != "instrumented-by" && stripped != "postprocess-job" {
				result[stripped] = value
			}
		}
	}
	return result
}

func (w *Watcher) createDataConfigMap(ctx context.Context, namespace, configMapName, jobName string, discoveries []string, datasets []string, annotations map[string]string, containersJSON string) error {
	// Build discovery data: array of discovery objects
	var discoveryArray []json.RawMessage
	for _, d := range discoveries {
		if d != "" {
			discoveryArray = append(discoveryArray, json.RawMessage(d))
		}
	}

	discoveryData := "[]"
	if len(discoveryArray) > 0 {
		bytes, err := json.Marshal(discoveryArray)
		if err == nil {
			discoveryData = string(bytes)
		}
	}

	// Merge dataset data
	datasetData := mergeDatasets(datasets)

	annotationsJSON, _ := json.Marshal(annotations)

	aggregateData := map[string]string{
		"discovery.json":   discoveryData,
		"dataset.json":     datasetData,
		"annotations.json": string(annotationsJSON),
		"containers.json":  containersJSON,
	}

	// The pods themselves may have already created this ConfigMap (writing their
	// own "discovery-<pod>.json" keys directly, see extractDataFromPod) before the
	// workload completed. Merge the aggregate keys in rather than blindly Create,
	// which would silently no-op on AlreadyExists and never add them.
	existing, err := w.clientset.CoreV1().ConfigMaps(namespace).Get(ctx, configMapName, metav1.GetOptions{})
	if err != nil {
		if !errors.IsNotFound(err) {
			return fmt.Errorf("get configmap %s: %w", configMapName, err)
		}
		cm := &corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name:      configMapName,
				Namespace: namespace,
				Labels: map[string]string{
					LabelPostprocessFor: jobName,
				},
			},
			Data: aggregateData,
		}
		if _, err := w.clientset.CoreV1().ConfigMaps(namespace).Create(ctx, cm, metav1.CreateOptions{}); err != nil && !errors.IsAlreadyExists(err) {
			return fmt.Errorf("create configmap %s: %w", configMapName, err)
		}
		return nil
	}

	if existing.Data == nil {
		existing.Data = map[string]string{}
	}
	for k, v := range aggregateData {
		existing.Data[k] = v
	}
	if existing.Labels == nil {
		existing.Labels = map[string]string{}
	}
	existing.Labels[LabelPostprocessFor] = jobName
	if _, err := w.clientset.CoreV1().ConfigMaps(namespace).Update(ctx, existing, metav1.UpdateOptions{}); err != nil {
		return fmt.Errorf("update configmap %s: %w", configMapName, err)
	}
	return nil
}

// mergeDatasets combines multiple dataset JSON strings into one.
func mergeDatasets(datasets []string) string {
	type datasetFile struct {
		Datasets    []json.RawMessage      `json:"datasets,omitempty"`
		RuntimeInfo map[string]interface{} `json:"runtime_info,omitempty"`
	}

	merged := datasetFile{
		RuntimeInfo: make(map[string]interface{}),
	}

	for _, raw := range datasets {
		if raw == "" {
			continue
		}
		var df datasetFile
		if err := json.Unmarshal([]byte(raw), &df); err != nil {
			continue
		}
		merged.Datasets = append(merged.Datasets, df.Datasets...)
		for k, v := range df.RuntimeInfo {
			if _, exists := merged.RuntimeInfo[k]; !exists {
				merged.RuntimeInfo[k] = v
			}
		}
	}

	if len(merged.Datasets) == 0 && len(merged.RuntimeInfo) == 0 {
		return "{}"
	}

	bytes, err := json.Marshal(merged)
	if err != nil {
		return "{}"
	}
	return string(bytes)
}

// buildPostprocessInputs reads the discovery data the pods themselves already
// wrote into the data ConfigMap, extracts dataset JSON from pod logs (still
// log-scraped for now), and serializes container command/args info for model
// detection.
func (w *Watcher) buildPostprocessInputs(ctx context.Context, namespace, configMapName string, pods []corev1.Pod) (discoveries, datasets []string, containersJSON string) {
	dataCM, err := w.clientset.CoreV1().ConfigMaps(namespace).Get(ctx, configMapName, metav1.GetOptions{})
	if err != nil {
		if !errors.IsNotFound(err) {
			log.Printf("warning: could not read data configmap %s/%s: %v", namespace, configMapName, err)
		}
		dataCM = nil
	}

	for _, pod := range pods {
		disc, ds := extractDataFromPod(&pod, dataCM)
		discoveries = append(discoveries, disc)
		datasets = append(datasets, ds)
	}

	type containerInfo struct {
		PodName string   `json:"pod_name"`
		Name    string   `json:"name"`
		Image   string   `json:"image"`
		Command []string `json:"command"`
		Args    []string `json:"args"`
	}
	var containers []containerInfo
	for _, pod := range pods {
		for _, c := range pod.Spec.Containers {
			containers = append(containers, containerInfo{
				PodName: pod.Name,
				Name:    c.Name,
				Image:   c.Image,
				Command: c.Command,
				Args:    c.Args,
			})
		}
	}
	raw, _ := json.Marshal(containers)
	return discoveries, datasets, string(raw)
}

// createPostprocessJobCore creates the data ConfigMap and the postprocess Job for a
// triggering resource (a Job or a bare Pod), identified only by name/namespace, using
// data gathered from the given pods. It does not patch AnnotationPostprocess back onto
// the trigger resource — callers must do that themselves, since the trigger's kind
// (Job vs Pod) determines which client to patch with.
func (w *Watcher) createPostprocessJobCore(ctx context.Context, namespace, triggerName string, pods []corev1.Pod, annotations map[string]string) (string, error) {
	postprocessName := postprocessJobName(triggerName)
	configMapName := aibomdata.ConfigMapName(triggerName)

	discoveries, datasets, containersJSON := w.buildPostprocessInputs(ctx, namespace, configMapName, pods)

	if err := w.createDataConfigMap(ctx, namespace, configMapName, triggerName, discoveries, datasets, annotations, containersJSON); err != nil {
		log.Printf("warning: could not create data configmap for %s/%s: %v", namespace, triggerName, err)
	}

	backoffLimit := int32(3)
	optional := true
	runAsNonRoot := true
	allowPrivilegeEscalation := false
	readOnlyRootFilesystem := true

	postprocessJob := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      postprocessName,
			Namespace: namespace,
			Labels: map[string]string{
				LabelPostprocessFor: triggerName,
			},
		},
		Spec: batchv1.JobSpec{
			BackoffLimit: &backoffLimit,
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						LabelPostprocessFor: triggerName,
					},
				},
				Spec: corev1.PodSpec{
					RestartPolicy:      corev1.RestartPolicyNever,
					ServiceAccountName: postprocessServiceAccountName,
					SecurityContext: &corev1.PodSecurityContext{
						RunAsNonRoot: &runAsNonRoot,
					},
					Containers: []corev1.Container{
						{
							Name:    postprocessContainerName,
							Image:   w.postprocessImage,
							Command: []string{"python3", "/app/postprocess.py"},
							SecurityContext: &corev1.SecurityContext{
								AllowPrivilegeEscalation: &allowPrivilegeEscalation,
								ReadOnlyRootFilesystem:   &readOnlyRootFilesystem,
								Capabilities: &corev1.Capabilities{
									Drop: []corev1.Capability{"ALL"},
								},
							},
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									corev1.ResourceCPU:    resource.MustParse("100m"),
									corev1.ResourceMemory: resource.MustParse("128Mi"),
								},
								Limits: corev1.ResourceList{
									corev1.ResourceCPU:    resource.MustParse("500m"),
									corev1.ResourceMemory: resource.MustParse("512Mi"),
								},
							},
							Env: []corev1.EnvVar{
								{Name: "AIBOM_JOB_NAME", Value: triggerName},
								{Name: "AIBOM_JOB_NAMESPACE", Value: namespace},
								{Name: "AIBOM_INPUT_DIR", Value: "/data/input"},
								{
									Name: "GRAFANA_URL",
									ValueFrom: &corev1.EnvVarSource{
										SecretKeyRef: &corev1.SecretKeySelector{
											LocalObjectReference: corev1.LocalObjectReference{Name: "aibom-config"},
											Key:                  "grafana-url",
											Optional:             &optional,
										},
									},
								},
								{
									Name: "GRAFANA_API_TOKEN",
									ValueFrom: &corev1.EnvVarSource{
										SecretKeyRef: &corev1.SecretKeySelector{
											LocalObjectReference: corev1.LocalObjectReference{Name: "aibom-config"},
											Key:                  "grafana-api-token",
											Optional:             &optional,
										},
									},
								},
							},
							VolumeMounts: []corev1.VolumeMount{
								{
									Name:      "aibom-input",
									MountPath: "/data/input",
									ReadOnly:  true,
								},
							},
						},
					},
					Volumes: []corev1.Volume{
						{
							Name: "aibom-input",
							VolumeSource: corev1.VolumeSource{
								ConfigMap: &corev1.ConfigMapVolumeSource{
									LocalObjectReference: corev1.LocalObjectReference{Name: configMapName},
								},
							},
						},
					},
				},
			},
		},
	}

	_, err := w.clientset.BatchV1().Jobs(namespace).Create(ctx, postprocessJob, metav1.CreateOptions{})
	if err != nil && !errors.IsAlreadyExists(err) {
		return "", fmt.Errorf("create postprocess job: %w", err)
	}

	log.Printf("created postprocess job %s/%s for %s", namespace, postprocessName, triggerName)
	return postprocessName, nil
}

func (w *Watcher) createPostprocessJob(ctx context.Context, job *batchv1.Job) error {
	// Extract data from pod logs — include sibling JobSet pods if applicable
	pods, err := w.getInstrumentedPods(job)
	if err != nil {
		log.Printf("warning: could not list pods for %s/%s: %v", job.Namespace, job.Name, err)
	}

	if jobsetName := job.Labels["jobset.sigs.k8s.io/jobset-name"]; jobsetName != "" {
		siblingPods, err := w.clientset.CoreV1().Pods(job.Namespace).List(ctx, metav1.ListOptions{
			LabelSelector: fmt.Sprintf("jobset.sigs.k8s.io/jobset-name=%s,%s=true", jobsetName, LabelInstrumented),
		})
		if err == nil {
			seen := make(map[string]bool)
			for _, p := range pods {
				seen[p.Name] = true
			}
			for _, p := range siblingPods.Items {
				if !seen[p.Name] {
					pods = append(pods, p)
				}
			}
		}
	}

	// Collect AIBOM annotations from the job and sibling jobs in the JobSet
	annotations := collectAIBOMAnnotations(job.Annotations)
	if jobsetName := job.Labels["jobset.sigs.k8s.io/jobset-name"]; jobsetName != "" && len(annotations) == 0 {
		siblingJobs, err := w.clientset.BatchV1().Jobs(job.Namespace).List(ctx, metav1.ListOptions{
			LabelSelector: fmt.Sprintf("jobset.sigs.k8s.io/jobset-name=%s", jobsetName),
		})
		if err == nil {
			for i := range siblingJobs.Items {
				if sa := collectAIBOMAnnotations(siblingJobs.Items[i].Annotations); len(sa) > 0 {
					annotations = sa
					break
				}
			}
		}
	}

	postprocessName, err := w.createPostprocessJobCore(ctx, job.Namespace, job.Name, pods, annotations)
	if err != nil {
		return err
	}

	patch := fmt.Sprintf(`{"metadata":{"annotations":{"%s":"%s"}}}`, AnnotationPostprocess, postprocessName)
	_, err = w.clientset.BatchV1().Jobs(job.Namespace).Patch(ctx, job.Name, types.MergePatchType, []byte(patch), metav1.PatchOptions{})
	if err != nil {
		return fmt.Errorf("annotate original job: %w", err)
	}

	return nil
}

// createPostprocessJobForPod is the Pod-level equivalent of createPostprocessJob, for
// bare/ReplicaSet-owned pods (e.g. KServe predictors) that have no owning Job to trigger
// postprocessing from. There is no JobSet-style sibling merging here since a bare pod has
// no sibling workload to pull additional data from.
func (w *Watcher) createPostprocessJobForPod(ctx context.Context, pod *corev1.Pod) error {
	annotations := collectAIBOMAnnotations(pod.Annotations)

	postprocessName, err := w.createPostprocessJobCore(ctx, pod.Namespace, pod.Name, []corev1.Pod{*pod}, annotations)
	if err != nil {
		return err
	}

	patch := fmt.Sprintf(`{"metadata":{"annotations":{"%s":"%s"}}}`, AnnotationPostprocess, postprocessName)
	_, err = w.clientset.CoreV1().Pods(pod.Namespace).Patch(ctx, pod.Name, types.MergePatchType, []byte(patch), metav1.PatchOptions{})
	if err != nil {
		return fmt.Errorf("annotate original pod: %w", err)
	}

	return nil
}

func postprocessJobName(jobName string) string {
	return aibomdata.PostprocessJobName(jobName)
}

// collectAIBOM runs once a postprocess Job succeeds. The AIBOM custom resource
// itself is created directly by postprocess.py via the Kubernetes API — a Job
// success here is proof that create call went through, so all that's left is
// bookkeeping: mark the Job as collected and clean up the Job/ConfigMap so a
// same-named rerun of the original workload doesn't collide with leftovers.
func (w *Watcher) collectAIBOM(ctx context.Context, job *batchv1.Job) {
	originalJobName := job.Labels[LabelPostprocessFor]
	if originalJobName == "" {
		return
	}

	patch := fmt.Sprintf(`{"metadata":{"annotations":{"%s":"%s"}}}`, AnnotationAIBOMCollected, time.Now().UTC().Format(time.RFC3339))
	if _, err := w.clientset.BatchV1().Jobs(job.Namespace).Patch(ctx, job.Name, types.MergePatchType, []byte(patch), metav1.PatchOptions{}); err != nil {
		log.Printf("warning: could not annotate postprocess job %s/%s as collected: %v", job.Namespace, job.Name, err)
	}

	background := metav1.DeletePropagationBackground
	if err := w.clientset.BatchV1().Jobs(job.Namespace).Delete(ctx, job.Name, metav1.DeleteOptions{PropagationPolicy: &background}); err != nil && !errors.IsNotFound(err) {
		log.Printf("warning: could not delete postprocess job %s/%s: %v", job.Namespace, job.Name, err)
	}
	configMapName := job.Name + configMapSuffix
	if err := w.clientset.CoreV1().ConfigMaps(job.Namespace).Delete(ctx, configMapName, metav1.DeleteOptions{}); err != nil && !errors.IsNotFound(err) {
		log.Printf("warning: could not delete postprocess data configmap %s/%s: %v", job.Namespace, configMapName, err)
	}
}
