package watcher

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"strings"
	"time"

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
	LabelEnabled          = "aibom.io/enabled"
	LabelInstrumented     = "aibom.io/instrumented"
	LabelPostprocessFor   = "aibom.io/postprocess-for"
	AnnotationPostprocess = "aibom.io/postprocess-job"

	annotationPrefix = "aibom.io/"

	discoveryStartMarker = "===AIBOM_DISCOVERY_START==="
	discoveryEndMarker   = "===AIBOM_DISCOVERY_END==="
	datasetStartMarker   = "===AIBOM_DATASET_START==="
	datasetEndMarker     = "===AIBOM_DATASET_END==="

	initContainerName = "aibom-discovery"

	finalizerName = "aibom.io/log-extraction"

	resyncPeriod      = 30 * time.Second
	maxJobNameLength  = 63
	postprocessSuffix = "-aibom-postprocess"
	configMapSuffix   = "-data"
)

// LogReader abstracts pod log access for testability.
type LogReader interface {
	GetLogs(ctx context.Context, namespace, podName, containerName string) (io.ReadCloser, error)
}

// kubeLogReader is the production LogReader using the Kubernetes API.
type kubeLogReader struct {
	clientset kubernetes.Interface
}

func (r *kubeLogReader) GetLogs(ctx context.Context, namespace, podName, containerName string) (io.ReadCloser, error) {
	req := r.clientset.CoreV1().Pods(namespace).GetLogs(podName, &corev1.PodLogOptions{
		Container: containerName,
	})
	return req.Stream(ctx)
}

type Watcher struct {
	clientset        kubernetes.Interface
	logReader        LogReader
	postprocessImage string
	factory          informers.SharedInformerFactory
}

func New(clientset kubernetes.Interface, postprocessImage string) *Watcher {
	w := &Watcher{
		clientset:        clientset,
		logReader:        &kubeLogReader{clientset: clientset},
		postprocessImage: postprocessImage,
		factory:          informers.NewSharedInformerFactory(clientset, resyncPeriod),
	}

	w.factory.Batch().V1().Jobs().Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    w.onJobEvent,
		UpdateFunc: func(_, newObj interface{}) { w.onJobEvent(newObj) },
		DeleteFunc: w.onJobEvent,
	})

	// Ensure the namespace informer is created so it syncs with the factory.
	w.factory.Core().V1().Namespaces().Informer()

	return w
}

func (w *Watcher) Start(ctx context.Context) error {
	w.factory.Start(ctx.Done())

	synced := w.factory.WaitForCacheSync(ctx.Done())
	for gvr, ok := range synced {
		if !ok {
			return fmt.Errorf("informer failed to sync: %v", gvr)
		}
	}

	log.Println("watcher started, watching for completed instrumented Jobs")
	<-ctx.Done()
	w.factory.Shutdown()
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

func (w *Watcher) shouldPostprocess(job *batchv1.Job) bool {
	pods, err := w.getInstrumentedPods(job)
	if err != nil || len(pods) == 0 {
		return false
	}
	return podsRequestGPU(pods) || len(collectAIBOMAnnotations(job)) > 0
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

// extractDelimitedJSON scans a log stream for a JSON block between start and end markers.
func extractDelimitedJSON(reader io.Reader, startMarker, endMarker string) string {
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	capturing := false

	for scanner.Scan() {
		line := scanner.Text()
		if strings.TrimSpace(line) == startMarker {
			capturing = true
			continue
		}
		if strings.TrimSpace(line) == endMarker {
			break
		}
		if capturing {
			trimmed := strings.TrimSpace(line)
			if json.Valid([]byte(trimmed)) {
				return trimmed
			}
			log.Printf("warning: content between %s markers is not valid JSON", startMarker)
			return ""
		}
	}
	return ""
}

// extractDataFromPod reads pod logs and extracts discovery and dataset JSON.
func (w *Watcher) extractDataFromPod(ctx context.Context, pod *corev1.Pod) (discoveryJSON, datasetJSON string) {
	// Discovery: from init container
	stream, err := w.logReader.GetLogs(ctx, pod.Namespace, pod.Name, initContainerName)
	if err != nil {
		log.Printf("warning: could not read logs for %s/%s container %s: %v",
			pod.Namespace, pod.Name, initContainerName, err)
	} else {
		discoveryJSON = extractDelimitedJSON(stream, discoveryStartMarker, discoveryEndMarker)
		stream.Close()
	}

	// Dataset: from application containers
	for _, c := range pod.Spec.Containers {
		stream, err := w.logReader.GetLogs(ctx, pod.Namespace, pod.Name, c.Name)
		if err != nil {
			continue
		}
		result := extractDelimitedJSON(stream, datasetStartMarker, datasetEndMarker)
		stream.Close()
		if result != "" {
			datasetJSON = result
			break
		}
	}

	return discoveryJSON, datasetJSON
}

// collectAIBOMAnnotations returns job annotations with the aibom.io/ prefix stripped.
func collectAIBOMAnnotations(job *batchv1.Job) map[string]string {
	result := make(map[string]string)
	for key, value := range job.Annotations {
		if strings.HasPrefix(key, annotationPrefix) {
			stripped := strings.TrimPrefix(key, annotationPrefix)
			if stripped != "" && stripped != "instrumented" && stripped != "instrumented-by" && stripped != "postprocess-job" {
				result[stripped] = value
			}
		}
	}
	return result
}

func (w *Watcher) createDataConfigMap(ctx context.Context, namespace, configMapName, jobName string, discoveries []string, datasets []string, annotations map[string]string) error {
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

	cm := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      configMapName,
			Namespace: namespace,
			Labels: map[string]string{
				LabelPostprocessFor: jobName,
			},
		},
		Data: map[string]string{
			"discovery.json":   discoveryData,
			"dataset.json":     datasetData,
			"annotations.json": string(annotationsJSON),
		},
	}

	_, err := w.clientset.CoreV1().ConfigMaps(namespace).Create(ctx, cm, metav1.CreateOptions{})
	if err != nil && !errors.IsAlreadyExists(err) {
		return fmt.Errorf("create configmap %s: %w", configMapName, err)
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

func (w *Watcher) createPostprocessJob(ctx context.Context, job *batchv1.Job) error {
	postprocessName := postprocessJobName(job.Name)
	configMapName := postprocessName + configMapSuffix
	if len(configMapName) > 253 {
		configMapName = strings.TrimRight(configMapName[:253], "-")
	}

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

	var discoveries []string
	var datasets []string
	for _, pod := range pods {
		disc, ds := w.extractDataFromPod(ctx, &pod)
		discoveries = append(discoveries, disc)
		datasets = append(datasets, ds)
	}

	// Collect AIBOM annotations from the job and sibling jobs in the JobSet
	annotations := collectAIBOMAnnotations(job)
	if jobsetName := job.Labels["jobset.sigs.k8s.io/jobset-name"]; jobsetName != "" && len(annotations) == 0 {
		siblingJobs, err := w.clientset.BatchV1().Jobs(job.Namespace).List(ctx, metav1.ListOptions{
			LabelSelector: fmt.Sprintf("jobset.sigs.k8s.io/jobset-name=%s", jobsetName),
		})
		if err == nil {
			for i := range siblingJobs.Items {
				if sa := collectAIBOMAnnotations(&siblingJobs.Items[i]); len(sa) > 0 {
					annotations = sa
					break
				}
			}
		}
	}

	// Create ConfigMap with extracted data
	if err := w.createDataConfigMap(ctx, job.Namespace, configMapName, job.Name, discoveries, datasets, annotations); err != nil {
		log.Printf("warning: could not create data configmap for %s/%s: %v", job.Namespace, job.Name, err)
	}

	backoffLimit := int32(3)
	optional := true

	postprocessJob := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      postprocessName,
			Namespace: job.Namespace,
			Labels: map[string]string{
				LabelPostprocessFor: job.Name,
			},
		},
		Spec: batchv1.JobSpec{
			BackoffLimit: &backoffLimit,
			Template: corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					RestartPolicy: corev1.RestartPolicyNever,
					Containers: []corev1.Container{
						{
							Name:    "aibom-postprocess",
							Image:   w.postprocessImage,
							Command: []string{"python3", "/app/postprocess.py"},
							Env: []corev1.EnvVar{
								{Name: "AIBOM_JOB_NAME", Value: job.Name},
								{Name: "AIBOM_JOB_NAMESPACE", Value: job.Namespace},
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

	_, err = w.clientset.BatchV1().Jobs(job.Namespace).Create(ctx, postprocessJob, metav1.CreateOptions{})
	if err != nil && !errors.IsAlreadyExists(err) {
		return fmt.Errorf("create postprocess job: %w", err)
	}

	patch := fmt.Sprintf(`{"metadata":{"annotations":{"%s":"%s"}}}`, AnnotationPostprocess, postprocessName)
	_, err = w.clientset.BatchV1().Jobs(job.Namespace).Patch(ctx, job.Name, types.MergePatchType, []byte(patch), metav1.PatchOptions{})
	if err != nil {
		return fmt.Errorf("annotate original job: %w", err)
	}

	log.Printf("created postprocess job %s/%s for completed job %s", job.Namespace, postprocessName, job.Name)
	return nil
}

func postprocessJobName(jobName string) string {
	maxBase := maxJobNameLength - len(postprocessSuffix)
	if len(jobName) > maxBase {
		jobName = jobName[:maxBase]
	}
	jobName = strings.TrimRight(jobName, "-")
	return jobName + postprocessSuffix
}
