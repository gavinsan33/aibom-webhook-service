package watcher

import (
	"context"
	"io"
	"strings"
	"testing"
	"time"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
)

// mockLogReader returns pre-canned log content for each container.
type mockLogReader struct {
	logs map[string]string // key: "namespace/pod/container"
}

func (m *mockLogReader) GetLogs(_ context.Context, namespace, podName, containerName string) (io.ReadCloser, error) {
	key := namespace + "/" + podName + "/" + containerName
	content, ok := m.logs[key]
	if !ok {
		return io.NopCloser(strings.NewReader("")), nil
	}
	return io.NopCloser(strings.NewReader(content)), nil
}

func enabledNamespace(name string) *corev1.Namespace {
	return &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name:   name,
			Labels: map[string]string{LabelEnabled: "true"},
		},
	}
}

func disabledNamespace(name string) *corev1.Namespace {
	return &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{Name: name},
	}
}

func completedJob(name, namespace string) *batchv1.Job {
	return &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: batchv1.JobSpec{
			Template: corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					RestartPolicy: corev1.RestartPolicyNever,
					Containers:    []corev1.Container{{Name: "test", Image: "busybox"}},
				},
			},
		},
		Status: batchv1.JobStatus{
			Conditions: []batchv1.JobCondition{
				{Type: batchv1.JobComplete, Status: corev1.ConditionTrue},
			},
		},
	}
}

func instrumentedPod(jobName, namespace string) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      jobName + "-pod",
			Namespace: namespace,
			Labels: map[string]string{
				"batch.kubernetes.io/job-name": jobName,
				LabelInstrumented:              "true",
			},
		},
		Spec: corev1.PodSpec{
			RestartPolicy:  corev1.RestartPolicyNever,
			InitContainers: []corev1.Container{{Name: initContainerName, Image: "pytorch:latest"}},
			Containers: []corev1.Container{{
				Name:  "training",
				Image: "busybox",
				Resources: corev1.ResourceRequirements{
					Limits: corev1.ResourceList{
						"nvidia.com/gpu": resource.MustParse("1"),
					},
				},
			}},
		},
	}
}

func startWatcher(t *testing.T, w *Watcher) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)

	w.factory.Start(ctx.Done())
	w.factory.WaitForCacheSync(ctx.Done())
	time.Sleep(50 * time.Millisecond)
}

// ---------------------------------------------------------------------------
// extractDelimitedJSON tests
// ---------------------------------------------------------------------------

func TestExtractDelimitedJSON_Discovery(t *testing.T) {
	logs := "Collecting info...\n===AIBOM_DISCOVERY_START===\n{\"gpu\":{\"count\":4}}\n===AIBOM_DISCOVERY_END===\nDone!\n"
	result := extractDelimitedJSON(strings.NewReader(logs), discoveryStartMarker, discoveryEndMarker)
	if result != `{"gpu":{"count":4}}` {
		t.Errorf("got %q, want %q", result, `{"gpu":{"count":4}}`)
	}
}

func TestExtractDelimitedJSON_Dataset(t *testing.T) {
	logs := "Training epoch 1...\n===AIBOM_DATASET_START===\n{\"datasets\":[{\"name\":\"cifar10\"}]}\n===AIBOM_DATASET_END===\n"
	result := extractDelimitedJSON(strings.NewReader(logs), datasetStartMarker, datasetEndMarker)
	if result != `{"datasets":[{"name":"cifar10"}]}` {
		t.Errorf("got %q, want %q", result, `{"datasets":[{"name":"cifar10"}]}`)
	}
}

func TestExtractDelimitedJSON_NoMarkers(t *testing.T) {
	logs := "Just some regular output\nnothing special here\n"
	result := extractDelimitedJSON(strings.NewReader(logs), discoveryStartMarker, discoveryEndMarker)
	if result != "" {
		t.Errorf("expected empty string, got %q", result)
	}
}

func TestExtractDelimitedJSON_InvalidJSON(t *testing.T) {
	logs := "===AIBOM_DISCOVERY_START===\nnot-valid-json\n===AIBOM_DISCOVERY_END===\n"
	result := extractDelimitedJSON(strings.NewReader(logs), discoveryStartMarker, discoveryEndMarker)
	if result != "" {
		t.Errorf("expected empty string for invalid JSON, got %q", result)
	}
}

func TestExtractDelimitedJSON_StartOnly(t *testing.T) {
	logs := "===AIBOM_DISCOVERY_START===\n{\"partial\":true}\n"
	result := extractDelimitedJSON(strings.NewReader(logs), discoveryStartMarker, discoveryEndMarker)
	if result != `{"partial":true}` {
		t.Errorf("got %q, want %q", result, `{"partial":true}`)
	}
}

func TestExtractDelimitedJSON_EmptyLogs(t *testing.T) {
	result := extractDelimitedJSON(strings.NewReader(""), discoveryStartMarker, discoveryEndMarker)
	if result != "" {
		t.Errorf("expected empty string for empty logs, got %q", result)
	}
}

// ---------------------------------------------------------------------------
// collectAIBOMAnnotations tests
// ---------------------------------------------------------------------------

func TestCollectAIBOMAnnotations_WithAnnotations(t *testing.T) {
	job := completedJob("j1", "ns")
	job.Annotations = map[string]string{
		"aibom.io/experiment-intent": "training",
		"aibom.io/model-name":        "llama-3",
		"aibom.io/instrumented-by":   "webhook",
		"aibom.io/postprocess-job":   "j1-aibom-postprocess",
		"other-annotation":           "ignored",
	}

	result := collectAIBOMAnnotations(job)

	if result["experiment-intent"] != "training" {
		t.Errorf("experiment-intent = %q, want %q", result["experiment-intent"], "training")
	}
	if result["model-name"] != "llama-3" {
		t.Errorf("model-name = %q, want %q", result["model-name"], "llama-3")
	}
	if _, ok := result["instrumented-by"]; ok {
		t.Error("should not include instrumented-by (internal annotation)")
	}
	if _, ok := result["postprocess-job"]; ok {
		t.Error("should not include postprocess-job (internal annotation)")
	}
	if _, ok := result["other-annotation"]; ok {
		t.Error("should not include non-aibom.io annotations")
	}
}

func TestCollectAIBOMAnnotations_NoAnnotations(t *testing.T) {
	job := completedJob("j1", "ns")
	result := collectAIBOMAnnotations(job)
	if len(result) != 0 {
		t.Errorf("expected empty map, got %v", result)
	}
}

// ---------------------------------------------------------------------------
// mergeDatasets tests
// ---------------------------------------------------------------------------

func TestMergeDatasets_Multiple(t *testing.T) {
	ds1 := `{"datasets":[{"dataset_name":"cifar10"}],"runtime_info":{"framework":"PyTorch"}}`
	ds2 := `{"datasets":[{"dataset_name":"imagenet"}],"runtime_info":{"batch_size":32}}`

	result := mergeDatasets([]string{ds1, ds2})
	if !strings.Contains(result, "cifar10") || !strings.Contains(result, "imagenet") {
		t.Errorf("merged result should contain both datasets: %s", result)
	}
	if !strings.Contains(result, "PyTorch") {
		t.Errorf("merged result should contain runtime_info: %s", result)
	}
}

func TestMergeDatasets_Empty(t *testing.T) {
	result := mergeDatasets([]string{"", ""})
	if result != "{}" {
		t.Errorf("expected {}, got %s", result)
	}
}

func TestMergeDatasets_Invalid(t *testing.T) {
	result := mergeDatasets([]string{"not-json", `{"datasets":[]}`})
	if result == "" {
		t.Error("should still produce output from valid entries")
	}
}

// ---------------------------------------------------------------------------
// Core watcher event tests
// ---------------------------------------------------------------------------

func TestIsJobComplete(t *testing.T) {
	tests := []struct {
		name     string
		job      *batchv1.Job
		expected bool
	}{
		{
			name:     "completed job",
			job:      completedJob("j1", "ns"),
			expected: true,
		},
		{
			name: "running job",
			job: &batchv1.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "j2", Namespace: "ns"},
			},
			expected: false,
		},
		{
			name: "failed job",
			job: &batchv1.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "j3", Namespace: "ns"},
				Status: batchv1.JobStatus{
					Conditions: []batchv1.JobCondition{
						{Type: batchv1.JobFailed, Status: corev1.ConditionTrue},
					},
				},
			},
			expected: false,
		},
	}

	w := &Watcher{}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := w.isJobComplete(tt.job); got != tt.expected {
				t.Errorf("isJobComplete() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestIsNamespaceEnabled(t *testing.T) {
	client := fake.NewSimpleClientset(enabledNamespace("enabled-ns"), disabledNamespace("disabled-ns"))
	w := New(client, "busybox:latest")
	startWatcher(t, w)

	if !w.isNamespaceEnabled("enabled-ns") {
		t.Error("expected enabled-ns to be enabled")
	}
	if w.isNamespaceEnabled("disabled-ns") {
		t.Error("expected disabled-ns to be disabled")
	}
	if w.isNamespaceEnabled("nonexistent") {
		t.Error("expected nonexistent namespace to be disabled")
	}
}

func TestOnJobEvent_CreatesPostprocessJob(t *testing.T) {
	ns := enabledNamespace("test-ns")
	job := completedJob("train-job", "test-ns")
	pod := instrumentedPod("train-job", "test-ns")

	client := fake.NewSimpleClientset(ns, job, pod)
	w := New(client, "aibom-postprocess:latest")

	discoveryJSON := `{"pod_metadata":{"name":"train-job-pod","uid":"abc123"},"gpu":{"gpu_count":"2"}}`
	w.logReader = &mockLogReader{
		logs: map[string]string{
			"test-ns/train-job-pod/aibom-discovery": "Starting...\n===AIBOM_DISCOVERY_START===\n" + discoveryJSON + "\n===AIBOM_DISCOVERY_END===\nDone\n",
			"test-ns/train-job-pod/training":        "Training...\n",
		},
	}
	startWatcher(t, w)

	w.onJobEvent(job)

	// Verify ConfigMap was created
	cm, err := client.CoreV1().ConfigMaps("test-ns").Get(context.TODO(), "train-job-aibom-postprocess-data", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("configmap not created: %v", err)
	}
	if !strings.Contains(cm.Data["discovery.json"], "abc123") {
		t.Errorf("configmap discovery.json should contain pod UID, got: %s", cm.Data["discovery.json"])
	}

	// Verify postprocess Job was created
	ppJob, err := client.BatchV1().Jobs("test-ns").Get(context.TODO(), "train-job-aibom-postprocess", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("postprocess job not created: %v", err)
	}

	if ppJob.Labels[LabelPostprocessFor] != "train-job" {
		t.Errorf("label %s = %q, want %q", LabelPostprocessFor, ppJob.Labels[LabelPostprocessFor], "train-job")
	}

	if *ppJob.Spec.BackoffLimit != 3 {
		t.Errorf("backoffLimit = %d, want 3", *ppJob.Spec.BackoffLimit)
	}

	container := ppJob.Spec.Template.Spec.Containers[0]
	if container.Image != "aibom-postprocess:latest" {
		t.Errorf("image = %q, want %q", container.Image, "aibom-postprocess:latest")
	}
	if len(container.Command) != 2 || container.Command[0] != "python3" {
		t.Errorf("command = %v, want [python3 /app/postprocess.py]", container.Command)
	}

	envNames := make(map[string]string)
	for _, e := range container.Env {
		envNames[e.Name] = e.Value
	}
	if envNames["AIBOM_JOB_NAME"] != "train-job" {
		t.Errorf("AIBOM_JOB_NAME = %q, want %q", envNames["AIBOM_JOB_NAME"], "train-job")
	}
	if envNames["AIBOM_JOB_NAMESPACE"] != "test-ns" {
		t.Errorf("AIBOM_JOB_NAMESPACE = %q, want %q", envNames["AIBOM_JOB_NAMESPACE"], "test-ns")
	}
	if envNames["AIBOM_INPUT_DIR"] != "/data/input" {
		t.Errorf("AIBOM_INPUT_DIR = %q, want %q", envNames["AIBOM_INPUT_DIR"], "/data/input")
	}

	// Verify volume mount
	if len(ppJob.Spec.Template.Spec.Volumes) != 1 {
		t.Fatalf("expected 1 volume, got %d", len(ppJob.Spec.Template.Spec.Volumes))
	}
	if ppJob.Spec.Template.Spec.Volumes[0].ConfigMap.Name != "train-job-aibom-postprocess-data" {
		t.Errorf("volume configmap name = %q, want %q", ppJob.Spec.Template.Spec.Volumes[0].ConfigMap.Name, "train-job-aibom-postprocess-data")
	}

	// Verify original job annotated
	updatedJob, _ := client.BatchV1().Jobs("test-ns").Get(context.TODO(), "train-job", metav1.GetOptions{})
	if updatedJob.Annotations[AnnotationPostprocess] != "train-job-aibom-postprocess" {
		t.Errorf("annotation %s = %q, want %q", AnnotationPostprocess, updatedJob.Annotations[AnnotationPostprocess], "train-job-aibom-postprocess")
	}
}

func TestOnJobEvent_WithAnnotations(t *testing.T) {
	ns := enabledNamespace("test-ns")
	job := completedJob("train-job", "test-ns")
	job.Annotations = map[string]string{
		"aibom.io/experiment-intent": "training",
		"aibom.io/model-name":        "llama-3",
	}
	pod := instrumentedPod("train-job", "test-ns")

	client := fake.NewSimpleClientset(ns, job, pod)
	w := New(client, "aibom-postprocess:latest")
	w.logReader = &mockLogReader{logs: map[string]string{}}
	startWatcher(t, w)

	w.onJobEvent(job)

	cm, err := client.CoreV1().ConfigMaps("test-ns").Get(context.TODO(), "train-job-aibom-postprocess-data", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("configmap not created: %v", err)
	}
	if !strings.Contains(cm.Data["annotations.json"], "training") {
		t.Errorf("annotations.json should contain experiment-intent, got: %s", cm.Data["annotations.json"])
	}
	if !strings.Contains(cm.Data["annotations.json"], "llama-3") {
		t.Errorf("annotations.json should contain model-name, got: %s", cm.Data["annotations.json"])
	}
}

func TestOnJobEvent_NonEnabledNamespace_Skips(t *testing.T) {
	ns := disabledNamespace("disabled-ns")
	job := completedJob("train-job", "disabled-ns")
	pod := instrumentedPod("train-job", "disabled-ns")

	client := fake.NewSimpleClientset(ns, job, pod)
	w := New(client, "busybox:latest")
	startWatcher(t, w)

	w.onJobEvent(job)

	_, err := client.BatchV1().Jobs("disabled-ns").Get(context.TODO(), "train-job-aibom-postprocess", metav1.GetOptions{})
	if err == nil {
		t.Error("postprocess job should not have been created in disabled namespace")
	}
}

func TestOnJobEvent_IncompleteJob_Skips(t *testing.T) {
	ns := enabledNamespace("test-ns")
	job := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{Name: "running-job", Namespace: "test-ns"},
	}

	client := fake.NewSimpleClientset(ns, job)
	w := New(client, "busybox:latest")
	startWatcher(t, w)

	w.onJobEvent(job)

	_, err := client.BatchV1().Jobs("test-ns").Get(context.TODO(), "running-job-aibom-postprocess", metav1.GetOptions{})
	if err == nil {
		t.Error("postprocess job should not have been created for incomplete job")
	}
}

func TestOnJobEvent_AlreadyPostprocessed_Skips(t *testing.T) {
	ns := enabledNamespace("test-ns")
	job := completedJob("train-job", "test-ns")
	job.Annotations = map[string]string{AnnotationPostprocess: "train-job-aibom-postprocess"}
	pod := instrumentedPod("train-job", "test-ns")

	client := fake.NewSimpleClientset(ns, job, pod)
	w := New(client, "busybox:latest")
	startWatcher(t, w)

	w.onJobEvent(job)

	jobs, _ := client.BatchV1().Jobs("test-ns").List(context.TODO(), metav1.ListOptions{})
	for _, j := range jobs.Items {
		if j.Name == "train-job-aibom-postprocess" {
			t.Error("should not create a second postprocess job")
		}
	}
}

func TestOnJobEvent_NoInstrumentedPods_Skips(t *testing.T) {
	ns := enabledNamespace("test-ns")
	job := completedJob("plain-job", "test-ns")
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "plain-job-pod",
			Namespace: "test-ns",
			Labels:    map[string]string{"batch.kubernetes.io/job-name": "plain-job"},
		},
		Spec: corev1.PodSpec{
			RestartPolicy: corev1.RestartPolicyNever,
			Containers:    []corev1.Container{{Name: "test", Image: "busybox"}},
		},
	}

	client := fake.NewSimpleClientset(ns, job, pod)
	w := New(client, "busybox:latest")
	startWatcher(t, w)

	w.onJobEvent(job)

	_, err := client.BatchV1().Jobs("test-ns").Get(context.TODO(), "plain-job-aibom-postprocess", metav1.GetOptions{})
	if err == nil {
		t.Error("postprocess job should not have been created for non-instrumented job")
	}
}

func TestOnJobEvent_NoGPU_Skips(t *testing.T) {
	ns := enabledNamespace("test-ns")
	job := completedJob("cpu-job", "test-ns")
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "cpu-job-pod",
			Namespace: "test-ns",
			Labels: map[string]string{
				"batch.kubernetes.io/job-name": "cpu-job",
				LabelInstrumented:              "true",
			},
		},
		Spec: corev1.PodSpec{
			RestartPolicy: corev1.RestartPolicyNever,
			Containers:    []corev1.Container{{Name: "test", Image: "busybox"}},
		},
	}

	client := fake.NewSimpleClientset(ns, job, pod)
	w := New(client, "busybox:latest")
	startWatcher(t, w)

	w.onJobEvent(job)

	_, err := client.BatchV1().Jobs("test-ns").Get(context.TODO(), "cpu-job-aibom-postprocess", metav1.GetOptions{})
	if err == nil {
		t.Error("postprocess job should not have been created for non-GPU job")
	}
}

func TestOnJobEvent_PostprocessJob_Skips(t *testing.T) {
	ns := enabledNamespace("test-ns")
	job := completedJob("train-job-aibom-postprocess", "test-ns")
	job.Labels = map[string]string{LabelPostprocessFor: "train-job"}
	pod := instrumentedPod("train-job-aibom-postprocess", "test-ns")

	client := fake.NewSimpleClientset(ns, job, pod)
	w := New(client, "busybox:latest")
	startWatcher(t, w)

	w.onJobEvent(job)

	jobs, _ := client.BatchV1().Jobs("test-ns").List(context.TODO(), metav1.ListOptions{})
	for _, j := range jobs.Items {
		if j.Name == "train-job-aibom-postprocess-aibom-postprocess" {
			t.Error("should not create a postprocess job for a postprocess job")
		}
	}
}

func TestFinalizerAddedToGPUJob(t *testing.T) {
	ns := enabledNamespace("test-ns")
	job := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{Name: "gpu-job", Namespace: "test-ns"},
	}
	pod := instrumentedPod("gpu-job", "test-ns")

	client := fake.NewSimpleClientset(ns, job, pod)
	w := New(client, "busybox:latest")
	startWatcher(t, w)

	w.onJobEvent(job)

	updated, err := client.BatchV1().Jobs("test-ns").Get(context.TODO(), "gpu-job", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("could not get job: %v", err)
	}
	if !hasFinalizer(updated) {
		t.Error("finalizer should have been added to GPU job")
	}

	_, err = client.BatchV1().Jobs("test-ns").Get(context.TODO(), "gpu-job-aibom-postprocess", metav1.GetOptions{})
	if err == nil {
		t.Error("postprocess job should not be created before job completes")
	}
}

func TestFinalizerNotAddedToNonGPUJob(t *testing.T) {
	ns := enabledNamespace("test-ns")
	job := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{Name: "cpu-job", Namespace: "test-ns"},
	}
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "cpu-job-pod",
			Namespace: "test-ns",
			Labels: map[string]string{
				"batch.kubernetes.io/job-name": "cpu-job",
				LabelInstrumented:              "true",
			},
		},
		Spec: corev1.PodSpec{
			RestartPolicy: corev1.RestartPolicyNever,
			Containers:    []corev1.Container{{Name: "test", Image: "busybox"}},
		},
	}

	client := fake.NewSimpleClientset(ns, job, pod)
	w := New(client, "busybox:latest")
	startWatcher(t, w)

	w.onJobEvent(job)

	updated, _ := client.BatchV1().Jobs("test-ns").Get(context.TODO(), "cpu-job", metav1.GetOptions{})
	if hasFinalizer(updated) {
		t.Error("finalizer should not be added to non-GPU job without AIBOM annotations")
	}
}

func TestPostprocessOnDeletion(t *testing.T) {
	ns := enabledNamespace("test-ns")
	now := metav1.Now()
	job := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "server-job",
			Namespace:         "test-ns",
			DeletionTimestamp: &now,
			Finalizers:        []string{finalizerName},
			Annotations: map[string]string{
				"aibom.io/experiment-intent": "inference",
				"aibom.io/model-name":        "granite-8b",
			},
		},
	}
	pod := instrumentedPod("server-job", "test-ns")

	client := fake.NewSimpleClientset(ns, job, pod)
	w := New(client, "aibom-postprocess:latest")
	w.logReader = &mockLogReader{logs: map[string]string{
		"test-ns/server-job-pod/aibom-discovery": "===AIBOM_DISCOVERY_START===\n{\"gpu\":{\"gpu_count\":\"1\"}}\n===AIBOM_DISCOVERY_END===\n",
	}}
	startWatcher(t, w)

	w.onJobEvent(job)

	ppJob, err := client.BatchV1().Jobs("test-ns").Get(context.TODO(), "server-job-aibom-postprocess", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("postprocess job not created on deletion: %v", err)
	}
	if ppJob.Labels[LabelPostprocessFor] != "server-job" {
		t.Errorf("label %s = %q, want %q", LabelPostprocessFor, ppJob.Labels[LabelPostprocessFor], "server-job")
	}

	cm, err := client.CoreV1().ConfigMaps("test-ns").Get(context.TODO(), "server-job-aibom-postprocess-data", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("configmap not created: %v", err)
	}
	if !strings.Contains(cm.Data["annotations.json"], "inference") {
		t.Errorf("annotations should contain experiment-intent: %s", cm.Data["annotations.json"])
	}
}

func TestFinalizerAddedToAnnotatedJob(t *testing.T) {
	ns := enabledNamespace("test-ns")
	job := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "annotated-job",
			Namespace: "test-ns",
			Annotations: map[string]string{
				"aibom.io/experiment-intent": "training",
			},
		},
	}
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "annotated-job-pod",
			Namespace: "test-ns",
			Labels: map[string]string{
				"batch.kubernetes.io/job-name": "annotated-job",
				LabelInstrumented:              "true",
			},
		},
		Spec: corev1.PodSpec{
			RestartPolicy: corev1.RestartPolicyNever,
			Containers:    []corev1.Container{{Name: "test", Image: "busybox"}},
		},
	}

	client := fake.NewSimpleClientset(ns, job, pod)
	w := New(client, "busybox:latest")
	startWatcher(t, w)

	w.onJobEvent(job)

	updated, _ := client.BatchV1().Jobs("test-ns").Get(context.TODO(), "annotated-job", metav1.GetOptions{})
	if !hasFinalizer(updated) {
		t.Error("finalizer should be added to job with AIBOM annotations even without GPU")
	}
}

func TestJobNameTruncation(t *testing.T) {
	longName := strings.Repeat("a", 60)
	result := postprocessJobName(longName)

	if len(result) > maxJobNameLength {
		t.Errorf("postprocess job name length %d exceeds max %d", len(result), maxJobNameLength)
	}

	if !strings.HasSuffix(result, postprocessSuffix) {
		t.Errorf("postprocess job name %q should end with %q", result, postprocessSuffix)
	}

	shortResult := postprocessJobName("my-job")
	if shortResult != "my-job-aibom-postprocess" {
		t.Errorf("postprocess job name = %q, want %q", shortResult, "my-job-aibom-postprocess")
	}

	// Name that would produce a trailing dash after truncation
	dashName := strings.Repeat("a", 44) + "-"
	dashResult := postprocessJobName(dashName)
	if strings.Contains(dashResult, "--") {
		t.Errorf("postprocess job name %q should not contain double dash", dashResult)
	}
	if len(dashResult) > maxJobNameLength {
		t.Errorf("postprocess job name length %d exceeds max %d", len(dashResult), maxJobNameLength)
	}
}
