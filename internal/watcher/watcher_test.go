package watcher

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/gavinsan33/aibom-webhook-service/internal/aibomdata"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
)

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

// instrumentedBarePod is like instrumentedPod but has no owning Job — the shape of a
// KServe predictor pod (ReplicaSet-owned) that the pod-level finalizer path targets.
func instrumentedBarePod(name, namespace string) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
			Labels:    map[string]string{LabelInstrumented: "true"},
		},
		Spec: corev1.PodSpec{
			RestartPolicy:  corev1.RestartPolicyNever,
			InitContainers: []corev1.Container{{Name: initContainerName, Image: "pytorch:latest"}},
			Containers: []corev1.Container{{
				Name:  "kserve-container",
				Image: "vllm:latest",
				Resources: corev1.ResourceRequirements{
					Limits: corev1.ResourceList{
						"nvidia.com/gpu": resource.MustParse("1"),
					},
				},
			}},
		},
	}
}

func TestShouldPostprocessPod_NoQualifyingSignal(t *testing.T) {
	pod := instrumentedBarePod("web-pod", "gavin-test")
	pod.Spec.Containers[0].Resources = corev1.ResourceRequirements{}

	w := New(fake.NewSimpleClientset(), Config{})
	if w.shouldPostprocessPod(pod) {
		t.Error("expected pod with no GPU request or annotations to be skipped")
	}
}

func TestShouldPostprocessPod_DebugPostprocessAllPods(t *testing.T) {
	pod := instrumentedBarePod("web-pod", "gavin-test")
	pod.Spec.Containers[0].Resources = corev1.ResourceRequirements{}

	w := New(fake.NewSimpleClientset(), Config{DebugPostprocessAllPods: true})
	if !w.shouldPostprocessPod(pod) {
		t.Error("expected debug-postprocess-all-pods to qualify a pod with no GPU request or annotations")
	}
}

func startWatcher(t *testing.T, w *Watcher) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)

	w.factory.Start(ctx.Done())
	w.factory.WaitForCacheSync(ctx.Done())
	w.podFactory.Start(ctx.Done())
	w.podFactory.WaitForCacheSync(ctx.Done())
	time.Sleep(50 * time.Millisecond)
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

	result := collectAIBOMAnnotations(job.Annotations)

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
	result := collectAIBOMAnnotations(job.Annotations)
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

func TestIsJobFinished(t *testing.T) {
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
			expected: true,
		},
	}

	w := &Watcher{}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := w.isJobFinished(tt.job); got != tt.expected {
				t.Errorf("isJobFinished() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestIsNamespaceEnabled(t *testing.T) {
	client := fake.NewSimpleClientset(enabledNamespace("enabled-ns"), disabledNamespace("disabled-ns"))
	w := New(client, Config{PostprocessImage: "busybox:latest"})
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

	// The aibom-discovery init container writes its own data directly into the
	// data ConfigMap (see extractDataFromPod) before the workload completes.
	discoveryJSON := `{"pod_metadata":{"name":"train-job-pod","uid":"abc123"},"gpu":{"gpu_count":"2"}}`
	dataConfigMap := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "train-job-aibom-postprocess-data",
			Namespace: "test-ns",
		},
		Data: map[string]string{
			"discovery-train-job-pod.json": discoveryJSON,
		},
	}

	client := fake.NewSimpleClientset(ns, job, pod, dataConfigMap)
	w := New(client, Config{PostprocessImage: "aibom-postprocess:latest"})
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

	if container.SecurityContext == nil {
		t.Fatal("container.SecurityContext = nil, want hardened SecurityContext")
	}
	if container.SecurityContext.AllowPrivilegeEscalation == nil || *container.SecurityContext.AllowPrivilegeEscalation {
		t.Error("AllowPrivilegeEscalation should be false")
	}
	if container.SecurityContext.ReadOnlyRootFilesystem == nil || !*container.SecurityContext.ReadOnlyRootFilesystem {
		t.Error("ReadOnlyRootFilesystem should be true")
	}
	if container.SecurityContext.Capabilities == nil || len(container.SecurityContext.Capabilities.Drop) != 1 || container.SecurityContext.Capabilities.Drop[0] != "ALL" {
		t.Errorf("Capabilities.Drop = %v, want [ALL]", container.SecurityContext.Capabilities)
	}
	podSC := ppJob.Spec.Template.Spec.SecurityContext
	if podSC == nil || podSC.RunAsNonRoot == nil || !*podSC.RunAsNonRoot {
		t.Error("pod SecurityContext.RunAsNonRoot should be true")
	}
	if container.Resources.Requests.Cpu().IsZero() || container.Resources.Limits.Cpu().IsZero() {
		t.Errorf("Resources = %+v, want non-zero CPU requests/limits", container.Resources)
	}
	if container.Resources.Requests.Memory().IsZero() || container.Resources.Limits.Memory().IsZero() {
		t.Errorf("Resources = %+v, want non-zero memory requests/limits", container.Resources)
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

	// Verify volume mounts: the data ConfigMap, plus the optional service-ca bundle
	// used to trust in-cluster Prometheus/Thanos Querier's TLS cert.
	if len(ppJob.Spec.Template.Spec.Volumes) != 2 {
		t.Fatalf("expected 2 volumes, got %d", len(ppJob.Spec.Template.Spec.Volumes))
	}
	if ppJob.Spec.Template.Spec.Volumes[0].ConfigMap.Name != "train-job-aibom-postprocess-data" {
		t.Errorf("volume configmap name = %q, want %q", ppJob.Spec.Template.Spec.Volumes[0].ConfigMap.Name, "train-job-aibom-postprocess-data")
	}
	serviceCAVolume := ppJob.Spec.Template.Spec.Volumes[1]
	if serviceCAVolume.ConfigMap.Name != serviceCAConfigMapName {
		t.Errorf("service-ca volume configmap name = %q, want %q", serviceCAVolume.ConfigMap.Name, serviceCAConfigMapName)
	}
	if serviceCAVolume.ConfigMap.Optional == nil || !*serviceCAVolume.ConfigMap.Optional {
		t.Error("service-ca volume should be optional")
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
	w := New(client, Config{PostprocessImage: "aibom-postprocess:latest"})
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
	w := New(client, Config{PostprocessImage: "busybox:latest"})
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
	w := New(client, Config{PostprocessImage: "busybox:latest"})
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
	w := New(client, Config{PostprocessImage: "busybox:latest"})
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
	w := New(client, Config{PostprocessImage: "busybox:latest"})
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
	w := New(client, Config{PostprocessImage: "busybox:latest"})
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
	w := New(client, Config{PostprocessImage: "busybox:latest"})
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
	w := New(client, Config{PostprocessImage: "busybox:latest"})
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
	w := New(client, Config{PostprocessImage: "busybox:latest"})
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
	w := New(client, Config{PostprocessImage: "aibom-postprocess:latest"})
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
	w := New(client, Config{PostprocessImage: "busybox:latest"})
	startWatcher(t, w)

	w.onJobEvent(job)

	updated, _ := client.BatchV1().Jobs("test-ns").Get(context.TODO(), "annotated-job", metav1.GetOptions{})
	if !hasFinalizer(updated) {
		t.Error("finalizer should be added to job with AIBOM annotations even without GPU")
	}
}

// ---------------------------------------------------------------------------
// Pod-level finalizer tests (bare/ReplicaSet-owned pods, e.g. KServe predictors)
// ---------------------------------------------------------------------------

func TestPodFinalizerAddedToGPUPod(t *testing.T) {
	ns := enabledNamespace("test-ns")
	pod := instrumentedBarePod("predictor-pod", "test-ns")

	client := fake.NewSimpleClientset(ns, pod)
	w := New(client, Config{PostprocessImage: "busybox:latest"})
	startWatcher(t, w)

	w.onPodEvent(pod)

	updated, err := client.CoreV1().Pods("test-ns").Get(context.TODO(), "predictor-pod", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("could not get pod: %v", err)
	}
	if !hasPodFinalizer(updated) {
		t.Error("finalizer should have been added to GPU pod")
	}

	_, err = client.BatchV1().Jobs("test-ns").Get(context.TODO(), "predictor-pod-aibom-postprocess", metav1.GetOptions{})
	if err == nil {
		t.Error("postprocess job should not be created before pod is deleted")
	}
}

// TestPostprocessReadsStorageInfoFromDiscoveryData reproduces the scenario
// that motivated writing storage.json in the discovery init container (see
// generate_snapshot.py's resolve_inference_service_storage) instead of the
// watcher looking it up lazily at pod-deletion time: deleting an
// InferenceService removes it from etcd immediately, well before Kubernetes'
// garbage collector cascades the delete down to the Pod, so by the time
// postprocessing runs at pod deletion, a live Get against the
// InferenceService would 404 almost every time. Model identity must instead
// come from what the init container already wrote at pod startup, while the
// InferenceService still existed — the watcher itself never talks to
// serving.kserve.io at all.
func TestPostprocessReadsStorageInfoFromDiscoveryData(t *testing.T) {
	ns := enabledNamespace("test-ns")
	now := metav1.Now()
	pod := instrumentedBarePod("granite-model-predictor-abc123", "test-ns")
	pod.Labels[aibomdata.LabelKServeInferenceService] = "granite-model"
	pod.Finalizers = []string{podFinalizerName}
	pod.DeletionTimestamp = &now

	storageJSON := `{"inference_service":"granite-model","storage_path":"models/tinyllama-1.1b-chat"}`
	dataConfigMap := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      pod.Name + "-aibom-postprocess-data",
			Namespace: "test-ns",
		},
		Data: map[string]string{
			fmt.Sprintf("storage-%s.json", pod.Name): storageJSON,
		},
	}

	client := fake.NewSimpleClientset(ns, pod, dataConfigMap)
	w := New(client, Config{PostprocessImage: "aibom-postprocess:latest"})
	startWatcher(t, w)

	w.onPodEvent(pod)

	cm, err := client.CoreV1().ConfigMaps("test-ns").Get(context.TODO(), pod.Name+"-aibom-postprocess-data", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("data configmap not found: %v", err)
	}
	var got map[string]string
	if err := json.Unmarshal([]byte(cm.Data["storage.json"]), &got); err != nil {
		t.Fatalf("invalid storage.json: %v", err)
	}
	if got["storage_path"] != "models/tinyllama-1.1b-chat" {
		t.Errorf("storage_path = %q, want %q", got["storage_path"], "models/tinyllama-1.1b-chat")
	}
}

func TestPostprocessDefaultsStorageInfoWhenAbsent(t *testing.T) {
	ns := enabledNamespace("test-ns")
	now := metav1.Now()
	pod := instrumentedBarePod("web-pod", "test-ns")
	pod.Finalizers = []string{podFinalizerName}
	pod.DeletionTimestamp = &now

	client := fake.NewSimpleClientset(ns, pod)
	w := New(client, Config{PostprocessImage: "aibom-postprocess:latest"})
	startWatcher(t, w)

	w.onPodEvent(pod)

	cm, err := client.CoreV1().ConfigMaps("test-ns").Get(context.TODO(), pod.Name+"-aibom-postprocess-data", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("data configmap not found: %v", err)
	}
	if cm.Data["storage.json"] != "{}" {
		t.Errorf("storage.json = %q, want %q for a workload with no InferenceService storage info", cm.Data["storage.json"], "{}")
	}
}

func TestPodFinalizerNotAddedToNonGPUPod(t *testing.T) {
	ns := enabledNamespace("test-ns")
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "cpu-pod",
			Namespace: "test-ns",
			Labels:    map[string]string{LabelInstrumented: "true"},
		},
		Spec: corev1.PodSpec{
			RestartPolicy: corev1.RestartPolicyNever,
			Containers:    []corev1.Container{{Name: "test", Image: "busybox"}},
		},
	}

	client := fake.NewSimpleClientset(ns, pod)
	w := New(client, Config{PostprocessImage: "busybox:latest"})
	startWatcher(t, w)

	w.onPodEvent(pod)

	updated, _ := client.CoreV1().Pods("test-ns").Get(context.TODO(), "cpu-pod", metav1.GetOptions{})
	if hasPodFinalizer(updated) {
		t.Error("finalizer should not be added to non-GPU pod without AIBOM annotations")
	}
}

func TestPodFinalizerAddedToAnnotatedPod(t *testing.T) {
	ns := enabledNamespace("test-ns")
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "annotated-pod",
			Namespace: "test-ns",
			Labels:    map[string]string{LabelInstrumented: "true"},
			Annotations: map[string]string{
				"aibom.io/experiment-intent": "inference",
			},
		},
		Spec: corev1.PodSpec{
			RestartPolicy: corev1.RestartPolicyNever,
			Containers:    []corev1.Container{{Name: "test", Image: "busybox"}},
		},
	}

	client := fake.NewSimpleClientset(ns, pod)
	w := New(client, Config{PostprocessImage: "busybox:latest"})
	startWatcher(t, w)

	w.onPodEvent(pod)

	updated, _ := client.CoreV1().Pods("test-ns").Get(context.TODO(), "annotated-pod", metav1.GetOptions{})
	if !hasPodFinalizer(updated) {
		t.Error("finalizer should be added to pod with AIBOM annotations even without GPU")
	}
}

func TestPostprocessOnPodDeletion(t *testing.T) {
	ns := enabledNamespace("test-ns")
	now := metav1.Now()
	pod := instrumentedBarePod("predictor-pod", "test-ns")
	pod.DeletionTimestamp = &now
	pod.Finalizers = []string{podFinalizerName}
	pod.Annotations = map[string]string{
		"aibom.io/experiment-intent": "inference",
		"aibom.io/model-name":        "granite-8b",
	}

	client := fake.NewSimpleClientset(ns, pod)
	w := New(client, Config{PostprocessImage: "aibom-postprocess:latest"})
	startWatcher(t, w)

	w.onPodEvent(pod)

	ppJob, err := client.BatchV1().Jobs("test-ns").Get(context.TODO(), "predictor-pod-aibom-postprocess", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("postprocess job not created on pod deletion: %v", err)
	}
	if ppJob.Labels[LabelPostprocessFor] != "predictor-pod" {
		t.Errorf("label %s = %q, want %q", LabelPostprocessFor, ppJob.Labels[LabelPostprocessFor], "predictor-pod")
	}

	cm, err := client.CoreV1().ConfigMaps("test-ns").Get(context.TODO(), "predictor-pod-aibom-postprocess-data", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("configmap not created: %v", err)
	}
	if !strings.Contains(cm.Data["annotations.json"], "inference") {
		t.Errorf("annotations should contain experiment-intent: %s", cm.Data["annotations.json"])
	}

	updated, err := client.CoreV1().Pods("test-ns").Get(context.TODO(), "predictor-pod", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("could not re-fetch pod: %v", err)
	}
	if hasPodFinalizer(updated) {
		t.Error("finalizer should have been removed after postprocess job creation")
	}
	if updated.Annotations[AnnotationPostprocess] != "predictor-pod-aibom-postprocess" {
		t.Errorf("annotation %s = %q, want %q", AnnotationPostprocess, updated.Annotations[AnnotationPostprocess], "predictor-pod-aibom-postprocess")
	}
}

func TestOnPodEvent_JobOwnedPod_Skipped(t *testing.T) {
	ns := enabledNamespace("test-ns")
	now := metav1.Now()
	pod := instrumentedBarePod("owned-pod", "test-ns")
	pod.Labels["batch.kubernetes.io/job-name"] = "some-job"
	pod.DeletionTimestamp = &now
	pod.Finalizers = []string{podFinalizerName}

	client := fake.NewSimpleClientset(ns, pod)
	w := New(client, Config{PostprocessImage: "busybox:latest"})
	startWatcher(t, w)

	w.onPodEvent(pod)

	_, err := client.BatchV1().Jobs("test-ns").Get(context.TODO(), "owned-pod-aibom-postprocess", metav1.GetOptions{})
	if err == nil {
		t.Error("postprocess job should not be created for a Job-owned pod via the pod path")
	}

	updated, _ := client.CoreV1().Pods("test-ns").Get(context.TODO(), "owned-pod", metav1.GetOptions{})
	if updated.Annotations[AnnotationPostprocess] != "" {
		t.Error("Job-owned pod should not be annotated by the pod path")
	}
}

func TestOnPodEvent_NotInstrumented_Skipped(t *testing.T) {
	ns := enabledNamespace("test-ns")
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "uninstrumented-pod",
			Namespace: "test-ns",
		},
		Spec: corev1.PodSpec{
			RestartPolicy: corev1.RestartPolicyNever,
			Containers: []corev1.Container{{
				Name:  "test",
				Image: "busybox",
				Resources: corev1.ResourceRequirements{
					Limits: corev1.ResourceList{"nvidia.com/gpu": resource.MustParse("1")},
				},
			}},
		},
	}

	client := fake.NewSimpleClientset(ns, pod)
	w := New(client, Config{PostprocessImage: "busybox:latest"})
	startWatcher(t, w)

	w.onPodEvent(pod)

	updated, _ := client.CoreV1().Pods("test-ns").Get(context.TODO(), "uninstrumented-pod", metav1.GetOptions{})
	if hasPodFinalizer(updated) {
		t.Error("finalizer should not be added to a pod the webhook never instrumented")
	}
}

func newAIBOMPostprocessFixtures(jobName, namespace string) (*batchv1.Job, *corev1.Pod, *corev1.ConfigMap) {
	ppJob := completedJob(jobName+postprocessSuffix, namespace)
	ppJob.Labels = map[string]string{LabelPostprocessFor: jobName}
	ppPod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      jobName + postprocessSuffix + "-pod",
			Namespace: namespace,
			Labels: map[string]string{
				"batch.kubernetes.io/job-name": jobName + postprocessSuffix,
			},
		},
		Spec: corev1.PodSpec{
			RestartPolicy: corev1.RestartPolicyNever,
			Containers:    []corev1.Container{{Name: postprocessContainerName, Image: "aibom-postprocess:latest"}},
		},
	}
	dataConfigMap := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      jobName + postprocessSuffix + configMapSuffix,
			Namespace: namespace,
			Labels:    map[string]string{LabelPostprocessFor: jobName},
		},
	}
	return ppJob, ppPod, dataConfigMap
}

// TestCollectAIBOM verifies the post-success bookkeeping: the AIBOM custom
// resource itself is created directly by postprocess.py via the Kubernetes
// API (not exercised here, see postprocess/), so collectAIBOM's only job is
// to mark the postprocess Job collected and clean up the Job/ConfigMap.
func TestCollectAIBOM(t *testing.T) {
	ns := enabledNamespace("test-ns")
	ppJob, ppPod, dataConfigMap := newAIBOMPostprocessFixtures("train-job", "test-ns")

	client := fake.NewSimpleClientset(ns, ppJob, ppPod, dataConfigMap)
	w := New(client, Config{PostprocessImage: "aibom-postprocess:latest"})
	startWatcher(t, w)

	w.onJobEvent(ppJob)

	_, err := client.BatchV1().Jobs("test-ns").Get(context.TODO(), "train-job-aibom-postprocess", metav1.GetOptions{})
	if !apierrors.IsNotFound(err) {
		t.Errorf("expected postprocess job to be deleted after collection, got err=%v", err)
	}
	_, err = client.CoreV1().ConfigMaps("test-ns").Get(context.TODO(), "train-job-aibom-postprocess-data", metav1.GetOptions{})
	if !apierrors.IsNotFound(err) {
		t.Errorf("expected postprocess data configmap to be deleted after collection, got err=%v", err)
	}

	// Guard against double-collection: if a postprocess job somehow still exists
	// with AnnotationAIBOMCollected already set (e.g. deletion failed), onJobEvent
	// must not run collectAIBOM a second time.
	alreadyCollected := completedJob("train-job-aibom-postprocess", "test-ns")
	alreadyCollected.Labels = map[string]string{LabelPostprocessFor: "train-job"}
	alreadyCollected.Annotations = map[string]string{AnnotationAIBOMCollected: "2026-01-01T00:00:00Z"}

	// Recreate so the second onJobEvent has something to (not) act on.
	if _, err := client.BatchV1().Jobs("test-ns").Create(context.TODO(), alreadyCollected, metav1.CreateOptions{}); err != nil {
		t.Fatalf("could not recreate postprocess job: %v", err)
	}
	w.onJobEvent(alreadyCollected)

	_, err = client.BatchV1().Jobs("test-ns").Get(context.TODO(), "train-job-aibom-postprocess", metav1.GetOptions{})
	if err != nil {
		t.Errorf("expected already-collected postprocess job to be left alone, got err=%v", err)
	}
}

// TestCollectAIBOM_DebugKeepPostprocessJobs verifies the debug escape hatch:
// with it set, the postprocess Job/data ConfigMap survive collection (still
// annotated as collected, so a resync doesn't try to collect it again) instead
// of being deleted — for inspecting postprocess pod logs/state after the fact.
func TestCollectAIBOM_DebugKeepPostprocessJobs(t *testing.T) {
	ns := enabledNamespace("test-ns")
	ppJob, ppPod, dataConfigMap := newAIBOMPostprocessFixtures("train-job", "test-ns")

	client := fake.NewSimpleClientset(ns, ppJob, ppPod, dataConfigMap)
	w := New(client, Config{PostprocessImage: "aibom-postprocess:latest", DebugKeepPostprocessJobs: true})
	startWatcher(t, w)

	w.onJobEvent(ppJob)

	gotJob, err := client.BatchV1().Jobs("test-ns").Get(context.TODO(), "train-job-aibom-postprocess", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("expected postprocess job to survive collection, got err=%v", err)
	}
	if gotJob.Annotations[AnnotationAIBOMCollected] == "" {
		t.Error("expected postprocess job to still be annotated as collected")
	}
	if _, err := client.CoreV1().ConfigMaps("test-ns").Get(context.TODO(), "train-job-aibom-postprocess-data", metav1.GetOptions{}); err != nil {
		t.Errorf("expected postprocess data configmap to survive collection, got err=%v", err)
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
