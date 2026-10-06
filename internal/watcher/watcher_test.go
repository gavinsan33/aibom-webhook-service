package watcher

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/gavinsan33/aibom-webhook-service/internal/aibomdata"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"
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
	pod := instrumentedBarePod("web-pod", "project-gavin-test")
	pod.Spec.Containers[0].Resources = corev1.ResourceRequirements{}

	if shouldPostprocessPod(pod) {
		t.Error("expected pod with no GPU request or annotations to be skipped")
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
	// used to trust in-cluster Prometheus/Thanos Querier's TLS cert, plus the
	// optional compiled-AIBOM Ed25519 signing key (#27).
	if len(ppJob.Spec.Template.Spec.Volumes) != 3 {
		t.Fatalf("expected 3 volumes, got %d", len(ppJob.Spec.Template.Spec.Volumes))
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
	signingVolume := ppJob.Spec.Template.Spec.Volumes[2]
	if signingVolume.Secret.SecretName != aibomdata.CompiledSigningKeySecretName {
		t.Errorf("signing key volume secret name = %q, want %q", signingVolume.Secret.SecretName, aibomdata.CompiledSigningKeySecretName)
	}
	if signingVolume.Secret.Optional == nil || !*signingVolume.Secret.Optional {
		t.Error("compiled signing key volume should be optional")
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

// --- Discovery signature verification ---

func discoverySigningSecret(namespace string, key []byte) *corev1.Secret {
	return &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      aibomdata.DiscoverySigningKeySecretName,
			Namespace: namespace,
		},
		Data: map[string][]byte{aibomdata.DiscoverySigningKeyDataKey: key},
	}
}

func TestVerifyDiscoverySignature_ValidSignatureVerifies(t *testing.T) {
	key := []byte("test-key")
	payload := `{"gpu":{"gpu_count":"2"}}`
	sig := hmacHex(t, key, payload)
	if !verifySignature(key, payload, sig) {
		t.Error("expected valid signature to verify")
	}
}

func TestVerifyDiscoverySignature_WrongKeyFails(t *testing.T) {
	payload := `{"gpu":{"gpu_count":"2"}}`
	sig := hmacHex(t, []byte("real-key"), payload)
	if verifySignature([]byte("wrong-key"), payload, sig) {
		t.Error("expected signature under a different key to fail verification")
	}
}

func TestVerifyDiscoverySignature_TamperedPayloadFails(t *testing.T) {
	key := []byte("test-key")
	sig := hmacHex(t, key, `{"gpu":{"gpu_count":"2"}}`)
	if verifySignature(key, `{"gpu":{"gpu_count":"8"}}`, sig) {
		t.Error("expected a payload that doesn't match the signed one to fail verification")
	}
}

func TestVerifyDiscoverySignature_EmptySignatureFails(t *testing.T) {
	if verifySignature([]byte("key"), "payload", "") {
		t.Error("expected an empty signature to never verify")
	}
}

// TestPostprocessDropsForgedDiscoveryData is the core regression test for
// #30/#29: a pod whose discovery-<pod>.json was overwritten by something
// other than the discovery init container (no valid matching .sig, since
// the app container never has the signing key) must not have that data
// merged into the aggregate discovery.json the AIBOM gets compiled from.
func TestPostprocessDropsForgedDiscoveryData(t *testing.T) {
	ns := enabledNamespace("test-ns")
	now := metav1.Now()
	pod := instrumentedBarePod("web-pod", "test-ns")
	pod.Finalizers = []string{podFinalizerName}
	pod.DeletionTimestamp = &now

	key := []byte("shared-secret")
	secret := discoverySigningSecret("test-ns", key)
	forgedDiscovery := `{"gpu":{"gpu_count":"8"}}`
	dataConfigMap := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      pod.Name + "-aibom-postprocess-data",
			Namespace: "test-ns",
		},
		Data: map[string]string{
			fmt.Sprintf("discovery-%s.json", pod.Name): forgedDiscovery,
			// No matching .sig -- or one that doesn't verify against the
			// real key -- either way this must be treated as unsigned.
			fmt.Sprintf("discovery-%s.sig", pod.Name): "not-a-real-signature",
		},
	}

	client := fake.NewSimpleClientset(ns, pod, dataConfigMap, secret)
	w := New(client, Config{PostprocessImage: "aibom-postprocess:latest"})
	startWatcher(t, w)

	w.onPodEvent(pod)

	cm, err := client.CoreV1().ConfigMaps("test-ns").Get(context.TODO(), pod.Name+"-aibom-postprocess-data", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("data configmap not found: %v", err)
	}
	if cm.Data["discovery.json"] != "[]" {
		t.Errorf("expected forged discovery data to be dropped, got discovery.json = %q", cm.Data["discovery.json"])
	}
}

// TestPostprocessKeepsValidlySignedDiscoveryData is the companion positive
// case: genuine discovery data with a signature that verifies against the
// namespace's own key must still make it into the compiled AIBOM inputs.
func TestPostprocessKeepsValidlySignedDiscoveryData(t *testing.T) {
	ns := enabledNamespace("test-ns")
	now := metav1.Now()
	pod := instrumentedBarePod("web-pod", "test-ns")
	pod.Finalizers = []string{podFinalizerName}
	pod.DeletionTimestamp = &now

	key := []byte("shared-secret")
	secret := discoverySigningSecret("test-ns", key)
	genuineDiscovery := `{"gpu":{"gpu_count":"2"}}`
	dataConfigMap := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      pod.Name + "-aibom-postprocess-data",
			Namespace: "test-ns",
		},
		Data: map[string]string{
			fmt.Sprintf("discovery-%s.json", pod.Name): genuineDiscovery,
			fmt.Sprintf("discovery-%s.sig", pod.Name):  hmacHex(t, key, genuineDiscovery),
		},
	}

	client := fake.NewSimpleClientset(ns, pod, dataConfigMap, secret)
	w := New(client, Config{PostprocessImage: "aibom-postprocess:latest"})
	startWatcher(t, w)

	w.onPodEvent(pod)

	cm, err := client.CoreV1().ConfigMaps("test-ns").Get(context.TODO(), pod.Name+"-aibom-postprocess-data", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("data configmap not found: %v", err)
	}
	if cm.Data["discovery.json"] != "["+genuineDiscovery+"]" {
		t.Errorf("expected genuine discovery data to survive verification, got discovery.json = %q", cm.Data["discovery.json"])
	}
}

// TestPostprocessKeepsUnverifiedDiscoveryWhenNoSigningKeyConfigured covers
// the gradual-rollout case: a namespace whose aibom-workload-namespace chart
// install predates signing.yaml has no Secret to verify against at all, so
// discovery data passes through unverified rather than being dropped.
func TestPostprocessKeepsUnverifiedDiscoveryWhenNoSigningKeyConfigured(t *testing.T) {
	ns := enabledNamespace("test-ns")
	now := metav1.Now()
	pod := instrumentedBarePod("web-pod", "test-ns")
	pod.Finalizers = []string{podFinalizerName}
	pod.DeletionTimestamp = &now

	unsignedDiscovery := `{"gpu":{"gpu_count":"2"}}`
	dataConfigMap := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      pod.Name + "-aibom-postprocess-data",
			Namespace: "test-ns",
		},
		Data: map[string]string{
			fmt.Sprintf("discovery-%s.json", pod.Name): unsignedDiscovery,
		},
	}

	// Deliberately no discoverySigningSecret in the fake clientset.
	client := fake.NewSimpleClientset(ns, pod, dataConfigMap)
	w := New(client, Config{PostprocessImage: "aibom-postprocess:latest"})
	startWatcher(t, w)

	w.onPodEvent(pod)

	cm, err := client.CoreV1().ConfigMaps("test-ns").Get(context.TODO(), pod.Name+"-aibom-postprocess-data", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("data configmap not found: %v", err)
	}
	if cm.Data["discovery.json"] != "["+unsignedDiscovery+"]" {
		t.Errorf("expected unsigned discovery data to pass through when no signing key is configured, got discovery.json = %q", cm.Data["discovery.json"])
	}
}

// TestPostprocessDropsForgedStorageData mirrors
// TestPostprocessDropsForgedDiscoveryData for #44: storage-<pod>.json is
// written by the same trusted discovery init container as
// discovery-<pod>.json (see generate_snapshot.py's
// resolve_inference_service_storage), so it's verified the same way and
// with the same key.
func TestPostprocessDropsForgedStorageData(t *testing.T) {
	ns := enabledNamespace("test-ns")
	now := metav1.Now()
	pod := instrumentedBarePod("granite-model-predictor-abc123", "test-ns")
	pod.Finalizers = []string{podFinalizerName}
	pod.DeletionTimestamp = &now

	key := []byte("shared-secret")
	secret := discoverySigningSecret("test-ns", key)
	forgedStorage := `{"inference_service":"forged","storage_path":"models/forged"}`
	dataConfigMap := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      pod.Name + "-aibom-postprocess-data",
			Namespace: "test-ns",
		},
		Data: map[string]string{
			fmt.Sprintf("storage-%s.json", pod.Name): forgedStorage,
			fmt.Sprintf("storage-%s.sig", pod.Name):  "not-a-real-signature",
		},
	}

	client := fake.NewSimpleClientset(ns, pod, dataConfigMap, secret)
	w := New(client, Config{PostprocessImage: "aibom-postprocess:latest"})
	startWatcher(t, w)

	w.onPodEvent(pod)

	cm, err := client.CoreV1().ConfigMaps("test-ns").Get(context.TODO(), pod.Name+"-aibom-postprocess-data", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("data configmap not found: %v", err)
	}
	if cm.Data["storage.json"] != "{}" {
		t.Errorf("expected forged storage data to be dropped, got storage.json = %q", cm.Data["storage.json"])
	}
}

// TestPostprocessKeepsValidlySignedStorageData is the companion positive
// case for #44.
func TestPostprocessKeepsValidlySignedStorageData(t *testing.T) {
	ns := enabledNamespace("test-ns")
	now := metav1.Now()
	pod := instrumentedBarePod("granite-model-predictor-abc123", "test-ns")
	pod.Finalizers = []string{podFinalizerName}
	pod.DeletionTimestamp = &now

	key := []byte("shared-secret")
	secret := discoverySigningSecret("test-ns", key)
	genuineStorage := `{"inference_service":"granite-model","storage_path":"models/tinyllama-1.1b-chat"}`
	dataConfigMap := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      pod.Name + "-aibom-postprocess-data",
			Namespace: "test-ns",
		},
		Data: map[string]string{
			fmt.Sprintf("storage-%s.json", pod.Name): genuineStorage,
			fmt.Sprintf("storage-%s.sig", pod.Name):  hmacHex(t, key, genuineStorage),
		},
	}

	client := fake.NewSimpleClientset(ns, pod, dataConfigMap, secret)
	w := New(client, Config{PostprocessImage: "aibom-postprocess:latest"})
	startWatcher(t, w)

	w.onPodEvent(pod)

	cm, err := client.CoreV1().ConfigMaps("test-ns").Get(context.TODO(), pod.Name+"-aibom-postprocess-data", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("data configmap not found: %v", err)
	}
	if cm.Data["storage.json"] != genuineStorage {
		t.Errorf("expected genuine storage data to survive verification, got storage.json = %q", cm.Data["storage.json"])
	}
}

// TestPostprocessKeepsUnverifiedStorageWhenNoSigningKeyConfigured mirrors
// the discovery-data gradual-rollout case for storage data.
func TestPostprocessKeepsUnverifiedStorageWhenNoSigningKeyConfigured(t *testing.T) {
	ns := enabledNamespace("test-ns")
	now := metav1.Now()
	pod := instrumentedBarePod("granite-model-predictor-abc123", "test-ns")
	pod.Finalizers = []string{podFinalizerName}
	pod.DeletionTimestamp = &now

	unsignedStorage := `{"inference_service":"granite-model","storage_path":"models/tinyllama-1.1b-chat"}`
	dataConfigMap := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      pod.Name + "-aibom-postprocess-data",
			Namespace: "test-ns",
		},
		Data: map[string]string{
			fmt.Sprintf("storage-%s.json", pod.Name): unsignedStorage,
		},
	}

	// Deliberately no discoverySigningSecret in the fake clientset.
	client := fake.NewSimpleClientset(ns, pod, dataConfigMap)
	w := New(client, Config{PostprocessImage: "aibom-postprocess:latest"})
	startWatcher(t, w)

	w.onPodEvent(pod)

	cm, err := client.CoreV1().ConfigMaps("test-ns").Get(context.TODO(), pod.Name+"-aibom-postprocess-data", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("data configmap not found: %v", err)
	}
	if cm.Data["storage.json"] != unsignedStorage {
		t.Errorf("expected unsigned storage data to pass through when no signing key is configured, got storage.json = %q", cm.Data["storage.json"])
	}
}

func datasetSigningSecret(namespace string, key []byte) *corev1.Secret {
	return &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      aibomdata.DatasetSigningKeySecretName,
			Namespace: namespace,
		},
		Data: map[string][]byte{aibomdata.DatasetSigningKeyDataKey: key},
	}
}

// TestPostprocessDropsForgedDatasetData is the #47 counterpart to
// TestPostprocessDropsForgedDiscoveryData: dataset-<pod>.json is signed by
// dataset_sidecar.py using its own separate key (not the discovery one --
// see aibomdata.DatasetSigningKeySecretName), so a pod's data with no
// matching valid .sig must not survive into the aggregate dataset.json.
func TestPostprocessDropsForgedDatasetData(t *testing.T) {
	ns := enabledNamespace("test-ns")
	now := metav1.Now()
	pod := instrumentedBarePod("web-pod", "test-ns")
	pod.Finalizers = []string{podFinalizerName}
	pod.DeletionTimestamp = &now

	key := []byte("dataset-shared-secret")
	secret := datasetSigningSecret("test-ns", key)
	forgedDataset := `{"datasets":[{"dataset_name":"forged-dataset"}]}`
	dataConfigMap := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      pod.Name + "-aibom-postprocess-data",
			Namespace: "test-ns",
		},
		Data: map[string]string{
			fmt.Sprintf("dataset-%s.json", pod.Name): forgedDataset,
			fmt.Sprintf("dataset-%s.sig", pod.Name):  "not-a-real-signature",
		},
	}

	client := fake.NewSimpleClientset(ns, pod, dataConfigMap, secret)
	w := New(client, Config{PostprocessImage: "aibom-postprocess:latest"})
	startWatcher(t, w)

	w.onPodEvent(pod)

	cm, err := client.CoreV1().ConfigMaps("test-ns").Get(context.TODO(), pod.Name+"-aibom-postprocess-data", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("data configmap not found: %v", err)
	}
	if cm.Data["dataset.json"] != "{}" {
		t.Errorf("expected forged dataset data to be dropped, got dataset.json = %q", cm.Data["dataset.json"])
	}
}

// TestPostprocessKeepsValidlySignedDatasetData is the companion positive case.
func TestPostprocessKeepsValidlySignedDatasetData(t *testing.T) {
	ns := enabledNamespace("test-ns")
	now := metav1.Now()
	pod := instrumentedBarePod("web-pod", "test-ns")
	pod.Finalizers = []string{podFinalizerName}
	pod.DeletionTimestamp = &now

	key := []byte("dataset-shared-secret")
	secret := datasetSigningSecret("test-ns", key)
	genuineDataset := `{"datasets":[{"dataset_name":"tatsu-lab/alpaca"}]}`
	dataConfigMap := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      pod.Name + "-aibom-postprocess-data",
			Namespace: "test-ns",
		},
		Data: map[string]string{
			fmt.Sprintf("dataset-%s.json", pod.Name): genuineDataset,
			fmt.Sprintf("dataset-%s.sig", pod.Name):  hmacHex(t, key, genuineDataset),
		},
	}

	client := fake.NewSimpleClientset(ns, pod, dataConfigMap, secret)
	w := New(client, Config{PostprocessImage: "aibom-postprocess:latest"})
	startWatcher(t, w)

	w.onPodEvent(pod)

	cm, err := client.CoreV1().ConfigMaps("test-ns").Get(context.TODO(), pod.Name+"-aibom-postprocess-data", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("data configmap not found: %v", err)
	}
	if !strings.Contains(cm.Data["dataset.json"], "tatsu-lab/alpaca") {
		t.Errorf("expected genuine dataset data to survive verification, got dataset.json = %q", cm.Data["dataset.json"])
	}
}

// TestPostprocessKeepsUnverifiedDatasetWhenNoSigningKeyConfigured mirrors
// the gradual-rollout case for dataset data.
func TestPostprocessKeepsUnverifiedDatasetWhenNoSigningKeyConfigured(t *testing.T) {
	ns := enabledNamespace("test-ns")
	now := metav1.Now()
	pod := instrumentedBarePod("web-pod", "test-ns")
	pod.Finalizers = []string{podFinalizerName}
	pod.DeletionTimestamp = &now

	unsignedDataset := `{"datasets":[{"dataset_name":"tatsu-lab/alpaca"}]}`
	dataConfigMap := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      pod.Name + "-aibom-postprocess-data",
			Namespace: "test-ns",
		},
		Data: map[string]string{
			fmt.Sprintf("dataset-%s.json", pod.Name): unsignedDataset,
		},
	}

	// Deliberately no datasetSigningSecret in the fake clientset.
	client := fake.NewSimpleClientset(ns, pod, dataConfigMap)
	w := New(client, Config{PostprocessImage: "aibom-postprocess:latest"})
	startWatcher(t, w)

	w.onPodEvent(pod)

	cm, err := client.CoreV1().ConfigMaps("test-ns").Get(context.TODO(), pod.Name+"-aibom-postprocess-data", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("data configmap not found: %v", err)
	}
	if !strings.Contains(cm.Data["dataset.json"], "tatsu-lab/alpaca") {
		t.Errorf("expected unsigned dataset data to pass through when no signing key is configured, got dataset.json = %q", cm.Data["dataset.json"])
	}
}

func hmacHex(t *testing.T, key []byte, payload string) string {
	t.Helper()
	mac := hmac.New(sha256.New, key)
	mac.Write([]byte(payload))
	return hex.EncodeToString(mac.Sum(nil))
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

// TestCollectAIBOM_DeletesWorkloadIdentity guards the other half of #43:
// once a job's postprocess Job succeeds, the per-job ServiceAccount/Role/
// RoleBinding/Secret the webhook provisioned at admission time (see
// internal/webhook/identity.go's ensureWorkloadIdentity) must be cleaned up
// too, both so a same-named rerun re-provisions a fresh token instead of
// reusing a stale one, and so these don't accumulate forever.
func TestCollectAIBOM_DeletesWorkloadIdentity(t *testing.T) {
	ns := enabledNamespace("test-ns")
	ppJob, ppPod, dataConfigMap := newAIBOMPostprocessFixtures("train-job", "test-ns")

	identityName := aibomdata.WorkloadIdentityName("train-job")
	sa := &corev1.ServiceAccount{ObjectMeta: metav1.ObjectMeta{Name: identityName, Namespace: "test-ns"}}
	role := &rbacv1.Role{ObjectMeta: metav1.ObjectMeta{Name: identityName, Namespace: "test-ns"}}
	roleBinding := &rbacv1.RoleBinding{ObjectMeta: metav1.ObjectMeta{Name: identityName, Namespace: "test-ns"}}
	secret := &corev1.Secret{ObjectMeta: metav1.ObjectMeta{Name: identityName, Namespace: "test-ns"}}

	client := fake.NewSimpleClientset(ns, ppJob, ppPod, dataConfigMap, sa, role, roleBinding, secret)
	w := New(client, Config{PostprocessImage: "aibom-postprocess:latest"})
	startWatcher(t, w)

	w.onJobEvent(ppJob)

	if _, err := client.CoreV1().ServiceAccounts("test-ns").Get(context.TODO(), identityName, metav1.GetOptions{}); !apierrors.IsNotFound(err) {
		t.Errorf("expected workload identity serviceaccount to be deleted after collection, got err=%v", err)
	}
	if _, err := client.RbacV1().Roles("test-ns").Get(context.TODO(), identityName, metav1.GetOptions{}); !apierrors.IsNotFound(err) {
		t.Errorf("expected workload identity role to be deleted after collection, got err=%v", err)
	}
	if _, err := client.RbacV1().RoleBindings("test-ns").Get(context.TODO(), identityName, metav1.GetOptions{}); !apierrors.IsNotFound(err) {
		t.Errorf("expected workload identity rolebinding to be deleted after collection, got err=%v", err)
	}
	if _, err := client.CoreV1().Secrets("test-ns").Get(context.TODO(), identityName, metav1.GetOptions{}); !apierrors.IsNotFound(err) {
		t.Errorf("expected workload identity token secret to be deleted after collection, got err=%v", err)
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

func TestBuildPostprocessInputs_CapturesOOMKilledStatus(t *testing.T) {
	pod := instrumentedPod("oom-job", "ns")
	pod.Status.ContainerStatuses = []corev1.ContainerStatus{
		{
			Name:    "training",
			ImageID: "busybox@sha256:abc",
			State: corev1.ContainerState{
				Terminated: &corev1.ContainerStateTerminated{
					Reason:   "OOMKilled",
					ExitCode: 137,
				},
			},
		},
	}
	w := &Watcher{clientset: fake.NewSimpleClientset()}

	_, _, containersJSON, _ := w.buildPostprocessInputs(context.Background(), "ns", "cm-name", []corev1.Pod{*pod}, nil)

	var containers []struct {
		PodName          string `json:"pod_name"`
		Name             string `json:"name"`
		TerminatedReason string `json:"terminated_reason"`
		ExitCode         *int32 `json:"exit_code"`
	}
	if err := json.Unmarshal([]byte(containersJSON), &containers); err != nil {
		t.Fatalf("unmarshal containers.json: %v", err)
	}
	if len(containers) != 1 {
		t.Fatalf("expected 1 container entry, got %d", len(containers))
	}
	if containers[0].TerminatedReason != "OOMKilled" {
		t.Errorf("terminated_reason = %q, want OOMKilled", containers[0].TerminatedReason)
	}
	if containers[0].ExitCode == nil || *containers[0].ExitCode != 137 {
		t.Errorf("exit_code = %v, want 137", containers[0].ExitCode)
	}
}

func TestBuildPostprocessInputs_CapturesFinishedAt(t *testing.T) {
	pod := instrumentedPod("done-job", "ns")
	finished := time.Date(2026, 1, 2, 3, 4, 5, 0, time.FixedZone("EST", -5*3600))
	pod.Status.ContainerStatuses = []corev1.ContainerStatus{
		{
			Name: "training",
			State: corev1.ContainerState{
				Terminated: &corev1.ContainerStateTerminated{
					Reason:     "Completed",
					FinishedAt: metav1.NewTime(finished),
				},
			},
		},
	}
	w := &Watcher{clientset: fake.NewSimpleClientset()}

	_, _, containersJSON, _ := w.buildPostprocessInputs(context.Background(), "ns", "cm-name", []corev1.Pod{*pod}, nil)

	var containers []struct {
		FinishedAt string `json:"finished_at"`
	}
	if err := json.Unmarshal([]byte(containersJSON), &containers); err != nil {
		t.Fatalf("unmarshal containers.json: %v", err)
	}
	if len(containers) != 1 {
		t.Fatalf("expected 1 container entry, got %d", len(containers))
	}
	// Always UTC, regardless of the zone the timestamp was recorded in.
	if want := "2026-01-02T08:04:05Z"; containers[0].FinishedAt != want {
		t.Errorf("finished_at = %q, want %q", containers[0].FinishedAt, want)
	}
}

func TestBuildPostprocessInputs_NoStatusOmitsTerminatedFields(t *testing.T) {
	pod := instrumentedPod("running-job", "ns")
	w := &Watcher{clientset: fake.NewSimpleClientset()}

	_, _, containersJSON, _ := w.buildPostprocessInputs(context.Background(), "ns", "cm-name", []corev1.Pod{*pod}, nil)

	if strings.Contains(containersJSON, "terminated_reason") {
		t.Errorf("expected no terminated_reason field when no container status is reported, got: %s", containersJSON)
	}
	if strings.Contains(containersJSON, "finished_at") {
		t.Errorf("expected no finished_at field when no container status is reported, got: %s", containersJSON)
	}
}

func TestBuildPostprocessInputs_CapturesResourceLimits(t *testing.T) {
	pod := instrumentedPod("limited-job", "ns")
	pod.Spec.Containers[0].Resources.Limits[corev1.ResourceMemory] = resource.MustParse("8Gi")
	pod.Spec.Containers[0].Resources.Limits[corev1.ResourceCPU] = resource.MustParse("2")
	w := &Watcher{clientset: fake.NewSimpleClientset()}

	_, _, containersJSON, _ := w.buildPostprocessInputs(context.Background(), "ns", "cm-name", []corev1.Pod{*pod}, nil)

	var containers []struct {
		MemoryLimitBytes *int64 `json:"memory_limit_bytes"`
		CPULimitMillis   *int64 `json:"cpu_limit_millis"`
	}
	if err := json.Unmarshal([]byte(containersJSON), &containers); err != nil {
		t.Fatalf("unmarshal containers.json: %v", err)
	}
	if len(containers) != 1 {
		t.Fatalf("expected 1 container entry, got %d", len(containers))
	}
	wantMem := int64(8 * 1024 * 1024 * 1024)
	if containers[0].MemoryLimitBytes == nil || *containers[0].MemoryLimitBytes != wantMem {
		t.Errorf("memory_limit_bytes = %v, want %d", containers[0].MemoryLimitBytes, wantMem)
	}
	if containers[0].CPULimitMillis == nil || *containers[0].CPULimitMillis != 2000 {
		t.Errorf("cpu_limit_millis = %v, want 2000", containers[0].CPULimitMillis)
	}
}

func TestBuildPostprocessInputs_NoResourceLimitsOmitsLimitFields(t *testing.T) {
	// instrumentedPod only sets an nvidia.com/gpu limit -- no memory/cpu limit.
	pod := instrumentedPod("no-limits-job", "ns")
	w := &Watcher{clientset: fake.NewSimpleClientset()}

	_, _, containersJSON, _ := w.buildPostprocessInputs(context.Background(), "ns", "cm-name", []corev1.Pod{*pod}, nil)

	if strings.Contains(containersJSON, "memory_limit_bytes") || strings.Contains(containersJSON, "cpu_limit_millis") {
		t.Errorf("expected no limit fields when the container sets no memory/cpu limit, got: %s", containersJSON)
	}
}

// ---------------------------------------------------------------------------
// Termination status and failed postprocess Jobs (#112)
// ---------------------------------------------------------------------------

type capturedContainer struct {
	PodName              string `json:"pod_name"`
	TerminatedReason     string `json:"terminated_reason"`
	JobResult            string `json:"job_result"`
	RestartCount         int32  `json:"restart_count"`
	LastTerminatedReason string `json:"last_terminated_reason"`
	LastExitCode         *int32 `json:"last_exit_code"`
}

func capturedContainers(t *testing.T, w *Watcher, pods []corev1.Pod, jobResults map[string]string) []capturedContainer {
	t.Helper()
	_, _, containersJSON, _ := w.buildPostprocessInputs(context.Background(), "ns", "cm-name", pods, jobResults)
	var containers []capturedContainer
	if err := json.Unmarshal([]byte(containersJSON), &containers); err != nil {
		t.Fatalf("unmarshal containers.json: %v", err)
	}
	return containers
}

func TestBuildPostprocessInputs_CapturesRestartHistory(t *testing.T) {
	pod := instrumentedPod("retry-job", "ns")
	pod.Status.ContainerStatuses = []corev1.ContainerStatus{{
		Name:         "training",
		RestartCount: 2,
		State: corev1.ContainerState{
			Terminated: &corev1.ContainerStateTerminated{Reason: "Completed", ExitCode: 0},
		},
		LastTerminationState: corev1.ContainerState{
			Terminated: &corev1.ContainerStateTerminated{Reason: "OOMKilled", ExitCode: 137},
		},
	}}
	w := &Watcher{clientset: fake.NewSimpleClientset()}

	got := capturedContainers(t, w, []corev1.Pod{*pod}, nil)

	if len(got) != 1 {
		t.Fatalf("want 1 container, got %d", len(got))
	}
	if got[0].TerminatedReason != "Completed" {
		t.Errorf("current state should still be reported, got %q", got[0].TerminatedReason)
	}
	if got[0].RestartCount != 2 || got[0].LastTerminatedReason != "OOMKilled" ||
		got[0].LastExitCode == nil || *got[0].LastExitCode != 137 {
		t.Errorf("earlier OOM kill not captured: %+v", got[0])
	}
}

func TestBuildPostprocessInputs_NoRestartHistoryWhenNeverRestarted(t *testing.T) {
	pod := instrumentedPod("clean-job", "ns")
	pod.Status.ContainerStatuses = []corev1.ContainerStatus{{
		Name:  "training",
		State: corev1.ContainerState{Terminated: &corev1.ContainerStateTerminated{Reason: "Completed"}},
	}}
	w := &Watcher{clientset: fake.NewSimpleClientset()}

	_, _, containersJSON, _ := w.buildPostprocessInputs(context.Background(), "ns", "cm-name", []corev1.Pod{*pod}, nil)

	for _, key := range []string{"restart_count", "last_terminated_reason", "last_exit_code", "job_result"} {
		if strings.Contains(containersJSON, key) {
			t.Errorf("%s should be omitted, got: %s", key, containersJSON)
		}
	}
}

func TestBuildPostprocessInputs_TagsEachPodWithItsJobsResult(t *testing.T) {
	failedAttempt := instrumentedPod("train-job", "ns")
	failedAttempt.Name = "train-job-attempt1"
	sibling := instrumentedPod("sibling-job", "ns")
	bare := instrumentedPod("x", "ns")
	bare.Name = "bare"
	bare.Labels = map[string]string{LabelInstrumented: "true"}
	for _, p := range []*corev1.Pod{failedAttempt, sibling, bare} {
		p.Status.ContainerStatuses = []corev1.ContainerStatus{{Name: "training"}}
	}
	w := &Watcher{clientset: fake.NewSimpleClientset()}

	got := capturedContainers(t, w, []corev1.Pod{*failedAttempt, *sibling, *bare},
		map[string]string{"train-job": "Complete", "sibling-job": "Failed"})

	want := map[string]string{"train-job-attempt1": "Complete", "sibling-job-pod": "Failed", "bare": ""}
	for _, c := range got {
		if c.JobResult != want[c.PodName] {
			t.Errorf("pod %s: job_result = %q, want %q", c.PodName, c.JobResult, want[c.PodName])
		}
	}
}

func TestJobResult(t *testing.T) {
	failed := completedJob("j", "ns")
	failed.Status.Conditions = []batchv1.JobCondition{{Type: batchv1.JobFailed, Status: corev1.ConditionTrue}}
	notTrue := completedJob("j", "ns")
	notTrue.Status.Conditions = []batchv1.JobCondition{{Type: batchv1.JobComplete, Status: corev1.ConditionFalse}}
	running := completedJob("j", "ns")
	running.Status.Conditions = nil

	for name, tc := range map[string]struct {
		job  *batchv1.Job
		want string
	}{
		"complete": {completedJob("j", "ns"), "Complete"},
		"failed":   {failed, "Failed"},
		"false":    {notTrue, ""},
		"running":  {running, ""},
	} {
		if got := jobResult(tc.job); got != tc.want {
			t.Errorf("%s: jobResult = %q, want %q", name, got, tc.want)
		}
	}
}

func failedPostprocessFixtures(jobName, namespace string) (*batchv1.Job, *corev1.ConfigMap) {
	ppJob, _, dataConfigMap := newAIBOMPostprocessFixtures(jobName, namespace)
	ppJob.Status.Conditions = []batchv1.JobCondition{{Type: batchv1.JobFailed, Status: corev1.ConditionTrue}}
	return ppJob, dataConfigMap
}

func TestCollectAIBOM_KeepsFailedPostprocessJobAndConfigMap(t *testing.T) {
	ns := enabledNamespace("test-ns")
	ppJob, dataConfigMap := failedPostprocessFixtures("train-job", "test-ns")
	identityName := aibomdata.WorkloadIdentityName("train-job")
	sa := &corev1.ServiceAccount{ObjectMeta: metav1.ObjectMeta{Name: identityName, Namespace: "test-ns"}}

	client := fake.NewSimpleClientset(ns, ppJob, dataConfigMap, sa)
	w := New(client, Config{PostprocessImage: "aibom-postprocess:latest"})
	startWatcher(t, w)

	w.onJobEvent(ppJob)

	got, err := client.BatchV1().Jobs("test-ns").Get(context.TODO(), "train-job-aibom-postprocess", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed postprocess job must be kept for inspection, got err=%v", err)
	}
	if got.Annotations[AnnotationAIBOMCollected] == "" {
		t.Error("kept job must still be marked collected so a resync doesn't collect it again")
	}
	if _, err := client.CoreV1().ConfigMaps("test-ns").Get(context.TODO(), "train-job-aibom-postprocess-data", metav1.GetOptions{}); err != nil {
		t.Errorf("data configmap must be kept so the run can be retried by hand, got err=%v", err)
	}
	if _, err := client.CoreV1().ServiceAccounts("test-ns").Get(context.TODO(), identityName, metav1.GetOptions{}); !apierrors.IsNotFound(err) {
		t.Errorf("workload identity should still be cleaned up, got err=%v", err)
	}

	// A resync must not collect it a second time.
	w.onJobEvent(got)
	if _, err := client.BatchV1().Jobs("test-ns").Get(context.TODO(), "train-job-aibom-postprocess", metav1.GetOptions{}); err != nil {
		t.Errorf("kept job disappeared on resync: %v", err)
	}
}

func TestCreatePostprocessJob_SetsTTLAndOwnsConfigMap(t *testing.T) {
	ns := enabledNamespace("test-ns")
	job := completedJob("train-job", "test-ns")
	pod := instrumentedPod("train-job", "test-ns")
	client := fake.NewSimpleClientset(ns, job, pod)
	// The fake API server assigns no UIDs, so give the Job one on create.
	client.PrependReactor("create", "jobs", func(action k8stesting.Action) (bool, runtime.Object, error) {
		obj := action.(k8stesting.CreateAction).GetObject().(*batchv1.Job)
		obj.UID = "pp-uid"
		return false, obj, nil
	})
	w := New(client, Config{PostprocessImage: "aibom-postprocess:latest"})

	if err := w.createPostprocessJob(context.TODO(), job); err != nil {
		t.Fatalf("createPostprocessJob: %v", err)
	}

	pp, err := client.BatchV1().Jobs("test-ns").Get(context.TODO(), "train-job-aibom-postprocess", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("get postprocess job: %v", err)
	}
	if pp.Spec.TTLSecondsAfterFinished == nil || *pp.Spec.TTLSecondsAfterFinished != failedPostprocessTTLSeconds {
		t.Errorf("TTLSecondsAfterFinished = %v, want %d", pp.Spec.TTLSecondsAfterFinished, failedPostprocessTTLSeconds)
	}
	cm, err := client.CoreV1().ConfigMaps("test-ns").Get(context.TODO(), aibomdata.ConfigMapName("train-job"), metav1.GetOptions{})
	if err != nil {
		t.Fatalf("get data configmap: %v", err)
	}
	if len(cm.OwnerReferences) != 1 || cm.OwnerReferences[0].Kind != "Job" ||
		cm.OwnerReferences[0].Name != "train-job-aibom-postprocess" || cm.OwnerReferences[0].UID != "pp-uid" {
		t.Errorf("data configmap should be owned by the postprocess job, got %+v", cm.OwnerReferences)
	}
}

func TestCreatePostprocessJob_ReplacesStaleFailedPostprocessJob(t *testing.T) {
	ns := enabledNamespace("test-ns")
	job := completedJob("train-job", "test-ns")
	pod := instrumentedPod("train-job", "test-ns")
	stale, staleCM := failedPostprocessFixtures("train-job", "test-ns")
	staleCM.Data = map[string]string{"annotations.json": "stale"}
	client := fake.NewSimpleClientset(ns, job, pod, stale, staleCM)
	w := New(client, Config{PostprocessImage: "aibom-postprocess:latest"})

	if err := w.createPostprocessJob(context.TODO(), job); err != nil {
		t.Fatalf("createPostprocessJob: %v", err)
	}

	pp, err := client.BatchV1().Jobs("test-ns").Get(context.TODO(), "train-job-aibom-postprocess", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("get postprocess job: %v", err)
	}
	if jobResult(pp) != "" {
		t.Errorf("the failed leftover should have been replaced by a fresh job, got conditions %+v", pp.Status.Conditions)
	}
	cm, err := client.CoreV1().ConfigMaps("test-ns").Get(context.TODO(), aibomdata.ConfigMapName("train-job"), metav1.GetOptions{})
	if err != nil {
		t.Fatalf("get data configmap: %v", err)
	}
	if cm.Data["annotations.json"] == "stale" {
		t.Error("data configmap still holds the previous run's data")
	}
}

func TestCreatePostprocessJob_LeavesRunningPostprocessJobAlone(t *testing.T) {
	ns := enabledNamespace("test-ns")
	job := completedJob("train-job", "test-ns")
	pod := instrumentedPod("train-job", "test-ns")
	running, _, _ := newAIBOMPostprocessFixtures("train-job", "test-ns")
	running.Status.Conditions = nil
	running.Annotations = map[string]string{"marker": "original"}
	client := fake.NewSimpleClientset(ns, job, pod, running)
	w := New(client, Config{PostprocessImage: "aibom-postprocess:latest"})

	if err := w.createPostprocessJob(context.TODO(), job); err != nil {
		t.Fatalf("createPostprocessJob: %v", err)
	}

	pp, err := client.BatchV1().Jobs("test-ns").Get(context.TODO(), "train-job-aibom-postprocess", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("get postprocess job: %v", err)
	}
	if pp.Annotations["marker"] != "original" {
		t.Error("an in-flight postprocess job must not be deleted and recreated")
	}
}

// ---------------------------------------------------------------------------
// Finalizer cleanup (#104)
// ---------------------------------------------------------------------------

func TestOnJobEvent_OptedOutNamespace_StripsFinalizer(t *testing.T) {
	now := metav1.Now()
	job := completedJob("train-job", "disabled-ns")
	job.DeletionTimestamp = &now
	job.Finalizers = []string{"other.io/keep", finalizerName}

	client := fake.NewSimpleClientset(disabledNamespace("disabled-ns"), job)
	w := New(client, Config{PostprocessImage: "busybox:latest"})
	startWatcher(t, w)

	w.onJobEvent(job)

	got, err := client.BatchV1().Jobs("disabled-ns").Get(context.TODO(), "train-job", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("get job: %v", err)
	}
	if hasFinalizer(got) {
		t.Error("our finalizer should be released when the namespace opted out")
	}
	if len(got.Finalizers) != 1 || got.Finalizers[0] != "other.io/keep" {
		t.Errorf("other finalizers must be preserved, got %v", got.Finalizers)
	}
	if _, err := client.BatchV1().Jobs("disabled-ns").Get(context.TODO(), "train-job-aibom-postprocess", metav1.GetOptions{}); err == nil {
		t.Error("no postprocess job should be created for an opted-out namespace")
	}
}

func TestOnPodEvent_OptedOutNamespace_StripsFinalizer(t *testing.T) {
	now := metav1.Now()
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "predictor",
			Namespace:         "disabled-ns",
			Labels:            map[string]string{LabelInstrumented: "true"},
			DeletionTimestamp: &now,
			Finalizers:        []string{podFinalizerName},
		},
	}

	client := fake.NewSimpleClientset(disabledNamespace("disabled-ns"), pod)
	w := New(client, Config{PostprocessImage: "busybox:latest"})
	startWatcher(t, w)

	w.onPodEvent(pod)

	got, err := client.CoreV1().Pods("disabled-ns").Get(context.TODO(), "predictor", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("get pod: %v", err)
	}
	if hasPodFinalizer(got) {
		t.Error("our pod finalizer should be released when the namespace opted out")
	}
}

func TestOnJobEvent_UnknownNamespace_KeepsFinalizer(t *testing.T) {
	// A namespace missing from the informer cache (not yet synced, or gone)
	// is not "opted out": stripping on a cache miss would discard the AIBOM of
	// a workload in a namespace that is still enabled.
	job := completedJob("train-job", "unknown-ns")
	job.Finalizers = []string{finalizerName}

	client := fake.NewSimpleClientset(job)
	w := New(client, Config{PostprocessImage: "busybox:latest"})
	startWatcher(t, w)

	w.onJobEvent(job)

	got, err := client.BatchV1().Jobs("unknown-ns").Get(context.TODO(), "train-job", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("get job: %v", err)
	}
	if !hasFinalizer(got) {
		t.Error("finalizer must be kept when the namespace can't be looked up")
	}
}

func TestStripAllFinalizers(t *testing.T) {
	withBoth := completedJob("a", "ns-one")
	withBoth.Finalizers = []string{"other.io/keep", finalizerName}
	withOurs := completedJob("b", "ns-two")
	withOurs.Finalizers = []string{finalizerName}
	untouched := completedJob("c", "ns-two")
	untouched.Finalizers = []string{"other.io/keep"}
	podWithOurs := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{
		Name: "p", Namespace: "ns-one", Finalizers: []string{podFinalizerName}}}
	podOther := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{
		Name: "q", Namespace: "ns-one", Finalizers: []string{"other.io/keep"}}}

	client := fake.NewSimpleClientset(withBoth, withOurs, untouched, podWithOurs, podOther)

	jobs, pods, err := StripAllFinalizers(context.TODO(), client)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if jobs != 2 || pods != 1 {
		t.Errorf("stripped %d jobs / %d pods, want 2 / 1", jobs, pods)
	}

	for _, tc := range []struct {
		ns, name string
		want     []string
	}{
		{"ns-one", "a", []string{"other.io/keep"}},
		{"ns-two", "b", nil},
		{"ns-two", "c", []string{"other.io/keep"}},
	} {
		got, _ := client.BatchV1().Jobs(tc.ns).Get(context.TODO(), tc.name, metav1.GetOptions{})
		if len(got.Finalizers) != len(tc.want) || (len(tc.want) == 1 && got.Finalizers[0] != tc.want[0]) {
			t.Errorf("job %s/%s finalizers = %v, want %v", tc.ns, tc.name, got.Finalizers, tc.want)
		}
	}
	p, _ := client.CoreV1().Pods("ns-one").Get(context.TODO(), "p", metav1.GetOptions{})
	if hasPodFinalizer(p) {
		t.Error("pod finalizer should be stripped")
	}
	q, _ := client.CoreV1().Pods("ns-one").Get(context.TODO(), "q", metav1.GetOptions{})
	if len(q.Finalizers) != 1 || q.Finalizers[0] != "other.io/keep" {
		t.Errorf("unrelated pod finalizers must be untouched, got %v", q.Finalizers)
	}

	// Idempotent: a second run (hook retry) has nothing left to do.
	jobs, pods, err = StripAllFinalizers(context.TODO(), client)
	if err != nil || jobs != 0 || pods != 0 {
		t.Errorf("second run = %d jobs / %d pods / %v, want 0 / 0 / nil", jobs, pods, err)
	}
}

// ---------------------------------------------------------------------------
// mergeDatasets across pods (#107)
// ---------------------------------------------------------------------------

func TestMergeDatasets_DedupesSameDatasetAcrossPods(t *testing.T) {
	pod := `{"datasets":[{"dataset_name":"alpaca","source":"datasets.load_dataset"}]}`
	result := mergeDatasets([]string{pod, pod, pod})

	var got struct {
		Datasets []map[string]interface{} `json:"datasets"`
	}
	if err := json.Unmarshal([]byte(result), &got); err != nil {
		t.Fatalf("invalid result %q: %v", result, err)
	}
	if len(got.Datasets) != 1 {
		t.Errorf("3 identical pods should merge to 1 dataset, got %d: %s", len(got.Datasets), result)
	}
}

func TestMergeDatasets_KeepsSameNameFromDifferentSources(t *testing.T) {
	a := `{"datasets":[{"dataset_name":"alpaca","source":"datasets.load_dataset"}]}`
	b := `{"datasets":[{"dataset_name":"alpaca","source":"torch.utils.data.DataLoader"}]}`
	var got struct {
		Datasets []map[string]interface{} `json:"datasets"`
	}
	if err := json.Unmarshal([]byte(mergeDatasets([]string{a, b})), &got); err != nil {
		t.Fatal(err)
	}
	if len(got.Datasets) != 2 {
		t.Errorf("same name via different sources are distinct entries, got %d", len(got.Datasets))
	}
}

func TestMergeDatasets_GitInfoComesFromOnePod(t *testing.T) {
	// Pod A has a commit but is on a detached HEAD (no branch); pod B has a
	// branch and dirty flag for a different checkout. Per-key merging would
	// label A's commit with B's branch.
	a := `{"runtime_info":{"git_commit":"aaa111","git_repository":"https://h/a"}}`
	b := `{"runtime_info":{"git_commit":"bbb222","git_branch":"main","git_dirty":true,"framework":"PyTorch"}}`

	var got struct {
		RuntimeInfo map[string]interface{} `json:"runtime_info"`
	}
	if err := json.Unmarshal([]byte(mergeDatasets([]string{a, b})), &got); err != nil {
		t.Fatal(err)
	}
	ri := got.RuntimeInfo
	if ri["git_commit"] != "aaa111" || ri["git_repository"] != "https://h/a" {
		t.Errorf("git group should come from the first pod, got %v", ri)
	}
	if _, ok := ri["git_branch"]; ok {
		t.Errorf("git_branch from another pod must not be mixed in, got %v", ri)
	}
	if _, ok := ri["git_dirty"]; ok {
		t.Errorf("git_dirty from another pod must not be mixed in, got %v", ri)
	}
	if ri["framework"] != "PyTorch" {
		t.Errorf("non-git keys still merge across pods, got %v", ri)
	}
}
