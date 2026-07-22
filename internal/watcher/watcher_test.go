package watcher

import (
	"context"
	"strings"
	"testing"
	"time"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
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
			RestartPolicy: corev1.RestartPolicyNever,
			Containers:    []corev1.Container{{Name: "test", Image: "busybox"}},
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
	w := New(client, "busybox:latest")
	startWatcher(t, w)

	w.onJobEvent(job)

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

	if ppJob.Spec.Template.Spec.Containers[0].Image != "busybox:latest" {
		t.Errorf("image = %q, want %q", ppJob.Spec.Template.Spec.Containers[0].Image, "busybox:latest")
	}

	envNames := make(map[string]string)
	for _, e := range ppJob.Spec.Template.Spec.Containers[0].Env {
		envNames[e.Name] = e.Value
	}
	if envNames["AIBOM_JOB_NAME"] != "train-job" {
		t.Errorf("AIBOM_JOB_NAME = %q, want %q", envNames["AIBOM_JOB_NAME"], "train-job")
	}
	if envNames["AIBOM_JOB_NAMESPACE"] != "test-ns" {
		t.Errorf("AIBOM_JOB_NAMESPACE = %q, want %q", envNames["AIBOM_JOB_NAMESPACE"], "test-ns")
	}

	updatedJob, _ := client.BatchV1().Jobs("test-ns").Get(context.TODO(), "train-job", metav1.GetOptions{})
	if updatedJob.Annotations[AnnotationPostprocess] != "train-job-aibom-postprocess" {
		t.Errorf("annotation %s = %q, want %q", AnnotationPostprocess, updatedJob.Annotations[AnnotationPostprocess], "train-job-aibom-postprocess")
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
	// Pod without the instrumented label
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
}
