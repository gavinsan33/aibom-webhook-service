package watcher

import (
	"context"
	"fmt"
	"log"
	"time"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
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

	resyncPeriod     = 30 * time.Second
	maxJobNameLength = 63
	postprocessSuffix = "-aibom-postprocess"
)

type Watcher struct {
	clientset        kubernetes.Interface
	postprocessImage string
	factory          informers.SharedInformerFactory
}

func New(clientset kubernetes.Interface, postprocessImage string) *Watcher {
	w := &Watcher{
		clientset:        clientset,
		postprocessImage: postprocessImage,
		factory:          informers.NewSharedInformerFactory(clientset, resyncPeriod),
	}

	w.factory.Batch().V1().Jobs().Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    w.onJobEvent,
		UpdateFunc: func(_, newObj interface{}) { w.onJobEvent(newObj) },
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

	if !w.isJobComplete(job) {
		return
	}

	if job.Annotations != nil && job.Annotations[AnnotationPostprocess] != "" {
		return
	}

	if !w.hasInstrumentedPods(job) {
		return
	}

	if err := w.createPostprocessJob(context.TODO(), job); err != nil {
		log.Printf("failed to create postprocess job for %s/%s: %v", job.Namespace, job.Name, err)
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

func (w *Watcher) hasInstrumentedPods(job *batchv1.Job) bool {
	pods, err := w.clientset.CoreV1().Pods(job.Namespace).List(context.TODO(), metav1.ListOptions{
		LabelSelector: fmt.Sprintf("batch.kubernetes.io/job-name=%s,%s=true", job.Name, LabelInstrumented),
		Limit:         1,
	})
	if err != nil {
		log.Printf("failed to list pods for job %s/%s: %v", job.Namespace, job.Name, err)
		return false
	}
	return len(pods.Items) > 0
}

func (w *Watcher) createPostprocessJob(ctx context.Context, job *batchv1.Job) error {
	postprocessName := postprocessJobName(job.Name)
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
							Command: []string{"/bin/sh", "-c"},
							Args:    []string{fmt.Sprintf("echo '[aibom] postprocess for job %s/%s'", job.Namespace, job.Name)},
							Env: []corev1.EnvVar{
								{Name: "AIBOM_JOB_NAME", Value: job.Name},
								{Name: "AIBOM_JOB_NAMESPACE", Value: job.Namespace},
								{Name: "AIBOM_BASE_DIR", Value: "/tmp/aibom"},
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
						},
					},
				},
			},
		},
	}

	_, err := w.clientset.BatchV1().Jobs(job.Namespace).Create(ctx, postprocessJob, metav1.CreateOptions{})
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
	return jobName + postprocessSuffix
}
