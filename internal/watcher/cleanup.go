package watcher

import (
	"context"
	"encoding/json"
	"fmt"
	"log"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/util/retry"
)

const cleanupPageSize = 500

// StripAllFinalizers removes this project's finalizers (finalizerName on
// Jobs, podFinalizerName on Pods) from every object in the cluster. It runs
// from the chart's post-delete hook: once the watcher is gone nothing is left
// to release them, and without this every Job and predictor Pod that still
// carries one stays Terminating forever (#104). It deliberately lists
// cluster-wide with no namespace or label filter, since the point is to
// catch objects the watcher would no longer look at.
//
// A failure on one object is logged and does not stop the rest; the returned
// error summarizes how many could not be cleaned up.
func StripAllFinalizers(ctx context.Context, c kubernetes.Interface) (jobs, pods int, err error) {
	failures := 0

	jobCont := ""
	for {
		list, lerr := c.BatchV1().Jobs(metav1.NamespaceAll).List(ctx, metav1.ListOptions{Limit: cleanupPageSize, Continue: jobCont})
		if lerr != nil {
			return jobs, pods, fmt.Errorf("list jobs: %w", lerr)
		}
		for i := range list.Items {
			j := &list.Items[i]
			if !containsString(j.Finalizers, finalizerName) {
				continue
			}
			removed, serr := stripJobFinalizer(ctx, c, j.Namespace, j.Name)
			if serr != nil {
				log.Printf("warning: could not strip finalizer from job %s/%s: %v", j.Namespace, j.Name, serr)
				failures++
			} else if removed {
				jobs++
			}
		}
		if jobCont = list.Continue; jobCont == "" {
			break
		}
	}

	podCont := ""
	for {
		list, lerr := c.CoreV1().Pods(metav1.NamespaceAll).List(ctx, metav1.ListOptions{Limit: cleanupPageSize, Continue: podCont})
		if lerr != nil {
			return jobs, pods, fmt.Errorf("list pods: %w", lerr)
		}
		for i := range list.Items {
			p := &list.Items[i]
			if !containsString(p.Finalizers, podFinalizerName) {
				continue
			}
			removed, serr := stripPodFinalizer(ctx, c, p.Namespace, p.Name)
			if serr != nil {
				log.Printf("warning: could not strip finalizer from pod %s/%s: %v", p.Namespace, p.Name, serr)
				failures++
			} else if removed {
				pods++
			}
		}
		if podCont = list.Continue; podCont == "" {
			break
		}
	}

	if failures > 0 {
		return jobs, pods, fmt.Errorf("%d object(s) could not be cleaned up", failures)
	}
	return jobs, pods, nil
}

// stripJobFinalizer re-reads the Job and patches with its resourceVersion so a
// concurrent change to the finalizer list (the API server's own, say) is
// detected as a conflict and retried instead of being overwritten.
func stripJobFinalizer(ctx context.Context, c kubernetes.Interface, namespace, name string) (bool, error) {
	removed := false
	err := retry.RetryOnConflict(retry.DefaultRetry, func() error {
		job, err := c.BatchV1().Jobs(namespace).Get(ctx, name, metav1.GetOptions{})
		if apierrors.IsNotFound(err) {
			return nil
		}
		if err != nil {
			return err
		}
		patch, ok := finalizerRemovalPatch(job.Finalizers, finalizerName, job.ResourceVersion)
		if !ok {
			return nil
		}
		if _, err := c.BatchV1().Jobs(namespace).Patch(ctx, name, types.MergePatchType, patch, metav1.PatchOptions{}); err != nil {
			return err
		}
		removed = true
		return nil
	})
	return removed, err
}

func stripPodFinalizer(ctx context.Context, c kubernetes.Interface, namespace, name string) (bool, error) {
	removed := false
	err := retry.RetryOnConflict(retry.DefaultRetry, func() error {
		pod, err := c.CoreV1().Pods(namespace).Get(ctx, name, metav1.GetOptions{})
		if apierrors.IsNotFound(err) {
			return nil
		}
		if err != nil {
			return err
		}
		patch, ok := finalizerRemovalPatch(pod.Finalizers, podFinalizerName, pod.ResourceVersion)
		if !ok {
			return nil
		}
		if _, err := c.CoreV1().Pods(namespace).Patch(ctx, name, types.MergePatchType, patch, metav1.PatchOptions{}); err != nil {
			return err
		}
		removed = true
		return nil
	})
	return removed, err
}

// finalizerRemovalPatch builds a merge patch that drops one finalizer and
// keeps the rest, or reports false if it isn't present.
func finalizerRemovalPatch(finalizers []string, remove, resourceVersion string) ([]byte, bool) {
	remaining := make([]string, 0, len(finalizers))
	for _, f := range finalizers {
		if f != remove {
			remaining = append(remaining, f)
		}
	}
	if len(remaining) == len(finalizers) {
		return nil, false
	}
	meta := map[string]interface{}{"finalizers": remaining}
	if resourceVersion != "" {
		meta["resourceVersion"] = resourceVersion
	}
	patch, _ := json.Marshal(map[string]interface{}{"metadata": meta})
	return patch, true
}

func containsString(list []string, s string) bool {
	for _, v := range list {
		if v == s {
			return true
		}
	}
	return false
}
