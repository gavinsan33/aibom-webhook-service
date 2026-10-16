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
			if !hasFinalizer(j) {
				continue
			}
			jc := c.BatchV1().Jobs(j.Namespace)
			removed, serr := stripFinalizer(finalizerName,
				func() (metav1.Object, error) { return jc.Get(ctx, j.Name, metav1.GetOptions{}) },
				func(patch []byte) error {
					_, err := jc.Patch(ctx, j.Name, types.MergePatchType, patch, metav1.PatchOptions{})
					return err
				})
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
			if !hasPodFinalizer(p) {
				continue
			}
			pc := c.CoreV1().Pods(p.Namespace)
			removed, serr := stripFinalizer(podFinalizerName,
				func() (metav1.Object, error) { return pc.Get(ctx, p.Name, metav1.GetOptions{}) },
				func(patch []byte) error {
					_, err := pc.Patch(ctx, p.Name, types.MergePatchType, patch, metav1.PatchOptions{})
					return err
				})
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

// stripFinalizer re-reads the object and patches with its resourceVersion so
// a concurrent change to the finalizer list (the API server's own, say) is
// detected as a conflict and retried instead of being overwritten.
func stripFinalizer(finalizer string, get func() (metav1.Object, error), patch func([]byte) error) (bool, error) {
	removed := false
	err := retry.RetryOnConflict(retry.DefaultRetry, func() error {
		obj, err := get()
		if apierrors.IsNotFound(err) {
			return nil
		}
		if err != nil {
			return err
		}
		p, ok := finalizerRemovalPatch(obj.GetFinalizers(), finalizer, obj.GetResourceVersion())
		if !ok {
			return nil
		}
		if err := patch(p); err != nil {
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
