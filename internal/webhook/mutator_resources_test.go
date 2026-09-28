package webhook

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const testGPUResource = corev1.ResourceName("nvidia.com/gpu")

// gpuContainer returns a container claiming gpuCount GPUs, via limits if
// viaLimits else requests only.
func gpuContainer(name string, gpuCount int, viaLimits bool) corev1.Container {
	c := corev1.Container{Name: name, Image: "pytorch:latest"}
	if viaLimits {
		c.Resources.Limits = corev1.ResourceList{testGPUResource: *resource.NewQuantity(int64(gpuCount), resource.BinarySI)}
	} else {
		c.Resources.Requests = corev1.ResourceList{testGPUResource: *resource.NewQuantity(int64(gpuCount), resource.BinarySI)}
	}
	return c
}

func gpuPod(containers ...corev1.Container) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "gpu-pod",
			Namespace: "default",
		},
		Spec: corev1.PodSpec{
			Containers: containers,
		},
	}
}

// TestMutate_DiscoveryInitContainer_ExplicitCPUMemoryResources guards the
// LimitRange fix: both requests AND limits must be set (a LimitRange's
// default/defaultRequest backfills whatever is missing, so leaving either
// side unset would let a workload namespace stamp its own defaults onto the
// container), and the values must be the small ones -- not a namespace's
// 1 CPU/2Gi request or 2 CPU/8Gi limit. A pod with no GPU must not get a
// nvidia.com/gpu claim at all.
func TestMutate_DiscoveryInitContainer_ExplicitCPUMemoryResources(t *testing.T) {
	m := newTestMutator()
	patches, err := m.Mutate(podWithOwner("Job"), "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	c := findInitContainer(patches, "aibom-discovery")
	if c == nil {
		t.Fatal("aibom-discovery init container not found")
	}
	assertExplicitResources(t, c,
		"250m", "128Mi", // requests
		"1", "512Mi", // limits
	)
	if _, ok := c.Resources.Limits[testGPUResource]; ok {
		t.Error("a pod with no GPU request must not get a nvidia.com/gpu claim on the discovery init container")
	}
}

// TestMutate_DiscoveryInitContainer_MirrorsPodTotalGPULimits is the
// multi-container regression: the init container must mirror the pod's
// TOTAL GPU allocation (the sum across containers, what the scheduler
// places the pod against), not just the first container's claim -- or
// nvidia-smi in the init container would see only a subset of the GPUs
// the pod actually gets.
func TestMutate_DiscoveryInitContainer_MirrorsPodTotalGPULimits(t *testing.T) {
	m := newTestMutator()
	pod := gpuPod(gpuContainer("train", 1, true), gpuContainer("aux", 1, true))

	patches, err := m.Mutate(pod, "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	c := findInitContainer(patches, "aibom-discovery")
	if c == nil {
		t.Fatal("aibom-discovery init container not found")
	}
	got, ok := c.Resources.Limits[testGPUResource]
	if !ok {
		t.Fatal("expected a nvidia.com/gpu limit on the discovery init container")
	}
	if want := resource.NewQuantity(2, resource.BinarySI); !got.Equal(*want) {
		t.Errorf("init container GPU limit = %s, want 2 (the sum across the pod's two 1-GPU containers)", got.String())
	}
}

// TestMutate_DiscoveryInitContainer_GPUMirrorsRequestsWhenNoLimits covers
// the requests-only declaration path (no container sets a GPU limit): the
// mirrored total still has to come from the requests.
func TestMutate_DiscoveryInitContainer_GPUMirrorsRequestsWhenNoLimits(t *testing.T) {
	m := newTestMutator()
	pod := gpuPod(gpuContainer("train", 1, false))

	patches, err := m.Mutate(pod, "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	c := findInitContainer(patches, "aibom-discovery")
	if c == nil {
		t.Fatal("aibom-discovery init container not found")
	}
	got, ok := c.Resources.Limits[testGPUResource]
	if !ok {
		t.Fatal("expected a nvidia.com/gpu limit on the discovery init container")
	}
	if want := resource.NewQuantity(1, resource.BinarySI); !got.Equal(*want) {
		t.Errorf("init container GPU limit = %s, want 1 (from the container's request, since it sets no limit)", got.String())
	}
}

// TestMutate_DiscoveryInitContainer_GPUSumsMixedLimitsAndRequests pins
// the per-container effective-claim rule (limit if set, else request)
// applied to the sum: 1 (limit) + 2 (request) = 3.
func TestMutate_DiscoveryInitContainer_GPUSumsMixedLimitsAndRequests(t *testing.T) {
	m := newTestMutator()
	pod := gpuPod(gpuContainer("train", 1, true), gpuContainer("aux", 2, false))

	patches, err := m.Mutate(pod, "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	c := findInitContainer(patches, "aibom-discovery")
	if c == nil {
		t.Fatal("aibom-discovery init container not found")
	}
	got, ok := c.Resources.Limits[testGPUResource]
	if !ok {
		t.Fatal("expected a nvidia.com/gpu limit on the discovery init container")
	}
	if want := resource.NewQuantity(3, resource.BinarySI); !got.Equal(*want) {
		t.Errorf("init container GPU limit = %s, want 3 (1 limit + 2 requests)", got.String())
	}
}

// TestMutate_DatasetSidecar_ExplicitCPUMemoryResources guards the same
// LimitRange fix for the dataset sidecar -- which, unlike the discovery
// init container, runs for the pod's entire lifetime, so its (small)
// request rides on the pod's scheduling footprint the whole time.
func TestMutate_DatasetSidecar_ExplicitCPUMemoryResources(t *testing.T) {
	m := newTestMutator()
	patches, err := m.Mutate(podWithOwner("Job"), "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	c := findInitContainer(patches, "aibom-dataset-sidecar")
	if c == nil {
		t.Fatal("aibom-dataset-sidecar init container not found")
	}
	assertExplicitResources(t, c,
		"50m", "64Mi", // requests
		"200m", "128Mi",
	)
}

// assertExplicitResources requires all four cpu/memory requirement fields
// to be present and exactly the expected values -- presence (not just
// non-zero) is the point, since a LimitRange backfills any field that is
// absent rather than any field that is merely zero.
func assertExplicitResources(t *testing.T, c *corev1.Container, cpuReq, memReq, cpuLim, memLim string) {
	t.Helper()
	check := func(list corev1.ResourceList, which, name string, want string) {
		q, ok := list[corev1.ResourceName(name)]
		if !ok {
			t.Errorf("container %q: missing %s %s -- a namespace LimitRange's %s would backfill it", c.Name, which, name, which)
			return
		}
		if w := resource.MustParse(want); !q.Equal(w) {
			t.Errorf("container %q: %s %s = %s, want %s", c.Name, which, name, q.String(), want)
		}
	}
	check(c.Resources.Requests, "request", "cpu", cpuReq)
	check(c.Resources.Requests, "request", "memory", memReq)
	check(c.Resources.Limits, "limit", "cpu", cpuLim)
	check(c.Resources.Limits, "limit", "memory", memLim)
}
