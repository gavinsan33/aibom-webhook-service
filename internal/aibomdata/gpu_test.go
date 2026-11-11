package aibomdata

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

func gpuCtr(name, res string, n int64, sidecar bool) corev1.Container {
	c := corev1.Container{Name: name, Resources: corev1.ResourceRequirements{
		Limits: corev1.ResourceList{corev1.ResourceName(res): *resource.NewQuantity(n, resource.DecimalSI)},
	}}
	if sidecar {
		always := corev1.ContainerRestartPolicyAlways
		c.RestartPolicy = &always
	}
	return c
}

func TestPodGPUClaims(t *testing.T) {
	cases := []struct {
		name                string
		init, app           []corev1.Container
		res                 corev1.ResourceName
		wantRunning, wantPk int64
	}{
		{"app only", nil, []corev1.Container{gpuCtr("a", "nvidia.com/gpu", 2, false)}, "nvidia.com/gpu", 2, 2},
		{"native sidecar adds to app", []corev1.Container{gpuCtr("vllm", "nvidia.com/gpu", 1, true)},
			[]corev1.Container{gpuCtr("a", "nvidia.com/gpu", 1, false)}, "nvidia.com/gpu", 2, 2},
		{"sidecar only", []corev1.Container{gpuCtr("vllm", "nvidia.com/gpu", 1, true)},
			[]corev1.Container{{Name: "client"}}, "nvidia.com/gpu", 1, 1},
		{"regular init raises peak only", []corev1.Container{gpuCtr("warm", "nvidia.com/gpu", 4, false)},
			[]corev1.Container{gpuCtr("a", "nvidia.com/gpu", 1, false)}, "nvidia.com/gpu", 1, 4},
		{"init after sidecar stacks", []corev1.Container{gpuCtr("s", "amd.com/gpu", 1, true), gpuCtr("i", "amd.com/gpu", 2, false)},
			nil, "amd.com/gpu", 1, 3},
		{"MIG slice", nil, []corev1.Container{gpuCtr("a", "nvidia.com/mig-1g.10gb", 1, false)}, "nvidia.com/mig-1g.10gb", 1, 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			pod := &corev1.Pod{Spec: corev1.PodSpec{InitContainers: tc.init, Containers: tc.app}}
			running, peak := PodGPUClaims(pod)
			if got := running[tc.res]; got.Value() != tc.wantRunning {
				t.Errorf("running = %d, want %d", got.Value(), tc.wantRunning)
			}
			if got := peak[tc.res]; got.Value() != tc.wantPk {
				t.Errorf("peak = %d, want %d", got.Value(), tc.wantPk)
			}
		})
	}
}

func TestPodGPUClaimsIgnoresNonGPU(t *testing.T) {
	pod := &corev1.Pod{Spec: corev1.PodSpec{Containers: []corev1.Container{gpuCtr("a", "example.com/fpga", 1, false)}}}
	if _, peak := PodGPUClaims(pod); len(peak) != 0 {
		t.Errorf("peak = %v, want empty", peak)
	}
}
