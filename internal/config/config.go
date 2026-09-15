package config

type Config struct {
	TLSCertPath      string
	TLSKeyPath       string
	Port             int
	DiscoveryImage   string
	DatasetDetection bool
	EnableWatcher    bool
	PostprocessImage string

	// TrustedWatcherIdentity is the full Kubernetes username (e.g.
	// "system:serviceaccount:aibom-system:aibom-webhook") of the
	// ServiceAccount this same binary's watcher runs as. It's used to
	// verify that a Job claiming aibom.io/postprocess-for was actually
	// created by the watcher, not by a workload spoofing the label to
	// dodge instrumentation -- see webhook's SanitizeJobPostprocessLabel.
	// Left empty, this check is disabled (fails open).
	TrustedWatcherIdentity string

	// TrustedJobControllerIdentity is the full username the cluster's
	// built-in Job controller uses when creating a Job's pods (typically
	// "system:serviceaccount:kube-system:job-controller", but this
	// depends on kube-controller-manager's --use-service-account-credentials
	// flag, so it isn't defaulted -- verify it on your own cluster before
	// setting it). Tightens the webhook's isPostprocessPod check: without
	// it, a raw Pod with a fabricated Job ownerReference can still dodge
	// instrumentation. Left empty, only the weaker ownerReference-only
	// check applies.
	TrustedJobControllerIdentity string
}
