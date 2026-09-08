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
}
