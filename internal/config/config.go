package config

type Config struct {
	TLSCertPath          string
	TLSKeyPath           string
	Port                 int
	DiscoveryImage       string
	DatasetDetection     bool
	EnableWatcher        bool
	PostprocessImage     string
	PrometheusURL        string
	GrafanaURL           string
	GrafanaDatasourceUID string
	// DebugKeepPostprocessJobs skips the usual cleanup of a succeeded postprocess
	// Job/data ConfigMap (see watcher.collectAIBOM) — for inspecting postprocess
	// pod logs/exit state or the data ConfigMap's contents after the fact. Left on,
	// this leaks a Job+ConfigMap per completed workload indefinitely; not meant for
	// routine production use.
	DebugKeepPostprocessJobs bool
	// DebugPostprocessAllPods bypasses the nvidia.com/gpu resource check used to
	// decide which pods qualify for postprocessing, so every instrumented pod
	// qualifies regardless of GPU requests. Intended for local testing against a
	// mock cluster (e.g. kind) with no GPU resources/device plugin at all, where
	// the normal GPU-request signal never fires.
	DebugPostprocessAllPods bool
}
