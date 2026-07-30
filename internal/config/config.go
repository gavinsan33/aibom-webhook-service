package config

type Config struct {
	TLSCertPath      string
	TLSKeyPath       string
	Port             int
	DiscoveryImage   string
	DatasetDetection bool
	EnableWatcher    bool
	PostprocessImage string
}
