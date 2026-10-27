package main

import (
	"crypto/tls"
	"log"
	"os"
	"sync"
	"time"
)

// certReloader serves the key pair at certPath/keyPath and re-reads it when the
// certificate file changes. service-ca and cert-manager both rotate the mounted
// Secret in place, but ListenAndServeTLS loads it only once, so a long-running
// pod would keep serving the old certificate past its expiry. The API server
// would then stop calling the webhook, and since it fails open, pods would just
// go uninstrumented with no error.
type certReloader struct {
	certPath, keyPath string

	mu      sync.Mutex
	modTime time.Time
	cert    *tls.Certificate
	// modTime of the last pair that failed to load, so a persistently broken
	// pair is warned about once rather than on every handshake.
	failedModTime time.Time
}

func (r *certReloader) GetCertificate(*tls.ClientHelloInfo) (*tls.Certificate, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	info, err := os.Stat(r.certPath)
	if err == nil && r.cert != nil && info.ModTime().Equal(r.modTime) {
		return r.cert, nil
	}
	if err == nil {
		var cert tls.Certificate
		cert, err = tls.LoadX509KeyPair(r.certPath, r.keyPath)
		if err == nil {
			r.cert, r.modTime = &cert, info.ModTime()
			return r.cert, nil
		}
	}
	// Mid-rotation the two files can briefly disagree; keep serving the last
	// good pair until they don't.
	if r.cert != nil {
		var mtime time.Time
		if info != nil {
			mtime = info.ModTime()
		}
		if !mtime.Equal(r.failedModTime) {
			r.failedModTime = mtime
			log.Printf("warning: could not reload TLS certificate, keeping the previous one: %v", err)
		}
		return r.cert, nil
	}
	return nil, err
}
