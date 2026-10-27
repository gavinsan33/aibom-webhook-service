package main

import (
	"bytes"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"log"
	"math/big"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func writeKeyPair(t *testing.T, dir, cn string, mtime time.Time) {
	t.Helper()
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	tmpl := &x509.Certificate{SerialNumber: big.NewInt(1), Subject: pkix.Name{CommonName: cn}, NotAfter: time.Now().Add(time.Hour)}
	der, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, &key.PublicKey, key)
	if err != nil {
		t.Fatal(err)
	}
	keyDER, _ := x509.MarshalECPrivateKey(key)
	crt, k := filepath.Join(dir, "tls.crt"), filepath.Join(dir, "tls.key")
	os.WriteFile(crt, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}), 0o600)
	os.WriteFile(k, pem.EncodeToMemory(&pem.Block{Type: "EC PRIVATE KEY", Bytes: keyDER}), 0o600)
	os.Chtimes(crt, mtime, mtime)
}

func commonName(t *testing.T, r *certReloader) string {
	t.Helper()
	c, err := r.GetCertificate(nil)
	if err != nil {
		t.Fatal(err)
	}
	leaf, err := x509.ParseCertificate(c.Certificate[0])
	if err != nil {
		t.Fatal(err)
	}
	return leaf.Subject.CommonName
}

// A rotated certificate is picked up without a restart, and a broken rotation
// keeps the last good one, warning once rather than on every handshake.
func TestCertReloaderPicksUpRotation(t *testing.T) {
	dir := t.TempDir()
	r := &certReloader{certPath: filepath.Join(dir, "tls.crt"), keyPath: filepath.Join(dir, "tls.key")}
	now := time.Now()

	writeKeyPair(t, dir, "first", now)
	if got := commonName(t, r); got != "first" {
		t.Fatalf("got %q, want first", got)
	}
	writeKeyPair(t, dir, "second", now.Add(time.Minute))
	if got := commonName(t, r); got != "second" {
		t.Fatalf("after rotation got %q, want second", got)
	}
	var logs bytes.Buffer
	log.SetOutput(&logs)
	defer log.SetOutput(os.Stderr)
	os.WriteFile(r.keyPath, []byte("garbage"), 0o600)
	os.Chtimes(r.certPath, now.Add(2*time.Minute), now.Add(2*time.Minute))
	for range 3 {
		if got := commonName(t, r); got != "second" {
			t.Fatalf("with a broken pair got %q, want the previous second", got)
		}
	}
	if n := strings.Count(logs.String(), "could not reload"); n != 1 {
		t.Fatalf("got %d reload warnings for one broken pair, want 1", n)
	}
}

// A certificate that disappears from a running pod (Secret deleted, mount
// unreadable) is warned about once, not silently served from the cache.
func TestCertReloaderWarnsWhenCertGoesMissing(t *testing.T) {
	dir := t.TempDir()
	r := &certReloader{certPath: filepath.Join(dir, "tls.crt"), keyPath: filepath.Join(dir, "tls.key")}
	writeKeyPair(t, dir, "first", time.Now())
	commonName(t, r)

	var logs bytes.Buffer
	log.SetOutput(&logs)
	defer log.SetOutput(os.Stderr)
	os.Remove(r.certPath)
	for range 3 {
		if got := commonName(t, r); got != "first" {
			t.Fatalf("with the certificate missing got %q, want the cached first", got)
		}
	}
	if n := strings.Count(logs.String(), "could not reload"); n != 1 {
		t.Fatalf("got %d reload warnings for a missing certificate, want 1", n)
	}
}
