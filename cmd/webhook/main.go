package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gavinsan33/aibom-webhook-service/internal/config"
	"github.com/gavinsan33/aibom-webhook-service/internal/webhook"
)

func main() {
	cfg := config.Config{}

	flag.StringVar(&cfg.TLSCertPath, "tls-cert", "/certs/tls.crt", "path to TLS certificate")
	flag.StringVar(&cfg.TLSKeyPath, "tls-key", "/certs/tls.key", "path to TLS private key")
	flag.IntVar(&cfg.Port, "port", 8443, "webhook server port")
	flag.StringVar(&cfg.DiscoveryImage, "discovery-image", "busybox:latest", "image for the discovery init container")
	flag.Parse()

	mutator := webhook.NewMutator(cfg.DiscoveryImage)
	handler := webhook.NewHandler(mutator)

	mux := http.NewServeMux()
	mux.Handle("/mutate", handler)
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("ok"))
	})

	server := &http.Server{
		Addr:         fmt.Sprintf(":%d", cfg.Port),
		Handler:      mux,
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 10 * time.Second,
	}

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGTERM, syscall.SIGINT)

	go func() {
		log.Printf("starting webhook server on :%d (discovery-image=%s)", cfg.Port, cfg.DiscoveryImage)
		if err := server.ListenAndServeTLS(cfg.TLSCertPath, cfg.TLSKeyPath); err != nil && err != http.ErrServerClosed {
			log.Fatalf("server error: %v", err)
		}
	}()

	<-stop
	log.Println("shutting down...")

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := server.Shutdown(ctx); err != nil {
		log.Fatalf("shutdown error: %v", err)
	}
	log.Println("server stopped")
}
