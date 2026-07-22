package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/gavinsan33/aibom-webhook-service/internal/config"
	"github.com/gavinsan33/aibom-webhook-service/internal/watcher"
	"github.com/gavinsan33/aibom-webhook-service/internal/webhook"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
)

func main() {
	cfg := config.Config{}

	flag.StringVar(&cfg.TLSCertPath, "tls-cert", "/certs/tls.crt", "path to TLS certificate")
	flag.StringVar(&cfg.TLSKeyPath, "tls-key", "/certs/tls.key", "path to TLS private key")
	flag.IntVar(&cfg.Port, "port", 8443, "webhook server port")
	flag.StringVar(&cfg.DiscoveryImage, "discovery-image", "pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime", "image for the discovery init container")
	flag.BoolVar(&cfg.DatasetDetection, "dataset-detection", true, "inject dataset detection hooks into application containers")
	flag.BoolVar(&cfg.EnableWatcher, "enable-watcher", true, "start the Job completion watcher")
	flag.StringVar(&cfg.PostprocessImage, "postprocess-image", "busybox:latest", "image for postprocess Jobs")
	flag.Parse()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	mutator := webhook.NewMutator(cfg.DiscoveryImage, cfg.DatasetDetection)
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
		log.Printf("starting webhook server on :%d (discovery-image=%s, dataset-detection=%v)", cfg.Port, cfg.DiscoveryImage, cfg.DatasetDetection)
		if err := server.ListenAndServeTLS(cfg.TLSCertPath, cfg.TLSKeyPath); err != nil && err != http.ErrServerClosed {
			log.Fatalf("server error: %v", err)
		}
	}()

	if cfg.EnableWatcher {
		clientset, err := buildClientset()
		if err != nil {
			log.Printf("WARNING: failed to create Kubernetes client, watcher disabled: %v", err)
		} else {
			w := watcher.New(clientset, cfg.PostprocessImage)
			go func() {
				if err := w.Start(ctx); err != nil {
					log.Printf("watcher error: %v", err)
				}
			}()
		}
	}

	<-stop
	log.Println("shutting down...")

	cancel()

	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer shutdownCancel()
	if err := server.Shutdown(shutdownCtx); err != nil {
		log.Fatalf("shutdown error: %v", err)
	}
	log.Println("server stopped")
}

func buildClientset() (kubernetes.Interface, error) {
	cfg, err := rest.InClusterConfig()
	if err != nil {
		kubeconfig := os.Getenv("KUBECONFIG")
		if kubeconfig == "" {
			home, _ := os.UserHomeDir()
			kubeconfig = filepath.Join(home, ".kube", "config")
		}
		cfg, err = clientcmd.BuildConfigFromFlags("", kubeconfig)
		if err != nil {
			return nil, fmt.Errorf("no in-cluster config and no kubeconfig found: %w", err)
		}
	}
	return kubernetes.NewForConfig(cfg)
}
