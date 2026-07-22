package webhook

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"

	admissionv1 "k8s.io/api/admission/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
)

var (
	scheme = runtime.NewScheme()
	codecs serializer.CodecFactory
)

func init() {
	_ = admissionv1.AddToScheme(scheme)
	codecs = serializer.NewCodecFactory(scheme)
}

type Handler struct {
	Mutator *Mutator
}

func NewHandler(mutator *Mutator) *Handler {
	return &Handler{Mutator: mutator}
}

func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	contentType := r.Header.Get("Content-Type")
	if contentType != "application/json" {
		http.Error(w, "expected application/json content type", http.StatusUnsupportedMediaType)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to read body: %v", err), http.StatusBadRequest)
		return
	}

	var review admissionv1.AdmissionReview
	if _, _, err := codecs.UniversalDeserializer().Decode(body, nil, &review); err != nil {
		http.Error(w, fmt.Sprintf("failed to decode admission review: %v", err), http.StatusBadRequest)
		return
	}

	response := h.handleAdmission(&review)

	review.Response = response
	review.Response.UID = review.Request.UID

	respBytes, err := json.Marshal(review)
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to marshal response: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.Write(respBytes)
}

func (h *Handler) handleAdmission(review *admissionv1.AdmissionReview) *admissionv1.AdmissionResponse {
	req := review.Request
	if req == nil {
		return allowResponse("no request in review")
	}

	if req.Resource != (metav1.GroupVersionResource{Group: "", Version: "v1", Resource: "pods"}) {
		return allowResponse("not a pod resource")
	}

	var pod corev1.Pod
	if err := json.Unmarshal(req.Object.Raw, &pod); err != nil {
		log.Printf("failed to unmarshal pod: %v", err)
		return allowResponse("failed to unmarshal pod")
	}

	patches, err := h.Mutator.Mutate(&pod)
	if err != nil {
		log.Printf("mutation error: %v", err)
		return allowResponse("mutation error")
	}

	if patches == nil {
		return allowResponse("no mutation needed")
	}

	patchBytes, err := json.Marshal(patches)
	if err != nil {
		log.Printf("failed to marshal patches: %v", err)
		return allowResponse("failed to marshal patches")
	}

	patchType := admissionv1.PatchTypeJSONPatch
	log.Printf("mutating pod %s/%s: %d patches", pod.Namespace, pod.Name, len(patches))
	return &admissionv1.AdmissionResponse{
		Allowed:   true,
		PatchType: &patchType,
		Patch:     patchBytes,
	}
}

func allowResponse(reason string) *admissionv1.AdmissionResponse {
	return &admissionv1.AdmissionResponse{
		Allowed: true,
		Result: &metav1.Status{
			Message: reason,
		},
	}
}
