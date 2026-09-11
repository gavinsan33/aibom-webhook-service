package webhook

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"

	admissionv1 "k8s.io/api/admission/v1"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
)

var (
	podGVR = metav1.GroupVersionResource{Group: "", Version: "v1", Resource: "pods"}
	jobGVR = metav1.GroupVersionResource{Group: "batch", Version: "v1", Resource: "jobs"}
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

	// TrustedWatcherIdentity is passed through to SanitizeJobPostprocessLabel
	// for every Job admission -- see its doc comment and config.Config's
	// TrustedWatcherIdentity field.
	TrustedWatcherIdentity string
}

func NewHandler(mutator *Mutator, trustedWatcherIdentity string) *Handler {
	return &Handler{Mutator: mutator, TrustedWatcherIdentity: trustedWatcherIdentity}
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
	if review.Request != nil {
		review.Response.UID = review.Request.UID
	}

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

	switch req.Resource {
	case podGVR:
		return h.handlePodAdmission(req)
	case jobGVR:
		return h.handleJobAdmission(req)
	default:
		return allowResponse("not a supported resource")
	}
}

func (h *Handler) handlePodAdmission(req *admissionv1.AdmissionRequest) *admissionv1.AdmissionResponse {
	var pod corev1.Pod
	if err := json.Unmarshal(req.Object.Raw, &pod); err != nil {
		log.Printf("failed to unmarshal pod: %v", err)
		return allowResponse("failed to unmarshal pod")
	}

	patches, err := h.Mutator.Mutate(&pod, req.UserInfo.Username)
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

// handleJobAdmission runs SanitizeJobPostprocessLabel against every Job
// creation in an opted-in namespace -- see that function's doc comment for
// why aibom.io/postprocess-for can't be trusted from the Job object alone.
func (h *Handler) handleJobAdmission(req *admissionv1.AdmissionRequest) *admissionv1.AdmissionResponse {
	var job batchv1.Job
	if err := json.Unmarshal(req.Object.Raw, &job); err != nil {
		log.Printf("failed to unmarshal job: %v", err)
		return allowResponse("failed to unmarshal job")
	}

	patches := SanitizeJobPostprocessLabel(&job, req.UserInfo.Username, h.TrustedWatcherIdentity)
	if patches == nil {
		return allowResponse("no mutation needed")
	}

	patchBytes, err := json.Marshal(patches)
	if err != nil {
		log.Printf("failed to marshal patches: %v", err)
		return allowResponse("failed to marshal patches")
	}

	patchType := admissionv1.PatchTypeJSONPatch
	log.Printf("stripping spoofed aibom.io/postprocess-for from job %s/%s (requester %q is not the trusted watcher identity)", job.Namespace, job.Name, req.UserInfo.Username)
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
