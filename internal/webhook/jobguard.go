package webhook

import (
	"github.com/gavinsan33/aibom-webhook-service/internal/aibomdata"
	batchv1 "k8s.io/api/batch/v1"
)

// SanitizeJobPostprocessLabel strips aibom.io/postprocess-for from both a
// Job's own labels and its pod template's labels, unless the Job was
// created by the trusted watcher identity (trustedIdentity, compared
// against the AdmissionReview's request.userInfo.username).
//
// This label is meant to mark a postprocess Job the watcher itself created
// (see watcher.go's createPostprocessJobCore) so the webhook's
// isPostprocessPod can avoid re-instrumenting its pod. But it's just Pod/Job
// metadata a requester can set on their own manifest -- without this check,
// any workload could self-label as a postprocess Job to dodge instrumentation
// entirely (and, via watcher.go's own Job-level check, dodge postprocessing
// too). request.userInfo is populated by the API server from real
// authentication and can't be spoofed by a requester the way labels can, so
// comparing it against the watcher's own identity is what actually closes
// this rather than just raising the bar.
//
// Both job.Labels and job.Spec.Template.ObjectMeta.Labels need stripping:
// the Job's own labels are what watcher.go's onJobEvent checks to skip
// postprocessing a Job entirely, while the pod template's labels are what
// the Job controller copies onto the actual pods mutator.go's
// isPostprocessPod later inspects. Both need to be sanitized at Job-creation
// time so the Job controller only ever propagates a clean value.
//
// If trustedIdentity is empty (not configured), this check is disabled and
// no patches are ever returned -- fails open rather than stripping a
// legitimate value when the operator hasn't wired up the identity to
// compare against, consistent with this webhook's general failurePolicy:
// Ignore posture.
//
// This does not close every path to spoofing the label -- a raw Pod
// submitted directly (not via a Job at all) never goes through this check.
// See isPostprocessPod's own doc comment for the complementary, narrower
// defense on that path, and its documented residual gap.
func SanitizeJobPostprocessLabel(job *batchv1.Job, requesterUsername, trustedIdentity string) []PatchOperation {
	if trustedIdentity == "" || requesterUsername == trustedIdentity {
		return nil
	}

	var patches []PatchOperation
	if _, ok := job.Labels[aibomdata.LabelPostprocessFor]; ok {
		patches = append(patches, PatchOperation{
			Op:   "remove",
			Path: "/metadata/labels/aibom.io~1postprocess-for",
		})
	}
	if _, ok := job.Spec.Template.ObjectMeta.Labels[aibomdata.LabelPostprocessFor]; ok {
		patches = append(patches, PatchOperation{
			Op:   "remove",
			Path: "/spec/template/metadata/labels/aibom.io~1postprocess-for",
		})
	}
	return patches
}
