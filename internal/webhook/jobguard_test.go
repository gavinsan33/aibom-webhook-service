package webhook

import (
	"testing"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const testTrustedIdentity = "system:serviceaccount:aibom-system:aibom-webhook"

func jobWithPostprocessLabel() *batchv1.Job {
	return &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "sneaky-job",
			Namespace: "default",
			Labels:    map[string]string{"aibom.io/postprocess-for": "train-job"},
		},
		Spec: batchv1.JobSpec{
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"aibom.io/postprocess-for": "train-job"},
				},
			},
		},
	}
}

func TestSanitizeJobPostprocessLabel_StripsFromUntrustedRequester(t *testing.T) {
	job := jobWithPostprocessLabel()
	patches := SanitizeJobPostprocessLabel(job, nil, "system:serviceaccount:default:some-user-sa", testTrustedIdentity)

	if len(patches) != 2 {
		t.Fatalf("expected 2 remove patches (job label + template label), got %d: %+v", len(patches), patches)
	}
	wantPaths := map[string]bool{
		"/metadata/labels/aibom.io~1postprocess-for":               false,
		"/spec/template/metadata/labels/aibom.io~1postprocess-for": false,
	}
	for _, p := range patches {
		if p.Op != "remove" {
			t.Errorf("expected remove op, got %q", p.Op)
		}
		if _, ok := wantPaths[p.Path]; !ok {
			t.Errorf("unexpected patch path %q", p.Path)
		}
		wantPaths[p.Path] = true
	}
	for path, seen := range wantPaths {
		if !seen {
			t.Errorf("expected a remove patch for %q", path)
		}
	}
}

func TestSanitizeJobPostprocessLabel_TrustedRequesterUntouched(t *testing.T) {
	job := jobWithPostprocessLabel()
	patches := SanitizeJobPostprocessLabel(job, nil, testTrustedIdentity, testTrustedIdentity)
	if patches != nil {
		t.Errorf("expected no patches for the trusted watcher identity, got %+v", patches)
	}
}

func TestSanitizeJobPostprocessLabel_DisabledWhenTrustedIdentityUnset(t *testing.T) {
	job := jobWithPostprocessLabel()
	patches := SanitizeJobPostprocessLabel(job, nil, "system:serviceaccount:default:some-user-sa", "")
	if patches != nil {
		t.Errorf("expected no patches when TrustedWatcherIdentity is unconfigured (fail open), got %+v", patches)
	}
}

func TestSanitizeJobPostprocessLabel_NoLabelsPresent_NoPatches(t *testing.T) {
	job := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{Name: "plain-job", Namespace: "default"},
	}
	patches := SanitizeJobPostprocessLabel(job, nil, "system:serviceaccount:default:some-user-sa", testTrustedIdentity)
	if patches != nil {
		t.Errorf("expected no patches when the label isn't present, got %+v", patches)
	}
}

// TestSanitizeJobPostprocessLabel_UpdateAddsLabel_Stripped covers the
// bypass the reviewer flagged: a Job (or its pod template) created without
// the label, then updated afterward to add it -- an UPDATE-time check must
// still catch this even though CREATE saw nothing to strip.
func TestSanitizeJobPostprocessLabel_UpdateAddsLabel_Stripped(t *testing.T) {
	oldJob := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{Name: "sneaky-job", Namespace: "default"},
	}
	newJob := jobWithPostprocessLabel()

	patches := SanitizeJobPostprocessLabel(newJob, oldJob, "system:serviceaccount:default:some-user-sa", testTrustedIdentity)
	if len(patches) != 2 {
		t.Fatalf("expected 2 remove patches for a label newly added on UPDATE, got %d: %+v", len(patches), patches)
	}
}

// TestSanitizeJobPostprocessLabel_UpdateChangesLabelValue_Stripped covers
// retargeting an existing, previously-legitimate label to point at a
// different Job.
func TestSanitizeJobPostprocessLabel_UpdateChangesLabelValue_Stripped(t *testing.T) {
	oldJob := jobWithPostprocessLabel()
	newJob := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "sneaky-job",
			Namespace: "default",
			Labels:    map[string]string{"aibom.io/postprocess-for": "some-other-job"},
		},
		Spec: batchv1.JobSpec{
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"aibom.io/postprocess-for": "train-job"},
				},
			},
		},
	}

	patches := SanitizeJobPostprocessLabel(newJob, oldJob, "system:serviceaccount:default:some-user-sa", testTrustedIdentity)
	if len(patches) != 1 {
		t.Fatalf("expected 1 remove patch (only the job label changed), got %d: %+v", len(patches), patches)
	}
	if patches[0].Path != "/metadata/labels/aibom.io~1postprocess-for" {
		t.Errorf("expected the job-label patch, got %+v", patches[0])
	}
}

// TestSanitizeJobPostprocessLabel_UpdateUnchangedLabel_NotStripped ensures
// an unrelated update to a Job that already legitimately carries the label
// (e.g. set by the watcher on a prior admission) doesn't get it stripped
// again just because the requester on this particular UPDATE isn't trusted.
func TestSanitizeJobPostprocessLabel_UpdateUnchangedLabel_NotStripped(t *testing.T) {
	oldJob := jobWithPostprocessLabel()
	newJob := jobWithPostprocessLabel()

	patches := SanitizeJobPostprocessLabel(newJob, oldJob, "system:serviceaccount:default:some-user-sa", testTrustedIdentity)
	if patches != nil {
		t.Errorf("expected no patches when the label is unchanged from oldJob, got %+v", patches)
	}
}
