package webhook

import (
	"fmt"

	"github.com/gavinsan33/aibom-webhook-service/internal/aibomdata"
	corev1 "k8s.io/api/core/v1"
)

// ValidateSigningPublicKeyConfigMap denies a create/update of the
// aibom-compiled-signing-public-key ConfigMap (see aibomdata's
// CompiledSigningPublicKeyConfigMapName, and CLAUDE.md's Compiled AIBOM
// Signing) from anyone other than that namespace's own aibom-postprocess
// ServiceAccount.
//
// Without this check, this ConfigMap has no RBAC of its own -- it's created
// via rbac.yaml's broad aibom-workload-data Role, bound to every
// ServiceAccount in the namespace (system:serviceaccounts:<ns>), including
// whatever ServiceAccount the workload's own untrusted app container runs
// as. RBAC is purely additive, so no narrower Role added elsewhere can
// subtract from that grant -- the only way to actually close this is
// admission-time enforcement, the same approach SanitizeJobPostprocessLabel
// already takes for aibom.io/postprocess-for.
//
// The attack this closes: a training pod PATCHes this ConfigMap to an
// attacker-generated public key, matching a private key only the attacker
// holds. A verifier (oc-aibom) checking a since-forged AIBOM -- signed with
// that attacker key, with the matching public key embedded in
// spec.signaturePublicKey -- would then report it fully Verified, since its
// embedded key now matches the (attacker-controlled) cluster anchor. Every
// legitimate AIBOM's cross-check would fail at the same time, since the
// anchor no longer matches the real signing key.
//
// Unlike SanitizeJobPostprocessLabel's TrustedWatcherIdentity, no operator
// configuration is needed here: aibom-postprocess is always named exactly
// that (aibomdata.PostprocessServiceAccountName) and always runs in the
// same namespace as the ConfigMap it publishes to -- watcher.go sets it as
// the postprocess Job's ServiceAccountName, and rbac.yaml creates it
// per-namespace. So the trusted identity is fully determined by the
// object's own namespace, not a separately-configured value that could be
// wrong for a given cluster.
//
// Returns "" (allow) for any ConfigMap other than this exact name, or when
// requesterUsername is the trusted identity for cm's namespace. Otherwise
// returns a non-empty denial reason.
//
// This does mean even a cluster-admin's own manual `oc apply`/`oc edit`
// against this ConfigMap gets denied unless done as the aibom-postprocess
// identity itself -- deliberate, mirroring the CRD's own spec immutability:
// the whole point is that nothing other than the trusted signer can produce
// a value this anchor will accept.
func ValidateSigningPublicKeyConfigMap(cm *corev1.ConfigMap, requesterUsername string) string {
	if cm.Name != aibomdata.CompiledSigningPublicKeyConfigMapName {
		return ""
	}
	trusted := fmt.Sprintf("system:serviceaccount:%s:%s", cm.Namespace, aibomdata.PostprocessServiceAccountName)
	if requesterUsername == trusted {
		return ""
	}
	return fmt.Sprintf("%s can only be created or modified by %s", aibomdata.CompiledSigningPublicKeyConfigMapName, trusted)
}
