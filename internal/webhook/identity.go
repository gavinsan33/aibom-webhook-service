package webhook

import (
	"context"
	"fmt"
	"time"

	"github.com/gavinsan33/aibom-webhook-service/internal/aibomdata"
	authenticationv1 "k8s.io/api/authentication/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes"
)

// WorkloadIdentityTokenTTL bounds how long the token minted below stays
// valid. Unlike the kubelet-managed ServiceAccountTokenProjection used for
// a pod's own identity (silently rotated for the life of the pod), this
// token is requested once, at admission time, and never rotated -- it has
// to outlive the longest training run this cluster expects to run, since a
// token expiring mid-run would break runtime_detector.py's atexit flush on
// an otherwise-successful job. The real enforcement against this identity
// outliving the job is ensureWorkloadIdentity's caller deleting the
// ServiceAccount once the job's postprocess Job succeeds (see watcher.go's
// collectAIBOM) -- that invalidates every token issued for it immediately,
// regardless of this TTL. This is a backstop for jobs that never reach that
// cleanup path (e.g. deleted without ever completing).
var WorkloadIdentityTokenTTL = 7 * 24 * time.Hour

// workloadIdentityTokenSecretKey is the data key holding the raw token
// inside the Secret ensureWorkloadIdentity produces -- matched by
// buildTokenVolume's SecretProjection.
const workloadIdentityTokenSecretKey = "token"

// ensureWorkloadIdentity idempotently provisions a per-job ServiceAccount,
// a Role scoped via resourceNames to exactly this job's own data
// ConfigMap, a RoleBinding tying the two together, and a Secret holding a
// token for that ServiceAccount -- minted via the TokenRequest API rather
// than a projected volume, since a projected ServiceAccountTokenProjection
// can only ever mint a token for the pod's own spec.serviceAccountName
// (which a workload may also need untouched for image pulls or cloud IAM
// federation, e.g. AWS IRSA/GCP Workload Identity).
//
// This exists so pods that write into this job's data ConfigMap can do so
// as an identity scoped to exactly that one ConfigMap, closing the
// cross-job interference described in #43 -- today, every pod in the
// namespace shares one broad Role granting create/get/patch on every
// ConfigMap, regardless of which job it belongs to.
//
// The returned Secret name is only ever valid for mounting the token; it
// is not itself sensitive metadata. Callers must treat any returned error
// as "identity provisioning unavailable" and fall back to the pod's own
// ServiceAccount for this write path (this service fails open -- see
// failurePolicy: Ignore in the webhook configuration -- so a Kubernetes API
// hiccup here must never block pod admission).
func ensureWorkloadIdentity(ctx context.Context, clientset kubernetes.Interface, namespace, triggerName, configMapName string, ownerRef metav1.OwnerReference) (secretName string, err error) {
	name := aibomdata.WorkloadIdentityName(triggerName)

	if err := ensureConfigMap(ctx, clientset, namespace, configMapName); err != nil {
		return "", fmt.Errorf("ensure data configmap %s/%s: %w", namespace, configMapName, err)
	}
	if err := ensureServiceAccount(ctx, clientset, namespace, name); err != nil {
		return "", fmt.Errorf("ensure serviceaccount %s/%s: %w", namespace, name, err)
	}
	if err := ensureRole(ctx, clientset, namespace, name, configMapName); err != nil {
		return "", fmt.Errorf("ensure role %s/%s: %w", namespace, name, err)
	}
	if err := ensureRoleBinding(ctx, clientset, namespace, name); err != nil {
		return "", fmt.Errorf("ensure rolebinding %s/%s: %w", namespace, name, err)
	}

	existing, err := clientset.CoreV1().Secrets(namespace).Get(ctx, name, metav1.GetOptions{})
	if err == nil && len(existing.Data[workloadIdentityTokenSecretKey]) > 0 {
		// Already provisioned by an earlier pod of the same job (e.g. a
		// JobSet replica, or a retried Job pod) -- reuse it rather than
		// minting (and having to track) a second live token per job.
		return name, nil
	}
	if err != nil {
		if !apierrors.IsNotFound(err) {
			return "", fmt.Errorf("get token secret %s/%s: %w", namespace, name, err)
		}
		existing = nil
	}

	token, err := requestServiceAccountToken(ctx, clientset, namespace, name, ownerRef)
	if err != nil {
		return "", fmt.Errorf("request token for %s/%s: %w", namespace, name, err)
	}

	if err := ensureTokenSecret(ctx, clientset, namespace, name, token, existing); err != nil {
		return "", fmt.Errorf("write token secret %s/%s: %w", namespace, name, err)
	}
	return name, nil
}

func ensureConfigMap(ctx context.Context, clientset kubernetes.Interface, namespace, name string) error {
	cm := &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace}}
	_, err := clientset.CoreV1().ConfigMaps(namespace).Create(ctx, cm, metav1.CreateOptions{})
	if err != nil && !apierrors.IsAlreadyExists(err) {
		return err
	}
	return nil
}

func ensureServiceAccount(ctx context.Context, clientset kubernetes.Interface, namespace, name string) error {
	sa := &corev1.ServiceAccount{ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace}}
	_, err := clientset.CoreV1().ServiceAccounts(namespace).Create(ctx, sa, metav1.CreateOptions{})
	if err != nil && !apierrors.IsAlreadyExists(err) {
		return err
	}
	return nil
}

// ensureRole grants get/patch (not create -- ensureConfigMap above already
// guarantees the ConfigMap exists by the time any pod using this identity
// starts, closing off the one gap resourceNames can't cover: RBAC
// resourceNames restrictions don't apply to the create verb, since the
// object doesn't exist yet for the API server to match a name against) on
// exactly one ConfigMap, named by resourceNames rather than a namespace-wide
// grant.
func ensureRole(ctx context.Context, clientset kubernetes.Interface, namespace, name, configMapName string) error {
	role := &rbacv1.Role{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace},
		Rules: []rbacv1.PolicyRule{
			{
				APIGroups:     []string{""},
				Resources:     []string{"configmaps"},
				Verbs:         []string{"get", "patch"},
				ResourceNames: []string{configMapName},
			},
		},
	}
	_, err := clientset.RbacV1().Roles(namespace).Create(ctx, role, metav1.CreateOptions{})
	if err != nil && !apierrors.IsAlreadyExists(err) {
		return err
	}
	return nil
}

func ensureRoleBinding(ctx context.Context, clientset kubernetes.Interface, namespace, name string) error {
	rb := &rbacv1.RoleBinding{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace},
		RoleRef: rbacv1.RoleRef{
			APIGroup: rbacv1.GroupName,
			Kind:     "Role",
			Name:     name,
		},
		Subjects: []rbacv1.Subject{
			{Kind: rbacv1.ServiceAccountKind, Name: name, Namespace: namespace},
		},
	}
	_, err := clientset.RbacV1().RoleBindings(namespace).Create(ctx, rb, metav1.CreateOptions{})
	if err != nil && !apierrors.IsAlreadyExists(err) {
		return err
	}
	return nil
}

// requestServiceAccountToken mints a token for the given ServiceAccount via
// the TokenRequest API. BoundObjectRef is set to the pod's owning Job (etc.)
// purely for audit/descriptive purposes -- Kubernetes only enforces live
// bound-object-existence checks for Pod/Secret references, not arbitrary
// Kinds, so this does not by itself invalidate the token when the Job is
// deleted. What does invalidate it is the ServiceAccount deletion in
// watcher.go's collectAIBOM: every token's validity is always tied to its
// issuing ServiceAccount still existing, independent of BoundObjectRef.
func requestServiceAccountToken(ctx context.Context, clientset kubernetes.Interface, namespace, saName string, ownerRef metav1.OwnerReference) (string, error) {
	expirationSeconds := int64(WorkloadIdentityTokenTTL.Seconds())
	tr := &authenticationv1.TokenRequest{
		ObjectMeta: metav1.ObjectMeta{Name: saName},
		Spec: authenticationv1.TokenRequestSpec{
			ExpirationSeconds: &expirationSeconds,
			BoundObjectRef: &authenticationv1.BoundObjectReference{
				Kind:       ownerRef.Kind,
				APIVersion: ownerRef.APIVersion,
				Name:       ownerRef.Name,
				UID:        types.UID(ownerRef.UID),
			},
		},
	}
	result, err := clientset.CoreV1().ServiceAccounts(namespace).CreateToken(ctx, saName, tr, metav1.CreateOptions{})
	if err != nil {
		return "", err
	}
	return result.Status.Token, nil
}

// ensureTokenSecret writes token into a Secret named name, creating it if
// absent or updating it in place if a prior (e.g. empty/corrupt) Secret of
// that name already exists -- existing is whatever ensureWorkloadIdentity's
// earlier Get returned (nil if that Get was a NotFound), reused here so
// this doesn't need a second round-trip just to learn the ResourceVersion
// an Update requires.
func ensureTokenSecret(ctx context.Context, clientset kubernetes.Interface, namespace, name, token string, existing *corev1.Secret) error {
	secret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace},
		Type:       corev1.SecretTypeOpaque,
		Data:       map[string][]byte{workloadIdentityTokenSecretKey: []byte(token)},
	}
	if existing != nil {
		secret.ResourceVersion = existing.ResourceVersion
		_, err := clientset.CoreV1().Secrets(namespace).Update(ctx, secret, metav1.UpdateOptions{})
		return err
	}
	_, err := clientset.CoreV1().Secrets(namespace).Create(ctx, secret, metav1.CreateOptions{})
	if apierrors.IsAlreadyExists(err) {
		// Lost a create race against another pod of the same job; whichever
		// write won is fine, both mint a token for the same ServiceAccount.
		return nil
	}
	return err
}
