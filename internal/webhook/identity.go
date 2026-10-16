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
func ensureWorkloadIdentity(ctx context.Context, clientset kubernetes.Interface, namespace, triggerName, configMapName string) (secretName string, err error) {
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

	token, err := requestServiceAccountToken(ctx, clientset, namespace, name)
	if err != nil {
		return "", fmt.Errorf("request token for %s/%s: %w", namespace, name, err)
	}

	if err := ensureTokenSecret(ctx, clientset, namespace, name, token); err != nil {
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
// the TokenRequest API, bounded only by WorkloadIdentityTokenTTL -- no
// BoundObjectRef. The apiserver only accepts BoundObjectRef.Kind values of
// "Pod" or "Secret" (anything else, including the owning Job this identity
// is for, is rejected outright: "cannot bind token to object of type
// batch/v1, Kind=Job"), and the pod doesn't exist as an object we could
// bind to yet at admission time anyway. Real invalidation instead comes
// from the ServiceAccount deletion in watcher.go's collectAIBOM: every
// token's validity is always tied to its issuing ServiceAccount still
// existing, independent of any BoundObjectRef.
func requestServiceAccountToken(ctx context.Context, clientset kubernetes.Interface, namespace, saName string) (string, error) {
	expirationSeconds := int64(WorkloadIdentityTokenTTL.Seconds())
	tr := &authenticationv1.TokenRequest{
		ObjectMeta: metav1.ObjectMeta{Name: saName},
		Spec: authenticationv1.TokenRequestSpec{
			ExpirationSeconds: &expirationSeconds,
		},
	}
	result, err := clientset.CoreV1().ServiceAccounts(namespace).CreateToken(ctx, saName, tr, metav1.CreateOptions{})
	if err != nil {
		return "", err
	}
	return result.Status.Token, nil
}

// ensureTokenSecret creates the Secret holding token. It never reads or
// updates: the webhook's namespaced Role can't scope get/update to this
// Secret by name (the name is per-job), and a namespace-wide get would also
// expose the signing-key Secrets. An existing Secret is left alone, so a later
// pod of the same job (a JobSet replica, a retried Job pod) reuses the first
// pod's token and the one minted here goes unused until it expires.
func ensureTokenSecret(ctx context.Context, clientset kubernetes.Interface, namespace, name, token string) error {
	secret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace},
		Type:       corev1.SecretTypeOpaque,
		Data:       map[string][]byte{workloadIdentityTokenSecretKey: []byte(token)},
	}
	_, err := clientset.CoreV1().Secrets(namespace).Create(ctx, secret, metav1.CreateOptions{})
	if apierrors.IsAlreadyExists(err) {
		return nil
	}
	return err
}
