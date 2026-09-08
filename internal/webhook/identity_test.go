package webhook

import (
	"context"
	"testing"

	"github.com/gavinsan33/aibom-webhook-service/internal/aibomdata"
	authenticationv1 "k8s.io/api/authentication/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"
)

// newFakeClientsetWithTokens returns a fake clientset whose CreateToken
// subresource actually fills in a token, since client-go's default fake
// reactor for the "serviceaccounts/token" subresource leaves Status.Token
// empty.
func newFakeClientsetWithTokens() *fake.Clientset {
	clientset := fake.NewSimpleClientset()
	clientset.PrependReactor("create", "serviceaccounts", func(action k8stesting.Action) (bool, runtime.Object, error) {
		if action.GetSubresource() != "token" {
			return false, nil, nil
		}
		tr := action.(k8stesting.CreateAction).GetObject().(*authenticationv1.TokenRequest).DeepCopy()
		tr.Status.Token = "fake-token-for-" + tr.Name
		return true, tr, nil
	})
	return clientset
}

func TestEnsureWorkloadIdentityProvisionsAllResources(t *testing.T) {
	clientset := newFakeClientsetWithTokens()
	ctx := context.Background()

	secretName, err := ensureWorkloadIdentity(ctx, clientset, "ns1", "train-job", "train-job-aibom-postprocess-data")
	if err != nil {
		t.Fatalf("ensureWorkloadIdentity() error = %v", err)
	}

	wantName := aibomdata.WorkloadIdentityName("train-job")
	if secretName != wantName {
		t.Errorf("secretName = %q, want %q", secretName, wantName)
	}

	if _, err := clientset.CoreV1().ConfigMaps("ns1").Get(ctx, "train-job-aibom-postprocess-data", metav1.GetOptions{}); err != nil {
		t.Errorf("data configmap not created: %v", err)
	}
	if _, err := clientset.CoreV1().ServiceAccounts("ns1").Get(ctx, wantName, metav1.GetOptions{}); err != nil {
		t.Errorf("serviceaccount not created: %v", err)
	}
	role, err := clientset.RbacV1().Roles("ns1").Get(ctx, wantName, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("role not created: %v", err)
	}
	if len(role.Rules) != 1 || len(role.Rules[0].ResourceNames) != 1 || role.Rules[0].ResourceNames[0] != "train-job-aibom-postprocess-data" {
		t.Errorf("role rules = %+v, want scoped to train-job-aibom-postprocess-data only", role.Rules)
	}
	for _, v := range role.Rules[0].Verbs {
		if v == "create" {
			t.Errorf("role grants create, which resourceNames can't restrict -- should be excluded")
		}
	}
	if _, err := clientset.RbacV1().RoleBindings("ns1").Get(ctx, wantName, metav1.GetOptions{}); err != nil {
		t.Errorf("rolebinding not created: %v", err)
	}
	secret, err := clientset.CoreV1().Secrets("ns1").Get(ctx, wantName, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("token secret not created: %v", err)
	}
	if len(secret.Data[workloadIdentityTokenSecretKey]) == 0 {
		t.Errorf("token secret has no token data")
	}
}

func TestEnsureWorkloadIdentityIsIdempotent(t *testing.T) {
	clientset := newFakeClientsetWithTokens()
	ctx := context.Background()

	if _, err := ensureWorkloadIdentity(ctx, clientset, "ns1", "train-job", "train-job-aibom-postprocess-data"); err != nil {
		t.Fatalf("first ensureWorkloadIdentity() error = %v", err)
	}
	secretName, err := ensureWorkloadIdentity(ctx, clientset, "ns1", "train-job", "train-job-aibom-postprocess-data")
	if err != nil {
		t.Fatalf("second ensureWorkloadIdentity() error = %v", err)
	}
	wantName := aibomdata.WorkloadIdentityName("train-job")
	if secretName != wantName {
		t.Errorf("secretName = %q, want %q", secretName, wantName)
	}
}

func TestEnsureWorkloadIdentityScopesDifferentJobsToDifferentConfigMaps(t *testing.T) {
	clientset := newFakeClientsetWithTokens()
	ctx := context.Background()

	if _, err := ensureWorkloadIdentity(ctx, clientset, "ns1", "job-a", "job-a-aibom-postprocess-data"); err != nil {
		t.Fatalf("ensureWorkloadIdentity(job-a) error = %v", err)
	}
	if _, err := ensureWorkloadIdentity(ctx, clientset, "ns1", "job-b", "job-b-aibom-postprocess-data"); err != nil {
		t.Fatalf("ensureWorkloadIdentity(job-b) error = %v", err)
	}

	roleA, err := clientset.RbacV1().Roles("ns1").Get(ctx, aibomdata.WorkloadIdentityName("job-a"), metav1.GetOptions{})
	if err != nil {
		t.Fatalf("role for job-a not found: %v", err)
	}
	roleB, err := clientset.RbacV1().Roles("ns1").Get(ctx, aibomdata.WorkloadIdentityName("job-b"), metav1.GetOptions{})
	if err != nil {
		t.Fatalf("role for job-b not found: %v", err)
	}
	if roleA.Rules[0].ResourceNames[0] == roleB.Rules[0].ResourceNames[0] {
		t.Errorf("job-a and job-b roles both scoped to %q, want distinct ConfigMaps", roleA.Rules[0].ResourceNames[0])
	}
}
