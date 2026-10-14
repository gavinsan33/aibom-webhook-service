package webhook

import (
	"context"
	"os"
	"strings"
	"testing"

	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/yaml"
)

// The webhook can only create the per-job Role if it holds every permission
// that Role grants (Kubernetes' RBAC escalation check). Those permissions come
// from the namespaced Role in aibom-workload-namespace, so this fails if
// ensureRole grows a rule that Role doesn't cover (#148).
func TestWorkloadNamespaceRoleCoversPerJobRole(t *testing.T) {
	raw, err := os.ReadFile("../../charts/aibom-workload-namespace/templates/webhook-access.yaml")
	if err != nil {
		t.Fatal(err)
	}
	// Drop the template lines; the first document is the Role.
	var lines []string
	for _, l := range strings.Split(string(raw), "\n") {
		if !strings.Contains(l, "{{") {
			lines = append(lines, l)
		}
	}
	doc := strings.SplitN(strings.Join(lines, "\n"), "\n---\n", 2)[0]
	var granted rbacv1.Role
	if err := yaml.Unmarshal([]byte(doc), &granted); err != nil {
		t.Fatal(err)
	}

	clientset := newFakeClientsetWithTokens()
	if err := ensureRole(context.Background(), clientset, "ns1", "r", "cm"); err != nil {
		t.Fatal(err)
	}
	perJob, err := clientset.RbacV1().Roles("ns1").Get(context.Background(), "r", metav1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}

	for _, want := range perJob.Rules {
		for _, res := range want.Resources {
			for _, verb := range want.Verbs {
				if !covers(granted.Rules, want.APIGroups[0], res, verb) {
					t.Errorf("webhook-access.yaml doesn't grant %s on %s/%s, which the per-job Role does", verb, want.APIGroups[0], res)
				}
			}
		}
	}
}

func covers(rules []rbacv1.PolicyRule, group, resource, verb string) bool {
	for _, r := range rules {
		if contains(r.APIGroups, group) && contains(r.Resources, resource) && contains(r.Verbs, verb) {
			return true
		}
	}
	return false
}

func contains(list []string, s string) bool {
	for _, v := range list {
		if v == s {
			return true
		}
	}
	return false
}
