# Installing with Argo CD

Both charts can be managed by Argo CD, including OpenShift GitOps, instead of `helm` commands. The webhook is one Application, and each workload namespace gets its own Application, which an ApplicationSet can generate. Argo CD renders charts without access to the cluster, so two values that Helm works out on its own have to be set:

- **`platform`**, on both charts. `auto` detects OpenShift by querying the cluster, which an offline render can't do, so it would render the plain-Kubernetes variant. Set `openshift` or `kubernetes`.
- **`signing.createKeys: false`**, on the namespace chart. The chart reuses existing signing keys by reading them from the cluster, so under Argo CD every sync would replace them, and data from pods running at that moment would fail verification. With it off, the webhook creates any missing key when it admits the namespace's first instrumented pod, and never changes it after that. See [Signing Keys](../reference/signing.md).

Nothing secret goes into Git: the webhook generates the HMAC keys, and the watcher generates the compiled-AIBOM key.

## Permissions

The webhook chart creates CRDs, ClusterRoles, ClusterRoleBindings and a MutatingWebhookConfiguration, and the namespace chart creates a RoleBinding in the webhook's namespace. The Argo CD instance that syncs them needs rights to create all of these, plus the Deployments, Jobs, Services, ServiceAccounts, ImageStreams, PodMonitors and NetworkPolicies the charts put in `project-aibom` and each workload namespace.

On OpenShift GitOps that is normally the default `openshift-gitops` instance, and out of the box it can't sync either chart. Its controller can create RBAC objects, Namespaces and ConfigMaps, but not CRDs or MutatingWebhookConfigurations, and it has no other rights in a namespace unless that namespace is labeled as managed by it. Grant one of:

- **cluster-admin**, the simplest:

    ```bash
    oc create clusterrolebinding openshift-gitops-cluster-admin --clusterrole=cluster-admin \
      --serviceaccount=openshift-gitops:openshift-gitops-argocd-application-controller
    ```

    Anyone who can create Applications in the `openshift-gitops` project then effectively has cluster-admin.

- **Only what the charts need.** A ClusterRole for the two missing cluster-scoped kinds:

    ```bash
    oc create clusterrole aibom-argocd \
      --verb='*' --resource=customresourcedefinitions.apiextensions.k8s.io,mutatingwebhookconfigurations.admissionregistration.k8s.io
    oc create clusterrolebinding aibom-argocd --clusterrole=aibom-argocd \
      --serviceaccount=openshift-gitops:openshift-gitops-argocd-application-controller
    ```

    Then label `project-aibom` and every workload namespace with `argocd.argoproj.io/managed-by=openshift-gitops` before its first sync. The operator then grants the instance admin rights in that namespace. The label can go in the ApplicationSet's `managedNamespaceMetadata` next to `aibom.io/enabled`, but the operator grants the rights only after the namespace exists, so the first sync can fail. Add a `syncPolicy.retry` or sync again.

## The webhook

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: aibom-webhook
  namespace: openshift-gitops
spec:
  project: default
  source:
    repoURL: quay.io/project-aibom
    chart: aibom-webhook
    targetRevision: <version>
    helm:
      valuesObject:
        platform: openshift
  destination:
    server: https://kubernetes.default.svc
    namespace: project-aibom
  syncPolicy:
    automated:
      prune: true
    syncOptions:
      - CreateNamespace=true
```

Add any other values you'd pass to `helm`, as listed in [Configuration](../reference/configuration.md). On Kubernetes without OpenShift, set `platform: kubernetes` and install cert-manager first, as in [Getting Started](getting-started.md#kubernetes-without-openshift).

To upgrade, change `targetRevision`. Argo CD applies the chart's CRDs as part of the sync, so the separate CRD step in [Upgrading](../reference/upgrading.md) isn't needed. Check with `oc get crd aiboms.aibom.io -o yaml` after an upgrade if a release notes a CRD change.

## Workload namespaces

One Application per namespace, generated from a list. Adding a namespace to the list enables it:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: ApplicationSet
metadata:
  name: aibom-namespaces
  namespace: openshift-gitops
spec:
  generators:
    - list:
        elements:
          - ns: team-a
          - ns: team-b
  template:
    metadata:
      name: 'aibom-ns-{{ns}}'
    spec:
      project: default
      source:
        repoURL: quay.io/project-aibom
        chart: aibom-workload-namespace
        targetRevision: <version>
        helm:
          valuesObject:
            platform: openshift
            signing:
              createKeys: false
      destination:
        server: https://kubernetes.default.svc
        namespace: '{{ns}}'
      syncPolicy:
        automated:
          prune: true
        syncOptions:
          - CreateNamespace=true
        managedNamespaceMetadata:
          labels:
            aibom.io/enabled: "true"
```

`managedNamespaceMetadata` adds the `aibom.io/enabled` label, which plain Helm can't. It applies only when Argo CD creates the namespace (`CreateNamespace=true`); label an existing namespace yourself or manage the Namespace object in Git. Set `aibomSystemNamespace` if the webhook isn't in `project-aibom`, and `vllmMetrics.enabled: false` on a cluster without the Prometheus Operator.

Keep `targetRevision` the same as the webhook's. A namespace that syncs before the webhook exists binds Roles to a ServiceAccount that isn't there yet, which starts working once the webhook syncs.

## Moving an existing namespace to `createKeys: false`

A namespace whose Application already synced with `createKeys: true` has the two key Secrets in its managed resources. With pruning on, switching to `false` deletes them, and the webhook creates new keys at the next pod admission. Pods running at that moment lose their hardware and dataset data from their AIBOMs once. To avoid that, switch while nothing is running, or sync that change with pruning off and the Secrets stay in place, no longer managed by Argo CD.

## Removing

Delete the namespace Applications first, then the webhook's. The webhook chart's uninstall cleanup, which removes the webhook's finalizers from Jobs and pods, is a Helm `post-delete` hook, and Argo CD doesn't run it. Delete running instrumented workloads before removing the webhook, or afterwards run the cleanup by hand as a cluster-admin:

```bash
helm template aibom-webhook oci://quay.io/project-aibom/aibom-webhook --version <version> \
  -n project-aibom --set platform=openshift \
  --show-only templates/finalizer-cleanup-hook.yaml | oc apply -f -
```

That applies the hook's own Job, ServiceAccount and ClusterRole, which grants only `list` and `patch` on Jobs and Pods. Once the Job has succeeded, delete it and those RBAC objects with the same command and `oc delete -f -`. As with Helm, the CRDs and existing AIBOMs are left in place.
