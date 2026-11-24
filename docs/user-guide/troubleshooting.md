# Troubleshooting

## Pods have no `aibom-discovery` init container

The webhook fails open: if it can't be reached, pods are created normally and no error is raised. The API server, not a pod, calls the webhook, so a NetworkPolicy in the install namespace that admits only same-namespace ingress makes every admission call time out silently. Workloads then run uninstrumented and no AIBOM is produced, with nothing logged.

The chart ships a NetworkPolicy named `aibom-webhook-admission` that allows ingress to the webhook pod's port 8443. It is on by default; set `networkPolicy.enabled=false` to opt out.

To check, run a server-side dry run of a bare GPU pod:

```bash
oc create --dry-run=server -o yaml -f <pod.yaml>
```

The result should carry the `aibom.io/instrumented` label, and a `mutating pod ...` line should appear in the webhook's log.

## Pods stay in `Init:ImagePullBackOff`

The `aibom-discovery` init container can't pull its image. Unless `image.discovery` is set, it is `quay.io/project-aibom/aibom-discovery` at the chart's version, which needs egress to Quay. Check the event:

```bash
oc describe pod <pod-name> -n <namespace>
```

Set `image.discovery.repository` and `image.discovery.tag` to an image the cluster can pull, as shown in [Getting Started](getting-started.md#cluster-requirements), and upgrade the webhook chart. Pods already created keep the old image, so recreate them.

## Install fails with "the AIBOM CRDs are not installed"

The webhook chart from the OpenShift Helm catalog doesn't include the `AIBOM` and `AIBOMTelemetry` CRDs, and refuses to install without them. Nothing has been created at that point. Apply the CRDs for the same version as a cluster-admin, then install again:

```bash
oc apply -f https://github.com/gavinsan33/aibom-webhook-service/releases/download/v<version>/aibom-crds.yaml
```

The error message prints this command with the version filled in. The chart from `oci://quay.io/project-aibom/aibom-webhook` includes the CRDs and doesn't hit this.

## Namespace chart install fails

On a first install, the namespace chart checks the webhook's configuration on the cluster and refuses to install into a namespace it would ignore. Nothing has been created when it fails. Each error says what to do:

- **"namespace … isn't labeled aibom.io/enabled=true"**: run the `oc label namespace` command it prints, then install again. The label is whatever the webhook chart's `webhook.namespaceSelectorLabel` is set to.
- **"the AIBOM webhook is installed in namespace …, but aibomSystemNamespace is …"**: set `aibomSystemNamespace` to the namespace it names. This is common after installing the webhook from the OpenShift console, which uses whichever project was selected.
- **"the AIBOM webhook isn't installed"**: install the webhook chart first.

The checks run only on a first install, and not where Helm has no cluster to look at, such as `helm template` and Argo CD.

## No AIBOMs after installing from the OpenShift console

Releases from 1.0.5 on refuse to install the namespace chart into an unlabeled namespace, and releases after 1.0.5 also refuse a wrong `aibomSystemNamespace` (see above), so this mostly applies to earlier releases and to Argo CD. Check the two steps the console can't do for you:

- **The namespace label.** `oc get namespace <namespace> --show-labels` must show `aibom.io/enabled=true`. Without it the webhook ignores the namespace's pods: they run with no `aibom-discovery` init container, and nothing is logged. Add it with `oc label namespace <namespace> aibom.io/enabled=true`, then rerun the workload.
- **The webhook's namespace.** The console installs into whichever project was selected, often `default`. Find it with `helm list -A`. If it isn't `project-aibom`, the namespace chart's `aibomSystemNamespace` must name it. Upgrade the namespace chart's release with that value set.

## Pods fail to start in an enabled namespace

A stale `aibom-scripts` ConfigMap can fail pod startup for every instrumented workload in the namespace. The dataset detector hook mounts `runtime_detector.py` from that ConfigMap, and the `aibom-discovery` and `aibom-dataset-sidecar` init containers each run a script from it.

Re-run the workload-namespace install (see [Getting Started](getting-started.md)) to refresh it.

## An AIBOM has no stored time series, or `AIBOMTelemetry` objects are left over

An AIBOM is still created without `spec.data.telemetry_series_ref` when Prometheus returned no data, the series would exceed the 900 KB cap, or creating the `AIBOMTelemetry` object was denied, for example by a ResourceQuota or a namespace whose chart predates the resource. The postprocess Job logs a warning in each case.

The series object is created before the AIBOM. If the AIBOM create then fails, the object is left behind: the postprocess Job can't delete these objects, by design. The Job logs the delete command, and its retry creates a fresh object. Leftovers have no owner, so the AIBOM column is empty:

```bash
oc get aibomtel -A
oc delete aibomtel <name> -n <namespace>
```

## Telemetry is empty

Outside OpenShift, `prometheus.url` has no default, so telemetry is off until you set it. Check what the webhook was given:

```bash
oc get deployment aibom-webhook -n project-aibom -o jsonpath='{.spec.template.spec.containers[0].args}'
```

On OpenShift, reading platform metrics from Thanos Querier requires a `cluster-monitoring-view` ClusterRoleBinding for the `aibom-postprocess` ServiceAccount in the webhook's namespace. The webhook chart creates it automatically, as long as the account running `helm install` is allowed to create ClusterRoleBindings and to grant that role, which in practice means a cluster-admin.

With an account that can't, the install fails on that binding. Skip it with `--set monitoringAccess.enabled=false`. Telemetry then comes back empty until a cluster-admin creates the binding once by hand. The chart's template can't be applied directly, since it's Helm-templated, so use:

```bash
oc create clusterrolebinding aibom-postprocess-monitoring-view \
  --clusterrole=cluster-monitoring-view \
  --serviceaccount=project-aibom:aibom-postprocess
```

On a cluster that isn't OpenShift, use a ClusterRole that grants the access your Prometheus requires instead.

## Inference telemetry is empty on OpenShift

The namespace chart's PodMonitor, which scrapes a serving engine's `/metrics`, is only picked up by OpenShift's user-workload monitoring, and that is off by default. Without it the PodMonitor is created and the install succeeds, but nothing scrapes it, so inference performance telemetry comes back empty with no error. GPU and other platform metrics are unaffected. Check whether it's on:

```bash
oc get pods -n openshift-user-workload-monitoring
```

No `prometheus-user-workload` pods means it's off. A cluster-admin enables it in `cluster-monitoring-config`. If that ConfigMap already exists, add `enableUserWorkload: true` to its `config.yaml` instead of replacing it:

```bash
oc apply -f - <<'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: cluster-monitoring-config
  namespace: openshift-monitoring
data:
  config.yaml: |
    enableUserWorkload: true
EOF
```
