# AIBOM Workload Namespace

Enables one namespace for the AIBOM webhook. Install it in every namespace whose workloads should get an AIBOM, after the `aibom-webhook` chart.

It creates the `aibom-scripts` ConfigMap that instrumented pods run, the HMAC keys that sign their hardware and dataset data, the RBAC that lets workload pods and the webhook write that data, the grants the postprocess Job needs to create the namespace's AIBOMs, and a PodMonitor for vLLM metrics.

Full documentation: https://gavinsan33.github.io/project-aibom/

## Install

A cluster-admin installs it, because it also creates a RoleBinding in the webhook's namespace. The namespace must already exist and carry the opt-in label, which Helm can't add:

```bash
oc label namespace <namespace> aibom.io/enabled=true
helm upgrade --install aibom-ns-<namespace> oci://quay.io/project-aibom/aibom-workload-namespace \
  -n <namespace> --version <version>
```

- Set `aibomSystemNamespace` if the webhook chart isn't in `project-aibom`.
- Set `vllmMetrics.enabled=false` on a cluster without the Prometheus Operator, or the install fails on the PodMonitor.

Run `helm test aibom-ns-<namespace> -n <namespace>` to check the scripts are in place. Upgrade the namespace chart whenever you upgrade the webhook chart: a stale `aibom-scripts` ConfigMap can stop instrumented pods from starting.
