# AIBOM Workload Namespace

Enables one namespace for the AIBOM webhook. Install it in every namespace whose workloads should get an AIBOM, after the `aibom-webhook` chart.

It creates the `aibom-scripts` ConfigMap that instrumented pods run, the HMAC keys that sign their hardware and dataset data, the RBAC that lets workload pods and the webhook write that data, the grants the postprocess Job needs to create the namespace's AIBOMs, and a PodMonitor for vLLM metrics.

Full documentation: https://gavinsan33.github.io/project-aibom/

**Prerequisite: label the namespace first.** Run `oc label namespace <namespace> aibom.io/enabled=true` before installing, including from the OpenShift console, which can't add the label itself. Without it the webhook would ignore the namespace's pods, so a first install refuses and prints the command. It also refuses if the webhook chart isn't installed yet, or if `aibomSystemNamespace` doesn't name the webhook's namespace (`project-aibom` by default); the error says what to set it to.

## Install

A cluster-admin installs it, because it also creates a RoleBinding in the webhook's namespace. The namespace must already exist and carry the opt-in label, which Helm can't add:

```bash
oc label namespace <namespace> aibom.io/enabled=true
helm upgrade --install aibom-ns-<namespace> oci://quay.io/project-aibom/aibom-workload-namespace \
  -n <namespace> --version <version>
```

- Set `aibomSystemNamespace` if the webhook chart isn't in `project-aibom`.
- Set `vllmMetrics.enabled=false` on a cluster without the Prometheus Operator, or the install fails on the PodMonitor.
- When rendering offline (Argo CD, `helm template`), set `platform` and `signing.createKeys=false`. The chart can't see the existing HMAC keys then and would replace them on every sync. With it off, the webhook creates any missing key when it admits the namespace's first instrumented pod.

Run `helm test aibom-ns-<namespace> -n <namespace>` to check the scripts are in place. Upgrade the namespace chart whenever you upgrade the webhook chart: a stale `aibom-scripts` ConfigMap can stop instrumented pods from starting.
