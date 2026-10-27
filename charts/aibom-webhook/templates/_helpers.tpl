{{/*
Namespace the webhook and its supporting resources are installed into.
Always the release namespace (the -n flag passed to helm install/upgrade) —
there is no separate values-based override, so there's only one place to set it.
*/}}
{{- define "aibom-webhook.namespace" -}}
{{- .Release.Namespace -}}
{{- end -}}

{{/*
Fully-qualified DNS name of the webhook Service, as seen from inside the cluster.
*/}}
{{- define "aibom-webhook.serviceDNSName" -}}
aibom-webhook.{{ include "aibom-webhook.namespace" . }}.svc
{{- end -}}

{{- define "aibom-webhook.labels" -}}
app.kubernetes.io/name: aibom-webhook-service
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{/*
"true" when installing on OpenShift, else empty. platform=auto checks for the
image.openshift.io API, which `helm template` without --api-versions can't see.
*/}}
{{- define "aibom-webhook.openshift" -}}
{{- $p := toString .Values.platform -}}
{{- if eq $p "openshift" -}}true
{{- else if eq $p "auto" -}}{{ if .Capabilities.APIVersions.Has "image.openshift.io/v1" }}true{{ end }}
{{- else if ne $p "kubernetes" -}}{{ fail (printf "platform must be auto, openshift or kubernetes, not %q" $p) }}
{{- end -}}
{{- end -}}

{{/*
Discovery init container image. Unset, it's OpenShift AI's runtime-pytorch
ImageStream where the OpenShift AI API exists, else the public PyTorch image.
The OpenShift AI API (datasciencecluster.opendatahub.io) is invisible to an
offline `helm template` just like image.openshift.io is, and -- unlike the
OpenShift defaults -- `platform` can't recover it: `platform=openshift` means
OpenShift, not OpenShift AI, and the runtime-pytorch ImageStream only exists on
the latter. So an offline render (e.g. Argo CD) always falls through to the
public image here regardless of `platform`; set `image.discovery` explicitly to
override it. See getting-started's offline note.
*/}}
{{- define "aibom-webhook.discoveryImage" -}}
{{- if .Values.image.discovery.repository -}}
{{ .Values.image.discovery.repository }}:{{ required "image.discovery.tag is required when image.discovery.repository is set" .Values.image.discovery.tag }}
{{- else if .Capabilities.APIVersions.Has "datasciencecluster.opendatahub.io/v1" -}}
image-registry.openshift-image-registry.svc:5000/redhat-ods-applications/runtime-pytorch:pytorch
{{- else -}}
pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime@sha256:e5c32997194edfc6fb0235223d1cd0433056045a468e7dfeaac390a3fcf8ae6d
{{- end -}}
{{- end -}}

{{/*
Prometheus URL. Unset, it's OpenShift's Thanos Querier on OpenShift and empty
(telemetry off) elsewhere; an explicit "" also turns telemetry off.
*/}}
{{- define "aibom-webhook.prometheusURL" -}}
{{- if not (kindIs "invalid" .Values.prometheus.url) -}}
{{ .Values.prometheus.url }}
{{- else if include "aibom-webhook.openshift" . -}}
https://thanos-querier.openshift-monitoring.svc:9091
{{- end -}}
{{- end -}}

{{/*
Image tags default to the chart's appVersion, so a released chart runs the
images built for that release. A checkout's Chart.yaml has appVersion "latest".
*/}}
{{- define "aibom-webhook.webhookTag" -}}
{{- .Values.image.webhook.tag | default .Chart.AppVersion -}}
{{- end -}}

{{- define "aibom-webhook.postprocessTag" -}}
{{- .Values.image.postprocess.tag | default .Chart.AppVersion -}}
{{- end -}}

{{/*
Who issues the webhook's TLS certificate: "service-ca" (OpenShift's built-in
service CA) or "cert-manager". tls.provider=auto picks service-ca on OpenShift.
*/}}
{{- define "aibom-webhook.tlsProvider" -}}
{{- $p := toString .Values.tls.provider -}}
{{- if eq $p "auto" -}}{{ if include "aibom-webhook.openshift" . }}service-ca{{ else }}cert-manager{{ end }}
{{- else if and (eq $p "service-ca") (not (include "aibom-webhook.openshift" .)) -}}
{{- fail "tls.provider=service-ca needs OpenShift; set platform=openshift (required for an offline helm template) or use cert-manager" -}}
{{- else -}}{{ $p }}
{{- end -}}
{{- end -}}
