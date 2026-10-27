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
"true" when the injected images should be served from the cluster's internal
registry (an OpenShift image cache) instead of pulled from Quay on every node.
On by default on OpenShift; skipped when build.enabled already produces the same
in-cluster ImageStreams from source, and empty off OpenShift (no internal
registry). templates/imagestream-cache.yaml creates the matching ImageStreams.
*/}}
{{- define "aibom-webhook.cacheImages" -}}
{{- if and (include "aibom-webhook.openshift" .) .Values.cacheImages.enabled (not .Values.build.enabled) -}}true{{- end -}}
{{- end -}}

{{- define "aibom-webhook.internalRegistry" -}}
image-registry.openshift-image-registry.svc:5000/{{ include "aibom-webhook.namespace" . }}
{{- end -}}

{{- define "aibom-webhook.discoveryImage" -}}
{{- if include "aibom-webhook.cacheImages" . -}}
{{ include "aibom-webhook.internalRegistry" . }}/aibom-discovery:{{ include "aibom-webhook.discoveryTag" . }}
{{- else -}}
{{ .Values.image.discovery.repository }}:{{ include "aibom-webhook.discoveryTag" . }}
{{- end -}}
{{- end -}}

{{- define "aibom-webhook.postprocessImage" -}}
{{- if include "aibom-webhook.cacheImages" . -}}
{{ include "aibom-webhook.internalRegistry" . }}/aibom-postprocess:{{ include "aibom-webhook.postprocessTag" . }}
{{- else -}}
{{ .Values.image.postprocess.repository }}:{{ include "aibom-webhook.postprocessTag" . }}
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

{{- define "aibom-webhook.discoveryTag" -}}
{{- .Values.image.discovery.tag | default .Chart.AppVersion -}}
{{- end -}}

{{/*
Who issues the webhook's TLS certificate: "service-ca" (OpenShift's built-in
service CA) or "cert-manager". tls.provider=auto picks service-ca on OpenShift
when the service-ca operator's API (operator.openshift.io/v1 ServiceCA) is
served too, since the image API alone doesn't prove the operator runs. An
explicit platform=openshift is trusted instead, since an offline render sees
no APIs.
*/}}
{{- define "aibom-webhook.tlsProvider" -}}
{{- $p := toString .Values.tls.provider -}}
{{- if eq $p "auto" -}}
{{- $serviceCA := or (eq (toString .Values.platform) "openshift") (.Capabilities.APIVersions.Has "operator.openshift.io/v1/ServiceCA") -}}
{{- if and (include "aibom-webhook.openshift" .) $serviceCA }}service-ca{{ else }}cert-manager{{ end }}
{{- else if and (eq $p "service-ca") (not (include "aibom-webhook.openshift" .)) -}}
{{- fail "tls.provider=service-ca needs OpenShift; set platform=openshift (required for an offline helm template) or use cert-manager" -}}
{{- else -}}{{ $p }}
{{- end -}}
{{- end -}}
