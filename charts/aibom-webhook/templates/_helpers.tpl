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
