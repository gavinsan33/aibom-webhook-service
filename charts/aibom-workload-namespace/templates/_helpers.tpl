{{/*
"true" when installing on OpenShift, else empty. Same rule as the
aibom-webhook chart's helper of the same name.
*/}}
{{- define "aibom-workload-namespace.openshift" -}}
{{- $p := toString .Values.platform -}}
{{- if eq $p "openshift" -}}true
{{- else if eq $p "auto" -}}{{ if .Capabilities.APIVersions.Has "image.openshift.io/v1" }}true{{ end }}
{{- else if ne $p "kubernetes" -}}{{ fail (printf "platform must be auto, openshift or kubernetes, not %q" $p) }}
{{- end -}}
{{- end -}}

{{/*
ClusterRole granting Prometheus read access: monitoringAccess.clusterRole, else
cluster-monitoring-view on OpenShift. Empty means no binding is created.
*/}}
{{- define "aibom-workload-namespace.monitoringClusterRole" -}}
{{- if .Values.monitoringAccess.clusterRole -}}
{{ .Values.monitoringAccess.clusterRole }}
{{- else if include "aibom-workload-namespace.openshift" . -}}
cluster-monitoring-view
{{- end -}}
{{- end -}}
