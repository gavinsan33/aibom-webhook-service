# Signing Keys

The workload-namespace chart creates three Secrets in each enabled namespace. Each signs a different part of the pipeline, and nothing creates or rotates them except the chart.

| Secret | Key field | Signs | Held by |
|--------|-----------|-------|---------|
| `aibom-discovery-hmac-key` | `hmac-key` | Hardware and storage data from the `aibom-discovery` init container | The init container only. The watcher reads it to verify. |
| `aibom-dataset-hmac-key` | `hmac-key` | Dataset data published by the `aibom-dataset-sidecar` | The sidecar only. The watcher reads it to verify. |
| `aibom-compiled-signing-key` | `ed25519-key` | The compiled AIBOM's `spec.data` | The `aibom-postprocess` Job only |

The two HMAC keys are shared secrets between the webhook's own components. The Ed25519 key is asymmetric so that `oc aibom` and anyone holding an archived AIBOM can verify a signature without being able to forge one. Its public half is embedded in each AIBOM (`spec.signaturePublicKey`) and published in the `aibom-compiled-signing-public-key` ConfigMap, the cluster-side anchor that `oc aibom describe` cross-checks. The `oc aibom describe` page in the CLI docs lists each result it can report.

The signatures are a tamper check, not proof the content is true. [Detected Fields](../CAPABILITIES.md) covers what each field does and doesn't guarantee.

## How keys are created and kept

The chart generates each Secret on first install. On later `helm upgrade --install` runs it looks the Secret up in the cluster and reuses the existing value, so re-running the install, for example to refresh the `aibom-scripts` ConfigMap, does not change the keys.

!!! warning "Rendering the chart without cluster access"
    Reuse depends on Helm's `lookup`, which returns nothing when the chart is rendered offline. Tools that render with `helm template` instead of talking to the cluster, which is Argo CD's default, as well as `helm template` itself and `--dry-run` without a cluster connection, generate fresh random keys on every render. Applied by such a tool, every sync would replace all three Secrets.

    If you manage namespaces this way, either use a tool that runs a real `helm upgrade`, or exclude the three Secrets from the sync and let the first install create them. To check that keys survive a sync, compare a hash before and after:

    ```bash
    oc get secret aibom-discovery-hmac-key -n <namespace> -o jsonpath='{.data.hmac-key}' | sha256sum
    ```

## What a key change does

There is no rotation command. A key changes only if its Secret is deleted and the chart runs again, or if it was replaced by an offline render as described above. This follows from how the chart and pipeline are built, and has not been exercised on a live cluster.

| Key | Effect of a change |
|-----|--------------------|
| Discovery or dataset HMAC key | Pods already running were signed with the old key, so the watcher can't verify their data. It drops those entries and logs them, and the AIBOM is missing that pod's hardware or dataset data. Pods started after the change are unaffected. |
| Compiled AIBOM key | New AIBOMs are signed with the new key, and the `aibom-compiled-signing-public-key` ConfigMap is overwritten with its public half. Existing AIBOMs keep their own embedded key and still verify against it, but `oc aibom describe` reports them as `KEY MISMATCH`, because the live anchor no longer matches. |

Namespaces installed before signing existed have no Secret. The watcher treats that as "verification unavailable" and passes the data through unverified instead of dropping it. A Secret that exists but can't be read for another reason, such as a missing grant, is also passed through, with a logged warning.

## Backup

The HMAC keys carry no long-term value: losing one costs only the in-flight pods described above. Back up `aibom-compiled-signing-key` if you want later AIBOMs to keep verifying against the same published key after a loss of the namespace. It is a private key, so store it in your secrets manager, not in Git.

Reading the Secrets takes access to the namespace's Secrets. The watcher's access is limited by `resourceNames` to the two HMAC Secrets, in namespaces that installed this chart. The `aibom-postprocess` ServiceAccount has no API grant on the Ed25519 Secret. The kubelet mounts it into the Job.
