# Taken from coldpress's discovery script in 'user_snapshot.yaml' and modified.

import hashlib
import hmac
import subprocess
import json
import time
import os
from datetime import datetime

try:
    import k8s_api
except ImportError:
    k8s_api = None

def run_cmd(command):
    try:
        out = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else "Not available"
    except Exception as e:
        return "Not available"

def cpu_benchmark():
    """Simple CPU compute benchmark - measures MFLOPS"""
    start = time.time()
    result = 0.0
    iterations = 10000000
    for i in range(iterations):
        result += i * 1.5 - 0.7
    elapsed = time.time() - start
    mflops = (iterations * 2) / (elapsed * 1000000)  # 2 ops per iteration
    return {
        "iterations": iterations,
        "time_seconds": f"{elapsed:.4f}",
        "mflops": f"{mflops:.2f}"
    }

def memory_bandwidth_benchmark():
    """Memory sequential read/write benchmark"""
    size_mb = 100
    size_bytes = size_mb * 1024 * 1024
    data = bytearray(size_bytes)

    # Write test
    start = time.time()
    for i in range(0, size_bytes, 8):
        data[i:i+8] = b'\x00\x01\x02\x03\x04\x05\x06\x07'
    write_time = time.time() - start
    write_bw = (size_mb / write_time) if write_time > 0 else 0

    # Read test
    start = time.time()
    checksum = 0
    for i in range(0, size_bytes, 8):
        checksum += data[i]
    read_time = time.time() - start
    read_bw = (size_mb / read_time) if read_time > 0 else 0

    return {
        "test_size_mb": size_mb,
        "write_bandwidth_mbps": f"{write_bw:.2f}",
        "read_bandwidth_mbps": f"{read_bw:.2f}",
        "write_time_sec": f"{write_time:.4f}",
        "read_time_sec": f"{read_time:.4f}"
    }

def disk_io_benchmark():
    """Disk I/O benchmark using /tmp"""
    test_file = "/tmp/io_bench_test.dat"
    size_mb = 50
    block_size = 1024 * 1024  # 1MB blocks

    try:
        # Write test
        start = time.time()
        with open(test_file, 'wb') as f:
            for _ in range(size_mb):
                f.write(os.urandom(block_size))
            f.flush()
            os.fsync(f.fileno())
        write_time = time.time() - start
        write_bw = (size_mb / write_time) if write_time > 0 else 0

        # Read test
        start = time.time()
        with open(test_file, 'rb') as f:
            while f.read(block_size):
                pass
        read_time = time.time() - start
        read_bw = (size_mb / read_time) if read_time > 0 else 0

        # Cleanup
        os.remove(test_file)

        return {
            "test_size_mb": size_mb,
            "write_bandwidth_mbps": f"{write_bw:.2f}",
            "read_bandwidth_mbps": f"{read_bw:.2f}",
            "write_time_sec": f"{write_time:.4f}",
            "read_time_sec": f"{read_time:.4f}"
        }
    except Exception as e:
        return {"error": str(e)}

def context_switch_benchmark():
    """Measure context switch overhead using subprocess spawning"""
    iterations = 100
    start = time.time()
    for _ in range(iterations):
        subprocess.run(['true'], capture_output=True)
    elapsed = time.time() - start
    avg_per_switch = (elapsed / iterations) * 1000  # milliseconds

    return {
        "iterations": iterations,
        "total_time_sec": f"{elapsed:.4f}",
        "avg_per_spawn_ms": f"{avg_per_switch:.4f}"
    }

print("Collecting static system information...")

snapshot = {
    "timestamp": datetime.now().isoformat(),
    "pod_metadata": {
        "name": os.environ.get("POD_NAME", "unknown"),
        "uid": os.environ.get("POD_UID", "unknown"),
        "namespace": os.environ.get("POD_NAMESPACE", "unknown"),
        "ip": os.environ.get("POD_IP", "unknown"),
        "node": os.environ.get("NODE_NAME", "unknown"),
        "start_time": datetime.now().isoformat()
    },
    "system": {
        "cpu_model": run_cmd("grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs"),
        "cpu_count": run_cmd("grep -c ^processor /proc/cpuinfo"),
        "cpu_cores_per_socket": run_cmd("lscpu | grep 'Core(s) per socket' | awk '{print $NF}' 2>/dev/null || echo 'N/A'"),
        "cpu_threads_per_core": run_cmd("lscpu | grep 'Thread(s) per core' | awk '{print $NF}' 2>/dev/null || echo 'N/A'"),
        "cpu_architecture": run_cmd("uname -m"),
        "cpu_current_freq_mhz": run_cmd("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null | awk '{print $1/1000}' || echo 'N/A'"),
        "cpu_max_freq_mhz": run_cmd("cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq 2>/dev/null | awk '{print $1/1000}' || echo 'N/A'"),
        "cpu_min_freq_mhz": run_cmd("cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq 2>/dev/null | awk '{print $1/1000}' || echo 'N/A'"),
        "cache_l1d": run_cmd("lscpu | grep 'L1d cache' | awk '{print $NF}' 2>/dev/null || echo 'N/A'"),
        "cache_l1i": run_cmd("lscpu | grep 'L1i cache' | awk '{print $NF}' 2>/dev/null || echo 'N/A'"),
        "cache_l2": run_cmd("lscpu | grep 'L2 cache' | awk '{print $NF}' 2>/dev/null || echo 'N/A'"),
        "cache_l3": run_cmd("lscpu | grep 'L3 cache' | awk '{print $NF}' 2>/dev/null || echo 'N/A'"),
        "memory_total_gb": run_cmd("grep MemTotal /proc/meminfo | awk '{printf \"%.2f\", $2/1024/1024}'"),
        "memory_available_gb": run_cmd("grep MemAvailable /proc/meminfo | awk '{printf \"%.2f\", $2/1024/1024}'"),
        "memory_free_gb": run_cmd("grep MemFree /proc/meminfo | awk '{printf \"%.2f\", $2/1024/1024}'"),
        "numa_node_count": run_cmd("ls -d /sys/devices/system/node/node* 2>/dev/null | wc -l"),
        "kernel_version": run_cmd("uname -r"),
        "uptime_seconds": run_cmd("cat /proc/uptime | awk '{print $1}'")
    },
    "gpu": {
        "gpu_count": run_cmd("nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l"),
        "gpu_models": run_cmd("nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null"),
        "gpu_memory_per_device_mb": run_cmd("nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null"),
        "gpu_driver_version": run_cmd("nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1"),
        "cuda_version": run_cmd("nvidia-smi | grep 'CUDA Version' | awk '{print $9}' 2>/dev/null || echo 'N/A'")
    },
    "network": {
        "interface_names": run_cmd("ls /sys/class/net 2>/dev/null | grep -v lo | tr '\\n' ', ' | sed 's/,$//'"),
        "rdma_devices": run_cmd("ls /sys/class/infiniband 2>/dev/null | tr '\\n' ', ' | sed 's/,$//' || echo 'None'"),
        "rdma_device_count": run_cmd("ls /sys/class/infiniband 2>/dev/null | wc -l"),
        "primary_mtu": run_cmd("cat /sys/class/net/eth0/mtu 2>/dev/null || echo 'N/A'"),
        "tcp_rmem": run_cmd("cat /proc/sys/net/ipv4/tcp_rmem 2>/dev/null || echo 'N/A'"),
        "tcp_wmem": run_cmd("cat /proc/sys/net/ipv4/tcp_wmem 2>/dev/null || echo 'N/A'"),
        "tcp_congestion_control": run_cmd("cat /proc/sys/net/ipv4/tcp_congestion_control 2>/dev/null || echo 'N/A'")
    },
    "storage": {
        "block_devices": run_cmd("lsblk -nd -o NAME,SIZE 2>/dev/null | grep -v loop"),
        "nvme_count": run_cmd("ls /dev/nvme* 2>/dev/null | grep -c nvme0n || echo '0'"),
        "tmpfs_size": run_cmd("df -h /tmp 2>/dev/null | tail -1 | awk '{print $2}'"),
        "tmpfs_avail": run_cmd("df -h /tmp 2>/dev/null | tail -1 | awk '{print $4}'"),
        "io_scheduler": run_cmd("cat /sys/block/sda/queue/scheduler 2>/dev/null | grep -o '\\[.*\\]' | tr -d '[]' || echo 'N/A'")
    },
    "performance_config": {
        "cpu_governor": run_cmd("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo 'N/A'"),
        "numa_balancing": run_cmd("cat /proc/sys/kernel/numa_balancing 2>/dev/null || echo 'N/A'"),
        "transparent_hugepages": run_cmd("cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null | grep -o '\\[.*\\]' | tr -d '[]' || echo 'N/A'"),
        "swappiness": run_cmd("cat /proc/sys/vm/swappiness 2>/dev/null || echo 'N/A'"),
        "dirty_ratio": run_cmd("cat /proc/sys/vm/dirty_ratio 2>/dev/null || echo 'N/A'"),
        "dirty_background_ratio": run_cmd("cat /proc/sys/vm/dirty_background_ratio 2>/dev/null || echo 'N/A'"),
        "max_map_count": run_cmd("cat /proc/sys/vm/max_map_count 2>/dev/null || echo 'N/A'"),
        "file_max": run_cmd("cat /proc/sys/fs/file-max 2>/dev/null || echo 'N/A'")
    },
    "process_limits": {
        "max_user_processes": run_cmd("ulimit -u 2>/dev/null || echo 'N/A'"),
        "max_open_files": run_cmd("ulimit -n 2>/dev/null || echo 'N/A'"),
        "max_stack_size_kb": run_cmd("ulimit -s 2>/dev/null || echo 'N/A'"),
        "max_memory_size_kb": run_cmd("ulimit -m 2>/dev/null || echo 'N/A'"),
        "cgroup_cpu_quota": run_cmd("cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || echo 'N/A'"),
        "cgroup_cpu_period": run_cmd("cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null || echo 'N/A'"),
        "cgroup_memory_limit_bytes": run_cmd("cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo 'N/A'")
    }
}

# Dynamic Benchmarks
print("Running CPU compute benchmark...")
snapshot["benchmarks"] = {
    "cpu_compute": cpu_benchmark()
}

print("Running memory bandwidth benchmark...")
snapshot["benchmarks"]["memory_bandwidth"] = memory_bandwidth_benchmark()

print("Running disk I/O benchmark...")
snapshot["benchmarks"]["disk_io"] = disk_io_benchmark()

print("Running context switch benchmark...")
snapshot["benchmarks"]["context_switch"] = context_switch_benchmark()

# Write JSON locally for debugging/parity with prior behavior
with open("/tmp/result/discovery.json", "w") as f:
    json.dump(snapshot, f, indent=2)

def resolve_inference_service_storage(namespace):
    """Resolve model identity from the owning KServe InferenceService's
    declared storage.path/storageUri, for predictors backed by an S3/MinIO
    data-connection bucket (spec.predictor.model.storage.key/path) rather
    than a custom serving container — the predictor's built-in vLLM container
    always runs with a fixed --model=/mnt/models mount, so there's no CLI arg
    identifying the actual model.

    This runs here, in the discovery init container at pod startup, rather
    than later from the always-on watcher at pod deletion: deleting the
    InferenceService is the normal way one of these pods gets deleted at all,
    and Kubernetes' garbage collector removes the InferenceService from etcd
    immediately, well before the delete cascades down to this pod — resolving
    it lazily at deletion time would lose that race almost every time.
    """
    isvc_name = os.environ.get("INFERENCESERVICE_NAME", "")
    if not k8s_api or not namespace or not isvc_name:
        return None
    try:
        isvc = k8s_api.get_custom_object(namespace, "serving.kserve.io", "v1beta1", "inferenceservices", isvc_name)
    except Exception as e:
        print(f"WARNING: could not get InferenceService {namespace}/{isvc_name}: {e}")
        return None
    if not isvc:
        return None

    model = ((isvc.get("spec") or {}).get("predictor") or {}).get("model") or {}
    storage = model.get("storage") or {}
    path = storage.get("path")
    storage_key = storage.get("key")
    storage_uri = model.get("storageUri")

    if not path and not storage_uri:
        return None

    result = {"inference_service": isvc_name}
    if path:
        result["storage_path"] = path
    if storage_key:
        result["storage_key"] = storage_key
    if storage_uri:
        result["storage_uri"] = storage_uri
    return result


_SIGNING_KEY_PATH = "/var/run/secrets/aibom/discovery-signing/hmac-key"


def sign_discovery_payload(payload):
    """HMAC-SHA256 the discovery payload's canonical bytes with the
    per-namespace signing key, so the watcher can later verify that a
    discovery-<pod>.json entry genuinely came from this (platform-controlled)
    discovery container rather than being fabricated or overwritten by the
    workload's own application container, which is never given this key.

    Returns None (unsigned) if the key isn't mounted -- e.g. a namespace
    whose aibom-workload-namespace chart install predates signing.yaml --
    so a missing key degrades to "unverifiable" rather than failing pod
    startup. The watcher treats an unsigned discovery payload as untrusted
    once a per-namespace key does exist, but passes it through unverified if
    the whole namespace has no key configured yet.
    """
    try:
        with open(_SIGNING_KEY_PATH, "rb") as f:
            key = f.read().strip()
    except OSError:
        return None
    if not key:
        return None
    return hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()


# Write directly into the AIBOM data ConfigMap the webhook told us about,
# rather than printing to stdout for the watcher to scrape from pod logs.
pod_name = os.environ.get("POD_NAME", "")
pod_namespace = os.environ.get("POD_NAMESPACE", "")
configmap_name = k8s_api.resolve_data_configmap_name() if k8s_api else ""

# Canonical (sorted-key, no incidental whitespace) serialization: this exact
# string is what gets signed and later re-hashed by the watcher, so it must
# be reproduced byte-for-byte, not just semantically equivalent JSON.
discovery_payload = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
data_updates = {f"discovery-{pod_name}.json": discovery_payload}
signature = sign_discovery_payload(discovery_payload)
if signature:
    data_updates[f"discovery-{pod_name}.sig"] = signature

storage_info = resolve_inference_service_storage(pod_namespace)
if storage_info:
    data_updates[f"storage-{pod_name}.json"] = json.dumps(storage_info)

if k8s_api and pod_name and pod_namespace and configmap_name:
    try:
        k8s_api.patch_configmap(pod_namespace, configmap_name, data_updates)
    except Exception as e:
        print(f"WARNING: could not write discovery data to ConfigMap {configmap_name}: {e}")
else:
    print("WARNING: k8s_api unavailable or POD_NAME/POD_NAMESPACE/AIBOM_DATA_CONFIGMAP not set, skipping ConfigMap write")

print("Snapshot generation complete!")
