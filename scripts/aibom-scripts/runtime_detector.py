"""
AIBOM Runtime Detector - Runtime shim for automatic training metadata collection.

Mounted into the app container as sitecustomize.py (on PYTHONPATH), so Python's
site module imports it at interpreter startup; can also be imported explicitly.
Monkey-patches common ML dataset
entry points to capture dataset name, source, and configuration without requiring
manual specification in intent.yaml, and inspects the training process's own
argv/config files to capture training args and parallelization strategy.

Supported dataset frameworks:
  - PyTorch DataLoader / Dataset
  - HuggingFace datasets.load_dataset
  - torchvision.datasets
  - webdataset.WebDataset

Also hooks transformers.TrainingArguments, transformers.PreTrainedModel.from_pretrained,
and peft.LoraConfig -- catching model/training config for scripts that build these
objects directly in Python, with no corresponding CLI flags for postprocess.py's
command-line detectors to see.

Writes detected metadata to $AIBOM_DATASET_OUTPUT (default: /results/dataset_detected.json)
only -- this process never talks to the Kubernetes API. The aibom-dataset-sidecar
container (dataset_sidecar.py) reads and signs that file and performs the actual
ConfigMap write from a process this application container doesn't control (#47) --
previously this module wrote directly to the ConfigMap itself using whatever
credentials the webhook gave the pod, which meant a compromised or malicious
training script could forge dataset-<pod-name>.json with a syntactically valid
(if meaningless) write of its own.

Output is written shortly after each detection (debounced), at exit, and on
SIGTERM, under an flock and via write-then-rename, since many processes share
the file and any of them can be killed without running atexit (#107).

All hooks are fault-tolerant — detection failures never interrupt training.
"""

import atexit
import hashlib
import importlib.util
import json
import os
import signal
import sys
import threading
import time
import traceback
import weakref

try:
    import fcntl
except ImportError:  # not POSIX; flush falls back to unlocked (still atomic)
    fcntl = None

_OUTPUT_PATH = os.environ.get(
    "AIBOM_DATASET_OUTPUT", "/results/dataset_detected.json"
)
_DEBUG = os.environ.get("AIBOM_DEBUG", "0") == "1"


def _dbg(msg):
    if _DEBUG:
        print(f"[AIBOM-DEBUG] {msg}", file=sys.stderr, flush=True)


def _dbg_exc(context):
    if _DEBUG:
        print(f"[AIBOM-DEBUG] EXCEPTION in {context}:", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)

class _NotifyingDict(dict):
    """runtime_info that schedules a (debounced) flush whenever it changes, so
    every one of the hook sites that records into it gets durable writes
    without each having to remember to ask for one (#107)."""

    # Only a real change notifies. _flush re-captures argv-derived values into
    # this dict on every run, and notifying on an identical write would
    # reschedule a flush forever.

    def __setitem__(self, key, value):
        changed = key not in self or self[key] != value
        super().__setitem__(key, value)
        if changed:
            _notify_change()

    def update(self, *args, **kwargs):
        incoming = dict(*args, **kwargs)
        changed = any(k not in self or self[k] != v for k, v in incoming.items())
        super().update(incoming)
        if changed:
            _notify_change()

    def setdefault(self, key, default=None):
        changed = key not in self
        result = super().setdefault(key, default)
        if changed:
            _notify_change()
        return result


_detected_datasets = []
_runtime_info = _NotifyingDict()
_lock = threading.Lock()
_hooks_installed = {}
# Maps a live dataset object (e.g. one returned by datasets.load_dataset) to
# its already-recorded entry, so a later DataLoader wrapping the same object
# updates that entry in place instead of recording it as a second dataset.
_dataset_registry = weakref.WeakKeyDictionary()
# Maps a HF dataset's (builder_name, config_name) to its already-recorded
# entry. Identity-based lookup above misses the very common case where a
# script transforms the loaded dataset (.map/.filter/.select/.shuffle, or
# re-fetching a split from a DatasetDict) before handing it to a DataLoader --
# each of those returns a *new* Dataset object, but builder_name/config_name
# survive the transform, so this catches it as the same underlying dataset.
_hf_dataset_key_registry = {}


def _hf_dataset_key(dataset):
    info = getattr(dataset, "info", None)
    builder_name = getattr(info, "builder_name", None) if info else None
    if not builder_name:
        return None
    return (builder_name, getattr(info, "config_name", None))


def _record(entry):
    _dbg(f"Recording dataset: {entry.get('dataset_name', '?')} via {entry.get('source', '?')}")
    with _lock:
        _detected_datasets.append(entry)
    _notify_change()


def _path_fingerprint(path):
    """SHA256 fingerprint from file metadata (names, sizes, mtimes) under path."""
    try:
        path = os.path.abspath(path)
        if not os.path.exists(path):
            return None
        h = hashlib.sha256()
        if os.path.isfile(path):
            st = os.stat(path)
            h.update(f"{path}\0{st.st_size}\0{int(st.st_mtime)}".encode())
        else:
            for dirpath, dirnames, filenames in os.walk(path):
                dirnames.sort()
                for fname in sorted(filenames):
                    fp = os.path.join(dirpath, fname)
                    try:
                        st = os.stat(fp)
                        rel = os.path.relpath(fp, path)
                        h.update(f"{rel}\0{st.st_size}\0{int(st.st_mtime)}".encode())
                    except OSError:
                        continue
        return h.hexdigest()
    except Exception:
        _dbg_exc("_path_fingerprint")
        return None


def _capture_training_args():
    """Best-effort extraction of common training args from sys.argv."""
    _arg_map = {
        "--epochs": ("epochs", int),
        "--num-epochs": ("epochs", int),
        "--num_epochs": ("epochs", int),
        "--num-train-epochs": ("epochs", int),
        "--num_train_epochs": ("epochs", int),
        "--batch-size": ("batch_size", int),
        "--batch_size": ("batch_size", int),
        "--per-device-train-batch-size": ("batch_size", int),
        "--per_device_train_batch_size": ("batch_size", int),
        "--lr": ("learning_rate", float),
        "--learning-rate": ("learning_rate", float),
        "--learning_rate": ("learning_rate", float),
    }
    try:
        argv = sys.argv[:]
        for i, arg in enumerate(argv):
            key, _, val = arg.partition("=")
            if key in _arg_map:
                name, conv = _arg_map[key]
                if not val and i + 1 < len(argv):
                    val = argv[i + 1]
                if val:
                    _runtime_info[name] = conv(val)
                    _dbg(f"Captured from argv: {name}={_runtime_info[name]}")
    except Exception:
        _dbg_exc("_capture_training_args")


_ACCELERATE_DISTRIBUTED_TYPE_STRATEGIES = {
    "fsdp": "fsdp",
    "deepspeed": "deepspeed",
    "multi_gpu": "data_parallel",
    "multi_cpu": "data_parallel",
}


def _capture_accelerate_config():
    """Resolve the real parallelization strategy from an --accelerate_config
    YAML file's actual content -- running here, inside the training
    container, is the only place this file is readable."""
    try:
        argv = sys.argv[:]
        path = None
        for i, arg in enumerate(argv):
            key, _, val = arg.partition("=")
            if key in ("--accelerate_config", "--accelerate-config"):
                if not val and i + 1 < len(argv):
                    val = argv[i + 1]
                path = val or None
                break
        if not path or not os.path.isfile(path):
            return
        try:
            import yaml
        except ImportError:
            _dbg("Accelerate config detection: PyYAML not available, skipping")
            return
        with open(path) as f:
            config = yaml.safe_load(f) or {}
        distributed_type = str(config.get("distributed_type", "")).lower()
        strategy = _ACCELERATE_DISTRIBUTED_TYPE_STRATEGIES.get(distributed_type)
        if strategy:
            _runtime_info["parallelization_strategy"] = strategy
            _dbg(f"Captured parallelization_strategy={strategy} from {path}")
    except Exception:
        _dbg_exc("_capture_accelerate_config")


def _find_git_dir(start=None):
    """Walk upward from `start` (default: CWD) looking for a .git directory,
    mirroring how `git` itself locates the repo root from any subdirectory
    -- the training script may not run from the exact directory a wrapper
    script `cd`'d into after cloning."""
    path = os.path.abspath(start or os.getcwd())
    for _ in range(20):  # bound the walk -- avoid looping forever on a broken/circular mount
        if os.path.isdir(os.path.join(path, ".git")):
            return os.path.join(path, ".git")
        parent = os.path.dirname(path)
        if parent == path:
            return None
        path = parent
    return None


def _resolve_git_ref(git_dir, ref):
    """Resolve "ref: refs/heads/<branch>" to a commit SHA via the loose ref
    file, falling back to packed-refs (git packs infrequently-updated refs,
    e.g. right after a shallow clone). A detached HEAD's file already
    contains the SHA directly."""
    ref = ref.strip()
    if not ref.startswith("ref:"):
        return ref
    ref_path = ref[len("ref:"):].strip()
    loose = os.path.join(git_dir, ref_path)
    if os.path.isfile(loose):
        with open(loose) as f:
            return f.read().strip()
    packed = os.path.join(git_dir, "packed-refs")
    if os.path.isfile(packed):
        with open(packed) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2 and parts[1] == ref_path:
                    return parts[0]
    return None


# SECURITY: keep this logic identical to postprocess.py's redact_git_url -- the two can't share a
# module (see tests/test_redact_git_url_sync.py, which fails if they drift).
def _redact_git_url(url):
    """Strip credentials from a git remote URL before it's recorded (#102).

    `git clone https://user:TOKEN@host/repo` stores that URL verbatim in
    .git/config and in the container command, and it would otherwise land in
    the signed, immutable AIBOM. For http(s)/ftp the whole userinfo goes (a
    bare username is often the token itself, e.g. a GitHub PAT); for other
    schemes (ssh://git@host/...) only a password is dropped, since the
    username isn't secret. scp-style `git@host:org/repo` has no scheme and is
    returned as-is for the same reason. Query string and fragment are dropped
    entirely (`?token=`, `?private_token=`), since they never identify the
    repository.
    """
    if not isinstance(url, str) or "://" not in url:
        return url
    # Imported here, not at module level: this runs as sitecustomize in the
    # user's own process at startup, and this is only needed at exit.
    import re
    import urllib.parse
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        # Unparseable (e.g. malformed IPv6 host): drop anything that looks
        # like userinfo, query and fragment rather than risk keeping a secret.
        return re.sub(r"[?#].*$", "", re.sub(r"^([^:/]+://)[^/]*@", r"\1", url))
    netloc = parts.netloc
    if "@" in netloc:
        userinfo, _, hostport = netloc.rpartition("@")
        if parts.scheme.lower() in ("http", "https", "ftp", "ftps"):
            netloc = hostport
        else:
            user = userinfo.partition(":")[0]
            netloc = f"{user}@{hostport}" if user else hostport
    return f"{parts.scheme}://{netloc}{parts.path}"


def _git_remote_url(git_dir):
    """Best-effort read of the "origin" remote URL straight out of
    .git/config, without shelling out to the git binary (which isn't
    guaranteed to be installed in the training image)."""
    config_path = os.path.join(git_dir, "config")
    if not os.path.isfile(config_path):
        return None
    try:
        import configparser
        parser = configparser.ConfigParser(strict=False)
        parser.read(config_path)
        for section in parser.sections():
            if section.startswith("remote") and "origin" in section:
                return _redact_git_url(parser.get(section, "url", fallback=None))
    except Exception:
        _dbg_exc("_git_remote_url")
    return None


def _git_dirty(worktree_dir):
    """Best-effort `git status --porcelain` if the git binary happens to be
    installed in the training image (not guaranteed). Returns None -- not
    False -- when it can't be determined, so callers don't misreport a clean
    tree they never actually checked."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=worktree_dir, capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None
        return bool(result.stdout.strip())
    except Exception:
        return None


def _capture_git_provenance(check_dirty=True):
    """Best-effort git provenance from a .git directory in the working
    tree -- covers workloads that `git clone` their training code at
    runtime rather than baking it into the image (see CLAUDE.md's git
    provenance section). Ranked above postprocess.py's command-line `git
    clone` parsing: this reflects the actual final checked-out state
    (after any `git checkout`/`git pull` beyond the initial clone), not
    just what the command said it intended to do.
    """
    try:
        git_dir = _find_git_dir()
        if not git_dir:
            return
        head_path = os.path.join(git_dir, "HEAD")
        if not os.path.isfile(head_path):
            return
        with open(head_path) as f:
            head = f.read().strip()
        commit = _resolve_git_ref(git_dir, head)
        if not commit:
            return
        info = {"git_commit": commit}
        if head.startswith("ref:"):
            info["git_branch"] = head[len("ref:"):].strip().rsplit("/", 1)[-1]
        remote = _git_remote_url(git_dir)
        if remote:
            info["git_repository"] = remote
        # `git status` can take up to its 10s timeout; skipped when flushing
        # from a signal handler, where the kubelet's grace period is ticking.
        dirty = _git_dirty(os.path.dirname(git_dir)) if check_dirty else None
        if dirty is not None:
            info["git_dirty"] = dirty
        with _lock:
            _runtime_info.update(info)
        _dbg(f"Captured git provenance: {info}")
    except Exception:
        _dbg_exc("_capture_git_provenance")


# ── Durable, concurrent-safe output (#107) ──────────────────────────────────
#
# This file is written by every Python process in every instrumented
# container of the pod -- each torchrun rank, spawned DataLoader worker, the
# launcher itself, `pip` -- so a write has to (a) take turns with the others
# and (b) never leave a half-written file for the sidecar to read. It also
# can't wait for atexit alone: SIGKILL (OOM, the kubelet's grace-period kill)
# and SIGTERM under Python's default handler skip it, so changes are flushed
# shortly after they happen, and SIGTERM flushes before dying.

_FLUSH_DEBOUNCE_S = float(os.environ.get("AIBOM_FLUSH_DEBOUNCE_S", "2"))
_LOCK_TIMEOUT_S = 5.0

_durable_flush_enabled = False
_flush_timer = None
_flush_timer_lock = threading.Lock()
_git_captured = False
_fork_handler_registered = False
_sigterm_installed = False


def _notify_change():
    """Schedule one flush ~_FLUSH_DEBOUNCE_S from now unless one is already
    pending, so a burst of detections costs a single write. A no-op until
    hooks are actually installed."""
    global _flush_timer
    if not _durable_flush_enabled:
        return
    # Bounded wait, not `with`: this is reachable from the SIGTERM handler
    # (via _flush's argv captures), which may have interrupted this very
    # function mid-hold.
    if not _flush_timer_lock.acquire(timeout=0.2):
        return
    try:
        if _flush_timer is not None:
            return
        timer = threading.Timer(_FLUSH_DEBOUNCE_S, _debounced_flush)
        timer.daemon = True
        _flush_timer = timer
    finally:
        _flush_timer_lock.release()
    try:
        timer.start()
    except RuntimeError:  # interpreter shutting down; atexit will flush
        with _flush_timer_lock:
            _flush_timer = None


def _debounced_flush():
    global _flush_timer
    with _flush_timer_lock:
        _flush_timer = None
    try:
        _flush(final=False)
    except Exception:
        _dbg_exc("_debounced_flush")


def _cancel_pending_flush():
    global _flush_timer
    with _flush_timer_lock:
        timer, _flush_timer = _flush_timer, None
    if timer is not None:
        timer.cancel()


def _after_fork_in_child():
    """A forked child (DataLoader worker, multiprocessing) inherits our locks
    in whatever state another thread left them -- including held by the
    flush timer thread, which doesn't exist in the child -- and a Timer
    object that will never fire. Start clean."""
    global _lock, _flush_timer, _flush_timer_lock, _git_captured
    _lock = threading.Lock()
    _flush_timer_lock = threading.Lock()
    _flush_timer = None
    _git_captured = False


def _on_sigterm(signum, frame):
    """Flush, then die exactly as the default handler would have: restore
    SIG_DFL and re-deliver, so the process still terminates by SIGTERM (exit
    status 143) rather than exiting cleanly. Only installed when the handler
    was SIG_DFL at startup; an application handler (uvicorn, torchrun) makes
    its own orderly exit, which runs atexit."""
    try:
        _flush(final=True, in_signal=True)
    except Exception:
        pass
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


def _register_durable_flush():
    """Wire up everything that gets data out before the process disappears.
    Called from each install path; safe to call more than once."""
    global _durable_flush_enabled, _fork_handler_registered, _sigterm_installed
    _durable_flush_enabled = True
    atexit.register(_flush)
    if not _fork_handler_registered and hasattr(os, "register_at_fork"):
        os.register_at_fork(after_in_child=_after_fork_in_child)
        _fork_handler_registered = True
    if not _sigterm_installed:
        try:
            if signal.getsignal(signal.SIGTERM) == signal.SIG_DFL:
                signal.signal(signal.SIGTERM, _on_sigterm)
                _sigterm_installed = True
        except (ValueError, OSError):  # not the main thread / no signals
            _dbg("SIGTERM handler not installed (not main thread)")


class _OutputLock:
    """Exclusive flock on a sidecar-file next to the output, taken with a
    bounded wait so a stuck holder can never hang process exit."""

    def __init__(self, path, timeout_s):
        self._path = path + ".lock"
        self._timeout_s = timeout_s
        self._file = None

    def __enter__(self):
        if fcntl is None:
            return self
        self._file = open(self._path, "a")
        deadline = time.monotonic() + self._timeout_s
        while True:
            try:
                fcntl.flock(self._file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except OSError:
                if time.monotonic() >= deadline:
                    self._file.close()
                    self._file = None
                    raise TimeoutError(f"could not lock {self._path} within {self._timeout_s}s")
                time.sleep(0.02)

    def __exit__(self, *exc):
        if self._file is not None:
            self._file.close()  # releases the flock
            self._file = None


def _merge_output(existing, data, info):
    """Fold this process's detections into what other processes already
    wrote. Datasets are deduped by (name, source). runtime_info is
    last-writer-wins per key, except the git_* keys, which are one
    observation of one checkout: a process that captured any replaces the
    whole group, so a commit from one working tree is never labelled with the
    branch (or dirty flag) of another (#107)."""
    existing_ds = existing.get("datasets", [])
    seen = {(e.get("dataset_name", ""), e.get("source", "")) for e in existing_ds}
    for entry in data:
        key = (entry.get("dataset_name", ""), entry.get("source", ""))
        if key not in seen:
            existing_ds.append(entry)
            seen.add(key)

    output = {"datasets": existing_ds}
    merged_info = existing.get("runtime_info", {})
    if any(k.startswith("git_") for k in info):
        merged_info = {k: v for k, v in merged_info.items() if not k.startswith("git_")}
    merged_info.update(info)
    if merged_info:
        output["runtime_info"] = merged_info
    return output


def _flush(final=True, in_signal=False):
    """Write what this process has detected into the shared output file.

    final=False is a periodic flush while the process is still running; it
    still writes everything, but only captures git provenance once. in_signal
    avoids blocking on `_lock` (the signal may have interrupted the very
    thread holding it) and skips the slow `git status`.
    """
    global _git_captured
    _capture_training_args()
    _capture_accelerate_config()

    got_lock = _lock.acquire(timeout=0.5 if in_signal else -1)
    try:
        if not _detected_datasets and not _runtime_info:
            _dbg("Flush: nothing detected, skipping write")
            return
    finally:
        if got_lock:
            _lock.release()

    # Only a process that detected something of its own records git
    # provenance. Capturing it unconditionally made every Python process that
    # happened to run in a checkout (`pip`, a launcher shim) overwrite the
    # training process's commit/branch with its own (#107).
    if final or not _git_captured:
        _capture_git_provenance(check_dirty=not in_signal)
        _git_captured = True

    got_lock = _lock.acquire(timeout=0.5 if in_signal else -1)
    try:
        data = _detected_datasets.copy()
        info = dict(_runtime_info)
    finally:
        if got_lock:
            _lock.release()

    _dbg(f"Flushing {len(data)} dataset(s) to {_OUTPUT_PATH} (final={final})")
    try:
        os.makedirs(os.path.dirname(_OUTPUT_PATH) or ".", exist_ok=True)
        with _OutputLock(_OUTPUT_PATH, 1.0 if in_signal else _LOCK_TIMEOUT_S):
            existing = {}
            if os.path.exists(_OUTPUT_PATH):
                try:
                    with open(_OUTPUT_PATH) as f:
                        existing = json.load(f)
                    _dbg(f"Merging with existing file ({len(existing.get('datasets', []))} datasets)")
                except Exception:
                    pass
            output = _merge_output(existing, data, info)
            # Write-then-rename in the same directory: the sidecar (and the
            # next writer) only ever sees a complete file, never a truncated
            # one mid-write.
            tmp_path = f"{_OUTPUT_PATH}.{os.getpid()}.tmp"
            with open(tmp_path, "w") as f:
                json.dump(output, f, indent=2, default=str)
            os.replace(tmp_path, _OUTPUT_PATH)
        _dbg(f"Flush succeeded: {_OUTPUT_PATH} ({len(output['datasets'])} datasets)")
        # The ConfigMap write itself now happens in the aibom-dataset-sidecar
        # container (dataset_sidecar.py), which watches this file from
        # outside this process's control and signs what it reads -- see the
        # module docstring above and #47.
    except Exception:
        _dbg_exc("_flush")


# ── PyTorch DataLoader hook ──────────────────────────────────────────────────

def _install_dataloader_hook():
    try:
        import torch.utils.data as tud
        import torch
    except ImportError:
        _dbg("DataLoader hook: torch not available, skipping")
        return

    if "dataloader" in _hooks_installed:
        return
    # Look up what's being patched before marking the hook installed, so a
    # failed lookup leaves it retryable instead of silently "installed".
    _orig_init = tud.DataLoader.__init__
    _hooks_installed["dataloader"] = True

    with _lock:
        _runtime_info["framework"] = "PyTorch"
        _runtime_info["framework_version"] = torch.__version__
    _dbg(f"DataLoader hook: installed (torch {torch.__version__})")

    def _patched_init(self, dataset=None, *args, **kwargs):
        _orig_init(self, dataset, *args, **kwargs)
        try:
            batch_size = getattr(self, "batch_size", None)
            with _lock:
                existing = _dataset_registry.get(dataset) if dataset is not None else None
                if existing is None and dataset is not None:
                    key = _hf_dataset_key(dataset)
                    if key is not None:
                        existing = _hf_dataset_key_registry.get(key)
            if existing is not None:
                _dbg(f"DataLoader hook: dataset already recorded via {existing.get('source', '?')}, merging")
                if batch_size is not None:
                    existing["batch_size"] = batch_size
                seen_via = existing.setdefault("seen_via", [])
                if "torch.utils.data.DataLoader" not in seen_via:
                    seen_via.append("torch.utils.data.DataLoader")
                _notify_change()
                return

            entry = _inspect_torch_dataset(dataset)
            if batch_size is not None:
                entry["batch_size"] = batch_size
            _record(entry)
        except Exception:
            _dbg_exc("DataLoader._patched_init")

    tud.DataLoader.__init__ = _patched_init


def _inspect_torch_dataset(dataset):
    cls = type(dataset)
    entry = {
        "source": "torch.utils.data.DataLoader",
        "dataset_class": f"{cls.__module__}.{cls.__qualname__}",
    }

    for attr in ("root", "data_path", "data_dir", "filename", "path"):
        val = getattr(dataset, attr, None)
        if val is not None:
            entry["path"] = str(val)
            break

    if hasattr(dataset, "train"):
        entry["split"] = "train" if dataset.train else "test"

    if hasattr(dataset, "transform"):
        t = dataset.transform
        entry["transform"] = type(t).__name__ if t else None

    name = getattr(dataset, "name", None) or cls.__name__
    entry["dataset_name"] = name

    if hasattr(dataset, "url"):
        entry["url"] = str(dataset.url)
    if hasattr(dataset, "urls"):
        urls = dataset.urls
        if isinstance(urls, (list, tuple)):
            entry["urls"] = [str(u) for u in urls[:5]]
        elif isinstance(urls, dict):
            entry["urls"] = {k: str(v) for k, v in list(urls.items())[:5]}

    if entry.get("path"):
        fp = _path_fingerprint(entry["path"])
        if fp:
            entry["fingerprint"] = fp

    return entry


# ── HuggingFace datasets.load_dataset hook ───────────────────────────────────

def _install_hf_datasets_hook():
    try:
        import datasets
    except ImportError:
        _dbg("HF datasets hook: datasets not available, skipping")
        return

    if "hf_datasets" in _hooks_installed:
        return
    _orig_load = datasets.load_dataset
    _hooks_installed["hf_datasets"] = True
    _dbg("HF datasets hook: installed, patching datasets.load_dataset")

    def _patched_load(path, *args, **kwargs):
        _dbg(f"HF datasets hook: load_dataset({path!r}) called")
        result = _orig_load(path, *args, **kwargs)
        try:
            entry = {
                "source": "datasets.load_dataset",
                "dataset_name": str(path),
            }
            name = args[0] if args else kwargs.get("name")
            if name:
                entry["config_name"] = str(name)

            split = kwargs.get("split")
            if split:
                entry["split"] = str(split)

            revision = kwargs.get("revision")
            if revision:
                entry["revision"] = str(revision)

            data_dir = kwargs.get("data_dir")
            if data_dir:
                entry["data_dir"] = str(data_dir)

            data_files = kwargs.get("data_files")
            if data_files:
                if isinstance(data_files, str):
                    entry["data_files"] = [data_files]
                elif isinstance(data_files, (list, tuple)):
                    entry["data_files"] = [str(f) for f in data_files[:10]]
                elif isinstance(data_files, dict):
                    entry["data_files"] = {
                        k: ([str(x) for x in v] if isinstance(v, list) else str(v))
                        for k, v in data_files.items()
                    }

            ds = result
            if isinstance(result, dict) and result:
                first_split = next(iter(result.values()))
                ds = first_split
                entry["splits"] = list(result.keys())
                _dbg(f"HF datasets hook: result is DatasetDict with splits {entry['splits']}, using first for metadata")

            if hasattr(ds, "_fingerprint"):
                entry["fingerprint"] = str(ds._fingerprint)
            elif hasattr(ds, "info") and ds.info:
                if hasattr(ds.info, "download_checksums") and ds.info.download_checksums:
                    checksums = list(ds.info.download_checksums.values())
                    if checksums and isinstance(checksums[0], dict):
                        entry["fingerprint"] = checksums[0].get("checksum")

            if hasattr(ds, "info") and ds.info:
                info = ds.info
                if hasattr(info, "version") and info.version:
                    entry["version"] = str(info.version)
                if hasattr(info, "license") and info.license:
                    entry["license"] = str(info.license)
                if hasattr(info, "description") and info.description:
                    entry["description"] = info.description[:200]

            _record(entry)
            try:
                with _lock:
                    _dataset_registry[ds] = entry
            except TypeError:
                _dbg("HF datasets hook: dataset object doesn't support weak references, skipping registry")
            key = _hf_dataset_key(ds)
            if key is not None:
                with _lock:
                    _hf_dataset_key_registry[key] = entry
        except Exception:
            _dbg_exc("HF datasets._patched_load")
        return result

    datasets.load_dataset = _patched_load


# ── torchvision.datasets hook ────────────────────────────────────────────────

_TORCHVISION_DATASETS = [
    "MNIST", "FashionMNIST", "KMNIST", "EMNIST",
    "CIFAR10", "CIFAR100",
    "ImageNet", "ImageFolder", "DatasetFolder",
    "CelebA", "LSUN", "STL10", "SVHN",
    "VOCDetection", "VOCSegmentation",
    "CocoDetection", "CocoCaptions",
    "Flickr8k", "Flickr30k",
    "Places365",
]


def _install_torchvision_hook():
    try:
        import torchvision.datasets as tvd
    except ImportError:
        _dbg("torchvision hook: torchvision not available, skipping")
        return

    if "torchvision" in _hooks_installed:
        return
    _hooks_installed["torchvision"] = True
    _dbg("torchvision hook: installed")

    for name in _TORCHVISION_DATASETS:
        cls = getattr(tvd, name, None)
        if cls is None:
            continue

        _orig = cls.__init__

        def _make_patched(original, dataset_name):
            def _patched(self, *args, **kwargs):
                original(self, *args, **kwargs)
                try:
                    entry = {
                        "source": f"torchvision.datasets.{dataset_name}",
                        "dataset_name": dataset_name,
                    }
                    if args:
                        entry["root"] = str(args[0])
                    elif "root" in kwargs:
                        entry["root"] = str(kwargs["root"])

                    train = kwargs.get("train")
                    if train is not None:
                        entry["split"] = "train" if train else "test"
                    elif len(args) > 1 and isinstance(args[1], bool):
                        entry["split"] = "train" if args[1] else "test"

                    if "split" in kwargs:
                        entry["split"] = str(kwargs["split"])

                    if "download" in kwargs:
                        entry["download"] = kwargs["download"]

                    if entry.get("root"):
                        fp = _path_fingerprint(entry["root"])
                        if fp:
                            entry["fingerprint"] = fp

                    _record(entry)
                except Exception:
                    _dbg_exc(f"torchvision._patched({dataset_name})")

            return _patched

        cls.__init__ = _make_patched(_orig, name)


# ── webdataset hook ──────────────────────────────────────────────────────────

def _install_webdataset_hook():
    try:
        import webdataset as wds
    except ImportError:
        _dbg("webdataset hook: webdataset not available, skipping")
        return

    if "webdataset" in _hooks_installed:
        return
    _orig_init = wds.WebDataset.__init__
    _hooks_installed["webdataset"] = True
    _dbg("webdataset hook: installed")

    def _patched_init(self, urls, *args, **kwargs):
        _orig_init(self, urls, *args, **kwargs)
        try:
            entry = {
                "source": "webdataset.WebDataset",
                "dataset_name": "WebDataset",
            }
            if isinstance(urls, str):
                entry["urls"] = [urls]
            elif isinstance(urls, (list, tuple)):
                entry["urls"] = [str(u) for u in urls[:10]]
            _record(entry)
        except Exception:
            _dbg_exc("webdataset._patched_init")

    wds.WebDataset.__init__ = _patched_init


# ── transformers Trainer / model hook ────────────────────────────────────────
#
# CLI-argument parsing (postprocess.py's detect_trl_from_command) only sees
# training config passed as literal flags on a recognized CLI (trl sft/dpo).
# The far more common case -- a custom script that builds TrainingArguments
# and calls Model.from_pretrained() directly in Python -- exposes nothing on
# the command line at all. Hooking the real objects here, inside the training
# process, catches that case regardless of how the script assembled its
# config (CLI, YAML, hardcoded).

_QUANT_CLASS_NAME_HINTS = (
    ("BitsAndBytes", "bitsandbytes"),
    ("GPTQ", "gptq"),
    ("AWQ", "awq"),
    ("HQQ", "hqq"),
    ("AQLM", "aqlm"),
)


def _install_transformers_hook():
    try:
        import transformers
    except ImportError:
        _dbg("transformers hook: transformers not available, skipping")
        return

    if "transformers" in _hooks_installed:
        return
    _orig_ta_post_init = transformers.TrainingArguments.__post_init__
    _orig_from_pretrained = transformers.PreTrainedModel.from_pretrained.__func__
    _hooks_installed["transformers"] = True
    _dbg("transformers hook: installed, patching TrainingArguments and PreTrainedModel.from_pretrained")

    # TrainingArguments is a dataclass, and so are its subclasses (trl's
    # SFTConfig/DPOConfig/GRPOConfig, Seq2SeqTrainingArguments, ...). Each
    # subclass gets its own generated __init__ that never calls the parent's,
    # so patching TrainingArguments.__init__ only ever caught the base class.
    # Every generated __init__ does call __post_init__, and subclasses that
    # override it call super().__post_init__(), so hooking that catches them
    # all -- after TrainingArguments' own normalization has run.
    def _patched_ta_post_init(self, *args, **kwargs):
        _orig_ta_post_init(self, *args, **kwargs)
        try:
            info = {"training_framework": "transformers.Trainer"}
            if getattr(self, "learning_rate", None) is not None:
                info["learning_rate"] = self.learning_rate
            if getattr(self, "per_device_train_batch_size", None) is not None:
                info["batch_size"] = self.per_device_train_batch_size
            if getattr(self, "num_train_epochs", None) is not None:
                info["epochs"] = self.num_train_epochs
            if getattr(self, "optim", None) is not None:
                # TrainingArguments.__post_init__ normalizes a plain string like
                # "adamw_torch_fused" into an OptimizerNames enum member, whose
                # default str() is "OptimizerNames.ADAMW_TORCH_FUSED" rather than
                # the plain value.
                info["optimizer"] = getattr(self.optim, "value", None) or str(self.optim)
            if getattr(self, "seed", None) is not None:
                info["random_seed"] = self.seed
            if getattr(self, "bf16", False):
                info["dtype"] = "bfloat16"
            elif getattr(self, "fp16", False):
                info["dtype"] = "float16"
            with _lock:
                _runtime_info.update(info)
            _dbg(f"transformers hook: captured TrainingArguments {info}")
        except Exception:
            _dbg_exc("transformers._patched_ta_post_init")

    transformers.TrainingArguments.__post_init__ = _patched_ta_post_init

    def _patched_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        model = _orig_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs)
        try:
            info = {"model_name": str(pretrained_model_name_or_path)}

            device_map = kwargs.get("device_map")
            if device_map is not None:
                info["model_device_map"] = str(device_map)

            config = getattr(model, "config", None)
            architectures = getattr(config, "architectures", None) if config else None
            if architectures:
                info["model_architecture"] = architectures[0]

            dtype = getattr(model, "dtype", None)
            if dtype is not None:
                info["dtype"] = str(dtype).replace("torch.", "")

            quant_config = getattr(config, "quantization_config", None) if config else None
            if quant_config is not None:
                method = getattr(quant_config, "quant_method", None)
                method = str(method) if method else None
                if not method and (
                    getattr(quant_config, "load_in_4bit", False)
                    or getattr(quant_config, "load_in_8bit", False)
                ):
                    method = "bitsandbytes"
                if not method:
                    cls_name = type(quant_config).__name__
                    for needle, name in _QUANT_CLASS_NAME_HINTS:
                        if needle.lower() in cls_name.lower():
                            method = name
                            break
                if method:
                    info["quantization_method"] = method

                bits = getattr(quant_config, "bits", None)
                if bits is None and getattr(quant_config, "load_in_4bit", False):
                    bits = 4
                elif bits is None and getattr(quant_config, "load_in_8bit", False):
                    bits = 8
                if bits is not None:
                    info["quantization_bits"] = bits

            with _lock:
                _runtime_info.update(info)
            _dbg(f"transformers hook: captured model {info}")
        except Exception:
            _dbg_exc("transformers._patched_from_pretrained")
        return model

    transformers.PreTrainedModel.from_pretrained = classmethod(_patched_from_pretrained)


# ── peft LoraConfig hook ──────────────────────────────────────────────────────

def _install_peft_hook():
    try:
        import peft
    except ImportError:
        _dbg("peft hook: peft not available, skipping")
        return

    if "peft" in _hooks_installed:
        return
    _orig_init = peft.LoraConfig.__init__
    _hooks_installed["peft"] = True
    _dbg("peft hook: installed, patching LoraConfig.__init__")

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        try:
            info = {}
            if getattr(self, "r", None) is not None:
                info["lora_rank"] = self.r
            if getattr(self, "lora_alpha", None) is not None:
                info["lora_alpha"] = self.lora_alpha

            with _lock:
                already_quantized = bool(
                    _runtime_info.get("quantization_method") or _runtime_info.get("quantization_bits")
                )
                if getattr(self, "use_dora", False):
                    info["adaptation_method"] = "dora"
                elif getattr(self, "use_rslora", False):
                    info["adaptation_method"] = "rslora"
                elif already_quantized:
                    info["adaptation_method"] = "qlora"
                else:
                    info["adaptation_method"] = "lora"
                _runtime_info.update(info)
            _dbg(f"peft hook: captured LoraConfig {info}")
        except Exception:
            _dbg_exc("peft._patched_init")

    peft.LoraConfig.__init__ = _patched_init


# ── Public API ───────────────────────────────────────────────────────────────

def install_hooks():
    """Install all available dataset detection hooks (immediate).

    Use when frameworks are already imported (e.g., mid-script activation).
    """
    _dbg("install_hooks: installing all hooks (immediate mode)")
    _install_dataloader_hook()
    _install_hf_datasets_hook()
    _install_torchvision_hook()
    _install_webdataset_hook()
    _install_transformers_hook()
    _install_peft_hook()
    _register_durable_flush()
    _dbg(f"install_hooks: done, output will go to {_OUTPUT_PATH}")


# Top-level module -> the hook to install once it has finished importing.
_LAZY_TRIGGERS = {
    "torch": _install_dataloader_hook,
    "datasets": _install_hf_datasets_hook,
    "torchvision": _install_torchvision_hook,
    "webdataset": _install_webdataset_hook,
    "transformers": _install_transformers_hook,
    "peft": _install_peft_hook,
}


class _PostImportHookFinder:
    """sys.meta_path finder that runs a trigger's hook the moment its module
    finishes executing -- before the import statement that loaded it binds
    anything.

    This replaced a builtins.__import__ wrapper that only installed hooks
    once the import stack unwound back to depth 0. That missed any module
    that first imported a framework from inside another import -- a user
    helper module, or trl's own CLI modules, doing `from datasets import
    load_dataset` at their top level bound the unpatched function before the
    hook ran. It also didn't see importlib.import_module() or lazy loaders,
    which bypass builtins.__import__, and its depth counter was shared across
    threads.

    The finder never loads anything itself: it asks the finders after it for
    the real spec and wraps that spec's loader's exec_module on the loader
    instance (keeping the loader's type, which importlib.resources and
    friends may check), then removes the wrapper once it has fired."""

    def __init__(self, triggers):
        self._pending = dict(triggers)
        self._claim_lock = threading.Lock()
        self._resolving = threading.local()

    def _claim(self, name):
        # Hooks run outside this lock: one may import another trigger (the
        # transformers hook pulls in torch), and holding a lock across an
        # import risks deadlocking against another thread's module lock.
        with self._claim_lock:
            hook = self._pending.pop(name, None)
            if not self._pending:
                try:
                    sys.meta_path.remove(self)
                except ValueError:
                    pass
            return hook

    def fire(self, name):
        hook = self._claim(name)
        if hook is None:
            return
        _dbg(f"Lazy hook: '{name}' finished importing, installing hook")
        try:
            hook()
        except Exception:
            _dbg_exc(f"install_hooks_lazy({name})")

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self._pending:
            return None
        resolving = getattr(self._resolving, "names", None)
        if resolving is None:
            resolving = self._resolving.names = set()
        if fullname in resolving:
            return None
        resolving.add(fullname)
        try:
            spec = None
            for finder in list(sys.meta_path):
                if finder is self:
                    continue
                find = getattr(finder, "find_spec", None)
                if find is None:
                    continue
                spec = find(fullname, path, target)
                if spec is not None:
                    break
        finally:
            resolving.discard(fullname)
        if spec is None or spec.loader is None:
            return spec
        self._wrap_loader(spec.loader, fullname)
        return spec

    def _wrap_loader(self, loader, fullname):
        orig_exec = getattr(loader, "exec_module", None)
        if orig_exec is None or isinstance(loader, type):
            _dbg(f"Lazy hook: can't wrap loader for '{fullname}' ({loader!r})")
            return
        finder = self

        def exec_module(module):
            try:
                orig_exec(module)
            finally:
                # A loader instance can be shared across modules (e.g. one
                # zipimporter per archive), so only fire for our module, and
                # restore the original method once it has run.
                if getattr(module, "__name__", None) == fullname:
                    try:
                        del loader.exec_module
                    except AttributeError:
                        pass
            if getattr(module, "__name__", None) == fullname:
                finder.fire(fullname)

        try:
            loader.exec_module = exec_module
        except (AttributeError, TypeError):
            _dbg(f"Lazy hook: can't wrap loader for '{fullname}' ({loader!r})")


def install_hooks_lazy():
    """Install hooks lazily, as each framework finishes importing.

    Use when activated early (e.g., as sitecustomize.py at interpreter
    startup) before frameworks are imported. Frameworks that are already
    imported get their hook immediately.
    """
    finder = _PostImportHookFinder(_LAZY_TRIGGERS)
    already = [mod for mod in _LAZY_TRIGGERS if mod in sys.modules]
    if len(already) < len(_LAZY_TRIGGERS):
        sys.meta_path.insert(0, finder)
    for mod in already:
        _dbg(f"Lazy hook: '{mod}' already imported, installing hook now")
        finder.fire(mod)

    _register_durable_flush()
    _dbg(f"install_hooks_lazy: done, output will go to {_OUTPUT_PATH}")


def get_detected_datasets():
    """Return a copy of detected datasets (for testing)."""
    with _lock:
        return list(_detected_datasets)


def flush():
    """Force-write detected datasets to disk."""
    _flush()


def reset():
    """Clear detected datasets and stop any scheduled flush (for testing)."""
    global _durable_flush_enabled, _git_captured
    _cancel_pending_flush()
    _durable_flush_enabled = False
    _git_captured = False
    with _lock:
        _detected_datasets.clear()
        _hf_dataset_key_registry.clear()


# Auto-install when loaded via PYTHONSTARTUP or env var activation.
# Uses lazy hooks so frameworks don't need to be imported yet.
if os.environ.get("AIBOM_DATASET_DETECT", "0") == "1":
    _dbg(f"Auto-activating dataset detection (pid={os.getpid()})")
    _dbg(f"  PYTHONPATH={os.environ.get('PYTHONPATH', '<unset>')}")
    _dbg(f"  AIBOM_DATASET_OUTPUT={_OUTPUT_PATH}")
    _dbg(f"  Loaded from: {__file__}")
    install_hooks_lazy()


def _run_shadowed_sitecustomize():
    """When mounted as sitecustomize.py, this file sits on PYTHONPATH ahead of
    site-packages, so Python imports it *instead of* any sitecustomize the
    image ships (Debian/Ubuntu's apport hook, a conda env's, a platform
    team's own). Find the next one on sys.path and run it, so the image
    behaves as it would without instrumentation. Errors are reported the way
    site.py reports its own sitecustomize errors, and never propagate."""
    here = os.path.dirname(os.path.abspath(__file__))
    for entry in sys.path:
        base = os.path.abspath(entry or os.getcwd())
        if base == here:
            continue
        for candidate in (
            os.path.join(base, "sitecustomize.py"),
            os.path.join(base, "sitecustomize", "__init__.py"),
        ):
            if not os.path.isfile(candidate):
                continue
            try:
                spec = importlib.util.spec_from_file_location("_aibom_shadowed_sitecustomize", candidate)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                _dbg(f"Ran shadowed sitecustomize: {candidate}")
            except Exception as exc:
                print(
                    f"Error in sitecustomize; set PYTHONVERBOSE for traceback:\n"
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
            return


if __name__ == "sitecustomize":
    _run_shadowed_sitecustomize()
