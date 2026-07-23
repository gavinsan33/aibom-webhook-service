"""
AIBOM Dataset Detector - Runtime shim for automatic dataset metadata collection.

Activated via PYTHONSTARTUP or explicit import. Monkey-patches common ML dataset
entry points to capture dataset name, source, and configuration without requiring
manual specification in intent.yaml.

Supported frameworks:
  - PyTorch DataLoader / Dataset
  - HuggingFace datasets.load_dataset
  - torchvision.datasets
  - webdataset.WebDataset

Writes detected metadata to $AIBOM_DATASET_OUTPUT (default: /results/dataset_detected.json).

All hooks are fault-tolerant — detection failures never interrupt training.
"""

import atexit
import hashlib
import json
import os
import sys
import threading
import traceback

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

_detected_datasets = []
_runtime_info = {}
_lock = threading.Lock()
_hooks_installed = {}


def _record(entry):
    _dbg(f"Recording dataset: {entry.get('dataset_name', '?')} via {entry.get('source', '?')}")
    with _lock:
        _detected_datasets.append(entry)


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


def _flush():
    _capture_training_args()
    with _lock:
        if not _detected_datasets and not _runtime_info:
            _dbg("Flush: nothing detected, skipping write")
            return
        data = _detected_datasets.copy()
        info = _runtime_info.copy()

    _dbg(f"Flushing {len(data)} dataset(s) to {_OUTPUT_PATH}")
    try:
        os.makedirs(os.path.dirname(_OUTPUT_PATH) or ".", exist_ok=True)

        existing = {}
        if os.path.exists(_OUTPUT_PATH):
            try:
                with open(_OUTPUT_PATH) as f:
                    existing = json.load(f)
                _dbg(f"Merging with existing file ({len(existing.get('datasets', []))} datasets)")
            except Exception:
                pass

        existing_ds = existing.get("datasets", [])
        seen = {(e.get("dataset_name", ""), e.get("source", "")) for e in existing_ds}
        for entry in data:
            key = (entry.get("dataset_name", ""), entry.get("source", ""))
            if key not in seen:
                existing_ds.append(entry)
                seen.add(key)

        output = {"datasets": existing_ds}

        merged_info = existing.get("runtime_info", {})
        merged_info.update(info)
        if merged_info:
            output["runtime_info"] = merged_info
            _dbg(f"Flushing runtime_info: {merged_info}")

        with open(_OUTPUT_PATH, "w") as f:
            json.dump(output, f, indent=2, default=str)
        _dbg(f"Flush succeeded: {_OUTPUT_PATH} ({len(existing_ds)} datasets)")

        # Print to stdout for log extraction by the watcher
        print("===AIBOM_DATASET_START===", flush=True)
        print(json.dumps(output, default=str), flush=True)
        print("===AIBOM_DATASET_END===", flush=True)
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
    _hooks_installed["dataloader"] = True

    with _lock:
        _runtime_info["framework"] = "PyTorch"
        _runtime_info["framework_version"] = torch.__version__
    _dbg(f"DataLoader hook: installed (torch {torch.__version__})")

    _orig_init = tud.DataLoader.__init__

    def _patched_init(self, dataset=None, *args, **kwargs):
        _orig_init(self, dataset, *args, **kwargs)
        try:
            entry = _inspect_torch_dataset(dataset)
            if hasattr(self, "batch_size") and self.batch_size is not None:
                entry["batch_size"] = self.batch_size
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
    _hooks_installed["hf_datasets"] = True
    _dbg("HF datasets hook: installed, patching datasets.load_dataset")

    _orig_load = datasets.load_dataset

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
    _hooks_installed["webdataset"] = True
    _dbg("webdataset hook: installed")

    _orig_init = wds.WebDataset.__init__

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
    atexit.register(_flush)
    _dbg(f"install_hooks: done, output will go to {_OUTPUT_PATH}")


def install_hooks_lazy():
    """Install hooks lazily — detects framework imports as they happen.

    Use when activated early (e.g., via PYTHONSTARTUP) before frameworks
    are imported. Wraps builtins.__import__ to intercept torch, datasets,
    torchvision, and webdataset imports, installing the appropriate hooks
    when each framework is first loaded.
    """
    import builtins

    _orig_import = builtins.__import__

    _triggers = {
        "torch": _install_dataloader_hook,
        "datasets": _install_hf_datasets_hook,
        "torchvision": _install_torchvision_hook,
        "webdataset": _install_webdataset_hook,
    }
    _pending = set(_triggers.keys())
    _import_depth = [0]

    def _hooked_import(name, *args, **kwargs):
        _import_depth[0] += 1
        try:
            result = _orig_import(name, *args, **kwargs)
        finally:
            _import_depth[0] -= 1

        if _import_depth[0] == 0:
            for mod in list(_pending):
                if mod in sys.modules:
                    _pending.discard(mod)
                    _dbg(f"Lazy hook: '{mod}' fully loaded (triggered by import of '{name}'), installing hook")
                    try:
                        _triggers[mod]()
                    except Exception:
                        _dbg_exc(f"install_hooks_lazy({mod})")
            if not _pending:
                builtins.__import__ = _orig_import
                _dbg("Lazy hook: all hooks installed, restoring original __import__")
        return result

    builtins.__import__ = _hooked_import

    for mod, hook in list(_triggers.items()):
        if mod in sys.modules:
            _pending.discard(mod)
            _dbg(f"Lazy hook: '{mod}' already imported, installing hook now")
            try:
                hook()
            except Exception:
                _dbg_exc(f"install_hooks_lazy({mod})")

    atexit.register(_flush)
    _dbg(f"install_hooks_lazy: done, output will go to {_OUTPUT_PATH}")


def get_detected_datasets():
    """Return a copy of detected datasets (for testing)."""
    with _lock:
        return list(_detected_datasets)


def flush():
    """Force-write detected datasets to disk."""
    _flush()


def reset():
    """Clear detected datasets (for testing)."""
    with _lock:
        _detected_datasets.clear()


# Auto-install when loaded via PYTHONSTARTUP or env var activation.
# Uses lazy hooks so frameworks don't need to be imported yet.
if os.environ.get("AIBOM_DATASET_DETECT", "0") == "1":
    _dbg(f"Auto-activating dataset detection (pid={os.getpid()})")
    _dbg(f"  PYTHONPATH={os.environ.get('PYTHONPATH', '<unset>')}")
    _dbg(f"  AIBOM_DATASET_OUTPUT={_OUTPUT_PATH}")
    _dbg(f"  Loaded from: {__file__}")
    install_hooks_lazy()
