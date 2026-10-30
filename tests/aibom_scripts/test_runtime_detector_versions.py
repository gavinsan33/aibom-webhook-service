"""Library and toolkit versions recorded by runtime_detector.py."""

import importlib.metadata
import json
import os
import subprocess
import sys
import types

import pytest

import runtime_detector as rd

_DETECTOR_SOURCE = os.path.join(os.path.dirname(rd.__file__), "runtime_detector.py")
_PYTHON = "%d.%d.%d" % sys.version_info[:3]


@pytest.fixture
def installed(monkeypatch):
    """Fake distribution metadata: {distribution: version}."""
    versions = {}

    def version(dist):
        if dist in versions:
            return versions[dist]
        raise importlib.metadata.PackageNotFoundError(dist)

    monkeypatch.setattr(importlib.metadata, "version", version)
    return versions


def _imported(monkeypatch, name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.fixture
def no_torch(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)


def _versions():
    rd._capture_software_versions()
    return rd._runtime_info["software_versions"]


def test_lists_only_imported_tracked_packages(installed, monkeypatch, no_torch):
    installed.update({"transformers": "4.57.1", "vllm": "0.11.0", "pyyaml": "6.0"})
    _imported(monkeypatch, "transformers")
    _imported(monkeypatch, "yaml")  # imported, but not tracked

    assert _versions() == {"python": _PYTHON, "packages": {"transformers": "4.57.1"}}


def test_falls_back_to_module_version(installed, monkeypatch, no_torch):
    # A source checkout on PYTHONPATH has no distribution metadata.
    _imported(monkeypatch, "trl", __version__="0.24.0.dev0")

    assert _versions()["packages"] == {"trl": "0.24.0.dev0"}


def test_module_under_a_different_name_than_its_distribution(installed, monkeypatch, no_torch):
    installed["flash-attn"] = "2.8.3"
    installed["megatron-core"] = "0.14.0"
    _imported(monkeypatch, "flash_attn")
    _imported(monkeypatch, "megatron.core")

    assert _versions()["packages"] == {"flash-attn": "2.8.3", "megatron-core": "0.14.0"}


def test_lazily_imported_package_appears_on_a_later_capture(installed, monkeypatch, no_torch):
    installed.update({"transformers": "4.57.1", "bitsandbytes": "0.48.1"})
    _imported(monkeypatch, "transformers")
    assert _versions()["packages"] == {"transformers": "4.57.1"}

    _imported(monkeypatch, "bitsandbytes")

    assert _versions()["packages"] == {"transformers": "4.57.1", "bitsandbytes": "0.48.1"}


def test_unknown_version_is_left_out(installed, monkeypatch, no_torch):
    _imported(monkeypatch, "peft")

    assert _versions() == {"python": _PYTHON}


def test_long_versions_are_truncated(installed, monkeypatch, no_torch):
    installed["peft"] = "1." + "9" * 100
    _imported(monkeypatch, "peft")

    assert len(_versions()["packages"]["peft"]) == 64


def _fake_torch(monkeypatch, cuda="12.8", hip=None, cudnn=91002, nccl=(2, 27, 3)):
    def nccl_version():
        if nccl is None:
            raise AttributeError("module 'torch._C' has no attribute '_nccl_version'")
        return nccl

    def public_cudnn_version():
        # Real torch checks device capability here, initializing CUDA.
        raise AssertionError("torch.backends.cudnn.version() initializes CUDA")

    # torch.backends.cudnn._cudnn is None in a build without cuDNN.
    binding = types.SimpleNamespace(getVersionInt=lambda: cudnn) if cudnn is not None else None
    return _imported(
        monkeypatch, "torch",
        __version__="2.9.0",
        version=types.SimpleNamespace(cuda=cuda, hip=hip),
        backends=types.SimpleNamespace(cudnn=types.SimpleNamespace(_cudnn=binding, version=public_cudnn_version)),
        cuda=types.SimpleNamespace(nccl=types.SimpleNamespace(version=nccl_version)),
    )


def test_records_torch_build(installed, monkeypatch):
    installed["torch"] = "2.9.0+cu128"
    _fake_torch(monkeypatch)

    assert _versions() == {
        "python": _PYTHON,
        "packages": {"torch": "2.9.0+cu128"},
        "torch_cuda": "12.8",
        "cudnn": "9.10.2",
        "nccl": "2.27.3",
    }


def test_cpu_only_torch_build(installed, monkeypatch):
    installed["torch"] = "2.9.0+cpu"
    _fake_torch(monkeypatch, cuda=None, cudnn=None, nccl=None)

    assert _versions() == {"python": _PYTHON, "packages": {"torch": "2.9.0+cpu"}}


def test_rocm_torch_build(installed, monkeypatch):
    # The cuDNN binding is MIOpen's on ROCm, so it isn't reported as cuDNN.
    _fake_torch(monkeypatch, cuda=None, hip="6.4.43482", cudnn=30400, nccl=(2, 26, 6))

    versions = _versions()
    assert versions["torch_hip"] == "6.4.43482"
    assert versions["nccl"] == "2.26.6"
    assert "torch_cuda" not in versions
    assert "cudnn" not in versions


@pytest.mark.parametrize("value,expected", [
    (91002, "9.10.2"),
    (90100, "9.1.0"),
    (8902, "8.9.2"),
    (7605, "7.6.5"),
    (None, None),
    (0, None),
    (True, None),
])
def test_format_cudnn_version(value, expected):
    assert rd._format_cudnn_version(value) == expected


@pytest.mark.parametrize("value,expected", [
    ((2, 27, 3), "2.27.3"),
    ([2, 21, 5, "+cuda12.4"], "2.21.5"),
    (21805, "2.18.5"),
    (2708, "2.7.8"),
    ((2, 1), None),
    (None, None),
])
def test_format_nccl_version(value, expected):
    assert rd._format_nccl_version(value) == expected


def test_capture_never_raises(monkeypatch, no_torch):
    monkeypatch.setattr(rd, "_TRACKED_PACKAGES", None)  # not iterable

    rd._capture_software_versions()

    assert "software_versions" not in rd._runtime_info


@pytest.fixture
def flush_to(tmp_path, monkeypatch):
    out = tmp_path / "out.json"
    monkeypatch.setattr(rd, "_OUTPUT_PATH", str(out))
    monkeypatch.setattr(rd, "_capture_git_provenance", lambda **_: None)
    monkeypatch.setattr(rd, "_capture_training_args", lambda: None)
    monkeypatch.setattr(rd, "_capture_accelerate_config", lambda: None)
    return out


def test_process_that_detected_nothing_writes_nothing(flush_to, no_torch):
    # pip, a shell helper: versions alone don't make a process worth recording.
    rd.flush()

    assert not flush_to.exists()


def test_flush_records_versions(flush_to, installed, monkeypatch, no_torch):
    installed["trl"] = "0.24.0"
    _imported(monkeypatch, "trl")
    rd._runtime_info["learning_rate"] = 1e-4

    rd.flush()

    info = json.loads(flush_to.read_text())["runtime_info"]
    assert info["software_versions"] == {"python": _PYTHON, "packages": {"trl": "0.24.0"}}


def test_signal_flush_skips_the_lookup(flush_to, installed, monkeypatch, no_torch):
    installed["trl"] = "0.24.0"
    _imported(monkeypatch, "trl")
    rd._runtime_info["learning_rate"] = 1e-4

    rd._flush(final=True, in_signal=True)

    info = json.loads(flush_to.read_text())["runtime_info"]
    assert "software_versions" not in info


def test_merge_unions_package_lists():
    # torchrun's launcher imports only torch and exits after its ranks.
    rank = {"python": "3.12.3", "packages": {"torch": "2.9.0", "trl": "0.24.0"}, "torch_cuda": "12.8"}
    launcher = {"python": "3.12.3", "packages": {"torch": "2.9.0"}, "torch_cuda": "12.8"}
    existing = {"runtime_info": {"software_versions": rank}}

    merged = rd._merge_output(existing, [], {"software_versions": launcher})

    assert merged["runtime_info"]["software_versions"] == rank


def test_merge_newer_version_wins():
    existing = {"runtime_info": {"software_versions": {"packages": {"trl": "0.23.0"}}}}

    merged = rd._merge_output(existing, [], {"software_versions": {"packages": {"trl": "0.24.0"}}})

    assert merged["runtime_info"]["software_versions"]["packages"] == {"trl": "0.24.0"}


def test_real_interpreter_reads_distribution_metadata(tmp_path):
    # Through sitecustomize and the real importlib.metadata, including a
    # distribution whose name differs from its module (flash-attn/flash_attn).
    hooks = tmp_path / "aibom-hooks"
    hooks.mkdir()
    (hooks / "sitecustomize.py").write_text(open(_DETECTOR_SOURCE).read())
    site = tmp_path / "site"
    for dist, module, version in (("trl", "trl", "0.24.0"), ("flash_attn", "flash_attn", "2.8.3")):
        (site / module).mkdir(parents=True)
        (site / module / "__init__.py").write_text("")
        info = site / f"{dist}-{version}.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {dist.replace('_', '-')}\nVersion: {version}\n")
    out = tmp_path / "out.json"
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": os.pathsep.join([str(hooks), str(site)]),
        "AIBOM_DATASET_DETECT": "1",
        "AIBOM_DATASET_OUTPUT": str(out),
    }
    script = "import trl, flash_attn, random\nrandom.seed(1)\n"
    subprocess.run([sys.executable, "-c", script], env=env, cwd=tmp_path, check=True, timeout=60)

    info = json.loads(out.read_text())["runtime_info"]
    assert info["software_versions"] == {
        "python": _PYTHON,
        "packages": {"trl": "0.24.0", "flash-attn": "2.8.3"},
    }
