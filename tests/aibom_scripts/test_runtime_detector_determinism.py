"""Seed hooks and determinism flags (#175)."""

import json
import os
import random
import subprocess
import sys
import types

import pytest

import runtime_detector as rd

_DETECTOR_SOURCE = os.path.join(os.path.dirname(rd.__file__), "runtime_detector.py")


@pytest.fixture
def fake_torch(fake_torch_module, monkeypatch):
    """The shared fake torch plus what the seed hook and flag capture read.
    As in real torch, torch.manual_seed is torch.random.manual_seed."""
    torch = sys.modules["torch"]
    calls = []

    def manual_seed(seed):
        calls.append(seed)
        return f"generator({seed})"

    random_module = types.ModuleType("torch.random")
    random_module.manual_seed = manual_seed
    torch.random = random_module
    torch.manual_seed = manual_seed

    cudnn = types.SimpleNamespace(deterministic=False, benchmark=False)
    torch.backends = types.SimpleNamespace(cudnn=cudnn)
    state = {"enabled": False, "warn_only": False}
    torch.are_deterministic_algorithms_enabled = lambda: state["enabled"]
    torch.is_deterministic_algorithms_warn_only_enabled = lambda: state["warn_only"]
    monkeypatch.setitem(sys.modules, "torch.random", random_module)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(rd, "_STARTUP_PYTHONHASHSEED", None)
    return types.SimpleNamespace(module=torch, calls=calls, cudnn=cudnn, algorithms=state)


@pytest.fixture
def fake_numpy(monkeypatch):
    calls = []

    def seed(seed=None):
        calls.append(seed)

    numpy_random = types.ModuleType("numpy.random")
    numpy_random.seed = seed
    numpy = types.ModuleType("numpy")
    numpy.random = numpy_random
    monkeypatch.setitem(sys.modules, "numpy", numpy)
    monkeypatch.setitem(sys.modules, "numpy.random", numpy_random)
    return types.SimpleNamespace(module=numpy_random, calls=calls)


class _NumpyInt:
    """Stands in for numpy.int64: not an int, but has __index__."""

    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value


def _seeds():
    return rd._runtime_info.get("seeds")


def test_records_each_generators_seed(fake_torch, fake_numpy):
    rd.install_hooks()
    import numpy as np
    import torch

    random.seed(42)
    np.random.seed(_NumpyInt(42))
    assert torch.manual_seed(42) == "generator(42)"

    assert _seeds() == {"values": {"python": 42, "numpy": 42, "torch": 42}}
    # The originals still ran.
    assert fake_numpy.calls[0].value == 42
    assert fake_torch.calls == [42]


def test_torch_random_manual_seed_is_patched_too(fake_torch):
    rd.install_hooks()
    import torch

    torch.random.manual_seed(7)

    assert torch.random.manual_seed is torch.manual_seed
    assert _seeds()["values"] == {"torch": 7}


def test_keyword_seed_is_recorded(fake_torch, fake_numpy):
    rd.install_hooks()
    import numpy as np

    random.seed(a=3)
    np.random.seed(seed=4)

    assert _seeds()["values"] == {"python": 3, "numpy": 4}


def test_reseeding_keeps_last_value_and_history():
    rd.install_hooks()

    random.seed(1234)
    random.seed(42)
    random.seed(1234)

    assert _seeds() == {"values": {"python": 1234}, "history": {"python": [1234, 42]}}


def test_repeating_one_seed_has_no_history():
    rd.install_hooks()

    random.seed(42)
    random.seed(42)

    assert _seeds() == {"values": {"python": 42}}


def test_history_is_capped():
    rd.install_hooks()

    for value in range(20):
        random.seed(value)

    assert _seeds()["values"] == {"python": 19}
    assert _seeds()["history"]["python"] == list(range(rd._SEED_HISTORY_MAX))


@pytest.mark.parametrize("seed", [None, b"bytes", True])
def test_unrecordable_seeds_are_skipped(seed):
    rd.install_hooks()

    random.seed(seed)

    assert _seeds() is None


def test_string_and_huge_seeds_stay_json_safe():
    rd.install_hooks()

    random.seed("x" * 100)
    assert _seeds()["values"]["python"] == "x" * 64

    random.seed(2 ** 64)
    # Ints past 2**53 can't be canonicalized under RFC 8785 (JCS).
    assert _seeds()["values"]["python"] == str(2 ** 64)


def test_failed_seed_call_records_nothing(fake_torch):
    rd.install_hooks()
    import torch

    def boom(seed):
        raise RuntimeError("bad seed")

    torch.manual_seed = rd._wrap_seed_fn(boom, "torch", "seed")
    with pytest.raises(RuntimeError):
        torch.manual_seed(1)

    assert _seeds() is None


def _run_as_dataloader_worker(source, **namespace):
    # torch's _worker_loop seeds each worker directly, then calls
    # worker_init_fn from the same frame.
    exec(compile(source, "worker.py", "exec"),
         {"__name__": "torch.utils.data._utils.worker", **namespace})


def test_dataloader_worker_seeds_are_ignored():
    rd.install_hooks()

    _run_as_dataloader_worker("random.seed(1000 + 3)", random=random)

    def worker_init_fn():  # e.g. transformers' seed_worker -> set_seed
        random.seed(99)

    _run_as_dataloader_worker("worker_init_fn()", worker_init_fn=worker_init_fn)

    assert _seeds() is None
    assert rd._training_activity is False


def test_non_primary_rank_does_not_record(monkeypatch):
    rd.install_hooks()
    monkeypatch.setenv("RANK", "3")

    random.seed(45)  # accelerate's set_seed(42, device_specific=True)

    assert _seeds() is None
    assert rd._training_activity is True


def test_rank_zero_records(monkeypatch):
    rd.install_hooks()
    monkeypatch.setenv("RANK", "0")

    random.seed(42)

    assert _seeds()["values"] == {"python": 42}


def test_hooks_install_once():
    rd.install_hooks()
    patched = random.seed
    rd.install_hooks()

    assert random.seed is patched


def test_torch_without_manual_seed_is_left_retryable(fake_torch_module):
    rd.install_hooks()

    assert "torch_seed" not in rd._hooks_installed
    assert "dataloader" in rd._hooks_installed


def test_training_arguments_data_seed_and_full_determinism(fake_transformers_module):
    rd.install_hooks()
    import transformers

    transformers.FakeSFTConfig(data_seed=7, full_determinism=True)

    assert rd._runtime_info["data_seed"] == 7
    assert rd._runtime_info["full_determinism"] is True


def test_training_arguments_without_them_records_neither(fake_transformers_module):
    rd.install_hooks()
    import transformers

    transformers.TrainingArguments()

    assert "data_seed" not in rd._runtime_info
    assert "full_determinism" not in rd._runtime_info


def test_flush_captures_flags_after_training_activity(fake_torch, tmp_path, monkeypatch):
    out = tmp_path / "out.json"
    monkeypatch.setattr(rd, "_OUTPUT_PATH", str(out))
    monkeypatch.setattr(rd, "_capture_git_provenance", lambda **_: None)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(rd, "_STARTUP_PYTHONHASHSEED", "0")
    rd.install_hooks()
    import torch

    torch.manual_seed(42)
    fake_torch.cudnn.deterministic, fake_torch.cudnn.benchmark = True, False
    fake_torch.algorithms.update(enabled=True, warn_only=True)
    rd.flush()

    info = json.loads(out.read_text())["runtime_info"]
    assert info["determinism_flags"] == {
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "deterministic_algorithms": True,
        "deterministic_algorithms_warn_only": True,
        "cublas_workspace_config": ":4096:8",
        "python_hash_seed": "0",
    }
    assert info["seeds"] == {"values": {"torch": 42}}


def test_warn_only_omitted_when_algorithms_disabled(fake_torch):
    rd.install_hooks()
    fake_torch.algorithms.update(enabled=False, warn_only=True)

    rd._capture_determinism_flags()

    assert rd._runtime_info["determinism_flags"] == {
        "cudnn_deterministic": False, "cudnn_benchmark": False, "deterministic_algorithms": False,
    }


def test_launcher_without_training_activity_records_no_flags(fake_torch, tmp_path, monkeypatch):
    # torchrun imports torch and exits after its ranks; its untouched
    # defaults must not overwrite what rank 0 wrote.
    out = tmp_path / "out.json"
    monkeypatch.setattr(rd, "_OUTPUT_PATH", str(out))
    monkeypatch.setattr(rd, "_capture_git_provenance", lambda **_: None)
    rd.install_hooks()
    fake_torch.cudnn.benchmark = True

    rd.flush()

    info = json.loads(out.read_text())["runtime_info"]
    assert "determinism_flags" not in info


def test_forked_child_starts_without_training_activity():
    # Set directly: a real seed call would schedule a flush timer, and
    # _after_fork_in_child drops it uncancelled (fine in a child, where the
    # thread doesn't exist; here it would fire into a later test).
    rd._training_activity = True

    rd._after_fork_in_child()

    assert rd._training_activity is False


def test_seeds_merge_as_one_observation():
    # A later process's seeds replace the whole group rather than mixing
    # generators from two processes.
    existing = {"runtime_info": {"seeds": {"values": {"python": 1, "torch": 1}}}}
    merged = rd._merge_output(existing, [], {"seeds": {"values": {"torch": 2}}})
    assert merged["runtime_info"]["seeds"] == {"values": {"torch": 2}}


def test_lazy_hooks_record_seeds_in_a_real_interpreter(tmp_path):
    # Through sitecustomize, with random imported after startup, and flushed
    # by atexit.
    hooks = tmp_path / "aibom-hooks"
    hooks.mkdir()
    (hooks / "sitecustomize.py").write_text(open(_DETECTOR_SOURCE).read())
    out = tmp_path / "out.json"
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(hooks),
        "AIBOM_DATASET_DETECT": "1",
        "AIBOM_DATASET_OUTPUT": str(out),
        "PYTHONHASHSEED": "123",
    }
    script = "import sys; assert 'random' not in sys.modules\nimport random\nrandom.seed(42)\n"
    subprocess.run([sys.executable, "-c", script], env=env, cwd=tmp_path, check=True, timeout=60)

    info = json.loads(out.read_text())["runtime_info"]
    assert info["seeds"] == {"values": {"python": 42}}
    assert info["determinism_flags"] == {"python_hash_seed": "123"}
