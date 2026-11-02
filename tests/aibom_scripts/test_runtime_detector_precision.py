"""Numerical precision: TF32 flags, autocast, Accelerator, DeepSpeed, FSDP and FP8 hooks."""

import dataclasses
import sys
import types

import pytest

import runtime_detector as rd


def _module(monkeypatch, name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.fixture
def fake_autocast(fake_torch_module, monkeypatch):
    class autocast:
        def __init__(self, device_type, dtype=None, enabled=True):
            self.device, self.fast_dtype, self._enabled = device_type, dtype, enabled

    class cuda_autocast(autocast):  # torch.cuda.amp.autocast
        def __init__(self, dtype=None):
            super().__init__("cuda", dtype)

    torch = sys.modules["torch"]
    torch.amp = types.SimpleNamespace(autocast_mode=types.SimpleNamespace(autocast=autocast))
    return types.SimpleNamespace(autocast=autocast, cuda_autocast=cuda_autocast)


def test_autocast_records_distinct_enabled_contexts(fake_autocast):
    rd.install_hooks()
    for _ in range(3):
        fake_autocast.autocast("cuda", dtype="torch.bfloat16")
    fake_autocast.cuda_autocast(dtype="torch.float16")
    fake_autocast.autocast("cpu", dtype="torch.bfloat16", enabled=False)

    assert rd._runtime_info["autocast"] == ["cuda:bfloat16", "cuda:float16"]


def test_accelerator_records_resolved_mixed_precision(monkeypatch):
    class Accelerator:
        def __init__(self, mixed_precision="no"):
            self.mixed_precision = mixed_precision
            self.fp8_backend = "TE" if mixed_precision == "fp8" else None

    _module(monkeypatch, "accelerate", Accelerator=Accelerator)
    rd.install_hooks()
    Accelerator(mixed_precision="fp8")
    Accelerator(mixed_precision="bf16")  # the first one wins

    assert rd._runtime_info["accelerator_precision"] == {
        "mixed_precision": "fp8", "fp8_backend": "transformer_engine"}


@pytest.mark.parametrize("bf16,fp16,expected", [(True, False, "bf16"), (False, True, "fp16"), (False, False, "no")])
def test_deepspeed_engine_records_its_precision(monkeypatch, bf16, fp16, expected):
    class DeepSpeedEngine:
        def __init__(self, config=None):
            self.config = config

        def bfloat16_enabled(self):
            return bf16

        def fp16_enabled(self):
            return fp16

    _module(monkeypatch, "deepspeed", DeepSpeedEngine=DeepSpeedEngine)
    rd.install_hooks()
    DeepSpeedEngine()

    assert rd._runtime_info["deepspeed_precision"] == expected


def test_fsdp_mixed_precision_policy(fake_torch_module, monkeypatch):
    @dataclasses.dataclass
    class MixedPrecision:
        param_dtype: object = None
        reduce_dtype: object = None
        buffer_dtype: object = None

    @dataclasses.dataclass(frozen=True)
    class MixedPrecisionPolicy:
        param_dtype: object = None
        output_dtype: object = None

    _module(monkeypatch, "torch.distributed.fsdp",
            MixedPrecision=MixedPrecision, MixedPrecisionPolicy=MixedPrecisionPolicy)
    rd.install_hooks()
    MixedPrecisionPolicy(param_dtype="torch.bfloat16", output_dtype="torch.float32")

    assert rd._runtime_info["fsdp_mixed_precision"] == {
        "policy": "MixedPrecisionPolicy", "param_dtype": "bfloat16", "output_dtype": "float32"}


def test_transformer_engine_fp8_autocast(monkeypatch):
    class DelayedScaling:
        pass

    calls = []
    te = _module(monkeypatch, "transformer_engine.pytorch",
                 fp8_autocast=lambda enabled=True, fp8_recipe=None: calls.append(enabled) or "ctx")
    rd.install_hooks()
    assert te.fp8_autocast(enabled=False) == "ctx"
    assert "fp8" not in rd._runtime_info
    te.fp8_autocast(enabled=True, fp8_recipe=DelayedScaling())

    assert calls == [False, True]
    assert rd._runtime_info["fp8"] == {"backend": "transformer_engine", "recipe": "DelayedScaling"}


def test_torchao_float8_conversion(monkeypatch):
    float8 = _module(monkeypatch, "torchao.float8", convert_to_float8_training=lambda m, **kw: m)
    rd.install_hooks()

    assert float8.convert_to_float8_training("model") == "model"
    assert rd._runtime_info["fp8"] == {"backend": "torchao"}


def test_tf32_flags_and_accelerate_env_read_at_flush(fake_torch_module, monkeypatch):
    torch = sys.modules["torch"]
    torch.backends = types.SimpleNamespace(
        cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=True)),
        cudnn=types.SimpleNamespace(allow_tf32=True, deterministic=False, benchmark=False),
    )
    torch.get_float32_matmul_precision = lambda: "high"
    monkeypatch.setenv("ACCELERATE_MIXED_PRECISION", "bf16")

    rd._capture_precision_flags()

    assert rd._runtime_info["precision_flags"] == {
        "tf32_matmul": True, "tf32_cudnn": True,
        "float32_matmul_precision": "high", "accelerate_mixed_precision": "bf16",
    }


def test_precision_flags_only_with_training_activity(fake_torch_module, monkeypatch, tmp_path):
    # torchrun's launcher imports torch and exits after its ranks; its
    # untouched defaults mustn't overwrite rank 0's.
    monkeypatch.setattr(rd, "_OUTPUT_PATH", str(tmp_path / "out.json"))
    monkeypatch.setenv("ACCELERATE_MIXED_PRECISION", "bf16")
    rd._runtime_info["model_name"] = "m"
    rd._flush()

    assert "precision_flags" not in rd._runtime_info
