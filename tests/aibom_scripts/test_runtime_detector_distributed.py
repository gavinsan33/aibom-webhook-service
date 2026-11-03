"""Distributed topology: world size, FSDP, DeepSpeed, Megatron and DeviceMesh hooks."""

import enum
import json
import sys
import types

import pytest

import runtime_detector as rd


def _module(monkeypatch, name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    monkeypatch.setitem(sys.modules, name, module)
    return module


class ShardingStrategy(enum.Enum):
    FULL_SHARD = 1
    SHARD_GRAD_OP = 2
    HYBRID_SHARD = 4


class CPUOffload:
    def __init__(self, offload_params=False):
        self.offload_params = offload_params


class CPUOffloadPolicy:
    pass


def test_accelerator_records_world_size_and_fsdp_plugin(monkeypatch):
    # accelerate 1.15: with reshard_after_forward set on FSDP1,
    # sharding_strategy is left at its FULL_SHARD default.
    plugin = types.SimpleNamespace(fsdp_version=1, sharding_strategy=ShardingStrategy.FULL_SHARD,
                                   reshard_after_forward=ShardingStrategy.SHARD_GRAD_OP,
                                   cpu_offload=CPUOffload(True))

    class Accelerator:
        def __init__(self):
            self.mixed_precision = "bf16"
            self.state = types.SimpleNamespace(
                distributed_type=types.SimpleNamespace(value="FSDP"), num_processes=4,
                fsdp_plugin=plugin, deepspeed_plugin=None)

    _module(monkeypatch, "accelerate", Accelerator=Accelerator)
    rd.install_hooks()
    Accelerator()

    assert rd._runtime_info["accelerator_distributed"] == {
        "distributed_type": "FSDP", "world_size": 4,
        "fsdp": {"version": 1, "sharding_strategy": "SHARD_GRAD_OP", "cpu_offload": True},
    }
    assert rd._runtime_info["accelerator_precision"] == {"mixed_precision": "bf16"}


def test_accelerator_records_deepspeed_plugin_fsdp2(monkeypatch):
    ds = types.SimpleNamespace(zero_stage=3, offload_optimizer_device="cpu", offload_param_device="none")

    class Accelerator:
        def __init__(self):
            self.state = types.SimpleNamespace(distributed_type="DEEPSPEED", num_processes=8, deepspeed_plugin=ds)

    _module(monkeypatch, "accelerate", Accelerator=Accelerator)
    rd.install_hooks()
    Accelerator()

    assert rd._runtime_info["accelerator_distributed"]["deepspeed"] == {
        "zero_stage": 3, "offload_optimizer": "cpu", "offload_param": "none"}


def test_deepspeed_engine_and_pipeline_stages(monkeypatch):
    class Device(str, enum.Enum):
        cpu = "cpu"

    class DeepSpeedEngine:
        def __init__(self):
            pass

        def bfloat16_enabled(self):
            return True

        def fp16_enabled(self):
            return False

        def zero_optimization_stage(self):
            return 2

        def zero_offload_optimizer(self):
            return types.SimpleNamespace(device=Device.cpu)

        def zero_offload_param(self):
            return None

    class PipelineEngine(DeepSpeedEngine):
        def __init__(self):
            super().__init__()
            self.num_stages = 4

    _module(monkeypatch, "deepspeed", DeepSpeedEngine=DeepSpeedEngine, PipelineEngine=PipelineEngine)
    rd.install_hooks()
    PipelineEngine()

    assert rd._runtime_info["deepspeed_engine"] == {"zero_stage": 2, "offload_optimizer": "cpu"}
    assert rd._runtime_info["deepspeed_pipeline_stages"] == 4


def test_fsdp1_module_and_fsdp2_fully_shard(fake_torch_module, monkeypatch):
    class FullyShardedDataParallel:
        def __init__(self, module, sharding_strategy=None, cpu_offload=None):
            self.sharding_strategy = sharding_strategy or ShardingStrategy.FULL_SHARD
            self.cpu_offload = cpu_offload or CPUOffload()

    fsdp = _module(monkeypatch, "torch.distributed.fsdp", FullyShardedDataParallel=FullyShardedDataParallel)
    rd.install_hooks()
    FullyShardedDataParallel("m", sharding_strategy=ShardingStrategy.HYBRID_SHARD)

    assert rd._runtime_info["fsdp_module"] == {"version": 1, "sharding_strategy": "HYBRID_SHARD", "cpu_offload": False}

    rd._runtime_info.pop("fsdp_module")
    fsdp.fully_shard = lambda module, **kwargs: module
    rd._hooks_installed.pop("fsdp_precision")
    rd._install_fsdp_hook()
    fsdp.fully_shard("m", mesh=types.SimpleNamespace(ndim=2), reshard_after_forward=False,
                     offload_policy=CPUOffloadPolicy())

    assert rd._runtime_info["fsdp_module"] == {
        "version": 2, "reshard_after_forward": False, "sharding_strategy": "HYBRID_SHARD", "cpu_offload": True}


def test_megatron_initialize_model_parallel(monkeypatch):
    calls = []
    parallel_state = _module(monkeypatch, "megatron.core.parallel_state",
                             initialize_model_parallel=lambda *a, **k: calls.append((a, k)))
    rd.install_hooks()
    parallel_state.initialize_model_parallel(4, 2, None, context_parallel_size=2, expert_model_parallel_size=8)

    assert calls  # the original still ran
    assert rd._runtime_info["megatron_parallel"] == {"tensor": 4, "pipeline": 2, "context": 2, "expert": 8}


def test_named_device_mesh_is_recorded_unnamed_is_not(fake_torch_module, monkeypatch):
    device_mesh = _module(monkeypatch, "torch.distributed.device_mesh",
                          init_device_mesh=lambda device_type, mesh_shape, **kwargs: mesh_shape)
    rd.install_hooks()
    device_mesh.init_device_mesh("cuda", (8,))
    assert "device_mesh" not in rd._runtime_info

    device_mesh.init_device_mesh("cuda", (2, 4), mesh_dim_names=("dp_shard", "tp"))
    assert rd._runtime_info["device_mesh"] == {"dp_shard": 2, "tp": 4}


def test_training_arguments_fsdp_and_deepspeed(tmp_path):
    ds = tmp_path / "ds.json"
    ds.write_text(json.dumps({"zero_optimization": {"stage": 3, "offload_param": {"device": "cpu"}}}))
    # transformers 5: fsdp=True plus fsdp_config.
    args = types.SimpleNamespace(fsdp=True, fsdp_config={"version": 2, "reshard_after_forward": True,
                                                          "cpu_offload": False}, deepspeed=str(ds))
    assert rd._training_args_distributed(args) == {
        "fsdp": {"version": 2, "reshard_after_forward": True, "cpu_offload": False},
        "deepspeed": {"zero_stage": 3, "offload_param": "cpu"},
    }
    # transformers 4.x: a list of FSDPOption.
    args = types.SimpleNamespace(fsdp=["full_shard", "offload", "auto_wrap"], fsdp_config={}, deepspeed=None)
    assert rd._training_args_distributed(args) == {"fsdp": {"sharding_strategy": "FULL_SHARD", "cpu_offload": True}}


def test_accelerate_config_file(tmp_path, monkeypatch):
    pytest.importorskip("yaml")
    config = tmp_path / "my_fsdp_config.yaml"
    config.write_text("distributed_type: FSDP\nnum_processes: 4\nnum_machines: 2\n"
                      "fsdp_config:\n  fsdp_version: 1\n  fsdp_sharding_strategy: 4\n  fsdp_offload_params: true\n")
    monkeypatch.setattr(sys, "argv", ["trl", "sft", "--accelerate_config", str(config)])

    rd._capture_accelerate_config()

    assert rd._runtime_info["accelerate_config_distributed"] == {
        "world_size": 4, "nodes": 2,
        "fsdp": {"version": 1, "sharding_strategy": "HYBRID_SHARD", "cpu_offload": True},
    }


@pytest.mark.parametrize("env, expected", [
    ({"WORLD_SIZE": "8", "LOCAL_WORLD_SIZE": "4", "GROUP_WORLD_SIZE": "2"},
     {"world_size": 8, "local_world_size": 4, "nodes": 2}),
    ({"WORLD_SIZE": "8", "LOCAL_SIZE": "4"}, {"world_size": 8, "local_world_size": 4, "nodes": 2}),  # deepspeed
    ({"OMPI_COMM_WORLD_SIZE": "2"}, {"world_size": 2}),
    ({}, None),
])
def test_distributed_env(monkeypatch, env, expected):
    for name in ("WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_WORLD_SIZE", "LOCAL_SIZE",
                 "OMPI_COMM_WORLD_SIZE", "OMPI_COMM_WORLD_LOCAL_SIZE"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    rd._capture_distributed_env()

    assert rd._runtime_info.get("distributed_env") == expected
