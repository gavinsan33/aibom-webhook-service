"""torch.optim.Optimizer hook and TrainingArguments' optimizer fields."""

import math
import sys
import types

import pytest

import runtime_detector as rd


class _Tensor0d:
    """A 0-d tensor lr, as capturable/fused optimizers allow."""

    ndim = 0

    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


@pytest.fixture
def fake_optim(fake_torch_module, monkeypatch):
    """torch.optim.Optimizer's own __init__ contract: merge `defaults` into
    each param group. Subclasses call it through super(), so a hook on the
    base class sees every one."""

    class Optimizer:
        def __init__(self, params, defaults):
            self.defaults = defaults
            groups = list(params)
            if not isinstance(groups[0], dict):
                groups = [{"params": groups}]
            self.param_groups = [{**defaults, **g} for g in groups]

    class AdamW(Optimizer):
        def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, fused=None):
            super().__init__(params, {"lr": lr, "betas": betas, "eps": eps,
                                      "weight_decay": weight_decay, "amsgrad": False, "fused": fused})

    optim = types.ModuleType("torch.optim")
    optim.Optimizer = Optimizer
    optim.AdamW = AdamW
    sys.modules["torch"].optim = optim
    monkeypatch.setitem(sys.modules, "torch.optim", optim)
    return optim


def _constructed():
    return rd._runtime_info.get("optimizer_constructed")


def test_records_class_and_hyperparameters(fake_optim):
    rd.install_hooks()
    fake_optim.AdamW(["w"], lr=2e-4, betas=(0.9, 0.95), weight_decay=0.1)

    assert _constructed() == {
        "class": f"{fake_optim.AdamW.__module__}.{fake_optim.AdamW.__qualname__}",
        # fused=None is unset, not a value.
        "params": {"amsgrad": False, "betas": [0.9, 0.95], "eps": 1e-8, "lr": 2e-4, "weight_decay": 0.1},
    }
    assert rd._training_activity


def test_reads_param_groups_not_defaults(fake_optim):
    # Trainer passes weight_decay per group, so `defaults` would show torch's
    # own 0.01 instead of what ran.
    rd.install_hooks()
    fake_optim.AdamW([{"params": ["w"], "weight_decay": 0.05}, {"params": ["b"], "weight_decay": 0.0}])

    assert _constructed()["params"]["weight_decay"] == 0.05
    assert _constructed()["varies_by_param_group"] == ["weight_decay"]


def test_first_optimizer_wins(fake_optim):
    rd.install_hooks()
    fake_optim.AdamW(["g"], lr=1e-4)
    fake_optim.AdamW(["d"], lr=4e-4)

    assert _constructed()["params"]["lr"] == 1e-4


def test_unusable_values_are_dropped(fake_optim):
    rd.install_hooks()
    fake_optim.Optimizer(["w"], {"lr": _Tensor0d(3e-4), "nan": math.nan, "big": 2 ** 60,
                                 "name": "x", "nested": {"a": 1}})

    assert _constructed()["params"] == {"lr": 3e-4}


def test_no_torch_optim_installs_nothing(fake_torch_module):
    rd.install_hooks()

    assert "torch_optimizer" not in rd._hooks_installed


def test_training_arguments_optimizer_and_scheduler_fields(fake_transformers_module):
    import dataclasses
    import enum

    class SchedulerType(str, enum.Enum):
        COSINE = "cosine"

    base = fake_transformers_module.TrainingArguments

    @dataclasses.dataclass
    class TrainingArguments(base):
        adam_beta1: float = 0.9
        adam_beta2: float = 0.999
        adam_epsilon: float = 1e-8
        weight_decay: float = 0.0
        optim_args: str = None
        lr_scheduler_type: str = "linear"
        warmup_ratio: float = 0.0
        warmup_steps: int = 0

        def __post_init__(self):
            super().__post_init__()
            self.lr_scheduler_type = SchedulerType(self.lr_scheduler_type)

    fake_transformers_module.TrainingArguments = TrainingArguments
    rd.install_hooks()
    TrainingArguments(adam_beta2=0.95, weight_decay=0.1, lr_scheduler_type="cosine", warmup_ratio=0.03)

    info = rd._runtime_info
    assert (info["adam_beta1"], info["adam_beta2"], info["adam_epsilon"]) == (0.9, 0.95, 1e-8)
    assert info["weight_decay"] == 0.1
    assert info["lr_scheduler_type"] == "cosine"
    assert (info["warmup_ratio"], info["warmup_steps"]) == (0.03, 0)
    assert "optim_args" not in info  # None
