import sys
import types

import pytest

import runtime_detector as rd


@pytest.fixture(autouse=True)
def _reset_runtime_detector_state():
    """Full isolation between tests: hook-install guards, weak/key dataset
    registries, and captured runtime_info are all module-level globals."""
    rd.reset()
    rd._hooks_installed.clear()
    rd._dataset_registry.clear()
    rd._runtime_info.clear()
    yield
    rd.reset()
    rd._hooks_installed.clear()
    rd._dataset_registry.clear()
    rd._runtime_info.clear()


class FakeDatasetInfo:
    def __init__(self, builder_name=None, config_name=None, version=None, license=None):
        self.builder_name = builder_name
        self.config_name = config_name
        self.version = version
        self.license = license


class FakeDataset:
    """Minimal stand-in for datasets.arrow_dataset.Dataset.

    Mirrors the one property this test suite cares about: .map()/.filter()/
    .select()/.shuffle() return a *new* object (new identity, new
    fingerprint) but preserve builder_name/config_name via .info -- the
    shape that broke identity-only dedup against the DataLoader hook.
    """

    def __init__(self, builder_name, config_name=None, fingerprint="fp-orig"):
        self.info = FakeDatasetInfo(builder_name=builder_name, config_name=config_name)
        self._fingerprint = fingerprint

    def map(self, fn):
        return FakeDataset(self.info.builder_name, self.info.config_name, fingerprint="fp-mapped")


@pytest.fixture
def fake_datasets_module(monkeypatch):
    def fake_load_dataset(path, *args, **kwargs):
        return FakeDataset(builder_name=path, config_name=kwargs.get("name"))

    module = types.ModuleType("datasets")
    module.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", module)
    return module


@pytest.fixture
def fake_torch_module(monkeypatch):
    class FakeDataLoader:
        def __init__(self, dataset=None, *args, **kwargs):
            self.dataset = dataset
            self.batch_size = kwargs.get("batch_size")

    data_module = types.ModuleType("torch.utils.data")
    data_module.DataLoader = FakeDataLoader

    utils_module = types.ModuleType("torch.utils")
    utils_module.data = data_module

    torch_module = types.ModuleType("torch")
    torch_module.__version__ = "0.0.0-fake"
    torch_module.utils = utils_module

    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "torch.utils", utils_module)
    monkeypatch.setitem(sys.modules, "torch.utils.data", data_module)
    return data_module


@pytest.fixture
def fake_torchvision_module(monkeypatch):
    class FakeMNIST:
        def __init__(self, root=None, train=True, download=False, **kwargs):
            self.root = root
            self.train = train
            self.download = download

    datasets_module = types.ModuleType("torchvision.datasets")
    datasets_module.MNIST = FakeMNIST

    torchvision_module = types.ModuleType("torchvision")
    torchvision_module.datasets = datasets_module

    monkeypatch.setitem(sys.modules, "torchvision", torchvision_module)
    monkeypatch.setitem(sys.modules, "torchvision.datasets", datasets_module)
    return datasets_module


@pytest.fixture
def fake_webdataset_module(monkeypatch):
    class FakeWebDataset:
        def __init__(self, urls, *args, **kwargs):
            self.urls = urls

    module = types.ModuleType("webdataset")
    module.WebDataset = FakeWebDataset
    monkeypatch.setitem(sys.modules, "webdataset", module)
    return module


class FakeQuantizationConfig:
    def __init__(self, quant_method=None, load_in_4bit=False, load_in_8bit=False, bits=None):
        self.quant_method = quant_method
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.bits = bits


class FakeModelConfig:
    def __init__(self, architectures=None, quantization_config=None):
        self.architectures = architectures
        self.quantization_config = quantization_config


class FakePreTrainedModel:
    def __init__(self, config=None, dtype=None):
        self.config = config
        self.dtype = dtype

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        return cls(config=kwargs.get("config"), dtype=kwargs.get("torch_dtype"))


class FakeTrainingArguments:
    def __init__(
        self, output_dir=None, learning_rate=5e-5, per_device_train_batch_size=8,
        num_train_epochs=3, optim="adamw_torch", seed=42, bf16=False, fp16=False, **kwargs,
    ):
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.per_device_train_batch_size = per_device_train_batch_size
        self.num_train_epochs = num_train_epochs
        self.optim = optim
        self.seed = seed
        self.bf16 = bf16
        self.fp16 = fp16


@pytest.fixture
def fake_transformers_module(monkeypatch):
    module = types.ModuleType("transformers")
    module.TrainingArguments = FakeTrainingArguments
    module.PreTrainedModel = FakePreTrainedModel
    monkeypatch.setitem(sys.modules, "transformers", module)
    return module


class FakeLoraConfig:
    def __init__(self, r=16, lora_alpha=32, use_dora=False, use_rslora=False, **kwargs):
        self.r = r
        self.lora_alpha = lora_alpha
        self.use_dora = use_dora
        self.use_rslora = use_rslora


@pytest.fixture
def fake_peft_module(monkeypatch):
    module = types.ModuleType("peft")
    module.LoraConfig = FakeLoraConfig
    monkeypatch.setitem(sys.modules, "peft", module)
    return module
