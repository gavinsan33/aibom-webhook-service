import json

import pytest

import runtime_detector as rd
from conftest import FakeModelConfig, FakeQuantizationConfig


# ---------------------------------------------------------------------------
# HF datasets + DataLoader dedup
# ---------------------------------------------------------------------------


def test_hf_dataset_wrapped_directly_merges_via_identity(fake_datasets_module, fake_torch_module):
    """Baseline: the untransformed object returned by load_dataset is passed
    straight to DataLoader -- dedup via the identity-keyed weak registry."""
    rd.install_hooks()
    import datasets
    import torch.utils.data as tud

    ds = datasets.load_dataset("tatsu-lab/alpaca")
    tud.DataLoader(ds, batch_size=4)

    entries = rd.get_detected_datasets()
    assert len(entries) == 1
    assert entries[0]["dataset_name"] == "tatsu-lab/alpaca"
    assert entries[0]["batch_size"] == 4
    assert entries[0]["seen_via"] == ["torch.utils.data.DataLoader"]


def test_transformed_dataset_still_merges_into_hf_entry(fake_datasets_module, fake_torch_module):
    """Regression test: a script that calls dataset.map(...) before handing
    the *new* object to DataLoader used to defeat identity-based dedup and
    produce a spurious second entry (dataset_name="Dataset",
    matches_declared=False) for what is really one dataset."""
    rd.install_hooks()
    import datasets
    import torch.utils.data as tud

    ds = datasets.load_dataset("tatsu-lab/alpaca")
    tokenized = ds.map(lambda x: x)
    tud.DataLoader(tokenized, batch_size=4)

    entries = rd.get_detected_datasets()
    assert len(entries) == 1
    assert entries[0]["dataset_name"] == "tatsu-lab/alpaca"
    assert entries[0]["source"] == "datasets.load_dataset"
    assert entries[0]["batch_size"] == 4
    assert entries[0]["seen_via"] == ["torch.utils.data.DataLoader"]


def test_dataset_wrapped_in_dataloader_twice_records_seen_via_once(fake_datasets_module, fake_torch_module):
    """Regression test: Trainer/accelerate commonly re-wrap the same dataset
    in a second DataLoader internally (e.g. via accelerator.prepare()) --
    each merge used to unconditionally append, producing a seen_via list
    with the same hook name duplicated instead of noting it once."""
    rd.install_hooks()
    import datasets
    import torch.utils.data as tud

    ds = datasets.load_dataset("tatsu-lab/alpaca")
    tud.DataLoader(ds, batch_size=4)
    tud.DataLoader(ds, batch_size=4)

    entries = rd.get_detected_datasets()
    assert len(entries) == 1
    assert entries[0]["seen_via"] == ["torch.utils.data.DataLoader"]


def test_distinct_datasets_are_not_merged(fake_datasets_module, fake_torch_module):
    """The key-based fallback must not over-match: two genuinely different
    HF datasets stay as two entries."""
    rd.install_hooks()
    import datasets
    import torch.utils.data as tud

    alpaca = datasets.load_dataset("tatsu-lab/alpaca").map(lambda x: x)
    dolly = datasets.load_dataset("databricks/dolly-15k").map(lambda x: x)
    tud.DataLoader(alpaca, batch_size=4)
    tud.DataLoader(dolly, batch_size=8)

    entries = rd.get_detected_datasets()
    assert {e["dataset_name"] for e in entries} == {
        "tatsu-lab/alpaca",
        "databricks/dolly-15k",
    }


def test_different_config_name_same_builder_not_merged(fake_datasets_module, fake_torch_module):
    """Same underlying builder but a different config (e.g. a dataset with
    multiple subsets) is a different dataset for dedup purposes."""
    rd.install_hooks()
    import datasets
    import torch.utils.data as tud

    en = datasets.load_dataset("wikitext", name="wikitext-2-raw-v1").map(lambda x: x)
    fr = datasets.load_dataset("wikitext", name="wikitext-103-raw-v1").map(lambda x: x)
    tud.DataLoader(en, batch_size=4)
    tud.DataLoader(fr, batch_size=4)

    entries = rd.get_detected_datasets()
    assert len(entries) == 2


def test_dataloader_without_prior_hf_hook_records_generic_entry(fake_torch_module):
    """No datasets library involved at all: DataLoader hook falls back to
    inspecting the raw torch Dataset object, as before."""
    rd.install_hooks()
    import torch.utils.data as tud

    class PlainDataset:
        name = "my-custom-dataset"

    tud.DataLoader(PlainDataset(), batch_size=2)

    entries = rd.get_detected_datasets()
    assert len(entries) == 1
    assert entries[0]["dataset_name"] == "my-custom-dataset"
    assert entries[0]["source"] == "torch.utils.data.DataLoader"


# ---------------------------------------------------------------------------
# torchvision / webdataset hooks
# ---------------------------------------------------------------------------


def test_torchvision_hook_records_entry(fake_torchvision_module):
    rd.install_hooks()
    import torchvision.datasets as tvd

    tvd.MNIST(root="/data/mnist", train=True, download=True)

    entries = rd.get_detected_datasets()
    assert len(entries) == 1
    assert entries[0]["source"] == "torchvision.datasets.MNIST"
    assert entries[0]["split"] == "train"
    assert entries[0]["root"] == "/data/mnist"


def test_webdataset_hook_records_entry(fake_webdataset_module):
    rd.install_hooks()
    import webdataset as wds

    wds.WebDataset(["s3://bucket/shard-{000..010}.tar"])

    entries = rd.get_detected_datasets()
    assert len(entries) == 1
    assert entries[0]["source"] == "webdataset.WebDataset"
    assert entries[0]["urls"] == ["s3://bucket/shard-{000..010}.tar"]


# ---------------------------------------------------------------------------
# Path fingerprinting
# ---------------------------------------------------------------------------


def test_path_fingerprint_stable_for_unchanged_file(tmp_path):
    f = tmp_path / "data.bin"
    f.write_bytes(b"hello")
    first = rd._path_fingerprint(str(f))
    second = rd._path_fingerprint(str(f))
    assert first is not None
    assert first == second


def test_path_fingerprint_changes_when_file_size_changes(tmp_path):
    f = tmp_path / "data.bin"
    f.write_bytes(b"hello")
    before = rd._path_fingerprint(str(f))
    f.write_bytes(b"hello world, now longer")
    after = rd._path_fingerprint(str(f))
    assert before != after


def test_path_fingerprint_missing_path_returns_none(tmp_path):
    assert rd._path_fingerprint(str(tmp_path / "does-not-exist")) is None


# ---------------------------------------------------------------------------
# Training-arg / accelerate-config capture
# ---------------------------------------------------------------------------


def test_capture_training_args_from_argv(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["train.py", "--num_train_epochs", "3", "--learning_rate=0.0002", "--batch_size", "8"],
    )
    rd._capture_training_args()
    assert rd._runtime_info["epochs"] == 3
    assert rd._runtime_info["learning_rate"] == 0.0002
    assert rd._runtime_info["batch_size"] == 8


def test_capture_accelerate_config_resolves_strategy_from_yaml(tmp_path, monkeypatch):
    pytest.importorskip("yaml")
    config_path = tmp_path / "fsdp_config.yaml"
    config_path.write_text("distributed_type: FSDP\nnum_processes: 4\n")
    monkeypatch.setattr("sys.argv", ["train.py", "--accelerate_config", str(config_path)])

    rd._capture_accelerate_config()

    assert rd._runtime_info["parallelization_strategy"] == "fsdp"


def test_capture_accelerate_config_missing_file_is_a_noop(monkeypatch):
    monkeypatch.setattr("sys.argv", ["train.py", "--accelerate_config", "/nope/missing.yaml"])
    rd._capture_accelerate_config()
    assert "parallelization_strategy" not in rd._runtime_info


def test_capture_accelerate_config_no_flag_is_a_noop(monkeypatch):
    monkeypatch.setattr("sys.argv", ["train.py"])
    rd._capture_accelerate_config()
    assert "parallelization_strategy" not in rd._runtime_info


# ---------------------------------------------------------------------------
# flush()
# ---------------------------------------------------------------------------


def test_flush_writes_datasets_and_runtime_info(tmp_path, monkeypatch):
    monkeypatch.setattr(rd, "_OUTPUT_PATH", str(tmp_path / "dataset_detected.json"))
    monkeypatch.setattr("sys.argv", ["train.py"])
    rd._record({"dataset_name": "tatsu-lab/alpaca", "source": "datasets.load_dataset"})

    rd.flush()

    with open(tmp_path / "dataset_detected.json") as f:
        output = json.load(f)
    assert output["datasets"] == [
        {"dataset_name": "tatsu-lab/alpaca", "source": "datasets.load_dataset"}
    ]


def test_flush_merges_with_existing_file_without_duplicating(tmp_path, monkeypatch):
    output_path = tmp_path / "dataset_detected.json"
    output_path.write_text(json.dumps({
        "datasets": [{"dataset_name": "tatsu-lab/alpaca", "source": "datasets.load_dataset"}]
    }))
    monkeypatch.setattr(rd, "_OUTPUT_PATH", str(output_path))
    monkeypatch.setattr("sys.argv", ["train.py"])
    rd._record({"dataset_name": "tatsu-lab/alpaca", "source": "datasets.load_dataset"})
    rd._record({"dataset_name": "some-other-dataset", "source": "torchvision.datasets.MNIST"})

    rd.flush()

    with open(output_path) as f:
        output = json.load(f)
    assert len(output["datasets"]) == 2
    names = {d["dataset_name"] for d in output["datasets"]}
    assert names == {"tatsu-lab/alpaca", "some-other-dataset"}


def test_flush_with_nothing_detected_does_not_write(tmp_path, monkeypatch):
    output_path = tmp_path / "dataset_detected.json"
    monkeypatch.setattr(rd, "_OUTPUT_PATH", str(output_path))
    monkeypatch.setattr("sys.argv", ["train.py"])

    rd.flush()

    assert not output_path.exists()


# ---------------------------------------------------------------------------
# transformers.TrainingArguments / PreTrainedModel.from_pretrained
# ---------------------------------------------------------------------------


def test_transformers_hook_captures_training_arguments(fake_transformers_module):
    rd.install_hooks()
    import transformers

    transformers.TrainingArguments(
        output_dir="/tmp/out",
        learning_rate=2e-4,
        per_device_train_batch_size=4,
        num_train_epochs=3,
        optim="adamw_bnb_8bit",
        seed=1234,
        bf16=True,
    )

    assert rd._runtime_info["training_framework"] == "transformers.Trainer"
    assert rd._runtime_info["learning_rate"] == 2e-4
    assert rd._runtime_info["batch_size"] == 4
    assert rd._runtime_info["epochs"] == 3
    assert rd._runtime_info["optimizer"] == "adamw_bnb_8bit"
    assert rd._runtime_info["random_seed"] == 1234
    assert rd._runtime_info["dtype"] == "bfloat16"


def test_transformers_hook_prefers_fp16_when_bf16_not_set(fake_transformers_module):
    rd.install_hooks()
    import transformers

    transformers.TrainingArguments(fp16=True)

    assert rd._runtime_info["dtype"] == "float16"


def test_transformers_hook_captures_model_identity_from_from_pretrained(fake_transformers_module):
    rd.install_hooks()
    import transformers

    config = FakeModelConfig(architectures=["GraniteForCausalLM"])
    transformers.PreTrainedModel.from_pretrained("ibm-granite/granite-3.3-2b-instruct", config=config)

    assert rd._runtime_info["model_name"] == "ibm-granite/granite-3.3-2b-instruct"
    assert rd._runtime_info["model_architecture"] == "GraniteForCausalLM"


def test_transformers_hook_infers_bitsandbytes_quantization_from_load_in_4bit(fake_transformers_module):
    rd.install_hooks()
    import transformers

    config = FakeModelConfig(quantization_config=FakeQuantizationConfig(load_in_4bit=True))
    transformers.PreTrainedModel.from_pretrained("some-model", config=config)

    assert rd._runtime_info["quantization_method"] == "bitsandbytes"
    assert rd._runtime_info["quantization_bits"] == 4


def test_transformers_hook_uses_explicit_quant_method_when_present(fake_transformers_module):
    rd.install_hooks()
    import transformers

    config = FakeModelConfig(quantization_config=FakeQuantizationConfig(quant_method="gptq", bits=4))
    transformers.PreTrainedModel.from_pretrained("some-model", config=config)

    assert rd._runtime_info["quantization_method"] == "gptq"
    assert rd._runtime_info["quantization_bits"] == 4


def test_transformers_hook_no_quantization_config_leaves_fields_unset(fake_transformers_module):
    rd.install_hooks()
    import transformers

    transformers.PreTrainedModel.from_pretrained("some-model", config=FakeModelConfig())

    assert "quantization_method" not in rd._runtime_info
    assert "quantization_bits" not in rd._runtime_info


def test_transformers_hook_captures_device_map(fake_transformers_module):
    rd.install_hooks()
    import transformers

    transformers.PreTrainedModel.from_pretrained(
        "some-model", config=FakeModelConfig(), device_map="auto"
    )

    assert rd._runtime_info["model_device_map"] == "auto"


def test_transformers_hook_no_device_map_leaves_field_unset(fake_transformers_module):
    rd.install_hooks()
    import transformers

    transformers.PreTrainedModel.from_pretrained("some-model", config=FakeModelConfig())

    assert "model_device_map" not in rd._runtime_info


# ---------------------------------------------------------------------------
# peft.LoraConfig
# ---------------------------------------------------------------------------


def test_peft_hook_captures_plain_lora(fake_peft_module):
    rd.install_hooks()
    import peft

    peft.LoraConfig(r=16, lora_alpha=32)

    assert rd._runtime_info["lora_rank"] == 16
    assert rd._runtime_info["lora_alpha"] == 32
    assert rd._runtime_info["adaptation_method"] == "lora"


def test_peft_hook_detects_dora(fake_peft_module):
    rd.install_hooks()
    import peft

    peft.LoraConfig(r=16, lora_alpha=32, use_dora=True)

    assert rd._runtime_info["adaptation_method"] == "dora"


def test_peft_hook_detects_rslora(fake_peft_module):
    rd.install_hooks()
    import peft

    peft.LoraConfig(r=16, lora_alpha=32, use_rslora=True)

    assert rd._runtime_info["adaptation_method"] == "rslora"


def test_peft_hook_detects_qlora_when_base_model_already_quantized(
    fake_transformers_module, fake_peft_module
):
    rd.install_hooks()
    import peft
    import transformers

    config = FakeModelConfig(quantization_config=FakeQuantizationConfig(load_in_4bit=True))
    transformers.PreTrainedModel.from_pretrained("some-model", config=config)
    peft.LoraConfig(r=16, lora_alpha=32)

    assert rd._runtime_info["adaptation_method"] == "qlora"
