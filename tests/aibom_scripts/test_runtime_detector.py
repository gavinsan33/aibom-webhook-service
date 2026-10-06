import importlib
import json
import os
import subprocess
import sys
import textwrap
import types

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


def test_capture_training_args_accepts_fractional_epochs(monkeypatch):
    monkeypatch.setattr("sys.argv", ["train.py", "--num_train_epochs", "0.5"])
    rd._capture_training_args()
    assert rd._runtime_info["epochs"] == 0.5


def test_capture_training_args_bad_value_keeps_later_args(monkeypatch):
    monkeypatch.setattr(
        "sys.argv", ["train.py", "--num_train_epochs", "$EPOCHS", "--batch_size", "8"]
    )
    rd._capture_training_args()
    assert "epochs" not in rd._runtime_info
    assert rd._runtime_info["batch_size"] == 8


def test_capture_accelerate_config_reads_config_file_flag(tmp_path, monkeypatch):
    pytest.importorskip("yaml")
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("distributed_type: DEEPSPEED\n")
    monkeypatch.setattr("sys.argv", ["accelerate", "launch", "--config_file", str(config_path)])
    rd._capture_accelerate_config()
    assert rd._runtime_info["parallelization_strategy"] == "deepspeed"


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
# Git provenance: .git-directory read
# ---------------------------------------------------------------------------


def test_find_git_dir_walks_upward(tmp_path):
    git_dir = tmp_path / "repo" / ".git"
    git_dir.mkdir(parents=True)
    subdir = tmp_path / "repo" / "sub" / "dir"
    subdir.mkdir(parents=True)
    assert rd._find_git_dir(str(subdir)) == str(git_dir)


def test_find_git_dir_returns_none_when_not_found(tmp_path):
    assert rd._find_git_dir(str(tmp_path)) is None


def test_resolve_git_ref_loose_ref_file(tmp_path):
    git_dir = tmp_path / ".git"
    (git_dir / "refs" / "heads").mkdir(parents=True)
    (git_dir / "refs" / "heads" / "main").write_text("deadbeefcafe0123\n")
    assert rd._resolve_git_ref(str(git_dir), "ref: refs/heads/main") == "deadbeefcafe0123"


def test_resolve_git_ref_falls_back_to_packed_refs(tmp_path):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "packed-refs").write_text(
        "# pack-refs with: peeled fully-peeled sorted\n"
        "cafef00dbeef0123 refs/heads/main\n"
        "0123456789abcdef refs/heads/other\n"
    )
    assert rd._resolve_git_ref(str(git_dir), "ref: refs/heads/main") == "cafef00dbeef0123"


def test_resolve_git_ref_detached_head_returns_sha_directly(tmp_path):
    assert rd._resolve_git_ref(str(tmp_path / ".git"), "deadbeefcafe0123") == "deadbeefcafe0123"


def test_resolve_git_ref_missing_ref_returns_none(tmp_path):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    assert rd._resolve_git_ref(str(git_dir), "ref: refs/heads/missing") is None


def test_git_remote_url_reads_origin_from_config(tmp_path):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text(
        '[remote "origin"]\n'
        "\turl = https://github.com/org/repo.git\n"
        "\tfetch = +refs/heads/*:refs/remotes/origin/*\n"
    )
    assert rd._git_remote_url(str(git_dir)) == "https://github.com/org/repo.git"


REDACTION_CASES = [
    ("https://user:ghp_SECRET@github.com/org/repo.git", "https://github.com/org/repo.git"),
    ("https://ghp_SECRET@github.com/org/repo", "https://github.com/org/repo"),
    ("https://x-access-token:ghp_SECRET@github.com:8443/org/repo", "https://github.com:8443/org/repo"),
    ("https://github.com/org/repo.git?private_token=SECRET", "https://github.com/org/repo.git"),
    ("https://github.com/org/repo.git?token=SECRET#frag", "https://github.com/org/repo.git"),
    ("ssh://git:SECRET@host/org/repo.git", "ssh://git@host/org/repo.git"),
    ("ssh://git@host/org/repo.git", "ssh://git@host/org/repo.git"),
    ("git@github.com:org/repo.git", "git@github.com:org/repo.git"),
    ("https://github.com/org/repo", "https://github.com/org/repo"),
    ("/local/path/repo", "/local/path/repo"),
    (None, None),
    ("", ""),
]


@pytest.mark.parametrize("url,expected", REDACTION_CASES)
def test_redact_git_url(url, expected):
    assert rd._redact_git_url(url) == expected


def test_redact_git_url_unparseable_still_drops_userinfo():
    out = rd._redact_git_url("https://user:SECRET@[::1/org/repo?token=SECRET")
    assert "SECRET" not in out


def test_git_remote_url_strips_credentials(tmp_path):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text(
        '[remote "origin"]\n'
        "\turl = https://user:ghp_SECRET@github.com/org/repo.git\n"
    )
    assert rd._git_remote_url(str(git_dir)) == "https://github.com/org/repo.git"


def test_git_remote_url_missing_config_returns_none(tmp_path):
    assert rd._git_remote_url(str(tmp_path / ".git")) is None


def test_git_dirty_returns_none_when_git_binary_unavailable(monkeypatch):
    import subprocess

    def fake_run(*args, **kwargs):
        raise FileNotFoundError("no such file: git")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert rd._git_dirty("/some/path") is None


def test_git_dirty_true_when_porcelain_output_nonempty(monkeypatch):
    import subprocess

    class FakeResult:
        returncode = 0
        stdout = " M train.py\n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeResult())
    assert rd._git_dirty("/some/path") is True


def test_git_dirty_false_when_porcelain_output_empty(monkeypatch):
    import subprocess

    class FakeResult:
        returncode = 0
        stdout = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeResult())
    assert rd._git_dirty("/some/path") is False


def _write_fake_repo(root):
    git_dir = root / ".git"
    (git_dir / "refs" / "heads").mkdir(parents=True)
    (git_dir / "refs" / "heads" / "main").write_text("deadbeefcafe0123\n")
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
    (git_dir / "config").write_text(
        '[remote "origin"]\n\turl = https://github.com/org/repo.git\n'
    )
    return git_dir


def test_capture_git_provenance_populates_runtime_info(tmp_path, monkeypatch):
    _write_fake_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rd, "_git_dirty", lambda worktree_dir: False)

    rd._capture_git_provenance()

    assert rd._runtime_info["git_commit"] == "deadbeefcafe0123"
    assert rd._runtime_info["git_branch"] == "main"
    assert rd._runtime_info["git_repository"] == "https://github.com/org/repo.git"
    assert rd._runtime_info["git_dirty"] is False


def test_capture_git_provenance_no_git_dir_is_a_noop(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rd._capture_git_provenance()
    assert "git_commit" not in rd._runtime_info


def test_capture_git_provenance_omits_dirty_when_undetermined(tmp_path, monkeypatch):
    _write_fake_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rd, "_git_dirty", lambda worktree_dir: None)

    rd._capture_git_provenance()

    assert "git_dirty" not in rd._runtime_info
    assert rd._runtime_info["git_commit"] == "deadbeefcafe0123"


# ---------------------------------------------------------------------------
# flush()
# ---------------------------------------------------------------------------


def test_flush_writes_datasets_and_runtime_info(tmp_path, monkeypatch):
    # Isolate from this repo's own .git -- otherwise _capture_git_provenance
    # (invoked by flush()) would walk up from the real CWD and pick up this
    # project's actual commit, polluting runtime_info.
    monkeypatch.chdir(tmp_path)
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
    monkeypatch.chdir(tmp_path)
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
    monkeypatch.chdir(tmp_path)
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


# ---------------------------------------------------------------------------
# Hook installation (#106)
# ---------------------------------------------------------------------------


def test_transformers_hook_captures_dataclass_subclass_config(fake_transformers_module):
    # trl's SFTConfig/DPOConfig are dataclass subclasses of TrainingArguments
    # with their own generated __init__, which a hook on the parent's
    # __init__ never saw.
    rd.install_hooks()
    import transformers

    transformers.FakeSFTConfig(learning_rate=1e-4, per_device_train_batch_size=2, seed=7)

    assert rd._runtime_info["learning_rate"] == 1e-4
    assert rd._runtime_info["batch_size"] == 2
    assert rd._runtime_info["random_seed"] == 7
    assert rd._runtime_info["optimizer"] == "adamw_torch"


def test_hook_not_marked_installed_when_patch_target_missing(monkeypatch):
    # A module without the attribute being patched (half-initialized, or an
    # incompatible version) must not leave the hook marked as installed.
    monkeypatch.setitem(sys.modules, "datasets", types.ModuleType("datasets"))
    with pytest.raises(AttributeError):
        rd._install_hf_datasets_hook()
    assert "hf_datasets" not in rd._hooks_installed


@pytest.fixture
def import_sandbox(tmp_path, monkeypatch):
    """A sys.path entry for throwaway modules, with the post-import finder
    and every module imported during the test removed afterwards."""
    monkeypatch.syspath_prepend(str(tmp_path))
    before_modules = set(sys.modules)
    before_meta_path = list(sys.meta_path)
    importlib.invalidate_caches()

    def write(name, source):
        path = tmp_path.joinpath(*name.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source))

    yield write
    sys.meta_path[:] = before_meta_path
    for name in set(sys.modules) - before_modules:
        del sys.modules[name]


_FAKE_DATASETS_PKG = """
    class Dataset:
        def __init__(self, path):
            self.path = path

    def load_dataset(path, *args, **kwargs):
        return Dataset(path)
"""


def test_post_import_hook_patches_before_nested_from_import_binds(import_sandbox):
    # A helper module that's itself being imported does
    # `from datasets import load_dataset`. The old depth-0 trigger installed
    # the hook only after that name was already bound to the original.
    import_sandbox("datasets/__init__.py", _FAKE_DATASETS_PKG)
    import_sandbox("aibom_userlib.py", """
        from datasets import load_dataset

        def load():
            return load_dataset("imdb")
    """)
    sys.meta_path.insert(0, rd._PostImportHookFinder({"datasets": rd._install_hf_datasets_hook}))

    import aibom_userlib

    aibom_userlib.load()
    assert [d["dataset_name"] for d in rd.get_detected_datasets()] == ["imdb"]


def test_post_import_hook_fires_for_importlib_import_module(import_sandbox):
    import_sandbox("datasets/__init__.py", _FAKE_DATASETS_PKG)
    sys.meta_path.insert(0, rd._PostImportHookFinder({"datasets": rd._install_hf_datasets_hook}))

    importlib.import_module("datasets").load_dataset("squad")

    assert [d["dataset_name"] for d in rd.get_detected_datasets()] == ["squad"]


def test_post_import_hook_removes_itself_once_every_trigger_fired(import_sandbox):
    import_sandbox("aibom_fake_a.py", "")
    import_sandbox("aibom_fake_b.py", "")
    fired = []
    finder = rd._PostImportHookFinder({
        "aibom_fake_a": lambda: fired.append("a"),
        "aibom_fake_b": lambda: fired.append("b"),
    })
    sys.meta_path.insert(0, finder)

    import aibom_fake_a  # noqa: F401
    assert finder in sys.meta_path
    import aibom_fake_b  # noqa: F401

    assert fired == ["a", "b"]
    assert finder not in sys.meta_path


def test_post_import_hook_waits_for_a_successful_import(import_sandbox):
    # A failed import doesn't fire the hook (the module never finished
    # executing); a later successful import of the same module does.
    import_sandbox("aibom_flaky.py", "import os\nif os.environ.get('AIBOM_FLAKY_FAIL'):\n    raise RuntimeError('boom')\n")
    fired = []
    sys.meta_path.insert(0, rd._PostImportHookFinder({"aibom_flaky": lambda: fired.append(1)}))

    os.environ["AIBOM_FLAKY_FAIL"] = "1"
    try:
        with pytest.raises(RuntimeError):
            import aibom_flaky  # noqa: F401
    finally:
        del os.environ["AIBOM_FLAKY_FAIL"]
    assert fired == []

    import aibom_flaky  # noqa: F401,F811
    assert fired == [1]


def test_post_import_hook_restores_the_loaders_exec_module(import_sandbox):
    import_sandbox("aibom_fake_c.py", "")
    sys.meta_path.insert(0, rd._PostImportHookFinder({"aibom_fake_c": lambda: None}))

    import aibom_fake_c

    assert "exec_module" not in vars(aibom_fake_c.__loader__)


# ---------------------------------------------------------------------------
# sitecustomize activation, in a real interpreter (#106)
# ---------------------------------------------------------------------------

_DETECTOR_SOURCE = os.path.join(os.path.dirname(rd.__file__), "runtime_detector.py")
_PROBE = (
    "import sys; m = sys.modules.get('sitecustomize'); "
    "print('ACTIVE' if m is not None and hasattr(m, 'install_hooks_lazy') else 'INACTIVE')"
)


@pytest.fixture
def hooks_dir(tmp_path):
    d = tmp_path / "aibom-hooks"
    d.mkdir()
    (d / "sitecustomize.py").write_text(open(_DETECTOR_SOURCE).read())
    return d


def _run_probe(python, hooks_dir, tmp_path, *flags, extra_env=None, extra_path=()):
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": os.pathsep.join([str(hooks_dir), *map(str, extra_path)]),
        "AIBOM_DATASET_DETECT": "1",
        "AIBOM_DATASET_OUTPUT": str(tmp_path / "out.json"),
        **(extra_env or {}),
    }
    result = subprocess.run(
        [str(python), *flags, "-c", _PROBE], env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


@pytest.mark.parametrize("flags,extra_env", [
    ((), {}),
    ((), {"PYTHONNOUSERSITE": "1"}),
    (("-s",), {}),
])
def test_detector_activates_with_user_site_disabled(hooks_dir, tmp_path, flags, extra_env):
    # usercustomize is skipped in all of these; sitecustomize isn't.
    assert _run_probe(sys.executable, hooks_dir, tmp_path, *flags, extra_env=extra_env) == "ACTIVE"


def test_detector_activates_inside_a_venv(hooks_dir, tmp_path):
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True, timeout=120)
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    assert _run_probe(python, hooks_dir, tmp_path) == "ACTIVE"


def test_detector_runs_the_sitecustomize_it_shadows(hooks_dir, tmp_path):
    other = tmp_path / "image-site"
    other.mkdir()
    marker = tmp_path / "image-sitecustomize-ran"
    (other / "sitecustomize.py").write_text(f"open({str(marker)!r}, 'w').close()\n")

    assert _run_probe(sys.executable, hooks_dir, tmp_path, extra_path=[other]) == "ACTIVE"
    assert marker.exists()


def test_shadowed_sitecustomize_errors_do_not_break_startup(hooks_dir, tmp_path):
    other = tmp_path / "image-site"
    other.mkdir()
    (other / "sitecustomize.py").write_text("raise RuntimeError('broken image sitecustomize')\n")

    assert _run_probe(sys.executable, hooks_dir, tmp_path, extra_path=[other]) == "ACTIVE"
