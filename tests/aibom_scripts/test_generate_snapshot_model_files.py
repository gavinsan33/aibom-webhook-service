import json
import os
import re
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "aibom-scripts" / "generate_snapshot.py"


def _load_reader():
    # generate_snapshot.py is a top-level script (benchmarks run at import),
    # so exec only the model-file reader section rather than importing it.
    src = SCRIPT.read_text()
    start = src.index("_MODEL_FILE_MAX_BYTES")
    end = src.index("_SIGNING_KEY_PATH")
    ns = {"os": os, "re": re, "json": json}
    exec(src[start:end], ns)
    return ns["read_model_source_files"]


read_model_source_files = _load_reader()

COMMIT = "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"

README = """---
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/blob/main/LICENSE
base_model: Qwen/Qwen2.5-32B
library_name: transformers
---

# Qwen2.5-32B-Instruct

## Introduction
"""


def _make_model(tmp_path, readme=README, with_cache=True):
    (tmp_path / "README.md").write_text(readme)
    (tmp_path / "config.json").write_text(json.dumps({
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "torch_dtype": "bfloat16",
        "max_position_embeddings": 32768,
    }))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 65527752704}, "weight_map": {}})
    )
    if with_cache:
        d = tmp_path / ".cache" / "huggingface" / "download"
        d.mkdir(parents=True)
        (d / "config.json.metadata").write_text(f"{COMMIT}\n989289c4\n1790092648.8\n")
    return tmp_path


def test_reads_identity_and_config_from_hf_snapshot(tmp_path):
    got = read_model_source_files(str(_make_model(tmp_path)))
    assert got == {
        "repo_id": "Qwen/Qwen2.5-32B-Instruct",
        "base_model": "Qwen/Qwen2.5-32B",
        "revision": COMMIT,
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "dtype": "bfloat16",
        "max_position_embeddings": 32768,
        "total_size_bytes": 65527752704,
    }


def test_repo_id_omitted_when_license_link_names_a_different_model_than_title(tmp_path):
    # A fine-tune's license_link commonly points at its base model.
    readme = README.replace("# Qwen2.5-32B-Instruct", "# my-finetune")
    got = read_model_source_files(str(_make_model(tmp_path, readme=readme)))
    assert "repo_id" not in got
    assert got["base_model"] == "Qwen/Qwen2.5-32B"


def test_list_valued_base_model_is_ignored(tmp_path):
    readme = README.replace("base_model: Qwen/Qwen2.5-32B", "base_model:\n- a/b\n- c/d")
    got = read_model_source_files(str(_make_model(tmp_path, readme=readme)))
    assert "base_model" not in got


def test_revision_ignores_malformed_metadata(tmp_path):
    _make_model(tmp_path, with_cache=False)
    d = tmp_path / ".cache" / "huggingface" / "download"
    d.mkdir(parents=True)
    (d / "config.json.metadata").write_text("not-a-commit\n")
    assert "revision" not in read_model_source_files(str(tmp_path))


def test_missing_or_empty_dir_returns_none(tmp_path):
    assert read_model_source_files("") is None
    assert read_model_source_files(str(tmp_path / "nope")) is None
    assert read_model_source_files(str(tmp_path)) is None


def test_malformed_json_files_are_skipped(tmp_path):
    (tmp_path / "config.json").write_text("{not json")
    (tmp_path / "model.safetensors.index.json").write_text("[]")
    assert read_model_source_files(str(tmp_path)) is None
