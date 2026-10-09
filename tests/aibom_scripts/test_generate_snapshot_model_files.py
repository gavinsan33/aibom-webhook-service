import json
import os
import re
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "aibom-scripts" / "generate_snapshot.py"


def _load_reader():
    # generate_snapshot.py is a top-level script (it runs at import),
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


# ---------------------------------------------------------------------------
# Fail-open behavior (#104): a bad model PVC or a failing stage must not hang
# or crash the init container.
# ---------------------------------------------------------------------------

import signal
import sys

import pytest


def _load_stage_runner(timeout="60"):
    src = SCRIPT.read_text()
    start = src.index("_STAGE_TIMEOUT_S")
    end = src.index("def run_cmd")
    ns = {"os": os, "signal": signal, "sys": sys}
    exec(src[start:end], ns)
    ns["_STAGE_TIMEOUT_S"] = int(timeout)
    return ns


def _guard(seconds=10):
    """Fail the test, rather than hang the suite, if something blocks."""
    def _boom(signum, frame):
        raise AssertionError("blocked: fail-open guard did not work")
    previous = signal.signal(signal.SIGALRM, _boom)
    signal.alarm(seconds)
    return previous


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs mkfifo")
def test_read_model_source_files_skips_fifo_readme(tmp_path):
    _make_model(tmp_path)
    (tmp_path / "README.md").unlink()
    os.mkfifo(tmp_path / "README.md")
    previous = _guard()
    try:
        result = read_model_source_files(str(tmp_path))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    # The FIFO is ignored, the rest of the model's files are still read.
    assert result["architectures"] == ["Qwen2ForCausalLM"]
    assert "repo_id" not in result


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs mkfifo")
def test_read_model_source_files_skips_symlink_to_fifo(tmp_path):
    _make_model(tmp_path)
    os.mkfifo(tmp_path / "pipe")
    (tmp_path / "config.json").unlink()
    (tmp_path / "config.json").symlink_to(tmp_path / "pipe")
    previous = _guard()
    try:
        result = read_model_source_files(str(tmp_path))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    assert "architectures" not in result


def test_run_stage_returns_value_on_success():
    ns = _load_stage_runner()
    assert ns["_run_stage"]("ok", lambda: 42) == 42


def test_run_stage_swallows_exception_and_uses_on_error():
    ns = _load_stage_runner()

    def boom():
        raise FileNotFoundError("true")

    assert ns["_run_stage"]("boom", boom) is None
    assert ns["_run_stage"]("boom", boom, lambda exc: {"error": str(exc)}) == {"error": "true"}


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs mkfifo")
def test_run_stage_times_out_a_blocked_open(tmp_path):
    ns = _load_stage_runner()
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)

    def blocked_open():
        with open(fifo, "rb") as f:  # blocks until a writer appears
            return f.read()

    previous = _guard()
    try:
        result = ns["_run_stage"]("blocked", blocked_open, lambda exc: "fallback", timeout_s=1)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    assert result == "fallback"


def test_run_stage_restores_previous_signal_handler():
    ns = _load_stage_runner()
    sentinel = lambda signum, frame: None
    previous = signal.signal(signal.SIGALRM, sentinel)
    try:
        ns["_run_stage"]("ok", lambda: None)
        assert signal.getsignal(signal.SIGALRM) is sentinel
    finally:
        signal.signal(signal.SIGALRM, previous)
