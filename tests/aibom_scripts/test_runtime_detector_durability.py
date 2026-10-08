"""Durable, concurrent-safe output from runtime_detector.py (#107)."""

import json
import multiprocessing
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

import runtime_detector as rd

# Captured before the `out_path` fixture stubs it out.
_REAL_CAPTURE_TRAINING_ARGS = rd._capture_training_args

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts" / "aibom-scripts"


@pytest.fixture
def out_path(tmp_path, monkeypatch):
    path = tmp_path / "out" / "dataset_detected.json"
    monkeypatch.setattr(rd, "_OUTPUT_PATH", str(path))
    # Real git/argv capture is exercised elsewhere; here it only adds noise
    # (and a `git status` per flush).
    monkeypatch.setattr(rd, "_capture_git_provenance", lambda check_dirty=True: None)
    monkeypatch.setattr(rd, "_capture_training_args", lambda: None)
    monkeypatch.setattr(rd, "_capture_accelerate_config", lambda: None)
    return path


def _wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


# ---------------------------------------------------------------------------
# Concurrent writers
# ---------------------------------------------------------------------------


def _writer(worker_id, per_worker, barrier):
    barrier.wait()  # start together so the read-merge-write windows overlap
    for i in range(per_worker):
        rd._detected_datasets.append({"dataset_name": f"ds-{worker_id}-{i}", "source": "test"})
        rd._flush()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
def test_concurrent_flushes_lose_nothing_and_never_leave_partial_json(out_path):
    ctx = multiprocessing.get_context("fork")
    workers, per_worker = 8, 15

    for trial in range(3):
        if out_path.exists():
            out_path.unlink()
        barrier = ctx.Barrier(workers)
        procs = [ctx.Process(target=_writer, args=(w, per_worker, barrier)) for w in range(workers)]
        for p in procs:
            p.start()
        # A reader racing the writers must only ever see complete files.
        while any(p.is_alive() for p in procs):
            if out_path.exists():
                try:
                    json.loads(out_path.read_text())
                except FileNotFoundError:
                    pass
                except ValueError as e:
                    raise AssertionError(f"trial {trial}: reader saw partial JSON: {e}")
        for p in procs:
            p.join()
            assert p.exitcode == 0

        names = {d["dataset_name"] for d in json.loads(out_path.read_text())["datasets"]}
        expected = {f"ds-{w}-{i}" for w in range(workers) for i in range(per_worker)}
        assert names == expected, f"trial {trial}: lost {len(expected - names)} of {len(expected)}"

    # No temp files left behind by the write-then-rename.
    assert [p.name for p in out_path.parent.iterdir() if p.name.endswith(".tmp")] == []


def test_flush_replaces_atomically_via_rename(out_path, monkeypatch):
    rd._runtime_info["lora_rank"] = 8
    replaced = []
    real_replace = os.replace
    monkeypatch.setattr(os, "replace", lambda a, b: (replaced.append((a, b)), real_replace(a, b))[1])

    rd._flush()

    assert len(replaced) == 1
    assert replaced[0][1] == str(out_path)
    assert replaced[0][0] != str(out_path)


# ---------------------------------------------------------------------------
# Merge semantics
# ---------------------------------------------------------------------------


def test_merge_git_group_is_replaced_not_mixed():
    existing = {"runtime_info": {
        "git_commit": "old111", "git_branch": "main", "git_dirty": True, "framework": "PyTorch",
    }}
    # A detached-HEAD checkout reports a commit but no branch.
    merged = rd._merge_output(existing, [], {"git_commit": "new222", "git_repository": "https://h/r"})
    ri = merged["runtime_info"]
    assert ri["git_commit"] == "new222"
    assert "git_branch" not in ri and "git_dirty" not in ri
    assert ri["framework"] == "PyTorch"


def test_merge_without_git_keys_leaves_existing_git_alone():
    existing = {"runtime_info": {"git_commit": "keep", "git_branch": "main"}}
    merged = rd._merge_output(existing, [], {"lora_rank": 8})
    assert merged["runtime_info"]["git_commit"] == "keep"
    assert merged["runtime_info"]["git_branch"] == "main"
    assert merged["runtime_info"]["lora_rank"] == 8


def test_merge_dedupes_datasets_by_name_and_source():
    existing = {"datasets": [{"dataset_name": "a", "source": "x"}]}
    merged = rd._merge_output(
        existing, [{"dataset_name": "a", "source": "x"}, {"dataset_name": "a", "source": "y"}], {}
    )
    assert [(d["dataset_name"], d["source"]) for d in merged["datasets"]] == [("a", "x"), ("a", "y")]


def test_process_with_nothing_detected_writes_nothing_and_skips_git(out_path, monkeypatch):
    def boom(check_dirty=True):
        raise AssertionError("git provenance must not be captured by a process that detected nothing")

    monkeypatch.setattr(rd, "_capture_git_provenance", boom)
    rd._flush()
    assert not out_path.exists()


def test_git_provenance_is_captured_once_for_periodic_flushes(out_path, monkeypatch):
    calls = []
    monkeypatch.setattr(rd, "_capture_git_provenance", lambda check_dirty=True: calls.append(check_dirty))
    rd._runtime_info["lora_rank"] = 8

    rd._flush(final=False)
    rd._flush(final=False)
    assert len(calls) == 1
    rd._flush(final=True)  # the exit-time flush re-captures: it is authoritative
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# Flush on detection (debounced)
# ---------------------------------------------------------------------------


def test_detection_is_flushed_without_waiting_for_exit(out_path, monkeypatch):
    monkeypatch.setattr(rd, "_FLUSH_DEBOUNCE_S", 0.05)
    monkeypatch.setattr(rd, "_durable_flush_enabled", True)

    rd._record({"dataset_name": "alpaca", "source": "test"})

    assert _wait_for(out_path.exists), "detection was never flushed"
    assert json.loads(out_path.read_text())["datasets"][0]["dataset_name"] == "alpaca"


def test_runtime_info_changes_are_flushed(out_path, monkeypatch):
    monkeypatch.setattr(rd, "_FLUSH_DEBOUNCE_S", 0.05)
    monkeypatch.setattr(rd, "_durable_flush_enabled", True)

    rd._runtime_info.update({"lora_rank": 16})

    assert _wait_for(out_path.exists)
    assert json.loads(out_path.read_text())["runtime_info"]["lora_rank"] == 16


def test_a_burst_of_detections_coalesces_into_one_flush(out_path, monkeypatch):
    monkeypatch.setattr(rd, "_FLUSH_DEBOUNCE_S", 0.2)
    monkeypatch.setattr(rd, "_durable_flush_enabled", True)
    calls = []
    monkeypatch.setattr(rd, "_flush", lambda final=True, in_signal=False: calls.append(final))

    for i in range(10):
        rd._record({"dataset_name": f"d{i}", "source": "test"})

    assert _wait_for(lambda: calls)
    time.sleep(0.4)
    assert calls == [False]


def test_flush_does_not_reschedule_itself_forever(out_path, monkeypatch):
    monkeypatch.setattr(rd, "_FLUSH_DEBOUNCE_S", 0.05)
    monkeypatch.setattr(rd, "_durable_flush_enabled", True)
    # Real argv capture writes the same values into runtime_info on every
    # flush; an unchanged value must not count as a change, or each flush
    # would schedule the next one forever.
    monkeypatch.setattr(rd, "_capture_training_args", _REAL_CAPTURE_TRAINING_ARGS)
    monkeypatch.setattr(rd.sys, "argv", ["train.py", "--learning_rate", "0.001"])

    rd._runtime_info["framework"] = "PyTorch"
    assert _wait_for(out_path.exists)
    time.sleep(0.3)
    assert rd._flush_timer is None
    mtime = out_path.stat().st_mtime
    time.sleep(0.3)
    assert out_path.stat().st_mtime == mtime


def test_unchanged_runtime_info_write_does_not_notify(monkeypatch):
    monkeypatch.setattr(rd, "_durable_flush_enabled", True)
    notified = []
    monkeypatch.setattr(rd, "_notify_change", lambda: notified.append(1))

    rd._runtime_info["x"] = 1
    rd._runtime_info["x"] = 1
    rd._runtime_info.update({"x": 1})

    assert len(notified) == 1


def test_notifications_are_inert_until_hooks_are_installed(out_path, monkeypatch):
    monkeypatch.setattr(rd, "_FLUSH_DEBOUNCE_S", 0.05)
    rd._runtime_info["x"] = 1
    time.sleep(0.2)
    assert rd._flush_timer is None
    assert not out_path.exists()


# ---------------------------------------------------------------------------
# Fork safety
# ---------------------------------------------------------------------------


def test_after_fork_in_child_resets_locks_and_timer():
    rd._lock.acquire()  # as if another thread held it at fork time
    rd._flush_timer = object()
    rd._git_captured = True

    rd._after_fork_in_child()

    assert rd._flush_timer is None
    assert rd._git_captured is False
    assert rd._lock.acquire(blocking=False), "child must not inherit a held lock"
    rd._lock.release()


# ---------------------------------------------------------------------------
# Signals, in a real process
# ---------------------------------------------------------------------------


def _run_child(tmp_path, body, debounce="60"):
    out = tmp_path / "child_out.json"
    script = tmp_path / "child.py"
    script.write_text(textwrap.dedent(f"""
        import os, sys, time, signal
        sys.path.insert(0, {str(SCRIPTS_DIR)!r})
        {textwrap.indent(textwrap.dedent(body), "        ").strip()}
    """))
    env = dict(
        os.environ,
        AIBOM_DATASET_DETECT="1",
        AIBOM_DATASET_OUTPUT=str(out),
        AIBOM_FLUSH_DEBOUNCE_S=debounce,
    )
    proc = subprocess.Popen(
        [sys.executable, str(script)], env=env, cwd=tmp_path,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    assert proc.stdout.readline().strip() == "ready", proc.stderr.read()
    return proc, out


def test_sigterm_flushes_then_dies_by_sigterm(tmp_path):
    proc, out = _run_child(tmp_path, """
        import runtime_detector as rd
        rd._runtime_info["lora_rank"] = 8
        print("ready", flush=True)
        time.sleep(60)
    """)
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=10)

    # Still terminated *by the signal* (what the default handler does), not
    # turned into a clean exit.
    assert proc.returncode == -signal.SIGTERM
    assert json.loads(out.read_text())["runtime_info"]["lora_rank"] == 8


def test_sigkill_loses_nothing_already_flushed(tmp_path):
    proc, out = _run_child(tmp_path, """
        import runtime_detector as rd
        rd._runtime_info["lora_rank"] = 8
        print("ready", flush=True)
        time.sleep(60)
    """, debounce="0.1")
    assert _wait_for(out.exists, timeout=10), "periodic flush never happened"
    proc.send_signal(signal.SIGKILL)
    proc.wait(timeout=10)

    assert proc.returncode == -signal.SIGKILL
    assert json.loads(out.read_text())["runtime_info"]["lora_rank"] == 8


def test_an_application_sigterm_handler_is_left_alone(tmp_path):
    proc, out = _run_child(tmp_path, """
        def app_handler(signum, frame):
            print("app handler ran", flush=True)
            sys.exit(0)
        signal.signal(signal.SIGTERM, app_handler)
        import runtime_detector as rd
        assert signal.getsignal(signal.SIGTERM) is app_handler
        rd._runtime_info["lora_rank"] = 8
        print("ready", flush=True)
        time.sleep(60)
    """)
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=10)

    # The app exits normally through its own handler, so atexit flushes.
    assert proc.returncode == 0
    assert json.loads(out.read_text())["runtime_info"]["lora_rank"] == 8
