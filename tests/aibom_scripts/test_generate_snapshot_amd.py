import glob
import os
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "aibom-scripts" / "generate_snapshot.py"


def _load():
    # generate_snapshot.py runs at import; exec only the AMD reader.
    src = SCRIPT.read_text()
    ns = {"os": os, "glob": glob}
    exec(src[src.index("def _read_sysfs"):src.index('print("Collecting static')], ns)
    return ns["amd_gpus"]


amd_gpus = _load()


def _gpu(drm, node, vendor, **files):
    dev = drm / node / "device"
    dev.mkdir(parents=True)
    (dev / "vendor").write_text(vendor + "\n")
    for name, value in files.items():
        (dev / name).write_text(value + "\n")


def test_amd_gpus_reads_only_allocated_amd_nodes(tmp_path):
    dri, drm = tmp_path / "dri", tmp_path / "drm"
    dri.mkdir()
    for node in ("renderD128", "renderD129"):  # allocated to the container
        (dri / node).touch()
    _gpu(drm, "renderD128", "0x1002", product_name="AMD Instinct MI300X", mem_info_vram_total=str(192 * 1024**3))
    _gpu(drm, "renderD129", "0x1002", device="0x74a1")  # no FRU name, no VRAM file
    _gpu(drm, "renderD130", "0x1002", product_name="not ours")  # host GPU, not mounted
    assert amd_gpus(str(dri), str(drm)) == [
        {"name": "AMD Instinct MI300X", "vram_mb": 192 * 1024},
        {"name": "AMD GPU 0x74a1", "vram_mb": None},
    ]


def test_amd_gpus_skips_other_vendors(tmp_path):
    dri, drm = tmp_path / "dri", tmp_path / "drm"
    dri.mkdir()
    (dri / "renderD128").touch()
    _gpu(drm, "renderD128", "0x10de", product_name="NVIDIA")
    assert amd_gpus(str(dri), str(drm)) == []
    assert amd_gpus(str(tmp_path / "missing"), str(drm)) == []
