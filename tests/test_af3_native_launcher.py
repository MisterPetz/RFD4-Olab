"""Exercise native AF3 child processes without importing the model or needing a GPU."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

AF3_DIR = Path(__file__).resolve().parents[1] / "src/rfproteina/af3"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, AF3_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    monkeypatch.delenv("AF3_NATIVE_PYTHON", raising=False)
    monkeypatch.delenv("AF3_IMAGE", raising=False)
    monkeypatch.delenv("APPTAINER_BIND_PATHS", raising=False)
    monkeypatch.delenv("JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES", raising=False)
    monkeypatch.setenv("AF3_MODEL_DIR", str(tmp_path / "weights"))
    monkeypatch.setenv("AF3_JAX_CACHE_DIR", str(tmp_path / "cache"))
    module = _load("af3_apptainer")
    monkeypatch.setitem(sys.modules, "af3_apptainer", module)
    return module


@pytest.fixture
def native_python(runtime, monkeypatch, tmp_path):
    python = tmp_path / "environment with spaces" / "python"
    python.parent.mkdir()
    python.symlink_to(sys.executable)
    monkeypatch.setenv("AF3_NATIVE_PYTHON", str(python))
    return python


def _worker(path):
    path.write_text(
        "import json, os, sys\n"
        "def result():\n"
        "    return {'status': 'ok', 'args': sys.argv[1:], 'gpu': os.environ['CUDA_VISIBLE_DEVICES'], "
        "'cache': os.environ['JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES'], "
        "'platform': os.environ['JAX_PLATFORMS']}\n"
        "if '--serve' in sys.argv:\n"
        "    print(json.dumps({'status': 'ready'}), flush=True)\n"
        "    for line in sys.stdin:\n"
        "        if line.strip() == 'shutdown': break\n"
        "        print(json.dumps(result()), flush=True)\n"
        "else:\n"
        "    print(json.dumps(result()))\n"
    )


@pytest.mark.parametrize("cache_override", [None, "all"])
def test_native_child_environment_and_flags(runtime, native_python, monkeypatch, tmp_path, cache_override):
    if cache_override is not None:
        monkeypatch.setenv("JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES", cache_override)
    script = tmp_path / "worker with spaces.py"
    _worker(script)
    manifest = tmp_path / "manifest with spaces.json"
    cmd = runtime.build_apptainer_fold_command(manifest, 3, True, script)
    assert str(native_python) in cmd  # Preserve a virtual environment's interpreter symlink.
    output = subprocess.run(cmd, check=True, capture_output=True, text=True)
    child = json.loads(output.stdout)
    assert child["gpu"] == "3"
    assert child["platform"] == "cuda"
    assert child["cache"] == (cache_override or "none")
    assert "--gpu_device=0" in child["args"]
    assert "--num_recycles=5" in child["args"]
    assert "--num_diffusion_samples=1" in child["args"]
    assert "--save_full_outputs=true" in child["args"]
    assert f"--manifest={manifest}" in child["args"]


def test_daemon_uses_native_worker_and_dynamic_requirements(runtime, native_python, monkeypatch, tmp_path):
    daemon = _load("af3_daemon")
    assert daemon.required_env_vars()[0] == "AF3_NATIVE_PYTHON"
    monkeypatch.delenv("AF3_NATIVE_PYTHON")
    assert daemon.required_env_vars()[0] == "AF3_IMAGE"
    monkeypatch.setenv("AF3_NATIVE_PYTHON", str(native_python))
    script = tmp_path / "resident_worker.py"
    _worker(script)
    proc = daemon._spawn_warm_worker(str(script), 2)
    try:
        daemon._wait_for_worker_ready(proc)
        child = daemon._handle_fold({"manifest": str(tmp_path / "manifest.json")}, proc)
        assert child["status"] == "ok"
        assert child["gpu"] == "2"
        assert child["cache"] == "none"
        assert "--serve" in child["args"]
        assert not any(flag.startswith("--manifest=") for flag in child["args"])
        proc.stdin.write("shutdown\n")
        proc.stdin.flush()
        assert proc.wait(timeout=10) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_apptainer_command_stays_unchanged(runtime, monkeypatch, tmp_path):
    monkeypatch.setenv("AF3_IMAGE", "/runtime/af3.sif")
    monkeypatch.setattr(runtime.tempfile, "gettempdir", lambda: str(tmp_path))
    cache = os.environ["AF3_JAX_CACHE_DIR"]
    weights = os.environ["AF3_MODEL_DIR"]
    script = AF3_DIR / "run_alphafold_batch.py"
    assert runtime.build_apptainer_fold_command("manifest.json", 1, False) == [
        "apptainer", "run", "--nv", "--env", "CUDA_VISIBLE_DEVICES=1",
        "--env", "XLA_PYTHON_CLIENT_PREALLOCATE=false", "--env", "JAX_PLATFORMS=cuda",
        "--env", f"AF3_MODEL_DIR={weights}", "--bind", "/tmp:/tmp",
        "--bind", f"{tmp_path}:{tmp_path}", "--bind", f"{weights}:{weights}",
        "--bind", f"{cache}:{cache}", "--bind", f"{AF3_DIR}:{AF3_DIR}",
        "/runtime/af3.sif", "python", str(script), "--num_recycles=5",
        f"--flash_attention_implementation={runtime.AF3_FLASH_ATTENTION_IMPLEMENTATION}",
        "--num_diffusion_samples=1", "--gpu_device=0", f"--jax_compilation_cache_dir={cache}",
        f"--model_dir={weights}", "--save_full_outputs=false", "--manifest=manifest.json",
    ]


@pytest.mark.parametrize("kind", ["missing", "not_executable"])
def test_native_interpreter_must_be_executable(runtime, monkeypatch, tmp_path, kind):
    python = tmp_path / "invalid_python"
    if kind == "not_executable":
        python.write_text("not an interpreter")
        python.chmod(0o600)
    monkeypatch.setenv("AF3_NATIVE_PYTHON", str(python))
    with pytest.raises(OSError, match="AF3_NATIVE_PYTHON"):
        runtime.build_apptainer_fold_command("manifest.json", 0, False)


def test_native_one_shot_requires_manifest(runtime, native_python):
    with pytest.raises(ValueError, match="manifest_path"):
        runtime.build_apptainer_fold_command(None, 0, False)
