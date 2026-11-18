import os, sys, importlib.util
REPO = os.environ.get("RFD4_REPO", os.path.expanduser("~/rfd4-proteina"))
sys.path.insert(0, os.path.join(REPO, "src/rfproteina/af3"))

def build(env):
    for k in ("AF3_IMAGE","AF3_NATIVE_PYTHON","AF3_JAX_CACHE_DIR","AF3_MODEL_DIR","PYTHONPATH","APPTAINER_BIND_PATHS"):
        os.environ.pop(k, None)
    os.environ.update(env)
    for m in list(sys.modules):
        if m == "af3_apptainer": del sys.modules[m]
    import af3_apptainer as A
    return A.build_apptainer_fold_command("/tmp/manifest.json", gpu_device=2, save_full_outputs=False)

base = dict(AF3_JAX_CACHE_DIR="/data/jaxcache", AF3_MODEL_DIR="/data/af3_weights")

print("=== apptainer route (unchanged behaviour) ===")
cmd = build({**base, "AF3_IMAGE":"/data/af3-open.sif", "APPTAINER_BIND_PATHS":"/mnt/data:/mnt/data"})
print(" ".join(cmd)); print()

print("=== native route (AF3_NATIVE_PYTHON set) ===")
cmd2 = build({**base, "AF3_NATIVE_PYTHON":"/opt/af3/venv/bin/python"})
print(" ".join(cmd2)); print()

print("=== native route must NOT require AF3_IMAGE ===")
try:
    build({**base, "AF3_NATIVE_PYTHON":"/opt/af3/venv/bin/python"}); print("  ok, no AF3_IMAGE needed")
except OSError as e: print("  FAILED:", e)

print("=== apptainer route still errors without AF3_IMAGE ===")
try:
    build(base); print("  FAILED: should have raised")
except OSError as e: print("  ok:", e)

print("\n=== flags identical between routes? ===")
fa = [c for c in cmd if c.startswith("--") and "=" in c and not c.startswith("--bind")]
fb = [c for c in cmd2 if c.startswith("--") and "=" in c]
print("  apptainer:", fa)
print("  native   :", fb)
print("  MATCH" if fa == fb else "  MISMATCH")
