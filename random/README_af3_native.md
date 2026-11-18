# Using a local AlphaFold 3 install as the RFD4 refolding oracle

Short answer: **yes**, with a ~60 line patch. The container was never doing much.

## Why it works

`build_apptainer_fold_command` in `src/rfproteina/af3/af3_apptainer.py` builds this:

```
apptainer run --nv --env ... <AF3_IMAGE> python run_alphafold_batch.py --num_recycles=5 ...
```

Two things make the container droppable:

1. **Everything RFD4-specific is bind-mounted, not baked in.** `run_alphafold_batch.py` and
   its sibling `af3_runtime.py` both live in this repo under `src/rfproteina/af3/`, and the
   command binds their directory in. The image supplies only a Python that can
   `import alphafold3` and `import jax` with CUDA — which is exactly what a local AF3
   install is.
2. **Every bind is an identity mapping** (`host:host`): `APPTAINER_BIND_PATHS`, `/tmp`,
   `$TMPDIR`, `AF3_MODEL_DIR`, `AF3_JAX_CACHE_DIR`, and the script directory. Nothing is
   remapped to a canonical in-container path, so removing the container changes no path
   anywhere. This is the part that makes it safe rather than merely plausible.

Your existing **weights** are already usable as-is: `AF3_MODEL_DIR` is passed straight
through to `--model_dir` and is read by AF3 itself, not by anything container-specific.
Only the runtime needed replacing.

Note there is no `AF3_RUNTIME_BACKEND` the way AF2 has `AF2_RUNTIME_BACKEND`
(apptainer / docker / pyxis). AF3 is hard-wired to apptainer upstream, which is why this
needs a patch rather than a setting.

## Apply it

```bash
cd /path/to/rfd4-proteina
git apply /path/to/af3-native-oracle.patch
```

Verified to apply cleanly against `main` at commit `6805869`.

The patch touches two files:

- `src/rfproteina/af3/af3_apptainer.py` — adds a native branch to
  `build_apptainer_fold_command`, keyed on a new `AF3_NATIVE_PYTHON` env var.
- `src/rfproteina/af3/af3_daemon.py` — `AF3_IMAGE` is no longer required when
  `AF3_NATIVE_PYTHON` is set.

Both the warm-worker daemon route and the direct `subprocess.run` fallback go through that
one function, so patching it covers both.

**When `AF3_NATIVE_PYTHON` is unset, the apptainer command is byte-for-byte what it was.**
This is additive; it cannot change existing behaviour.

## Configure

In `.env.local` (or whichever env file you use):

```bash
# the interpreter of your local AF3 install - must import alphafold3 and jax[cuda]
export AF3_NATIVE_PYTHON=/opt/af3/venv/bin/python

# unchanged, and the same values you already use
export AF3_MODEL_DIR=/data/af3_weights
export AF3_JAX_CACHE_DIR=/data/af3_jax_cache

# AF3_IMAGE is no longer needed
```

Then run as normal:

```bash
rfd4-generate inputs=rfd4_stereocomplementary.yaml \
    subset_to_examples=t4_core_noligand \
    recycle_strategy=medium/midpeak \
    num_replicates=200 batch_size=8 \
    out_dir=./runs/t4_noligand \
    metrics_tags='[ame_refolding,motif_scaffold_refolding]'
```

`motif_scaffold_refolding` is the tag for motif scaffolding and enzyme design;
`ame_refolding` is the MCSA active-site benchmark tag. Both run the `af3` oracle group.

## Check it before a real run

```bash
python test_af3_native_cmd.py
```

Prints the argv for both routes and asserts the flags match. Expected native command:

```
env CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
    AF3_MODEL_DIR=/data/af3_weights PYTHONPATH=<repo>/src/rfproteina/af3 \
    /opt/af3/venv/bin/python <repo>/src/rfproteina/af3/run_alphafold_batch.py \
    --num_recycles=5 --flash_attention_implementation=cudnn --num_diffusion_samples=1 \
    --gpu_device=0 --jax_compilation_cache_dir=/data/jax_cache \
    --model_dir=/data/af3_weights --save_full_outputs=false --manifest=/tmp/manifest.json
```

Then a one-design smoke test before scaling:

```bash
rfd4-generate inputs=rfd4_stereocomplementary.yaml \
    subset_to_examples=t4_core_noligand num_replicates=1 batch_size=1 \
    out_dir=./runs/af3_smoke metrics_tags='[motif_scaffold_refolding]'
```

Check `runs/af3_smoke/inference_metrics/per_example_metrics.csv` actually has the
refolding columns, not just that the job exited 0.

## Design notes

- **`env` rather than mutating `os.environ`.** Neither call site passes `env=` to
  `subprocess`, so the child would otherwise inherit the parent's environment. Setting
  `CUDA_VISIBLE_DEVICES` in-process would retarget the *parent's* GPU — the one running
  the diffusion model. The `env` launcher scopes it to the child and keeps the builder a
  pure function, matching its existing signature.
- **`--gpu_device=0` is still correct.** `CUDA_VISIBLE_DEVICES=N` remaps the chosen device
  to index 0 for the child, natively just as inside the container.
- **`PYTHONPATH`** is set to the batch script's directory, replacing the container's bind,
  so `import af3_runtime` resolves. Any existing `PYTHONPATH` is appended, not clobbered.
- `AF3_NUM_RECYCLES = 5` and the attention backend are module constants, unchanged. The
  daemon JAX-compiles against the recycle count at startup, so per-call overrides are not
  possible — edit the constant if you need a different value. The attention backend is
  already overridable via `AF3_FLASH_ATTENTION_IMPLEMENTATION` (default `cudnn`; `triton`
  only on datacenter cards).

## What I could not verify

No GPU, no AF3 install and no image in this container, so I tested **command construction
only** — that the native argv is well-formed, that the flags are identical between routes,
that `AF3_IMAGE` is required in one route and not the other, and that the apptainer output
is unchanged. The first real run is still the test of whether your AF3 build and
`run_alphafold_batch.py` agree: that script pins `alphafold3.constants.chemical_components`
and `alphafold3.data.featurisation`, so a sufficiently different AF3 version could have
moved them. If it fails, the error will be an `ImportError` from those two lines rather
than anything to do with this patch.

## If you would rather not patch

`rfd4-evaluate` scores a folder of structures you already have, with the same
`metrics_tags`:

```bash
rfd4-evaluate inputs=./runs/t4_noligand/inference_samples \
    metrics_tags='[motif_scaffold_refolding]' out_dir=./runs/t4_eval
```

That still uses the same oracle, so it needs the same runtime — it decouples *when* you
fold, not *how*. Genuinely avoiding the oracle means folding with your own AF3 CLI and
computing scRMSD yourself, which gives up the pass criteria and the search reward.
