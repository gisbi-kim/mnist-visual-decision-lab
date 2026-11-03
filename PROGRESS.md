# Work in progress — 2026-11-03

User scope: original MNIST Visual Decision Lab design, emphasis on good data,
approximately 1–3 hours of local GPU training; do not expand the comparison study.

Hardware verified locally: NVIDIA RTX 4070 Ti SUPER, 16,376 MiB VRAM.
At setup, other apps used about 4,988 MiB; 11,074 MiB was free.
Windows, Python 3.11, PyTorch 2.14.1+cu130, NVIDIA driver 610.88.

Dataset prepared: 18,048 bilingual training rows from 2,265 scenes and 6,380
distinct original MNIST training images. Split source images before composing scenes.
Balanced task × language × feasible action quotas; duplicate maximum/minimum and
multiple-match instructions are rejected. Withheld expression templates and two
condition combinations have separate final evaluation suites. Renderer variations
are independent of targets. Five semantic/leakage/rendering tests pass.

Initial environment troubleshooting: pandas 3.1.0 stalled during import in this
Windows environment. pandas 2.3.3 loaded successfully. Unsloth imports successfully;
the installed xformers binary mismatches this PyTorch version, so Unsloth uses
PyTorch attention. No security settings or other application processes were changed.

Base model pinned to Qwen/Qwen3.5-0.8B revision
2fc06364715b967f1860aea9cf38778875588b17, downloaded successfully.

Smoke training, saved-weight reload on CPU/CUDA, and 16-example overfit passed.
Overfit achieved 16/16 correct. Unsloth vs portable CUDA max probability difference
on eight validation examples: 0.000862. CPU vs CUDA: 0.001920. All actions agreed.

Main training is running: `train.py --run main --steps 6000 --seconds 7200
--batch 16 --accum 1 --validate-every 200 --val-count 256`.
Process started 2026-11-03 10:14 KST; actual optimizer work started about 10:15.
Live files: `main.log`, `runs/main/status.json`, `runs/main/validation.json`.
At step 2400, best validation subset accuracy is 98.83% (selected at step 2000).
No final VLM test result yet.
The training process uses about 2.8 GiB allocated / 3.0 GiB reserved with checkpointing.

Original-scope CPU CNN+rules baseline completed (`results/baseline/`). It trains on
the same training scenes, parses the limited template vocabulary, and detects
border colors from pixels. It is never used in the VLM app inference path.

Portable Flask UI and direct HTTP real-inference test passed. Browser automation
is disabled in Work settings, so no browser screenshot verification was performed.
The temporary smoke-model web server was stopped after the check.

Fresh environment `.verify-env` contains no Unsloth/bitsandbytes/Triton, but
Windows App Control returned WinError 4551 when loading both the CPU wheel and
the subsequently installed CUDA wheel there. The CUDA DLL hash matches the
working environment. Security settings were not changed. Clean-environment
execution is therefore blocked on this computer, and must not be reported as
verified. CPU inference works in `.venv` with the CUDA PyTorch build.
Cold-cache, unauthenticated base download and CPU smoke inference passed in that
working environment with `HF_HOME=.verify-cache` (`cold-cache.log`); the initial
load including download took about 72.8 seconds. This is not a fresh Python
environment success claim.

`finish_local.ps1` is waiting for the main training process (PID 36892), then
automatically runs calibration, all four test suites, CUDA/CPU diagnostics,
platform comparison and `make_report.py`. It writes `final-evaluation.log`.
It uses the working `.venv` for both CPU and CUDA; no claim of isolated-env
execution should be made. This pipeline does not publish anything.

Remaining: finish main training; calibrate best weights on calibration partition;
evaluate all four held-out suites; run diagnostics and cross-device comparison;
package/hash weights; prepare final results; create new public GitHub repo and
release, verify public download in a new folder, and start the finished local app.
GitHub connector has no repository/release creation tool; use authenticated gh CLI
for those operations. No HF token is configured; GitHub Release is an authorized
alternative distribution host under the original spec (no LFS usage needed).

Finalization tools are implemented: `finish_local.ps1` runs evaluation and
`make_report.py`; `package_release.py` stages a versioned ZIP and
`release_manifest.json` (after evaluation, not yet run); `download_model.py`
verifies both the weight archive and all pinned base model files. The base files
are 1,769,980,465 bytes; the measured Windows CUDA inference wheels total
2,052,544,052 bytes. See `docs/base-model-files.json` and `docs/download-sizes.json`.
Seven project tests pass, actual encoder leakage/image tests pass, source compiles.

Mac MPS has not been tested on a physical Mac.
