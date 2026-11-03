# Execution notes — 2026-11-03

Original scope retained: Qwen3.5-0.8B + LoRA + Clef, with effort concentrated on
balanced bilingual data and held-out evaluation. No paid GPU or hosting was used.

## Completed locally

- Actual hardware: RTX 4070 Ti SUPER, 16,376 MiB VRAM; Ryzen 7 9800X3D;
  approximately 64 GB system RAM. This machine is not a 5070 Ti.
- 18,048 training instructions, 2,265 scenes and 6,380 original MNIST images.
  Source-image partitions, feasible action quotas, held-out wording/combinations,
  renderer variations and metadata are described in README and data-summary.json.
- Actual image encoder audit and seven project tests passed. Smoke training,
  separate-process reload and 16-example overfit passed; evidence is in
  results/preflight and results/encoder-audit.json.
- Main training: 6,000 steps, 4,139.333 seconds (69.0 minutes), batch 16,
  LoRA rank/alpha 16, bfloat16, no 4-bit quantization. Peak allocated/reserved
  GPU memory: 2.77/2.97 GiB. Best validation checkpoint: step 2,000, 253/256.
- Calibration used 752 separate rows. Four final test suites contain 5,056
  instructions; accuracy is 99.14%, 97.87%, 98.83%, and 97.85%.
- Actual CPU/CUDA predictions, latency, input ablations, same-image instruction
  changes, CNN+rules baseline and high-confidence failures are recorded in RESULTS.md.
- Unsloth and portable CUDA agreed on all 16 checked validation inputs for the
  selected trained weights; max uncalibrated probability difference 0.001142.
- Portable CPU/CUDA agreed on all 16 matched diagnostic inputs; max calibrated
  probability difference 0.028303. This is a sampled check, not a universal guarantee.
- Versioned weights and pinned base files have SHA-256 manifests. The archive is
  86,549,227 bytes. Windows inference wheels + base + weights total 3,909,073,744
  bytes, excluding Python/Git installers, source transfer and network metadata.

## Environment findings and limits

pandas 3.1.0 stalled during import; pandas 2.3.3 worked. The installed xformers
binary did not match PyTorch, so Unsloth used PyTorch attention. Reference PyTorch
implementations were used for unavailable optimized causal-convolution/linear-
attention kernels. These were actual successful runs, not simulated outputs.

A second new Python environment encountered Windows App Control WinError 4551
while loading both CPU-only and CUDA PyTorch DLLs. The latter DLL's hash matched
the working environment. Security settings were not changed. Therefore a clean
Python-environment launch is **not verified** on this computer. The dedicated
working environment performs CPU and CUDA inference successfully. A separate
empty Hugging Face cache, anonymous base download and CPU smoke inference passed;
that is a cold-cache test, not a new-environment success claim.

Apple Silicon MPS is **unverified** because no physical Mac is available.
Browser automation is disabled in Work settings, so visual browser verification
was not performed; server/HTTP checks are recorded separately. No settings were
changed to bypass either restriction.

The fixed 7/4/3 example has actual errors and remains published. Final test outputs
were not used to retrain or select the checkpoint. A text-generation baseline and
head-only training ablation were omitted to keep the requested scope focused.

## Publication

Code and model release are prepared for the new public repository. GitHub's
connected tool lacks repository/release creation, so the authenticated GitHub CLI
will be used. No Hugging Face write token is configured; a GitHub Release is the
alternative weight host allowed by the original specification. No Git LFS is used.
Anonymous released-source/weight download verification will be recorded after
publication. The original GitHub repository and Hugging Face Space are untouched.
