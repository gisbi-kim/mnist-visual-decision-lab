# MNIST Visual Decision Lab

Three handwritten digits, colored boxes, and an instruction. The same image can
require LEFT, CENTER, RIGHT, or NONE when the instruction changes.

This project trains **Qwen3.5-0.8B + LoRA + a Clef decision head** on bilingual
Korean/English synthetic MNIST scenes. The local app sends the actual image and
the user's original text to that model. It does not parse the instruction into
a rule or pass MNIST labels to the model.

**Work in progress:** full training/evaluation and a public weight release are
being prepared. No final test accuracy or Mac GPU performance is claimed yet.

## Local app

Python 3.11 was used for the verified Windows environment. Install PyTorch first,
then the portable inference dependencies. The local app does not require Unsloth,
bitsandbytes, Triton, or CUDA-only kernels.

### Windows / NVIDIA CUDA (PowerShell)

```powershell
git clone https://github.com/gisbi-kim/mnist-visual-decision-lab.git
cd mnist-visual-decision-lab
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe download_model.py
.\.venv\Scripts\python.exe app.py --device cuda
```

### Apple Silicon Mac (MPS or CPU)

These are the intended installation commands. Physical Mac verification is pending.
The pinned PyTorch CPython 3.11 Apple Silicon wheel targets **macOS 14 or newer**.

```bash
git clone https://github.com/gisbi-kim/mnist-visual-decision-lab.git
cd mnist-visual-decision-lab
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.14.1 torchvision==0.29.1
python -m pip install -r requirements.txt
python download_model.py
python app.py --device mps
# CPU alternative:
# python app.py --device cpu
```

Open **http://127.0.0.1:7860**. Use `--device auto` for CUDA → MPS → CPU
selection, or explicitly request a device. Explicit requests fail clearly if
unavailable; they are not silently replaced by CPU. Stop with Ctrl+C.

The initial model load may download the pinned Qwen base model. Its cache is
managed by Hugging Face (`HF_HOME` / `HF_HUB_CACHE`, or the default user cache).
The extra weights are stored in `weights/`. Offline inference requires both
downloads to be complete; set `HF_HUB_OFFLINE=1` after that. The model executes on
the computer running the server, even if the page is viewed on a small screen.
The server binds only to localhost.

On this Windows machine, a second freshly installed environment encountered
Windows App Control **WinError 4551** while loading PyTorch DLLs, for both the
CPU-only wheel and the CUDA wheel. The original dedicated environment runs CPU
and CUDA inference successfully. Security settings were not changed. A truly
fresh-environment launch on this machine is therefore not yet verified; an
organization-managed machine may require its normal software approval process.

## Dataset design

- Preserve the official MNIST train/test distinction. Split original training
  image IDs into train, validation, and calibration **before** composing scenes.
- Train: **18,048 instructions, 2,265 scenes, 6,380 distinct source images**.
- Equal quotas for each feasible **condition × language × action** cell. Global
  counts: LEFT/CENTER/RIGHT 4,608 each, NONE 4,224. Unfiltered minimum/maximum
  always exists, so those tasks cannot honestly have NONE examples.
- 2,197 training scenes have instructions with different correct actions.
- Reject multi-match instructions and tied extrema. Duplicate digit values are
  permitted when they do not make the requested answer ambiguous.
- Vary digit enlargement, border shades/thickness, and small position jitter
  independently of targets. Share the renderer between training and the app.
- Hold out two complete combinations: red + even + maximum, and blue + odd +
  minimum. Their individual attributes occur in training.
- Hold out two sentence structures per language. Final sets separately evaluate
  new handwriting, new wording, new combinations, and their combination.
- Store the raw IDs, labels, colors, render seed and parameters for reproduction.
  Labels and the oracle stay in preparation/evaluation; `engine.py` does not
  import them. Each model payload contains only image, instruction and options.

The actual saved processor/Clef encoder was also probed: changing the gold
label leaves input tokens/pixels identical; blanking the image changes pixel
tensors; changing the user instruction changes input tokens. The tested image
produced 70 visual tokens within a 279-token combined input. See
`results/encoder-audit.json`. This complements the source-boundary tests.

| Split | Instructions | Scenes |
|---|---:|---:|
| Validation | 1,128 | 146 |
| Calibration | 752 | 102 |
| Test: known wording/conditions | 1,504 | 195 |
| Test: unseen wording | 1,504 | 198 |
| Test: unseen condition combinations | 1,024 | 323 |
| Test: unseen wording and combinations | 1,024 | 330 |

All final test suites use the official test image pool. They can share original
test glyphs with each other; none occur in the train/validation/calibration pools.
"Unseen" refers to this fine-tuning run; the base model's pretraining corpus was
not audited for prior exposure to public MNIST images.
Instructions on one scene are correlated. Reports therefore use a scene-cluster
bootstrap; reuse of source glyphs between scenes is a remaining dependence.

## Reproduce training

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-train.txt
.\.venv\Scripts\python.exe prepare_data.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe train.py --run main --steps 6000 --seconds 7200 --batch 16 --accum 1 --validate-every 200 --val-count 256
```

Training uses a fresh head on the pretrained base, not the earlier MNIST
`-1/STOP/+1` adapter. The script records the exact base revision, versions,
trainable layers, losses, validation, memory peaks, file hashes and checkpoints.
LoRA rank/alpha 16, bfloat16 base, no 4-bit quantization in this run. The training
wall-clock cap is checked after each optimizer step and includes validation time.
Initial download, model load and dataset encoding are outside this cap.

The best checkpoint is chosen using a fixed 256-row validation subset. Calibration
uses a different source-image partition, never the final test set.

```powershell
.\.venv\Scripts\python.exe evaluate.py --checkpoint runs/main/best --device cuda --calibrate --output results
```

## Model and platform limits

This is a controlled educational benchmark. A digit recognizer and rules can
also solve it. Neither an intrinsic accuracy advantage nor a speed advantage
over text generation is assumed. Unsupported instructions, multiple valid
answers, arbitrary uploaded layouts, and out-of-distribution handwriting may fail.
NONE is **absence of a qualifying digit**, not a general uncertainty or refusal
signal. Model probabilities are not a guarantee of correctness.

CPU uses float32; portable CUDA uses bfloat16; the provisional MPS path uses
float32. Portable inference loads the same LoRA and actual Clef head with the
saved processor and option schema. It compares saved/loaded adapter tensors.
The app never quietly enables CPU fallback for MPS. A user who explicitly enables
`PYTORCH_ENABLE_MPS_FALLBACK=1` may execute some operations on CPU; this must not
be described as all-GPU inference. MPS success and speed require a real Mac test.

## Sources and licensing

Independent successor to [the original MNIST decision tutorial](https://github.com/gisbi-kim/jev-mnist-tutorial).
No original repository or Space is modified.

- [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B), Apache-2.0.
- [Cloudflare Clef](https://huggingface.co/Cloudflare/clef), Apache-2.0. The
  unmodified reference encoder/head is `clef_reference.py`; its license and
  upstream README are included. Obtained from the original project's public
  [CPU demo](https://huggingface.co/spaces/giseopkim/mnist-decision-demo).
- [Unsloth decision training](https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth).
  Unsloth includes differently licensed components; this project uses the
  installed packages and does not relicense their code. Its decision implementation
  identifies itself as AGPL-3.0-only; Unsloth Zoo identifies LGPL-3.0-or-later.
- MNIST: Yann LeCun, Corinna Cortes and Christopher J.C. Burges; downloaded from
  the PyTorch OSSCI mirror, preserving original train/test data. See dataset notes.

See `PROGRESS.md` for current execution status. Final result and release details
will replace the work-in-progress notice after verification.
