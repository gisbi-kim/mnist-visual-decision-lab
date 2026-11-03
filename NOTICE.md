# Sources and notices

New project code: Copyright 2026 giseop kim. Apache License 2.0 (`LICENSE`).

`clef_reference.py` is an unmodified copy of Cloudflare's Clef
`joint_schema_model.py`, retrieved through
https://huggingface.co/spaces/giseopkim/mnist-decision-demo.
The upstream Apache-2.0 license is `CLEF_LICENSE.txt`; provenance is in
`CLEF_README.md` and `clef_manifest.json`. The README's original relative
filename `LICENSE` refers to that upstream license.

Qwen base model: Qwen/Qwen3.5-0.8B, revision
2fc06364715b967f1860aea9cf38778875588b17, Apache-2.0. Its upstream license is
preserved in the model download. Adapters are new fine-tuning weights.

Unsloth and Unsloth Zoo are installed dependencies, not vendored project code.
Their original notices and licenses remain in the installed distributions.
The Unsloth decision modules carry AGPL-3.0-only notices; Unsloth Zoo carries
LGPL-3.0-or-later notices. The portable app uses standard Transformers/PEFT and
the Apache-2.0 Clef reference rather than importing Unsloth.

MNIST authors: Yann LeCun, Corinna Cortes, Christopher J.C. Burges.
Original: https://yann.lecun.com/exdb/mnist/
Download mirror: https://ossci-datasets.s3.amazonaws.com/mnist/
The maintained https://huggingface.co/datasets/ylecun/mnist card labels the
dataset MIT. `examples.npz` contains the first 1,000 official test examples;
the app's rendered scenes transform their scale, background and colored frames.
Source image IDs and authorship are preserved. Training data can be regenerated
from checksummed originals; it is not included as a large Git artifact.

Project inspiration and API usage reference:
https://github.com/gisbi-kim/jev-mnist-tutorial
No files in that project or its original Space were changed.
