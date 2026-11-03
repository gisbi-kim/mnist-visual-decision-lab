# Measured results

These numbers come from actual saved-weight inference. Test data were not used to select the checkpoint or fit the temperature.

Training GPU: NVIDIA GeForce RTX 4070 Ti SUPER; 69.0 minutes including validation/checkpoint overhead.
Peak training allocation: 2.77 GiB; reserved: 2.97 GiB.

| Held-out set | N | VLM + decision head | Scene bootstrap 95% CI | CNN + rules |
|---|---:|---:|---|---:|
| test_id | 1504 | 99.14% | 98.45%–99.67% | 95.21% |
| test_language | 1504 | 97.87% | 96.50%–98.95% | 97.01% |
| test_composition | 1024 | 98.83% | 97.95%–99.60% | 95.70% |
| test_combined | 1024 | 97.85% | 96.66%–98.86% | 96.48% |

The CNN baseline knows the fixed slot geometry and a limited template vocabulary. Its parser matched all benchmark semantics, including held-out phrases; this is not evidence of unrestricted language understanding.
The scene bootstrap does not fully account for original glyph reuse across scenes. Different test suites share the official test pool.

| Set | Korean | English | NONE precision / recall | Both changed instructions correct | NLL | Brier | ECE |
|---|---:|---:|---|---:|---:|---:|---:|
| test_id | 99.20% | 99.07% | 98.05% / 100.00% | 97.90% | 0.066 | 0.021 | 0.017 |
| test_language | 97.61% | 98.14% | 99.71% / 96.88% | 95.81% | 0.115 | 0.041 | 0.007 |
| test_composition | 98.63% | 99.02% | 98.43% / 98.05% | 97.06% | 0.080 | 0.026 | 0.016 |
| test_combined | 98.44% | 97.27% | 95.83% / 98.83% | 96.56% | 0.096 | 0.039 | 0.020 |

## Same image, different instructions

![Canonical 7, 4, 3 example](docs/canonical.png)

| Instruction | Expected | Predicted | LEFT | CENTER | RIGHT | NONE |
|---|---|---|---:|---:|---:|---:|
| 빨간 상자에 있는 홀수를 골라. | LEFT | LEFT | 98.2% | 0.6% | 0.6% | 0.6% |
| Choose the odd digit in a red box. | LEFT | LEFT | 98.3% | 0.6% | 0.6% | 0.5% |
| 가장 작은 홀수를 골라. | RIGHT | LEFT | 98.2% | 0.6% | 0.6% | 0.5% |
| Choose the smallest odd digit. | RIGHT | LEFT | 98.2% | 0.6% | 0.6% | 0.5% |
| 짝수를 골라. | CENTER | CENTER | 0.7% | 97.8% | 0.8% | 0.7% |
| Choose the even digit. | CENTER | CENTER | 5.1% | 73.7% | 16.2% | 5.0% |
| 파란 상자에 있는 짝수를 골라. | NONE | RIGHT | 0.7% | 0.7% | 98.0% | 0.6% |
| Choose the even digit in a blue box. | NONE | RIGHT | 0.6% | 0.6% | 98.3% | 0.5% |

This fixed example deliberately remains in the report despite failures. The right-hand MNIST image is labeled 3; misreading its handwriting is a plausible explanation, not a directly measured internal recognition result. No final-test example was used for retraining or checkpoint selection.

## Input dependence

Accuracy on a fixed diagnostic subset, scored against the original targets. Blank/shuffled inputs remove required information; these are not new valid-task benchmarks.

- normal: 100.00%
- blank_image: 21.88%
- shuffled_image: 23.44%
- blank_instruction: 26.56%

## Device and calibration measurements

Matched CPU/CUDA comparison: 16 inputs, action agreement 100.0%, maximum probability difference 0.0283.
Median latency in that matched set: CUDA 70.5 ms (bfloat16), CPU 770.2 ms (float32).
Batch size 1, 336 × 112 RGB images; preprocessing, tensor transfer, forward pass, probabilities and device synchronization included. Downloads, initialization and image disk reads excluded. Canonical examples may retain warm-up effects; full test-suite timings are in `results/report.json`.
Calibration: temperature 1.7278 fitted on 752 separate calibration rows; calibration NLL 0.1806 → 0.1279.
Unsloth/portable CUDA parity for the selected trained weights is recorded in `results/backend-comparison.json`. This sample check is not a guarantee of identical outputs on all inputs.

## Validation history

| Step | Fixed validation subset accuracy |
|---:|---:|
| 200 | 96.09% |
| 400 | 94.92% |
| 600 | 98.05% |
| 800 | 98.05% |
| 1000 | 98.05% |
| 1200 | 97.27% |
| 1400 | 98.05% |
| 1600 | 97.66% |
| 1800 | 98.44% |
| 2000 | 98.83% |
| 2200 | 97.66% |
| 2400 | 98.05% |
| 2600 | 98.05% |
| 2800 | 98.05% |
| 3000 | 98.44% |
| 3200 | 98.05% |
| 3400 | 98.05% |
| 3600 | 97.66% |
| 3800 | 98.44% |
| 4000 | 98.05% |
| 4200 | 98.05% |
| 4400 | 98.44% |
| 4600 | 98.44% |
| 4800 | 98.83% |
| 5000 | 98.44% |
| 5200 | 98.83% |
| 5400 | 98.83% |
| 5600 | 98.83% |
| 5800 | 98.83% |
| 6000 | 98.83% |
| 6000 | 98.83% |

## Representative failures

Selected by high model confidence among incorrect final test predictions; not a random sample.

### test_id: test_id-00103

![Failure scene](docs/failures/test_id-00103.png)

Instruction: 가장 큰 숫자를 찾아. 없으면 NONE을 선택해.
Expected **RIGHT**; predicted **CENTER** at 98.4% confidence.

### test_id: test_id-00051

![Failure scene](docs/failures/test_id-00051.png)

Instruction: Pick the smallest odd digit, or choose NONE if there is none.
Expected **RIGHT**; predicted **NONE** at 98.3% confidence.

### test_id: test_id-00151

![Failure scene](docs/failures/test_id-00151.png)

Instruction: Pick the smallest digit, or choose NONE if there is none.
Expected **LEFT**; predicted **RIGHT** at 98.2% confidence.

### test_language: test_language-00093

![Failure scene](docs/failures/test_language-00093.png)

Instruction: 빨간 상자에 있는 숫자 중 최댓값의 위치를 알려 줘. 없으면 NONE으로 답해.
Expected **LEFT**; predicted **CENTER** at 98.5% confidence.

### test_language: test_language-00176

![Failure scene](docs/failures/test_language-00176.png)

Instruction: 파란 상자에 있는 짝수 중 최솟값의 위치를 알려 줘. 없으면 NONE으로 답해.
Expected **NONE**; predicted **LEFT** at 98.4% confidence.

### test_language: test_language-00129

![Failure scene](docs/failures/test_language-00129.png)

Instruction: Look for the largest even digit; indicate its position, or NONE if absent.
Expected **CENTER**; predicted **RIGHT** at 98.4% confidence.

### test_composition: test_composition-00040

![Failure scene](docs/failures/test_composition-00040.png)

Instruction: Select the smallest odd digit in a blue box.
Expected **NONE**; predicted **RIGHT** at 98.4% confidence.

### test_composition: test_composition-00010

![Failure scene](docs/failures/test_composition-00010.png)

Instruction: Choose the largest even digit in a red box.
Expected **LEFT**; predicted **RIGHT** at 98.4% confidence.

### test_composition: test_composition-00010

![Failure scene](docs/failures/test_composition-00010.png)

Instruction: 빨간 상자에 있는 가장 큰 짝수를 선택해 줘.
Expected **LEFT**; predicted **RIGHT** at 98.3% confidence.

### test_combined: test_combined-00055

![Failure scene](docs/failures/test_combined-00055.png)

Instruction: 선택할 대상은 파란 상자에 있는 가장 작은 홀수야. 해당하는 상자의 위치로 답해 줘. 없으면 NONE이야.
Expected **NONE**; predicted **RIGHT** at 98.3% confidence.

### test_combined: test_combined-00066

![Failure scene](docs/failures/test_combined-00066.png)

Instruction: 빨간 상자에 있는 짝수 중 최댓값의 위치를 알려 줘. 없으면 NONE으로 답해.
Expected **LEFT**; predicted **CENTER** at 97.7% confidence.

### test_combined: test_combined-00244

![Failure scene](docs/failures/test_combined-00244.png)

Instruction: Which box contains the smallest odd digit in a blue box? If no box qualifies, select NONE.
Expected **LEFT**; predicted **NONE** at 97.2% confidence.

## Scope and platform verification

- Windows NVIDIA CUDA and Windows CPU: independent-process real inference verified; detailed comparison is in `results/platform-comparison.json`.
- Apple Silicon MPS: implementation provided, physical Mac unavailable and **unverified**.
- A separately fine-tuned text-generation baseline and a head-only ablation were not run; the user requested focusing effort on data and the original model rather than expanding the study.
- No claim that the VLM is necessary, or that the decision head is inherently faster or more accurate than alternatives.
- Full probability outputs, confusion matrices, per-condition accuracy and timings are in `results/`.
