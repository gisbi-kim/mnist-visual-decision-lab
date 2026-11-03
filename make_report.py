"""Render source-backed result tables and copy representative failure examples."""
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent

def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))

def main():
    report=read('results/report.json')
    baseline=read('results/baseline/report.json')
    diagnostics=read('results/diagnostics-cuda.json')
    history=read('runs/main/validation.json')
    training=read('runs/main/metrics.json')
    config=read('runs/main/config.json')
    lines=['# Measured results','',
        'These numbers come from actual saved-weight inference. Test data were not used to select the checkpoint or fit the temperature.','',
        f"Training GPU: {config['gpu']}; {training['train_runtime']/60:.1f} minutes including validation/checkpoint overhead.",
        f"Peak training allocation: {training['peak_allocated_gb']:.2f} GiB; reserved: {training['peak_reserved_gb']:.2f} GiB.",'',
        '| Held-out set | N | VLM + decision head | Scene bootstrap 95% CI | CNN + rules |',
        '|---|---:|---:|---|---:|']
    for name,m in report['suites'].items():
        ci=m['scene_bootstrap_ci95'];base=baseline['suites'][name]['accuracy']
        lines.append(f"| {name} | {m['count']} | {m['accuracy']:.2%} | {ci[0]:.2%}–{ci[1]:.2%} | {base:.2%} |")
    lines+=['','The CNN baseline knows the fixed slot geometry and a limited template vocabulary. Its parser matched all benchmark semantics, including held-out phrases; this is not evidence of unrestricted language understanding.',
        'The scene bootstrap does not fully account for original glyph reuse across scenes. Different test suites share the official test pool.',
        '', '| Set | Korean | English | NONE precision / recall | Both changed instructions correct | NLL | Brier | ECE |',
        '|---|---:|---:|---|---:|---:|---:|---:|']
    for name,m in report['suites'].items():
        lines.append(f"| {name} | {m['language_accuracy']['ko']:.2%} | {m['language_accuracy']['en']:.2%} | {m['none_precision']:.2%} / {m['none_recall']:.2%} | {m['changed_instruction_both_correct']:.2%} | {m['nll']:.3f} | {m['brier']:.3f} | {m['ece_10_bins']:.3f} |")
    lines+=['','## Same image, different instructions','','![Canonical 7, 4, 3 example](docs/canonical.png)','',
            '| Instruction | Expected | Predicted | LEFT | CENTER | RIGHT | NONE |','|---|---|---|---:|---:|---:|---:|']
    for case in diagnostics['canonical']:
        p=case['prediction'];probs=p['probabilities']
        lines.append(f"| {case['instruction']} | {case['gold']} | {p['action']} | "+' | '.join(f'{probs[a]:.1%}' for a in ('LEFT','CENTER','RIGHT','NONE'))+' |')
    lines+=['','## Input dependence','',
            'Accuracy on a fixed diagnostic subset, scored against the original targets. Blank/shuffled inputs remove required information; these are not new valid-task benchmarks.','']
    lines += [f'- {key}: {value:.2%}' for key,value in diagnostics['ablation_accuracy'].items()]
    lines+=['','## Validation history','','| Step | Fixed validation subset accuracy |','|---:|---:|']
    lines += [f"| {v['step']} | {v['accuracy']:.2%} |" for v in history]
    lines+=['','## Representative failures','',
            'Selected by high model confidence among incorrect final test predictions; not a random sample.','']
    failure_dir=ROOT/'docs/failures';failure_dir.mkdir(parents=True,exist_ok=True)
    for name in report['suites']:
        predictions=read(f'results/{name}_predictions.json')
        failures=[p for p in predictions if p['action']!=p['prediction']['action']]
        failures.sort(key=lambda p:max(p['prediction']['probabilities'].values()),reverse=True)
        for row in failures[:3]:
            destination=failure_dir/Path(row['image']).name
            shutil.copy2(ROOT/'data'/row['image'],destination)
            lines.extend([f"### {name}: {row['scene_id']}",'',f"![Failure scene](docs/failures/{destination.name})",'',
                f"Instruction: {row['instruction']}",f"Expected **{row['action']}**; predicted **{row['prediction']['action']}** at {max(row['prediction']['probabilities'].values()):.1%} confidence.",''])
    lines+=['## Scope and platform verification','',
            '- Windows NVIDIA CUDA and Windows CPU: independent-process real inference verified; detailed comparison is in `results/platform-comparison.json`.',
            '- Apple Silicon MPS: implementation provided, physical Mac unavailable and **unverified**.',
            '- A separately fine-tuned text-generation baseline and a head-only ablation were not run; the user requested focusing effort on data and the original model rather than expanding the study.',
            '- No claim that the VLM is necessary, or that the decision head is inherently faster or more accurate than alternatives.',
            '- Full probability outputs, confusion matrices, per-condition accuracy and timings are in `results/`.']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':main()
