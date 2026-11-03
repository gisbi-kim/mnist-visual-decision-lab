"""Package measured best weights for a public release. Does not publish anything."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import zipfile
from schema import ACTIONS, questions

ROOT=Path(__file__).resolve().parent

def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--version',default='v0.1.0');args=p.parse_args()
    report=read('results/report.json');history=read('runs/main/validation.json')
    training=read('runs/main/metrics.json');config=read('runs/main/config.json')
    best=max(history,key=lambda v:v['accuracy'])
    folder=ROOT/'dist'/args.version/'model'
    if folder.exists():raise RuntimeError('Version already staged; preserve it and choose a new staging version')
    shutil.copytree(ROOT/'runs/main/best',folder)
    dump(folder/'inference_schema.json',{'actions':ACTIONS,'questions':questions(),
        'renderer':read('data/summary.json')['renderer'],'input_fields':['image','instruction','questions']})
    shutil.copy2(ROOT/'CLEF_LICENSE.txt',folder/'CLEF_LICENSE.txt')
    shutil.copy2(ROOT/'LICENSE',folder/'LICENSE')
    card=['---','license: apache-2.0','base_model: Qwen/Qwen3.5-0.8B','tags: [mnist, lora, clef, decision-model, vision-language]','---','',
          '# MNIST Visual Decision Lab weights','',
          'Actual LoRA adapter and Clef joint head, trained on bilingual three-digit selection scenes. These files require the pinned base model and project inference code.','',
          f"Base revision: `{config['base_revision']}`.",
          f"Best validation checkpoint: step {best['step']}, {best['accuracy']:.2%} on a fixed 256-row validation subset.",
          f"Training: {training['train_runtime']/60:.1f} minutes on {config['gpu']}; 18,048 training rows, rank/alpha 16, bfloat16 base.",'',
          '| Held-out set | Accuracy |','|---|---:|']
    card += [f"| {name} | {m['accuracy']:.2%} |" for name,m in report['suites'].items()]
    card += ['','See https://github.com/gisbi-kim/mnist-visual-decision-lab for code, limitations, example outputs and failure cases.',
             'CPU and CUDA real inference verified. Apple Silicon MPS is unverified without a physical Mac.',
             'Windows App Control blocked loading PyTorch in a second newly installed environment on the development computer; working-environment and cold-cache inference were verified.',
             'NONE means no qualifying digit, not uncertainty. Probabilities do not guarantee correctness.',
             'Calibration uses a separate calibration partition. Test sets did not select the checkpoint or temperature.']
    (folder/'README.md').write_text('\n'.join(card)+'\n',encoding='utf-8')
    files={f.name:{'bytes':f.stat().st_size,'sha256':sha(f)} for f in folder.iterdir() if f.is_file()}
    name=f'mnist-visual-decision-lab-{args.version}.zip'
    archive=folder.parent/name
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in sorted(folder.iterdir()):
            if f.is_file():z.write(f,f.name)
    manifest={'version':args.version,'url':f'https://github.com/gisbi-kim/mnist-visual-decision-lab/releases/download/{args.version}/{name}',
              'bytes':archive.stat().st_size,'sha256':sha(archive),'files':files,
              'unpacked_bytes':sum(f['bytes'] for f in files.values()),'base_model':config['base_model'],
              'base_revision':config['base_revision'],'base_files':read('docs/base-model-files.json'),
              'selected_step':best['step']}
    dump(ROOT/'release_manifest.json',manifest)
    sizes=read('docs/download-sizes.json')
    sizes['new_weight_archive_bytes']=manifest['bytes']
    sizes['windows_cuda_core_download_total_bytes']=sizes['python_dependencies_download_bytes']+sizes['base_model_download_bytes']+manifest['bytes']
    sizes['note']='Measured package wheels + all pinned base snapshot files + new weight archive. Excludes Python/Git installers, repository transfer and small network metadata. Other platforms differ.'
    dump(ROOT/'docs/download-sizes.json',sizes)
    hardware={'platform':platform.platform(),'python':sys.version,'host':read('docs/environment.json'),
              'nvidia_smi':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv'],text=True),
              'training':training,'configuration':config,'selected_step':best['step']}
    dump(ROOT/'results/training-summary.json',hardware)
    for name in ('history.json','validation.json','requirements.lock.txt'):
        shutil.copy2(ROOT/'runs/main'/name,ROOT/'results'/('training-'+name))
    print(json.dumps({'archive':str(archive),'bytes':manifest['bytes'],'sha256':manifest['sha256'],
                      'selected_step':best['step'],'windows_core_download_bytes':sizes['windows_cuda_core_download_total_bytes']},indent=2))

if __name__=='__main__':main()
