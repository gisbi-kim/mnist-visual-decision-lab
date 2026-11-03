"""Actual example predictions, input ablations and cross-device comparison."""
import argparse
import json
from pathlib import Path
import random
import numpy as np
import psutil
import torch
from PIL import Image
from engine import DecisionEngine
from oracle import instruction
from render import render

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--output',default='results/diagnostics-cuda.json');p.add_argument('--count',type=int,default=64)
    args=p.parse_args()
    engine=DecisionEngine(args.checkpoint,args.device)
    arrays=np.load(ROOT/'examples.npz')
    ids=[int(np.flatnonzero(arrays['labels']==d)[0]) for d in (7,4,3)]
    image=render(arrays['images'][ids],['red','red','blue'],0)
    (ROOT/'docs').mkdir(exist_ok=True);image.save(ROOT/'docs/canonical.png')
    cases=[(('red','odd',None),'LEFT'),((None,'odd','min'),'RIGHT'),((None,'even',None),'CENTER'),(('blue','even',None),'NONE')]
    canonical=[]
    for task,gold in cases:
        for language in ('ko','en'):
            text=instruction(task,language,0)
            canonical.append(dict(instruction=text,gold=gold,language=language,prediction=engine.predict(image,text)))
    rows=[json.loads(s) for s in (ROOT/'data/test_id.jsonl').read_text(encoding='utf-8').splitlines()]
    # Fixed subset; no model selection uses these final diagnostics.
    selected=random.Random(413).sample(rows,min(args.count,len(rows)))
    ablations=[]
    for i,row in enumerate(selected):
        with Image.open(ROOT/'data'/row['image']) as im:
            normal=im.convert('RGB')
        with Image.open(ROOT/'data'/selected[(i+1)%len(selected)]['image']) as im:
            swapped=im.convert('RGB')
        predictions={
            'normal':engine.predict(normal,row['instruction']),
            'blank_image':engine.predict(Image.new('RGB',normal.size,'white'),row['instruction']),
            'shuffled_image':engine.predict(swapped,row['instruction']),
            'blank_instruction':engine.predict(normal,'Choose the correct option.')}
        ablations.append(dict(scene_id=row['scene_id'],instruction=row['instruction'],gold=row['action'],predictions=predictions))
        if (i+1)%16==0:print('DIAGNOSTICS',i+1,flush=True)
    memory=psutil.Process().memory_info()
    report=dict(device=str(engine.device),dtype=str(engine.dtype),canonical=canonical,ablations=ablations,
        process_rss_bytes=memory.rss,process_peak_working_set_bytes=getattr(memory,'peak_wset',None),
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if engine.device.type=='cuda' else None,
        ablation_accuracy={k:sum(r['predictions'][k]['action']==r['gold'] for r in ablations)/len(ablations) for k in predictions},
        note='Ablation accuracy is against the original target. Blank/shuffled inputs deliberately remove required information; this is not a new valid-task benchmark.')
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('DIAGNOSTICS_SAVED',out,flush=True)

if __name__=='__main__':main()
