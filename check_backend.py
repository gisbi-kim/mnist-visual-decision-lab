"""Check trained-checkpoint parity in separate Unsloth and portable processes."""
import os
os.environ.setdefault('UNSLOTH_COMPILE_DISABLE','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image
from schema import ACTIONS, questions

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--backend',choices=['unsloth','portable'],required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    if args.backend=='unsloth':
        from unsloth import FastDecisionModel
        import torch
        model,processor=FastDecisionModel.from_pretrained(args.checkpoint,load_in_4bit=False,dtype=torch.bfloat16)
        FastDecisionModel.for_inference(model)
        def predict(image,text):
            result=FastDecisionModel.predict(model,processor,text,questions(),images=[image])['select']
            return result
    else:
        from engine import DecisionEngine
        engine=DecisionEngine(args.checkpoint,'cuda')
        def predict(image,text):
            # Compare raw softmax before the portable external calibration file.
            probs=engine.logits(image,text).softmax(-1).cpu().tolist()
            return {'answer':ACTIONS[max(range(4),key=lambda i:probs[i])],
                    'probabilities':dict(zip(ACTIONS,probs))}
    rows=[json.loads(s) for s in (ROOT/'data/validation.jsonl').read_text(encoding='utf-8').splitlines()][:16]
    outputs=[]
    for row in rows:
        with Image.open(ROOT/'data'/row['image']) as im:
            result=predict(im.convert('RGB'),row['instruction'])
        outputs.append({'scene_id':row['scene_id'],'instruction':row['instruction'],'gold':row['action'],'prediction':result})
    hashes={name:hashlib.sha256((Path(args.checkpoint)/name).read_bytes()).hexdigest()
            for name in ('adapter_model.safetensors','joint_head.safetensors')}
    Path(args.output).write_text(json.dumps({'backend':args.backend,'weight_sha256':hashes,
        'temperature_scope':'Before external calibration.json; saved internal temperatures are one.',
        'rows':outputs},ensure_ascii=False,indent=2),encoding='utf-8')
    print('BACKEND_CHECK_SAVED',args.output,flush=True)

if __name__=='__main__': main()
