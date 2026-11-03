"""Separate limited-grammar CNN + color detection + rule baseline; never used by the app."""
import json
from pathlib import Path
import random
import time
import numpy as np
from PIL import Image
import torch
from torch import nn
from oracle import answer

ROOT=Path(__file__).resolve().parent

class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Flatten(),
            nn.Linear(32*7*7,128),nn.ReLU(),nn.Linear(128,10))
    def forward(self,x):return self.layers(x)

def pixels(image):
    # Fixed three-slot renderer geometry is part of this baseline's supported input grammar.
    crops=[];colors=[]
    array=np.asarray(image)
    for i in range(3):
        crop=image.crop((i*112+12,12,i*112+100,100)).convert('L').resize((28,28),Image.Resampling.BILINEAR)
        crops.append((255-np.asarray(crop,dtype=np.float32))/255)
        border=array[5:9,i*112+15:i*112+95].mean(axis=(0,1))
        colors.append('red' if border[0]>border[2] else 'blue')
    return torch.tensor(np.stack(crops))[:,None],colors

def parse(text):
    # Only the documented template vocabulary; no claim of unrestricted NL understanding.
    lower=text.lower()
    color='red' if 'red' in lower or '빨간' in text else 'blue' if 'blue' in lower or '파란' in text else None
    parity='odd' if 'odd' in lower or '홀수' in text else 'even' if 'even' in lower or '짝수' in text else None
    extreme='min' if 'smallest' in lower or '가장 작은' in text or '최솟값' in text else 'max' if 'largest' in lower or '가장 큰' in text or '최댓값' in text else None
    if not any((color,parity,extreme)):raise ValueError('Unsupported baseline grammar')
    return color,parity,extreme

def main():
    torch.set_num_threads(4);torch.manual_seed(3407);random.seed(3407)
    model=DigitCNN()
    scenes=json.loads((ROOT/'data/train_scenes.json').read_text(encoding='utf-8'))
    x=[];y=[]
    for scene in scenes:
        with Image.open(ROOT/'data'/scene['image']) as im:batch,_=pixels(im)
        x.extend(batch);y.extend(scene['digits'])
    data=torch.utils.data.TensorDataset(torch.stack(x),torch.tensor(y))
    loader=torch.utils.data.DataLoader(data,batch_size=128,shuffle=True)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    start=time.perf_counter()
    for epoch in range(8):
        model.train();hits=0
        for images,labels in loader:
            optimizer.zero_grad();logits=model(images);loss=nn.functional.cross_entropy(logits,labels)
            loss.backward();optimizer.step();hits+=int((logits.argmax(-1)==labels).sum())
        print('CNN',epoch+1,hits/len(data),flush=True)
    runtime=time.perf_counter()-start
    model.eval()
    out=ROOT/'results/baseline';out.mkdir(parents=True,exist_ok=True)
    torch.save(model.state_dict(),out/'cnn.pt')
    report={'type':'CNN + border color + limited vocabulary rules','device':'cpu','train_seconds':runtime,
            'training_glyph_occurrences':len(data),'epochs':8,'suites':{},
            'limitations':'Known three-slot layout and template vocabulary. Deterministic rule outputs are not calibrated probabilities; NLL/ECE are not reported.'}
    for name in ('test_id','test_language','test_composition','test_combined'):
        rows=[json.loads(s) for s in (ROOT/'data'/f'{name}.jsonl').read_text(encoding='utf-8').splitlines()]
        predictions=[];parse_correct=0
        for row in rows:
            then=time.perf_counter()
            with Image.open(ROOT/'data'/row['image']) as im:batch,colors=pixels(im)
            with torch.no_grad():digits=model(batch).argmax(-1).tolist()
            task=parse(row['instruction'])
            parse_correct+=tuple(row['task'])==task
            action=answer(digits,colors,task)
            # Recognition can create a tie; emit INVALID and count as an error separately.
            predictions.append(dict(gold=row['action'],pred=action,task_name=row['task_name'],language=row['language'],
                                    latency_ms=(time.perf_counter()-then)*1000))
        hits=[p['gold']==p['pred'] for p in predictions]
        report['suites'][name]={'count':len(rows),'accuracy':float(np.mean(hits)),
            'invalid_ties':sum(p['pred'] is None for p in predictions),'parser_accuracy':parse_correct/len(rows),
            'median_latency_ms':float(np.median([p['latency_ms'] for p in predictions])),
            'latency_scope':'Disk image read, crop preprocessing, CNN, color detection and rule execution; CPU'}
        print(name,report['suites'][name],flush=True)
        (out/f'{name}.json').write_text(json.dumps(predictions,ensure_ascii=False),encoding='utf-8')
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

if __name__=='__main__':main()
