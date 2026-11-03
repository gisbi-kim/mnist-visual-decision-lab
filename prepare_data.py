"""Reproducible image-disjoint, task/action/language-balanced bilingual data."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import random
import struct
import urllib.request
import numpy as np
from PIL import Image, ImageDraw
from oracle import TASKS, HELD_OUT, answer, instruction, task_name
from render import render
from schema import ACTIONS

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
FILES = {'train-images-idx3-ubyte.gz':'f68b3c2dcbeaaa9fbdd348bbdeb94873',
         'train-labels-idx1-ubyte.gz':'d53e105ee54ea40749a09fcbcd1e9432',
         't10k-images-idx3-ubyte.gz':'9fb629c4189551a2d022fa330f9573f3',
         't10k-labels-idx1-ubyte.gz':'ec29112dd5afa0611ce80d1b7f02629c'}

def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def download():
    if (DATA/'mnist.npz').exists():
        return dict(np.load(DATA/'mnist.npz',allow_pickle=False))
    arrays = {}
    for name,md5 in FILES.items():
        path = DATA/'raw'/name
        path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            urllib.request.urlretrieve('https://ossci-datasets.s3.amazonaws.com/mnist/'+name,path)
        content = path.read_bytes()
        assert hashlib.md5(content).hexdigest() == md5
        raw = gzip.decompress(content)
        magic,n = struct.unpack('>II',raw[:8])
        array = np.frombuffer(raw[16 if magic==2051 else 8:],dtype=np.uint8)
        key = ('train_' if name.startswith('train') else 'test_') + ('images' if magic==2051 else 'labels')
        arrays[key] = array.reshape(n,28,28) if magic==2051 else array
    np.savez_compressed(DATA/'mnist.npz',**arrays)
    return arrays

def splits(arrays):
    rng = random.Random(3407)
    groups = {k:[] for k in ('train','validation','calibration')}
    for d in range(10):
        ids = np.flatnonzero(arrays['train_labels']==d).tolist()
        rng.shuffle(ids)
        groups['validation'] += ids[:400]
        groups['calibration'] += ids[400:600]
        groups['train'] += ids[600:]
    groups['test'] = list(range(len(arrays['test_labels'])))
    for a in ('train','validation','calibration'):
        for b in ('train','validation','calibration'):
            if a != b: assert not set(groups[a]) & set(groups[b])
    return groups

def generate(arrays, ids, name, tasks, quota, variants, seed):
    rng = random.Random(seed)
    source = 'test' if name.startswith('test') else 'train'
    by_digit = {d:[i for i in ids if int(arrays[source+'_labels'][i]) == d] for d in range(10)}
    wanted = {(task_name(t),lang,a):quota for t in tasks for lang in ('ko','en') for a in ACTIONS
              if not (a=='NONE' and t[0] is None and t[1] is None)}
    counts = Counter()
    rows, scenes = [],[]
    attempts = 0
    while any(counts[k] < v for k,v in wanted.items()):
        attempts += 1
        if attempts > quota*20000: raise RuntimeError('Could not fill balanced quotas')
        digits = [rng.randrange(10) for _ in range(3)]
        colors = [rng.choice(('red','blue')) for _ in range(3)]
        candidates = []
        for task in tasks:
            action = answer(digits,colors,task)
            if action is None: continue
            for language in ('ko','en'):
                key = (task_name(task),language,action)
                if counts[key] < wanted[key]:
                    candidates.append((task,language,action,key))
        rng.shuffle(candidates)
        candidates.sort(key=lambda c:counts[c[3]])
        selected = candidates[:8]
        if not selected: continue
        scene_id = f'{name}-{len(scenes):05d}'
        raw_ids = [rng.choice(by_digit[d]) for d in digits]
        scene = dict(scene_id=scene_id,source=source,image_ids=raw_ids,digits=digits,colors=colors,
                     render_seed=rng.randrange(2**31),image=f'images/{scene_id}.png')
        image = render([arrays[source+'_images'][i] for i in raw_ids], colors,scene['render_seed'])
        (DATA/'images').mkdir(exist_ok=True)
        image.save(DATA/scene['image'])
        scenes.append(scene)
        for task,language,action,key in selected:
            variant = rng.choice(variants)
            rows.append(dict(scene_id=scene_id,image=scene['image'],instruction=instruction(task,language,variant),
                             action=action,task=list(task),task_name=task_name(task),language=language,variant=variant))
            counts[key] += 1
    rng.shuffle(rows)
    with (DATA/f'{name}.jsonl').open('w',encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False)+'\n')
    dump(DATA/f'{name}_scenes.json',scenes)
    digit_counts = Counter(d for s in scenes for d in s['digits'])
    scene_answers = defaultdict(set)
    for row in rows: scene_answers[row['scene_id']].add(row['action'])
    summary = dict(rows=len(rows),scenes=len(scenes),source=source,seed=seed,
                   unique_source_images=len({i for s in scenes for i in s['image_ids']}),
                   actions=dict(Counter(r['action'] for r in rows)),digits=dict(digit_counts),
                   per_task_language_action={'|'.join(k):v for k,v in sorted(counts.items())},
                   slot_digit={str(i):dict(Counter(s['digits'][i] for s in scenes)) for i in range(3)},
                   slot_color={str(i):dict(Counter(s['colors'][i] for s in scenes)) for i in range(3)},
                   multi_instruction_scenes=sum(n>1 for n in Counter(r['scene_id'] for r in rows).values()),
                   different_answer_scenes=sum(len(v)>1 for v in scene_answers.values()))
    print(name,summary['rows'],'rows',summary['scenes'],'scenes',summary['actions'],flush=True)
    return rows,scenes,summary

def validate(all_data, split_ids):
    used = defaultdict(set)
    split_ids = {k:set(v) for k,v in split_ids.items()}
    for name,(rows,scenes,_) in all_data.items():
        scene_by_id = {s['scene_id']:s for s in scenes}
        for row in rows:
            s = scene_by_id[row['scene_id']]
            assert row['action'] == answer(s['digits'],s['colors'],row['task'])
            assert (DATA/row['image']).is_file()
            expected_split = 'test' if name.startswith('test') else name
            assert set(s['image_ids']) <= split_ids[expected_split]
            used[expected_split].update((s['source'],i) for i in s['image_ids'])
        if name in ('train','validation','calibration','test_id','test_language'):
            assert all(tuple(r['task']) not in HELD_OUT for r in rows)
    names = list(used)
    for i,a in enumerate(names):
        for b in names[i+1:]: assert not used[a] & used[b],(a,b)
    assert not {r['instruction'] for r in all_data['train'][0]} & {r['instruction'] for r in all_data['test_language'][0]}
    return {'image_disjoint_splits':True,'oracle_labels_checked':True,'heldout_templates_disjoint':True,
            'heldout_combinations_absent_from_training':True}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-quota',type=int,default=96)
    args = parser.parse_args()
    DATA.mkdir(exist_ok=True)
    arrays = download()
    ids = splits(arrays)
    dump(DATA/'splits.json',ids)
    known = [t for t in TASKS if t not in HELD_OUT]
    configs = [('train',known,args.train_quota,[0,1,2,3]),('validation',known,6,[0,1,2,3]),
               ('calibration',known,4,[0,1,2,3]),('test_id',known,8,[0,1,2,3]),
               ('test_language',known,8,[4,5]),('test_composition',sorted(HELD_OUT),64,[0,1,2,3]),
               ('test_combined',sorted(HELD_OUT),64,[4,5])]
    result = {}
    for n,(name,tasks,q,variants) in enumerate(configs):
        result[name] = generate(arrays,ids['test' if name.startswith('test') else name],name,tasks,q,variants,3407+n)
    summary = {k:v[2] for k,v in result.items()}
    summary['checks'] = validate(result,ids)
    summary['heldout_combinations'] = sorted(HELD_OUT)
    summary['renderer'] = {'size':[336,112],'digit_sizes':[76,80,84],'border_widths':[5,6,7], 'interpolation':'NEAREST'}
    summary['note'] = 'NONE is balanced within feasible task/language cells; unfiltered min/max cannot yield NONE. Test suites share the official test pool, never training images.'
    dump(DATA/'summary.json',summary)
    preview = Image.new('RGB',(672,112*6),'white')
    for i,s in enumerate(result['train'][1][:12]):
        preview.paste(Image.open(DATA/s['image']),(i%2*336,i//2*112))
    preview.save(DATA/'preview.png')
    print(json.dumps(summary['checks']),flush=True)

if __name__ == '__main__': main()
