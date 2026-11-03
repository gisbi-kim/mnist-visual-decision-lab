"""Compare actual CPU/CUDA predictions for exactly matched inputs."""
import json
import statistics
from pathlib import Path

root=Path(__file__).resolve().parent
cuda=json.loads((root/'results/diagnostics-cuda.json').read_text(encoding='utf-8'))
cpu=json.loads((root/'results/diagnostics-cpu.json').read_text(encoding='utf-8'))
pairs=[]
for a,b in zip(cuda['canonical'],cpu['canonical']):
    assert a['instruction']==b['instruction']
    pairs.append((a['prediction'],b['prediction']))
for a,b in zip(cuda['ablations'],cpu['ablations']):
    assert a['scene_id']==b['scene_id'] and a['instruction']==b['instruction']
    pairs.append((a['predictions']['normal'],b['predictions']['normal']))
report={'examples':len(pairs),'cuda_dtype':cuda['dtype'],'cpu_dtype':cpu['dtype'],
        'max_probability_difference':max(abs(a['probabilities'][k]-b['probabilities'][k]) for a,b in pairs for k in a['probabilities']),
        'action_agreement':sum(a['action']==b['action'] for a,b in pairs)/len(pairs),
        'cpu_median_ms':statistics.median(b['latency_ms'] for a,b in pairs),
        'cuda_median_ms':statistics.median(a['latency_ms'] for a,b in pairs),
        'mps':'UNVERIFIED: no physical Apple Silicon Mac available',
        'note':'Canonical and fixed test-subset inputs, matched weights and processor. Timings include preprocessing and synchronization; warm-up effects can remain in canonical examples.'}
(root/'results/platform-comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
