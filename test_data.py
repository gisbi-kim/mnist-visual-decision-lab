"""Semantic, leakage-boundary and rendering checks."""
import ast
from pathlib import Path
from itertools import product
import numpy as np
from oracle import answer, TASKS
from schema import model_input
from render import render

def test_spec_example():
    digits=[7,4,3];colors=['red','red','blue']
    assert answer(digits,colors,('red','odd',None))=='LEFT'
    assert answer(digits,colors,(None,'odd','min'))=='RIGHT'
    assert answer(digits,colors,(None,'even',None))=='CENTER'
    assert answer(digits,colors,('blue','even',None))=='NONE'

def test_ties_and_empty():
    assert answer([7,7,3],['red']*3,(None,None,'max')) is None
    assert answer([2,4,6],['red']*3,(None,'odd','max'))=='NONE'
    assert answer([7,7,3],['red']*3,(None,None,'min'))=='RIGHT'

def test_position_equivariance():
    actions=['LEFT','CENTER','RIGHT']
    for task in TASKS:
        for digits in product(range(10),repeat=3):
            colors=['red','blue','red']
            a=answer(digits,colors,task)
            b=answer(digits[::-1],colors[::-1],task)
            assert b == (actions[2-actions.index(a)] if a in actions else a)

def test_inference_boundary():
    image=render([np.zeros((28,28),dtype=np.uint8)]*3,['red','blue','red'])
    text='내가 쓴 지시 그대로: 파란 상자의 홀수를 골라'
    payload=model_input(image,text)
    assert payload['state']==text
    assert set(payload)=={'state','images','questions'}
    for filename in ('engine.py','schema.py'):
        tree=ast.parse(Path(filename).read_text(encoding='utf-8'))
        modules=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        assert not {'oracle','prepare_data'} & set(modules)

def test_render_is_label_free_and_repeatable():
    pixels=[np.zeros((28,28),dtype=np.uint8)]*3
    a=render(pixels,['red','red','blue'],1)
    b=render(pixels,['red','red','blue'],1)
    assert a.size==(336,112) and a.tobytes()==b.tobytes()
