"""Local web UI. User text is forwarded verbatim to the actual decision model."""
import argparse
import base64
from io import BytesIO
import json
from pathlib import Path
import random
import threading
import numpy as np
from PIL import Image, UnidentifiedImageError
from flask import Flask, jsonify, render_template, request
from render import render

ROOT=Path(__file__).resolve().parent

def create_app(engine):
    app=Flask(__name__)
    app.json.sort_keys=False
    app.config['MAX_CONTENT_LENGTH']=3*1024*1024
    lock=threading.Lock()
    examples=np.load(ROOT/'examples.npz',allow_pickle=False)
    rng=random.Random()
    @app.get('/')
    def index():return render_template('index.html',device=str(engine.device),dtype=str(engine.dtype),
        fallback=getattr(engine,'cpu_fallback_enabled',False))
    @app.post('/scene')
    def scene():
        canonical=(request.get_json(silent=True) or {}).get('canonical',False)
        if canonical:
            ids=[int(np.flatnonzero(examples['labels']==d)[0]) for d in (7,4,3)]
            colors=['red','red','blue']
        else:
            ids=[rng.randrange(len(examples['images'])) for _ in range(3)]
            colors=[rng.choice(('red','blue')) for _ in range(3)]
        image=render(examples['images'][ids],colors,0 if canonical else rng.randrange(2**31))
        buffer=BytesIO();image.save(buffer,format='PNG')
        return jsonify(image='data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode())
    @app.post('/predict')
    def predict():
        payload=request.get_json()
        text=payload.get('instruction','')
        if not isinstance(text,str) or not text.strip() or len(text)>1000:
            return jsonify(error='지시를 1~1000자로 입력해 주세요.'),400
        try:
            data=payload['image'].split(',',1)[1]
            image=Image.open(BytesIO(base64.b64decode(data,validate=True)))
            if image.width*image.height>1500000:raise ValueError('이미지가 너무 큽니다.')
            with lock: prediction=engine.predict(image,text)
            return jsonify(prediction)
        except (ValueError,KeyError,IndexError,UnidentifiedImageError) as exc:return jsonify(error=str(exc)),400
        except RuntimeError as exc:
            hint=' CPU 실행 방법: python app.py --device cpu' if str(engine.device)!='cpu' else ''
            return jsonify(error=str(exc)+hint),500
    return app

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',default='weights')
    parser.add_argument('--device',default='auto',choices=['auto','cuda','cpu','mps'])
    parser.add_argument('--port',type=int,default=7860)
    args=parser.parse_args()
    if not Path(args.checkpoint).exists():
        raise SystemExit('Weights are missing. Run: python download_model.py')
    print('Loading Qwen + trained LoRA + Clef head. The first run may download the base model.',flush=True)
    from engine import DecisionEngine
    engine=DecisionEngine(args.checkpoint,args.device)
    print(f'Ready: {engine.device}, {engine.dtype}; http://127.0.0.1:{args.port}',flush=True)
    create_app(engine).run(host='127.0.0.1',port=args.port,debug=False,threaded=True)

if __name__=='__main__':main()
