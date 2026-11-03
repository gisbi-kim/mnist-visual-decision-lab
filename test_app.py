"""Transport tests use a named test double; the runnable app always loads real weights."""
from app import create_app

class RecordingTestEngine:
    device='cpu';dtype='test-only'
    def predict(self,image,instruction):
        self.received=(image,instruction)
        return {'action':'LEFT','probabilities':{'LEFT':1.,'CENTER':0.,'RIGHT':0.,'NONE':0.},'latency_ms':0.,'device':'cpu'}

def test_user_text_is_forwarded_verbatim():
    engine=RecordingTestEngine()
    client=create_app(engine).test_client()
    scene=client.post('/scene',json={'canonical':True}).json
    text='  내가 직접 쓴 조건: 가장 작은 홀수?  '
    response=client.post('/predict',json={'image':scene['image'],'instruction':text})
    assert response.status_code==200
    assert engine.received[1]==text
    assert engine.received[0].size==(336,112)

def test_home_and_bad_input():
    client=create_app(RecordingTestEngine()).test_client()
    assert client.get('/').status_code==200
    assert client.post('/predict',json={'instruction':''}).status_code==400
    assert client.post('/predict',json={'instruction':'x','image':'not-an-image'}).status_code==400
