from pathlib import Path
import json
from fastapi import FastAPI, HTTPException

app=FastAPI(title='CardioLens API',description='Research API for the CardioLens ECG intelligence prototype.',version='0.1.0')

@app.get('/')
def root():
    return {'name':'CardioLens','status':'research-prototype','message':'From raw ECG signals to interpretable evidence.'}

@app.get('/health')
def health():
    return {'ok':True}

@app.get('/results')
def results():
    path=Path('results/test_evaluation.json')
    if not path.exists():
        raise HTTPException(status_code=404,detail='No evaluation results exist yet. Run training and evaluation first.')
    return json.loads(path.read_text())
