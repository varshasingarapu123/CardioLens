from pathlib import Path
import json
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from models.cnn import ECGCNN
from preprocessing.labels import CLASS_NAMES
from explainability.attribution import gradient_attribution

def main():
    z=np.load('data/processed/test.npz',allow_pickle=True)
    X,y=z['X'],z['y']
    model=ECGCNN()
    model.load_state_dict(torch.load('results/checkpoints/cnn.pt',map_location='cpu'))
    model.eval()
    with torch.no_grad():
        logits=model(torch.tensor(X,dtype=torch.float32).unsqueeze(1))
        probs=torch.softmax(logits,dim=1).numpy()
        pred=probs.argmax(1)

    report=classification_report(y,pred,output_dict=True,zero_division=0)

    # Export several real held-out test beats so judges can interact with
    # multiple model outputs without pretending the static site is a live API.
    sample_indices=[]
    for cls in range(5):
        idx=np.flatnonzero(y==cls)
        if len(idx):
            sample_indices.append(int(idx[0]))
    sample_indices=sample_indices[:5]

    samples=[]
    for sample_index in sample_indices:
        target,attr=gradient_attribution(model,X[sample_index])
        samples.append({
            'index':sample_index,
            'true_class':CLASS_NAMES[int(y[sample_index])],
            'predicted_class':CLASS_NAMES[int(target)],
            'confidence':float(probs[sample_index,target]),
            'waveform':[round(float(v),6) for v in X[sample_index].tolist()],
            'attribution':[round(float(v),6) for v in attr.tolist()]
        })

    first=samples[0]
    payload={
        'status':'trained-evaluation',
        'dataset':'MIT-BIH Arrhythmia Database',
        'sample_index':first['index'],
        'sample_true_class':first['true_class'],
        'sample_predicted_class':first['predicted_class'],
        'sample_confidence':first['confidence'],
        'sample_waveform':first['waveform'],
        'sample_attribution':first['attribution'],
        'samples':samples,
        'accuracy':float(accuracy_score(y,pred)),
        'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
        'class_names':CLASS_NAMES,
        'classification_report':report
    }
    out=Path('docs/data'); out.mkdir(parents=True,exist_ok=True)
    (out/'results.json').write_text(json.dumps(payload))
    print(json.dumps({'accuracy':payload['accuracy'],'macro_f1':payload['macro_f1'],'samples':len(samples)}))

if __name__=='__main__':
    main()
