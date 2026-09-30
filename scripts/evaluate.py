from pathlib import Path
import json,joblib,numpy as np,torch
from sklearn.metrics import classification_report,confusion_matrix
from models.baseline import features
from models.cnn import ECGCNN

def main():
    z=np.load('data/processed/test.npz',allow_pickle=True); X,y=z['X'],z['y']
    baseline=joblib.load('results/checkpoints/baseline.joblib'); bp=baseline.predict(features(X))
    cnn=ECGCNN(); cnn.load_state_dict(torch.load('results/checkpoints/cnn.pt',map_location='cpu')); cnn.eval()
    with torch.no_grad(): cp=cnn(torch.tensor(X,dtype=torch.float32).unsqueeze(1)).argmax(1).numpy()
    out={'baseline':classification_report(y,bp,output_dict=True,zero_division=0),'cnn':classification_report(y,cp,output_dict=True,zero_division=0),'baseline_confusion_matrix':confusion_matrix(y,bp).tolist(),'cnn_confusion_matrix':confusion_matrix(y,cp).tolist()}
    Path('results').mkdir(exist_ok=True); Path('results/test_evaluation.json').write_text(json.dumps(out,indent=2))

if __name__=='__main__': main()
