from pathlib import Path
import json,joblib,numpy as np
from sklearn.metrics import classification_report
from models.baseline import features,make_model
def main():
    z=np.load('data/processed/train.npz',allow_pickle=True); zv=np.load('data/processed/validation.npz',allow_pickle=True)
    model=make_model(); model.fit(features(z['X']),z['y'])
    report=classification_report(zv['y'],model.predict(features(zv['X'])),output_dict=True,zero_division=0)
    Path('results/checkpoints').mkdir(parents=True,exist_ok=True)
    joblib.dump(model,'results/checkpoints/baseline.joblib')
    Path('results/baseline_validation.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__': main()
