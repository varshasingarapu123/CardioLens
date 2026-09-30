from pathlib import Path
import numpy as np,torch,matplotlib.pyplot as plt
from models.cnn import ECGCNN
from explainability.attribution import gradient_attribution

def main(index=0):
    z=np.load('data/processed/test.npz',allow_pickle=True); X,y=z['X'],z['y']
    model=ECGCNN(); model.load_state_dict(torch.load('results/checkpoints/cnn.pt',map_location='cpu'))
    target,attr=gradient_attribution(model,X[index])
    Path('results/figures').mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(12,4)); ax.plot(X[index]); ax.set_title(f'Model attribution | true={int(y[index])} | predicted={target}'); ax.set_xlabel('Sample'); ax.set_ylabel('Normalized ECG amplitude'); fig.tight_layout(); fig.savefig('results/figures/attribution.png',dpi=180); plt.close(fig)

if __name__=='__main__': main()
