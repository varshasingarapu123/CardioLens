from pathlib import Path
import json,numpy as np,torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
from models.cnn import ECGCNN

def load(split):
    z=np.load(Path('data/processed')/f'{split}.npz',allow_pickle=True)
    return torch.tensor(z['X'],dtype=torch.float32).unsqueeze(1),torch.tensor(z['y'],dtype=torch.long)

def main(epochs=10,batch_size=128,seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    X,y=load('train'); Xv,yv=load('validation')
    train_loader=DataLoader(TensorDataset(X,y),batch_size=batch_size,shuffle=True)
    val_loader=DataLoader(TensorDataset(Xv,yv),batch_size=batch_size)
    model=ECGCNN(); counts=torch.bincount(y,minlength=5).float()
    loss_fn=nn.CrossEntropyLoss(weight=counts.sum()/(5*counts.clamp_min(1)))
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    best=float('inf'); bad=0; history=[]
    Path('results/checkpoints').mkdir(parents=True,exist_ok=True)
    for epoch in range(1,epochs+1):
        model.train(); total=0
        for xb,yb in train_loader:
            opt.zero_grad(); loss=loss_fn(model(xb),yb); loss.backward(); opt.step(); total+=loss.item()*len(xb)
        train_loss=total/len(X); model.eval(); total=0
        with torch.no_grad():
            for xb,yb in val_loader: total+=loss_fn(model(xb),yb).item()*len(xb)
        val_loss=total/len(Xv); history.append({'epoch':epoch,'train_loss':train_loss,'val_loss':val_loss})
        if val_loss<best: best=val_loss; bad=0; torch.save(model.state_dict(),'results/checkpoints/cnn.pt')
        else:
            bad+=1
            if bad>=3: break
    Path('results/cnn_history.json').write_text(json.dumps(history,indent=2))

if __name__=='__main__': main()
