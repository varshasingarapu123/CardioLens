from pathlib import Path
import json
import numpy as np
import wfdb
from preprocessing.pipeline import build_dataset
from preprocessing.split import split_records, assert_disjoint

RAW=Path('data/raw/mitdb')
OUT=Path('data/processed')
MAX_PER_CLASS=600

def cap_dataset(path):
    z=np.load(path,allow_pickle=True)
    X,y=z['X'],z['y']
    rng=np.random.default_rng(42)
    keep=[]
    for cls in range(5):
        idx=np.flatnonzero(y==cls)
        if len(idx)>MAX_PER_CLASS:
            idx=rng.choice(idx,MAX_PER_CLASS,replace=False)
        keep.extend(idx.tolist())
    keep=np.array(sorted(keep))
    np.savez_compressed(path,X=X[keep],y=y[keep],record_id=z['record_id'][keep])
    csv=path.with_suffix('.csv')
    if csv.exists():
        import pandas as pd
        meta=pd.read_csv(csv)
        meta.iloc[keep].to_csv(csv,index=False)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    records=wfdb.get_record_list('mitdb')
    train_ids,val_ids,test_ids=split_records(records,seed=42)
    assert_disjoint(train_ids,val_ids,test_ids)
    splits={'train':train_ids,'validation':val_ids,'test':test_ids}
    (OUT/'splits.json').write_text(json.dumps(splits,indent=2))
    for name,ids in splits.items():
        path=OUT/f'{name}.npz'
        build_dataset(RAW,ids,path)
        cap_dataset(path)
        z=np.load(path,allow_pickle=True)
        print(f'{name}: {len(z["y"])} beats; classes={np.bincount(z["y"],minlength=5).tolist()}')

if __name__=='__main__':
    main()
