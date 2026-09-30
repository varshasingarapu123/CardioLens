from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter,filtfilt
import wfdb
from .labels import class_id

def bandpass(signal,fs,low=0.5,high=40.0,order=3):
    nyquist=fs/2; high=min(high,nyquist*0.95)
    if not 0<low<high: raise ValueError('Invalid band-pass frequencies.')
    b,a=butter(order,[low/nyquist,high/nyquist],btype='band')
    return filtfilt(b,a,signal).astype(np.float32)

def normalize(segment):
    x=segment.astype(np.float32); x=x-x.mean(); return x/(x.std()+1e-8)

def load_record(data_dir,record_id):
    base=str(Path(data_dir)/record_id); return wfdb.rdrecord(base),wfdb.rdann(base,'atr')

def extract_beats(data_dir,record_id,channel=0,left=90,right=180):
    record,annotation=load_record(data_dir,record_id); fs=float(record.fs); signal=record.p_signal[:,channel].astype(np.float32); filtered=bandpass(signal,fs); rows=[]
    for sample,symbol in zip(annotation.sample,annotation.symbol):
        cid=class_id(symbol); start=int(sample)-left; end=int(sample)+right
        if cid is None or start<0 or end>len(filtered): continue
        rows.append({'record_id':record_id,'sample':int(sample),'symbol':symbol,'label':cid,'segment':normalize(filtered[start:end]),'fs':fs})
    return rows

def build_dataset(data_dir,record_ids,output_path,channel=0):
    rows=[]
    for rid in record_ids:
        try: rows.extend(extract_beats(data_dir,rid,channel)); print(f'processed {rid}: {len(rows)} total beats')
        except Exception as exc: print(f'SKIP {rid}: {exc}')
    if not rows: raise RuntimeError('No usable ECG segments were produced.')
    X=np.stack([r.pop('segment') for r in rows]).astype(np.float32); meta=pd.DataFrame(rows); output_path=Path(output_path); output_path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output_path,X=X,y=meta.label.to_numpy(),record_id=meta.record_id.to_numpy()); meta.to_csv(output_path.with_suffix('.csv'),index=False); return X,meta
