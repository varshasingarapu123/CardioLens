import numpy as np

def split_records(record_ids,train=0.7,val=0.15,seed=42):
    ids=np.array(sorted(set(record_ids)))
    if len(ids)<3: raise ValueError('At least 3 records are required.')
    rng=np.random.default_rng(seed); rng.shuffle(ids); n=len(ids)
    n_train=max(1,int(n*train)); n_val=max(1,int(n*val))
    if n_train+n_val>=n: n_train=max(1,n-2); n_val=1
    return ids[:n_train].tolist(),ids[n_train:n_train+n_val].tolist(),ids[n_train+n_val:].tolist()

def assert_disjoint(*groups):
    sets=[set(g) for g in groups]
    for i in range(len(sets)):
        for j in range(i+1,len(sets)):
            if sets[i]&sets[j]: raise AssertionError('Record leakage detected between splits.')
