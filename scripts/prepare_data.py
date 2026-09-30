from pathlib import Path
import json,wfdb
from preprocessing.pipeline import build_dataset
from preprocessing.split import split_records,assert_disjoint
RAW=Path('data/raw/mitdb'); OUT=Path('data/processed')
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    records=wfdb.get_record_list('mitdb')
    train_ids,val_ids,test_ids=split_records(records,seed=42)
    assert_disjoint(train_ids,val_ids,test_ids)
    splits={'train':train_ids,'validation':val_ids,'test':test_ids}
    (OUT/'splits.json').write_text(json.dumps(splits,indent=2))
    for name,ids in splits.items(): build_dataset(RAW,ids,OUT/f'{name}.npz')
if __name__=='__main__': main()
