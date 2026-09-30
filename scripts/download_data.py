from pathlib import Path
import wfdb
OUT=Path('data/raw/mitdb')
def main():
    OUT.mkdir(parents=True,exist_ok=True); print('Downloading MIT-BIH Arrhythmia Database...'); wfdb.dl_database('mitdb',dl_dir=str(OUT)); print(f'Dataset downloaded to {OUT}')
if __name__=='__main__': main()
