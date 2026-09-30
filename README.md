# CardioLens — From Raw ECG Signals to Interpretable Evidence

CardioLens is an explainable ECG intelligence research prototype that turns public ECG signals into reproducible machine-learning experiments and signal-level evidence.

Medical safety: this is a research/educational prototype, not a medical device and not for diagnosis or clinical decision-making.

## Pipeline
PhysioNet MIT-BIH -> WFDB ingestion -> validation/filtering -> beat segmentation -> record-level split -> classical baseline + 1D CNN -> held-out evaluation -> gradient-based attribution -> FastAPI.

## Dataset
The default dataset is the MIT-BIH Arrhythmia Database from PhysioNet. The dataset is not redistributed in this repository.

## Quick start
1. Create a Python virtual environment.
2. Install requirements.txt.
3. Run scripts/download_data.py.
4. Run scripts/prepare_data.py.
5. Train the baseline and CNN.
6. Run scripts/evaluate.py and scripts/explain.py.
7. Start the API with uvicorn backend.app:app --reload.

## Research design
- Fixed random seeds for reproducibility.
- Record-level train/validation/test separation to reduce beat-level leakage.
- Classical logistic-regression baseline using deterministic waveform statistics.
- Compact 1D CNN operating directly on ECG beat windows.
- Precision, recall, F1 and confusion matrices generated from actual runs.
- Gradient-based temporal attribution for the trained CNN.
- No hard-coded performance numbers.

## Limitations
MIT-BIH is a historical research dataset and does not represent every acquisition environment or patient population. Dataset performance must not be interpreted as clinical performance.

## Team
- Varsha Singarapu
- Vishnupriya Singarapu
