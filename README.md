# Peptide Absorbance Prediction

Machine learning pipeline for predicting peptide absorbance values from amino acid sequence features. Developed as part of the **CASFER REU 2025** research program at Case Western Reserve University.

## Project Overview

This project predicts the absorbance of 15-mer peptide sequences using sequence-derived biophysical features. The dataset consists of 96 peptide sequences measured across two independent experimental trials, each providing absorbance readings at the last 20 time points.

### Key Results

- **Random Forest regression** achieves R² ≈ 0.86 in cross-trial validation (train on Trial 1, test on Trial 2)
- **Binary classification** (high vs low absorbance) achieves strong F1 scores using an ensemble voting classifier
- **Top predictive features**: negative charge ratio, glutamic acid content, flexibility, instability index

## Repository Structure

```
casfer-sdle-repo/
├── data/                          # Experimental data files
│   ├── peptide_csv1_last20.csv    # Trial 1 absorbance data
│   ├── peptide2_csv1_last20.csv   # Trial 2 absorbance data
│   └── Data_Vertical_Trial1_FORSDLE.xlsx
├── notebooks/                     # Analysis notebooks
│   ├── 01_eda_and_feature_engineering.ipynb
│   ├── 02_regression_models.ipynb
│   ├── 03_classification_and_clustering.ipynb
│   └── 04_cross_trial_validation.ipynb
├── src/                           # Reusable Python modules
│   ├── __init__.py
│   ├── feature_extraction.py      # Sequence feature extraction
│   ├── data_loading.py            # Data loading utilities
│   ├── models.py                  # Model definitions & training
│   └── visualization.py           # Plotting utilities
├── requirements.txt
├── .gitignore
└── README.md
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_eda_and_feature_engineering` | Load data, extract features, explore distributions and correlations |
| `02_regression_models` | Train Ridge, RF, GBR regressors; dipeptide + PCA hybrid features; stacking ensemble |
| `03_classification_and_clustering` | Binary classification (high/low absorbance), KMeans clustering, cluster-based prediction |
| `04_cross_trial_validation` | Train on Trial 1 / test on Trial 2 (and vice versa) for cross-experiment generalization |

## Setup

```bash
# Clone the repo
git clone https://github.com/Aartikkk/peptide-absorbance-prediction.git
cd peptide-absorbance-prediction

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Launch notebooks
jupyter notebook notebooks/
```

## Features Extracted

The pipeline extracts the following feature categories from peptide sequences:

- **Biophysical**: GRAVY, isoelectric point, molecular weight, instability index, aromaticity, secondary structure fractions, flexibility
- **Compositional**: Individual amino acid percentages, group ratios (hydrophobic, charged, polar, aromatic)
- **Positional**: N-terminal and C-terminal charge, hydrophobicity, polarity
- **Physicochemical**: Kyte-Doolittle hydrophobicity, Van der Waals volume
- **Dipeptide**: 400 normalized dipeptide frequency features (optional, PCA-compressed)

## Author

Aarti Khatri — CASFER REU 2025, Case Western Reserve University
