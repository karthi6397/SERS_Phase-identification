# Deep Learning Models for Microalgal Growth-Phase Classification from SERS Spectral Data

Supplementary code for classifying microalgal growth phase (**Lag**, **Log**, **Stationary**) from Surface-Enhanced Raman Spectroscopy (SERS) spectra using four deep learning architectures, evaluated under a common train/validation/test protocol with a follow-up robustness analysis across multiple split ratios.

## Models Implemented

| # | Model | Framework | Section |
|---|-------|-----------|---------|
| 1 | Autoencoder (unsupervised feature extraction) + feed-forward classifier | TensorFlow / Keras | `Section 1` |
| 2 | 1D Convolutional Neural Network (1D-CNN) | PyTorch | `Section 2` |
| 3 | 1D Residual Network (1D-ResNet) | PyTorch | `Section 3` |
| 4 | Transformer Encoder | PyTorch | `Section 4` |

Each section is **self-contained** — it can be run independently, provided `Merged_1_PL.csv` is present in the working directory and the relevant dependencies are installed.

## Data

**Input file:** `Merged_1_PL.csv` — a SERS spectral intensity matrix. Columns are named `Day<N>_Sample<M>`, identifying the cultivation day (`N`) and biological replicate (`M`) for each spectrum.

**Label assignment** (from acquisition day):

| Growth phase | Day range | Label |
|---|---|---|
| Lag | day ≤ 8 | 0 |
| Log | 9 ≤ day ≤ 20 | 1 |
| Stationary | day > 20 | 2 |

The raw data file is not included in this repository — place your own `Merged_1_PL.csv` in the working directory before running any section.

## Preprocessing (applied consistently across all four models)

1. **Savitzky–Golay filtering** — window length 25, polynomial order 2, 1st derivative
2. **Standard Normal Variate (SNV)** normalization
3. **Z-score standardization** (`StandardScaler`), where used

## Evaluation Protocol

- **Primary split:** 70:15:15 (train:validation:test), stratified by class, `random_state = 42`
- **Robustness analysis:** each model is re-trained and re-evaluated across seven additional train:test split ratios — 50:50, 60:40, 65:35, 70:30, 75:25, 80:20, 90:10 — to assess sensitivity to sample size
- **Reproducibility:** all random seeds fixed at `SEED = 42` (Python, NumPy, TensorFlow, and PyTorch, including CUDA/cuDNN determinism flags where applicable)

## Outputs

Each section exports:

**Quantitative results (Excel)**
- Accuracy, macro-averaged precision/recall/F1, Cohen's kappa
- ROC curve data (FPR/TPR per class, per split)
- Per-epoch training/validation loss and accuracy
- Training time, test/inference time, and peak memory usage (via `psutil`)
- Per-split-ratio performance tables from the robustness analysis (one workbook per ratio)

**Figures (PNG, 300–400 dpi)**
- Confusion matrix (test set, publication-styled with `seaborn`)
- Accuracy/loss vs. epoch curves

## Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Dependencies

```
numpy
pandas
scipy
scikit-learn
matplotlib
seaborn
psutil
tqdm                # used in Section 2
tensorflow          # Section 1 (autoencoder + classifier)
torch                # Sections 2–4 (CNN, ResNet, Transformer)
openpyxl             # Excel export (Section 4)
xlsxwriter           # Excel export (Section 1)
```

Pin exact versions in `requirements.txt` once your environment is finalized — TensorFlow/PyTorch version drift is a common source of non-reproducible results even with fixed seeds.

## Usage

1. Place THE DATASET in the working directory.
2. Run each of the four sections (independently or in sequence). Each section:
   - Loads and preprocesses the spectra,
   - Trains on the 70:15:15 split and exports metrics/plots for that split,
   - Re-runs training across the seven additional split ratios and exports one Excel file per ratio.

## Notes on Model Architectures

- **Autoencoder + classifier:** a 3-layer encoder (256 → 128 → 64) trained to reconstruct input spectra (MSE loss), followed by a separately trained feed-forward classifier (128 → 64 → 32 → 3, softmax) on the learned 64-dimensional embedding.
- **1D-CNN:** two convolutional blocks (32, 64 filters, kernel size 5) with max-pooling, followed by a fully connected head.
- **1D-ResNet:** three residual blocks (32 → 64 → 128 channels) with skip connections and stride-2 downsampling, global average pooling, and a linear classification head.
- **Transformer encoder:** a single-layer `TransformerEncoder` (d_model = 64, 4 attention heads, feed-forward dim = 128) operating on the whole spectrum as a single token, followed by a 2-layer classification head. Robustness here is assessed via 5-fold stratified cross-validation rather than the split-ratio sweep used in the other three sections.



## Contact

**Karthikeyan Meenatchisundaram**
Postdoctoral Fellow, HiSpec Lab, Department of Electronics and Communication Engineering, SRM University–AP
