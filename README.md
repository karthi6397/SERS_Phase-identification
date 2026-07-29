# Microalgal Biomass Forecasting: A Comparative Modeling Framework

This repository contains a comparative implementation of seven forecasting approaches — classical statistical, machine learning, deep learning, and physics-informed — applied to time-series prediction of microalgal biomass concentration (dry cell weight, mg/L) in a photobioreactor/cultivation setting.

## Overview

Accurate short- and medium-horizon forecasting of biomass growth is central to intelligent process control in microalgal cultivation. This codebase benchmarks the following models on the same biomass time series to compare predictive accuracy, generalization, and interpretability trade-offs:

| # | Model | Category | Notes |
|---|-------|----------|-------|
| 1 | ARIMA | Classical statistical | Order (1,1,3); includes iterative multi-step rolling forecast variant |
| 2 | LSTM | Deep learning (RNN) | Single-layer LSTM on min-max scaled univariate sequence |
| 3 | XGBoost | Gradient-boosted trees | Lag features, rolling mean, and FFT-derived seasonality features; hyperparameters tuned via `GridSearchCV` |
| 4 | Prophet | Additive statistical | Includes changepoint tuning and time-series cross-validation (`cross_validation`, `performance_metrics`) |
| 5 | PINN | Physics-informed deep learning | Neural network constrained by a Monod-kinetics ODE residual (learnable `mu_max`, `K_s`) |
| 6 | RVFL | Randomized neural network | Custom NumPy implementation (random-weight hidden layer + closed-form ridge output layer) |
| 7 | Transformer | Deep learning (attention) | Implemented via [Darts](https://unit8co.github.io/darts/) `TransformerModel` |

A Friedman test is included to assess whether replicate measurements (e.g., biological/technical replicates of dry cell weight) differ significantly, which is relevant for validating the reliability of the ground-truth data feeding the forecasts.

## Repository Structure

```
.
├── Python_codes.ipynb        # Main notebook: all seven models + evaluation + Friedman test
├── data/                     # (add) input CSVs — see "Data" section
├── results/                  # (add) output figures and prediction/residual spreadsheets
├── requirements.txt          # (add) Python dependencies
└── README.md
```

> The notebook is organized into clearly labeled sections (`### ARIMA`, `### LSTM`, `### XGBoost`, `### Prophet`, `### PINN`, `### RVFL`, `### Transformers model`, `### Sensitivity`) using raw markdown cells, followed by a `## Supporting files` block with Prophet cross-validation and the Friedman test.

## Data

The notebook expects three input files (paths are currently hardcoded to a local Windows environment and **should be updated to relative paths before use**):

| File | Used by | Expected columns |
|------|---------|-------------------|
| `Primary data_Timeseries.csv` | ARIMA, LSTM, XGBoost, PINN, RVFL, Transformer | `Hours`, `Biomass` (mg/L) |
| `prophet.csv` | Prophet | `Days` (datetime), `Biomass` |
| `stat.csv` | Friedman test | `DCW value 1`, `DCW value 2`, `DCW value 3` (replicate measurements) |

Missing values are handled via linear interpolation (`df.interpolate(method='linear', axis=0)`) prior to modeling. Raw data is not included in this repository — add your own files under `data/` and update the file paths in the notebook accordingly.

## Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Core dependencies

```
numpy
pandas
matplotlib
seaborn
scikit-learn
scipy
statsmodels
prophet
xgboost
tensorflow          # for keras LSTM
torch                # for the PINN
darts                # for the Transformer model
```

Pin exact versions in `requirements.txt` once your environment is finalized, to keep results reproducible — Prophet, Darts, and TensorFlow/PyTorch version mismatches are a common source of silent behavior changes.

## Usage

1. Place the three input CSVs under `data/` and update the `pd.read_csv(...)` paths in each model section of `Python_codes.ipynb`.
2. Run the notebook top to bottom, or execute individual model sections independently — each section (ARIMA, LSTM, XGBoost, Prophet, PINN, RVFL, Transformer) is self-contained after the shared import/preprocessing cells.
3. Each model section:
   - Fits the model on a train/test split (last 195 hours held out as the test set in most sections),
   - Generates forecast plots (saved as `.png`, 400 dpi),
   - Computes MAE, RMSE, and MAPE on train and test sets,
   - Exports actual-vs-predicted values and residuals to `.xlsx`.
4. Update the `plt.savefig(...)` and `.to_excel(...)` paths to point to a local `results/` directory.

## Evaluation Metrics

All models are assessed using:
- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error
- **MAPE** — Mean Absolute Percentage Error (custom implementation, since `sklearn`'s MAPE was not used directly)

Train and test metrics are reported separately to flag overfitting.

## Statistical Validation

The Friedman test (`scipy.stats.friedmanchisquare`) is applied to biological/technical replicate dry cell weight values (`stat.csv`) to test for significant differences among replicates (α = 0.05), supporting the reliability of the biomass measurements used to train and evaluate the forecasting models.

## Known Limitations / To-Do Before Publishing the Repository

- **Hardcoded local paths**: all `pd.read_csv` and `plt.savefig` calls reference local Windows directories (e.g., `C:\Users\karth\...`) and must be replaced with relative paths (e.g., `data/...`, `results/...`) for the repository to run on another machine.
- **Duplicate import cells**: the import/preprocessing block appears twice at the top of the notebook; consider consolidating.
- **No `requirements.txt` / `environment.yml` yet**: add one with pinned versions for reproducibility.
- **Random seeds**: seeds are set for XGBoost/TensorFlow/NumPy in the XGBoost section but not uniformly across all models (e.g., RVFL, LSTM) — consider standardizing for full reproducibility across models.
- **RVFL and PINN train/test evaluation** use a single split each; k-fold or walk-forward cross-validation would strengthen the comparison, consistent with the cross-validation already used for Prophet.


## Contact

**Karthikeyan Meenatchisundaram**
Postdoctoral Fellow, HiSpec Lab, Department of Electronics and Communication Engineering, SRM University–AP
