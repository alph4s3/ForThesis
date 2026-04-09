# Extreme Heat Prediction Using Impact-Centric Variables
### Bachelor of Science in Computer Science — Thesis Project

---

## Overview

This repository contains the complete Python implementation for the thesis:

> **"Enhancing Extreme Heat Prediction Using Impact-Centric Variables
> in Machine Learning Models"**

The system trains and compares two LSTM models:
- **Baseline LSTM** — conventional meteorological inputs only
  (temperature, humidity, wind speed)
- **Impact-Centric LSTM** — adds UHI intensity, wet-bulb temperature,
  PM 2.5, and heat index

It includes SHAP-based feature explainability and an alert dispatch module.

---

## Architecture (9-Module Design)

```
WeatherRecord ──► DataPipeline ──► LSTMModel ──► EvaluationModule
ImpactRecord  ──►                              ──► XAIModule
HeatEvent     ──► EvaluationModule ──────────► AlertModule
                  DataFetchModule (live feeds)
```

| Module | File | Purpose |
|---|---|---|
| WeatherRecord | `entities.py` | Meteorological observation |
| ImpactRecord | `entities.py` | UHI, wet-bulb, PM2.5, heat index |
| HeatEvent | `entities.py` | Classified heat event + alerts |
| DataPipeline | `data_pipeline.py` | Load → merge → sequence → split → scale |
| LSTMModel | `lstm_model.py` | Baseline & impact-centric LSTM |
| EvaluationModule | `evaluation_module.py` | RMSE, MAE, F1, AUC, comparison |
| XAIModule | `xai_module.py` | SHAP importance + reports |
| AlertModule | `alert_data_modules.py` | Threshold alerts + logging |
| DataFetchModule | `alert_data_modules.py` | Live API data retrieval |

---

## Project File Structure

```
heat_prediction/
├── entities.py              # WeatherRecord, ImpactRecord, HeatEvent
├── data_generator.py        # Synthetic Davao City dataset
├── data_pipeline.py         # DataPipeline class
├── lstm_model.py            # LSTMModel (baseline + impact-centric)
├── evaluation_module.py     # EvaluationModule
├── xai_module.py            # XAIModule (SHAP)
├── alert_data_modules.py    # AlertModule + DataFetchModule
├── main.py                  # Full experiment runner
├── requirements.txt
└── README.md
```

Output files (generated on run):
```
outputs/
├── baseline_final.weights.h5
├── impact_final.weights.h5
├── model_comparison.csv
├── shap_importance.png
├── xai_report.txt
└── alert_log.json
data/
├── weather_records.csv
└── impact_records.csv
```

---

## Setup & Installation

### 1. Prerequisites
- Python 3.10 or higher
- pip package manager

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU acceleration (optional):** Replace `tensorflow` with
> `tensorflow[and-cuda]` for NVIDIA GPU support.

---

## Running the Experiment

### Full experiment (recommended)

```bash
python main.py
```

### Custom options

```bash
# 2 years of data, 50 epochs, 48-hour lookback
python main.py --days 730 --epochs 50 --seq 48

# Quick debug run (no SHAP, fewer epochs)
python main.py --days 90 --epochs 5 --no-shap

# Custom output directories
python main.py --outdir results/ --datadir datasets/
```

### Available flags

| Flag | Default | Description |
|---|---|---|
| `--days` | 365 | Synthetic simulation days |
| `--epochs` | 30 | Max LSTM training epochs |
| `--seq` | 24 | Lookback window (hours) |
| `--batch` | 64 | Mini-batch size |
| `--no-shap` | False | Skip SHAP (faster debugging) |
| `--outdir` | `outputs/` | Output directory |
| `--datadir` | `data/` | Data directory |

---

## Expected Runtime (CPU)

| Configuration | Approx. Time |
|---|---|
| 1 year, 30 epochs, no SHAP | 5–10 min |
| 1 year, 30 epochs, with SHAP | 10–20 min |
| 2 years, 50 epochs, with SHAP | 20–40 min |

---

## Real Data Sources

To replace the synthetic generator with real observational data,
obtain data from the following sources and format them as CSV files
matching `weather_records.csv` and `impact_records.csv` schemas.

| Variable | Source | URL |
|---|---|---|
| Temperature, Humidity, Wind | PAGASA | pagasa.dost.gov.ph |
| Temperature, Humidity, Wind | ERA5 (ECMWF) | cds.climate.copernicus.eu |
| Urban Heat Island (LST) | MODIS via GEE | earthengine.google.com |
| Urban Heat Island (LST) | USGS EarthExplorer | earthexplorer.usgs.gov |
| PM 2.5 Air Quality | OpenAQ | openaq.org |
| PM 2.5 Air Quality | AirVisual | iqair.com/air-pollution-data-api |
| Solar Radiation | NASA POWER | power.larc.nasa.gov |

---

## Methodology Notes

### Time-Series Cross-Validation
Data is split **chronologically** (70% train / 15% val / 15% test) to
prevent look-ahead bias. No random shuffling is applied.

### Class Imbalance Handling
Extreme heat events are rare (~5–15% of observations). The system
applies **inverse-frequency class weights** during training.
To use SMOTE instead, uncomment `imbalanced-learn` in requirements.txt.

### SHAP Explainability
A `KernelExplainer` is used (model-agnostic, compatible with LSTM/RNN).
It operates on flattened sequences; SHAP values are then averaged
across time steps to produce per-feature importance scores.

---


```
John Wallace Aceres | Danna Mishna Lledo
Enhancing Extreme Heat Prediction Using
Impact-Centric Variables in Machine Learning Models.
Bachelor of Science in Computer Science Thesis,
Ateneo de Zamboanga University, Zamboanga City,  Zamboanga Del Sur, Philippines.
```

---

## License
This project is developed for academic purposes.
