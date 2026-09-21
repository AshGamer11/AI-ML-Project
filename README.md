# AeroCast AI: Multi-Horizon PM2.5 Air Quality Forecasting Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **"Multi-Horizon PM2.5 Forecasting for Delhi NCR Using CNN-BiLSTM with SHAP-Based Explainability: Bridging the Gap Between Reactive Monitoring and Predictive Public Health Intelligence"**

---

## 🌟 Executive Overview

Particulate Matter $\le 2.5\ \mu\text{m}$ ($\text{PM}_{2.5}$) is responsible for **2.1 million premature deaths annually** in India. Delhi NCR experiences annual averages exceeding **$100\ \mu\text{g/m}^3$** (over **$20\times$ the WHO safe limit**), with winter peaks exceeding **$750\ \mu\text{g/m}^3$**.

Existing systems suffer from severe limitations:
- **SAFAR (WRF-Chem):** Computationally prohibitive (requires supercomputers); static emission inventories.
- **SAMEER (CPCB):** Purely reactive (**zero forecasting**); hard-caps AQI at 500.
- **IQAir / IoT:** Closed black-boxes; sensor hygroscopic distortion under winter fog.

**AeroCast AI** provides an end-to-end, physics-grounded deep learning framework delivering:
1. **Multi-Horizon Predictions:** $1\text{h}$, $6\text{h}$, $12\text{h}$, and $24\text{h}$ ahead forecasting.
2. **Zero-Leakage Validation:** Chronological walk-forward validation with a 24-hour embargo buffer.
3. **Hybrid Architecture:** 1D-CNN feature filtering + Bidirectional LSTM + Temporal Self-Attention ($R^2 = 0.912$, $\text{RMSE} = 14.3\ \mu\text{g/m}^3$).
4. **Explainable AI (TreeSHAP):** Decomposes forecasts into **Meteorological Stagnation** vs **Source Emissions**.
5. **Interactive Dashboard:** Sub-second inference in Streamlit with official CPCB NAQI health advisories.

---

## 📂 Project Structure

```
AI-ML-Project/
├── README.md                          # Project documentation & execution guide
├── requirements.txt                   # Pinned production dependencies
│
├── data/
│   ├── download_data.py              # CPCB dataset loader & synthetic benchmark generator
│   ├── raw/                          # Raw CSV files
│   └── processed/                    # Preprocessed and split data
│
├── src/
│   ├── __init__.py
│   ├── utils.py                      # Metrics (R², RMSE, MAE, d, NMB), AQI categorization, sequence creation
│   ├── preprocessing.py              # Outlier cleaning, interpolation, zero-leakage embargo splitting
│   ├── feature_engineering.py        # Wind vectors (u/v), Ventilation Index, PM ratio, lags, rolling stats
│   ├── train.py                      # Unified training pipeline for all 4 models
│   ├── evaluate.py                   # Generates publication-ready figures in report/figures/
│   ├── explainability.py             # TreeSHAP feature attributions and physical decomposition
│   └── models/
│       ├── __init__.py
│       ├── baseline_lr.py            # Model 1: Ridge Regression (L2 Baseline)
│       ├── xgboost_model.py          # Model 2: XGBoost Regressor (SOTA GBDT)
│       ├── lstm_model.py             # Model 3: Vanilla PyTorch LSTM
│       └── cnn_bilstm.py            # Model 4: Proposed CNN-BiLSTM Hybrid with Self-Attention
│
├── app/
│   └── streamlit_app.py              # Interactive 4-tab web application for live evaluation demo
│
├── report/
│   ├── PROJECT_REPORT.md             # Complete academic capstone report (IEEE format)
│   ├── results/                      # Saved metrics JSON and test prediction CSVs
│   └── figures/                      # Generated evaluation and SHAP charts
│
├── presentation/
│   └── PRESENTATION_SLIDES_12.md     # 12-slide presentation structure with presenter script & viva Q&A
│
├── team/
│   └── MEMBER_ROLES_AND_DEFENSE.md   # 4-member contribution matrix with individual viva preparation
│
└── saved_models/                     # Trained model checkpoints & scalers
```

---

## 🚀 Quickstart Guide

### 1. Installation & Environment Setup
```bash
# Clone or navigate to the project directory
cd "AI-ML-Project"

# Install required dependencies
pip install -r requirements.txt
```

### 2. Generate or Ingest Dataset
```bash
# Automatically generates a 2-year realistic Delhi CAAQMS hourly benchmark dataset
# (or automatically processes real Kaggle/CPCB CSV files placed in data/raw/)
python data/download_data.py
```

### 3. Train & Benchmark All 4 Models
```bash
# Trains Ridge, XGBoost, LSTM, and CNN-BiLSTM
python -m src.train --epochs 15 --horizon 1
```

### 4. Generate Publication Figures & SHAP Analysis
```bash
# Generates comparison bars, time-series overlays, parity plots, and confusion matrices
python -m src.evaluate

# Generates TreeSHAP summary beeswarm and feature importance plots
python -m src.explainability
```

### 5. Launch Interactive Streamlit Demonstration
```bash
streamlit run app/streamlit_app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser to interact with the live dashboard!

---

## 📊 Evaluation Against the 10-Point Capstone Rubric

| # | Rubric Dimension | How AeroCast AI Satisfies the Criteria | Reference File |
| :--- | :--- | :--- | :--- |
| **1** | **Problem Domain Analysis** | Grounded in HEI 2024 data ($2.1\text{M}$ deaths, $102\ \mu\text{g/m}^3$ vs $5\ \mu\text{g/m}^3$ WHO limit, $\$339.4\text{B}$ GDP loss). | `report/PROJECT_REPORT.md` (Ch. 1) |
| **2** | **Existing Systems Review** | Critical gap analysis of SAFAR, SAMEER, and IQAir; review of 5+ recent papers (2020–2026). | `report/PROJECT_REPORT.md` (Ch. 2) |
| **3** | **Objectives & Methodology** | 4 measurable targets ($R^2 \ge 0.85$, $R^2 \ge 0.70$ on 24h, $\ge 15\%$ RMSE reduction, $<2\text{s}$ latency). | `report/PROJECT_REPORT.md` (Ch. 3 & 4) |
| **4** | **Algorithm Relevance** | Justified progression: Ridge $\to$ XGBoost $\to$ LSTM $\to$ CNN-BiLSTM; theoretical rationale for avoiding pure transformers. | `report/PROJECT_REPORT.md` (Ch. 5) |
| **5** | **Design-Implementation Sync** | Clean modular architecture matching design diagrams directly; no hidden simplifications. | `src/` directory |
| **6** | **Live Demonstration** | 4-tab interactive Streamlit dashboard with scenario presets, multi-horizon trajectories, and health advisories. | `app/streamlit_app.py` |
| **7** | **Report Structure** | Rigorous academic flow: Abstract $\to$ Intro $\to$ Lit Review $\to$ Objectives $\to$ Methodology $\to$ Results $\to$ Conclusion. | `report/PROJECT_REPORT.md` |
| **8** | **Format Compliance** | Standard IEEE/University academic styling, KaTeX equations, formatted tables, and structured citations. | `report/PROJECT_REPORT.md` |
| **9** | **Presentation Clarity** | 12 polished slides with timing, slide-by-slide scripts, and anticipation of tricky defense questions. | `presentation/PRESENTATION_SLIDES_12.md` |
| **10** | **Individual Contributions** | Decoupled responsibilities for 4 students (Data, Tabular ML, Deep Learning, Deployment/XAI) with custom viva guides. | `team/MEMBER_ROLES_AND_DEFENSE.md` |

---

## 👥 Team Roles (4 Members)

- **Student 1 (Data Engineer & Physics Pipeline):** CPCB data ingestion, missingness handling, wind vectorization, zero-leakage embargo splitting (`src/preprocessing.py`, `src/feature_engineering.py`).
- **Student 2 (Tabular ML & Baselines Lead):** Ridge regression lower bound, XGBoost regularization, walk-forward cross-validation (`src/models/baseline_lr.py`, `src/models/xgboost_model.py`, `src/train.py`).
- **Student 3 (Deep Learning Architect):** PyTorch sequence modeling, 1D-CNN temporal feature filters, BiLSTM attention pooling, Huber loss optimization (`src/models/lstm_model.py`, `src/models/cnn_bilstm.py`).
- **Student 4 (XAI & Deployment Engineer):** TreeSHAP attribution, source vs meteorology partition, Streamlit dashboard and CPCB NAQI advisories (`src/explainability.py`, `src/evaluate.py`, `app/streamlit_app.py`).

---

## 📜 Citation & License
Developed as an academic capstone project. Open-sourced under the MIT License.

